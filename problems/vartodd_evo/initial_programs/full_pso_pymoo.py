import random
from typing import Iterable

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from helper import (
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    ExplorationScore,
    FinalizationScore,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
)

np.random.seed(43)
random.seed(41)

LEGACY_ONE_HOT_CAP = 1 << 20
PSO_POP = 16
N_EVAL = 144
Z_RESERVE_CAP_MIN = 50_000
Z_RESERVE_CAP_SPAN = 150_000
Z_HARD_CAP_MIN = 100_000
Z_HARD_CAP_SPAN = 400_000
TODD_RESERVE_MAX = 3


def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))


def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0 / beta) * np.log(np.exp(-beta * (xs - m)).sum()))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def budget_int(budget: float, minimum: int, span: int) -> int:
    return int(minimum + int(span * float(budget)))


class Evaluator(BaseEvaluator):
    """Pymoo twin of the lean full-length stochastic descent tuner.

    Identical policy shape to the pyswarms port: one constant profile,
    pow=1, one_hot="all" generation, TOHPE keep=2/z_choices=2, TODD keep=0
    (reserve-only lifeline), final_size=12, beamwidth=2 softmax@0.2 for the
    entire descent. Only the optimizer backend differs (pymoo PSO), so the
    pair isolates the optimizer-library variable."""

    seeds = [random.randint(1, 10000) for _ in range(2)]

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float):
        # Legacy C++ behavior: every one-hot basis vector is tried; the
        # fraction parameter is consumed only for x0 layout compatibility.
        _ = one_hot_fraction
        return [LEGACY_ONE_HOT_CAP, 0, max(0, int(sample_count))]

    def policy_mapping(self):
        ranks = [0]
        pool_score = ExplorationScore([self.map_par(_w_tanh) for _ in range(5)], pow=1)
        final_weights = [self.map_par(_w_tanh) for _ in range(6)]
        final_centers = [self.map_par(sigmoid) for _ in range(6)]
        final_score = FinalizationScore(final_weights, final_centers, pow=1)
        self.set_scores(ranks, [PolicyScores(exploration=pool_score, final=final_score)])

        z_budget = self.map_par(sigmoid)
        z_research_budget = self.map_par(sigmoid)
        todd_reserve_budget = self.map_par(sigmoid)
        sample_budget = self.map_par(sigmoid)
        one_hot_fraction = 0.1 + 0.5 * self.map_par(sigmoid)
        sample_count = 8 + int(48 * sample_budget)
        sample_caps = self._sample_caps(sample_count, one_hot_fraction)
        z_reserve_cap = budget_int(z_research_budget, Z_RESERVE_CAP_MIN, Z_RESERVE_CAP_SPAN)
        z_hard_cap = max(z_reserve_cap, budget_int(z_research_budget, Z_HARD_CAP_MIN, Z_HARD_CAP_SPAN))
        todd_reserve = min(TODD_RESERVE_MAX, int(TODD_RESERVE_MAX * todd_reserve_budget))

        sampling = SamplingBudget(one_hot="all", sparse=sample_caps[1], dense=sample_caps[2], sparse_max_weight=2)
        self.set_action_selection(ActionSelection(beamwidth=2, mode="softmax", temperature=0.2))
        self.set_action_pool(ActionPool(final_size=12))
        self.set_tohpe_search(TohpeSearch(sampling=sampling, pool=SourcePool(keep=2, reserve=0), z_choices=2))
        self.set_todd_search(
            ToddSearch(
                sampling=sampling,
                pool=SourcePool(keep=0, reserve=todd_reserve),
                actions_per_bucket=1,
                buckets=ZBucketSearch(
                    min_buckets=10 + int(250 * z_budget),
                    max_buckets=z_reserve_cap,
                    limit_bucket=z_hard_cap,
                ),
            )
        )

    def __call__(self, params: Iterable[float]) -> float:
        tcounts = self.run(params, self.seeds)
        bestish = softmin(tcounts, beta=6.0)
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return bestish + 0.02 * spread


class Problem(ElementwiseProblem):
    def __init__(self, fun: Evaluator):
        n = len(fun.extract_active())
        super().__init__(n_var=n, n_obj=1, xl=np.full(n, -1.0), xu=np.full(n, 1.0))
        self.fun = fun

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.fun(np.asarray(x, dtype=float))


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=500)
    if len(fun.extract_active()) == 0:
        fun.run([], fun.seeds)
        return fun.get_best()
    algorithm = PSO(pop_size=PSO_POP, w=0.7, c1=0.4, c2=0.4, adaptive=False)
    minimize(Problem(fun), algorithm, termination=("n_eval", N_EVAL), seed=7, verbose=False)
    return fun.get_best()
