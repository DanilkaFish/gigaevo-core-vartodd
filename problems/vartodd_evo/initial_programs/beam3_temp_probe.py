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

SEEDS = [23, 24]
PSO_POP = 12
N_EVAL = 128


def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))


def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0 / beta) * np.log(np.exp(-beta * (xs - m)).sum()))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Evaluator(BaseEvaluator):
    """Wider, hotter stochastic beam over the lean descent shape.

    beamwidth=3 with softmax@0.35 and a slightly larger merged pool (16)
    trades evaluation count (96 instead of 144-160; each descent is ~1.5x
    the beam-2 cost) for more within-descent path branching. Probes whether
    extra width and heat find better mid-band branches than the beam-2
    default, or whether beam-2's cheap decorrelated volume wins."""

    def policy_mapping(self):
        pool_score = ExplorationScore([self.map_par(_w_tanh) for _ in range(5)], pow=1)
        final_weights = [self.map_par(_w_tanh) for _ in range(6)]
        final_centers = [self.map_par(sigmoid) for _ in range(6)]
        self.set_scores(PolicyScores(exploration=pool_score, final=FinalizationScore(final_weights, final_centers, pow=1)))

        z_budget = self.map_par(sigmoid)
        z_research_budget = self.map_par(sigmoid)
        todd_reserve = min(3, int(3 * self.map_par(sigmoid)))
        dense_samples = 8 + int(48 * self.map_par(sigmoid))
        z_reserve_cap = 50_000 + int(150_000 * z_research_budget)
        z_hard_cap = max(z_reserve_cap, 100_000 + int(400_000 * z_research_budget))

        sampling = SamplingBudget(one_hot="all", sparse=0, dense=dense_samples, sparse_max_weight=2)
        self.set_action_selection(ActionSelection(beamwidth=3, mode="softmax", temperature=0.35))
        self.set_action_pool(ActionPool(final_size=16))
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
        tcounts = self.run(params, SEEDS)
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
        fun.run([], SEEDS)
        return fun.get_best()
    algorithm = PSO(pop_size=PSO_POP, w=0.7, c1=0.4, c2=0.4, adaptive=False)
    minimize(Problem(fun), algorithm, termination=("n_eval", N_EVAL), seed=23, verbose=False)
    return fun.get_best()
