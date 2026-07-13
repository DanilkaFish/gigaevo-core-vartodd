from typing import Iterable

import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
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

SEEDS = [17, 18]
DE_POP = 16
N_EVAL = 144
TODD_RESERVE_MAX = 3


def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))


def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0 / beta) * np.log(np.exp(-beta * (xs - m)).sum()))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Evaluator(BaseEvaluator):
    """Lean stochastic full-length descent tuned by differential evolution.

    Same winning shape as the full_pso pair — constant profile, pow=1,
    one_hot="all" generation with tiny retention, TODD keep=0 reserve-only,
    beamwidth=2 softmax — but a hotter temperature (0.3), TOHPE keep=3, and
    DE/rand/1/bin instead of PSO. DE's difference vectors handle the noisy,
    multimodal objective differently than swarm attraction; this probes
    whether the lean shape's gains are optimizer-family robust."""

    def policy_mapping(self):
        pool_score = ExplorationScore([self.map_par(_w_tanh) for _ in range(5)], pow=1)
        final_weights = [self.map_par(_w_tanh) for _ in range(6)]
        final_centers = [self.map_par(sigmoid) for _ in range(6)]
        self.set_scores(PolicyScores(exploration=pool_score, final=FinalizationScore(final_weights, final_centers, pow=1)))

        z_budget = self.map_par(sigmoid)
        z_research_budget = self.map_par(sigmoid)
        todd_reserve_budget = self.map_par(sigmoid)
        dense_samples = 8 + int(40 * self.map_par(sigmoid))
        z_reserve_cap = 50_000 + int(150_000 * z_research_budget)
        z_hard_cap = max(z_reserve_cap, 100_000 + int(400_000 * z_research_budget))
        todd_reserve = min(TODD_RESERVE_MAX, int(TODD_RESERVE_MAX * todd_reserve_budget))

        sampling = SamplingBudget(one_hot="all", sparse=0, dense=dense_samples, sparse_max_weight=2)
        self.set_action_selection(ActionSelection(beamwidth=2, mode="softmax", temperature=0.3))
        self.set_action_pool(ActionPool(final_size=12))
        self.set_tohpe_search(TohpeSearch(sampling=sampling, pool=SourcePool(keep=8, reserve=0), z_choices=2))
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
    algorithm = DE(pop_size=DE_POP, variant="DE/rand/1/bin", CR=0.9, F=0.6)
    minimize(Problem(fun), algorithm, termination=("n_eval", N_EVAL), seed=17, verbose=False)
    return fun.get_best()
