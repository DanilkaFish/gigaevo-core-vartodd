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

INITIAL_RANK = 567
TAIL_RANK = 450
SEEDS = [37, 38]
PSO_POP = 12
N_EVAL = 600


def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))


def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0 / beta) * np.log(np.exp(-beta * (xs - m)).sum()))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Evaluator(BaseEvaluator):
    """Lean beam-2 descent with a real TODD lifeline only below rank 1300.

    Above 1300 the descent is pure light TOHPE (keep=2, one_hot="all") —
    prior runs show TODD buys nothing there. Below 1300 TODD switches on
    with full one-hot recall, keep=4/reserve=2 and actions_per_bucket=3,
    z capped at ~60k: prior terminal bands starved (apz~0.01) using
    *sampled* y recall over huge z research; this instead retains several
    y per researched bucket so accepted actions survive to the merge.
    Tests whether a retention-rich tail TODD beats the pure-TOHPE tail."""

    def policy_mapping(self):
        pool_score = ExplorationScore([self.map_par(_w_tanh) for _ in range(5)], pow=1)
        final_weights = [self.map_par(_w_tanh) for _ in range(6)]
        final_centers = [self.map_par(sigmoid) for _ in range(6)]
        self.set_scores(PolicyScores(exploration=pool_score, final=FinalizationScore(final_weights, final_centers, pow=1)))

        dense_samples = 8 + int(48 * self.map_par(sigmoid))
        tail_min_buckets = 256 + int(1792 * self.map_par(sigmoid))
        tail_z_cap = 20_000 + int(40_000 * self.map_par(sigmoid))

        sampling = SamplingBudget(one_hot="all", sparse=0, dense=dense_samples, sparse_max_weight=2)
        self.set_action_selection(ActionSelection(beamwidth=2, mode="softmax", temperature=0.2))
        self.set_action_pool(ActionPool(final_size=12))
        self.set_tohpe_search(TohpeSearch(sampling=sampling, pool=SourcePool(keep=12, reserve=0), z_choices=2))
        self.set_todd_search(
            [INITIAL_RANK, TAIL_RANK],
            [
                ToddSearch(
                    sampling=SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=2),
                    pool=SourcePool(keep=0, reserve=0),
                    actions_per_bucket=1,
                ),
                ToddSearch(
                    sampling=sampling,
                    pool=SourcePool(keep=4, reserve=2),
                    actions_per_bucket=3,
                    buckets=ZBucketSearch(
                        min_buckets=tail_min_buckets,
                        max_buckets=tail_z_cap,
                        limit_bucket=tail_z_cap,
                    ),
                ),
            ],
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
    minimize(Problem(fun), algorithm, termination=("n_eval", N_EVAL), seed=37, verbose=False)
    return fun.get_best()
