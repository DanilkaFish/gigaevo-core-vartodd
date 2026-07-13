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


INITIAL_RANK = 1701
MID_RANK = 1515
LOW_RANK = 1365
SEEDS = [33, 34]
SCHEDULE = [INITIAL_RANK, MID_RANK, LOW_RANK]


def signed(x: float, scale: float = 2.8) -> float:
    return float(scale * np.tanh(float(x) / 1.25))


def unit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20.0, 20.0)))))


def score_rank(ranks: list[int]) -> float:
    if not ranks:
        return float(INITIAL_RANK)
    values = np.asarray(ranks, dtype=float)
    return float(values.min() + 0.01 * values.mean())


class Evaluator(BaseEvaluator):
    """TOHPE ladder plus tiny TODD scouts. The score surface is fully weighted
    (5 exploration + 6 final weights), with separate centers for dim/yw/zw and
    tohpe. The TODD side is intentionally capped at small finite limits so this
    explores cheap rare-z assistance rather than heavy tail grinding."""

    def policy_mapping(self):
        explore_w = [self.map_par(signed) for _ in range(5)]
        final_w = [self.map_par(signed) for _ in range(6)]
        dim_center = self.map_par(unit)
        y_center = self.map_par(unit)
        z_center = 0.0
        tohpe_center = 0.0
        tohpe_sparse = self.map_par(lambda x: 12 + int(92 * unit(x)))
        tohpe_dense = self.map_par(lambda x: 8 + int(80 * unit(x)))
        tohpe_keep = self.map_par(lambda x: 18 + int(46 * unit(x)))
        z_choices = self.map_par(lambda x: 4 + int(12 * unit(x)))
        scout_cap = self.map_par(lambda x: 5120 + int(153600 * unit(x)))
        scout_keep = self.map_par(lambda x: 4 + int(12 * unit(x)))
        pool_size = self.map_par(lambda x: 30 + int(42 * unit(x)))

        self.set_scores(
            PolicyScores(
                ExplorationScore(
                    explore_w,
                    centers=[0.0, dim_center, 0.0, y_center, z_center],
                    pow=1,
                ),
                FinalizationScore(
                    final_w,
                    centers=[0.0, dim_center, 0.0, y_center, z_center, tohpe_center],
                    pow=1,
                ),
            )
        )
        self.set_action_selection(
            SCHEDULE,
            [
                ActionSelection(beamwidth=1, mode="best", temperature=0.0),
                ActionSelection(beamwidth=1, mode="best", temperature=0.0),
                ActionSelection(beamwidth=2, mode="best", temperature=0.0),
            ],
        )
        self.set_action_pool(
            SCHEDULE,
            [
                ActionPool(final_size=pool_size),
                ActionPool(final_size=pool_size + 8),
                ActionPool(final_size=pool_size + 16),
            ],
        )
        self.set_tohpe_search(
            SCHEDULE,
            [
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=tohpe_sparse, dense=tohpe_dense, sparse_max_weight=5),
                    SourcePool(keep=tohpe_keep, reserve=1),
                    z_choices=z_choices,
                ),
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=tohpe_sparse // 2, dense=tohpe_dense // 2, sparse_max_weight=4),
                    SourcePool(keep=max(10, tohpe_keep // 2), reserve=1),
                    z_choices=max(4, z_choices // 2),
                ),
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=20, dense=12, sparse_max_weight=3),
                    SourcePool(keep=10, reserve=0),
                    z_choices=4,
                ),
            ],
        )
        self.set_todd_search(
            SCHEDULE,
            [
                ToddSearch(
                    SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=2),
                    SourcePool(keep=0, reserve=0),
                    actions_per_bucket=1,
                ),
                ToddSearch(
                    SamplingBudget(one_hot=24, sparse=4, dense=0, sparse_max_weight=2),
                    SourcePool(keep=scout_keep, reserve=1),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=64, max_buckets=scout_cap, limit_bucket=scout_cap),
                ),
                ToddSearch(
                    SamplingBudget(one_hot=32, sparse=6, dense=0, sparse_max_weight=2),
                    SourcePool(keep=scout_keep + 6, reserve=2),
                    actions_per_bucket=2,
                    buckets=ZBucketSearch(min_buckets=128, max_buckets=2 * scout_cap, limit_bucket=2 * scout_cap),
                ),
            ],
        )

    def __call__(self, params: Iterable[float]) -> float:
        return score_rank(self.run(params, SEEDS))


class Problem(ElementwiseProblem):
    def __init__(self, fun: Evaluator):
        n = len(fun.extract_active())
        super().__init__(n_var=n, n_obj=1, xl=np.full(n, -2.2), xu=np.full(n, 2.2))
        self.fun = fun

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.fun(np.asarray(x, dtype=float))


def optimize(fun: Evaluator, n_eval: int, seed: int = 33) -> np.ndarray:
    active = fun.extract_active()
    if len(active) == 0:
        fun.run([], SEEDS)
        return active
    algorithm = PSO(pop_size=10, w=0.58, c1=0.7, c2=0.35, adaptive=False)
    result = minimize(Problem(fun), algorithm, termination=("n_eval", n_eval), seed=seed, verbose=False)
    return np.asarray(result.X if result.X is not None else fun.extract_active(), dtype=float)


def restart(fun: BaseEvaluator, rank_thr: int, xopt: np.ndarray | None = None) -> bool:
    return bool(fun.best_paths and fun.set_up_new_init(0, rank_thr=rank_thr, xopt=xopt) is not None)


def entrypoint():
    fun = Evaluator(path_name="init")
    xopt = optimize(fun, n_eval=28)
    for i, margin in enumerate([55, 20]):
        threshold = fun.best_rank + margin
        if restart(fun, threshold, xopt=xopt):
            xopt = optimize(fun, n_eval=8, seed=34 + i)
    return fun.get_best()
