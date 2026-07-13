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


INITIAL_RANK = 567
SWITCH_RANK = 490
SEEDS = [23, 24]
SCHEDULE = [INITIAL_RANK, SWITCH_RANK]


def signed(x: float, scale: float = 3.4) -> float:
    return float(scale * np.tanh(float(x) / 1.5))


def unit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20.0, 20.0)))))


def score_rank(ranks: list[int]) -> float:
    if not ranks:
        return float(INITIAL_RANK)
    values = np.asarray(ranks, dtype=float)
    return float(values.min() + 0.015 * values.std())


class Evaluator(BaseEvaluator):
    """Hard tail split. The upper band is a TOHPE-heavy descent; below
    `SWITCH_RANK`, TODD gets a bounded tail budget. The earlier full bucket-space
    tail was still running past the 10k-second target, so this version preserves
    the split strategy but caps bucket research and trims optimizer restarts."""

    def policy_mapping(self):
        explore_w = [self.map_par(signed) for _ in range(5)]
        final_w = [self.map_par(signed) for _ in range(6)]
        dim_center = self.map_par(unit)
        y_center = self.map_par(unit)
        z_center = self.map_par(unit)
        tail_min = self.map_par(lambda x: 256 + int(2048 * unit(x)))
        tail_keep = self.map_par(lambda x: 12 + int(24 * unit(x)))
        tail_reserve_frac = self.map_par(lambda x: 0.12 + 0.24 * unit(x))
        tail_apb = self.map_par(lambda x: 1 + int(2 * unit(x)))
        early_keep = self.map_par(lambda x: 16 + int(28 * unit(x)))
        pool_size = self.map_par(lambda x: 28 + int(28 * unit(x)))
        tail_dense = 8
        early_sparse = 36
        early_dense = 28
        tail_cap = 16_384 + 32 * tail_min

        tail_reserve = max(2, min(tail_keep, int(tail_keep * tail_reserve_frac)))
        self.set_scores(
            PolicyScores(
                ExplorationScore(
                    explore_w,
                    centers=[0.0, dim_center, 0.0, y_center, z_center],
                    pow=1,
                ),
                FinalizationScore(
                    final_w,
                    centers=[0.0, dim_center, 0.0, y_center, z_center, 0.0],
                    pow=1,
                ),
            )
        )
        self.set_action_selection(
            SCHEDULE,
            [
                ActionSelection(beamwidth=1, mode="best", temperature=0.0),
                ActionSelection(beamwidth=2, mode="best", temperature=0.0),
            ],
        )
        self.set_action_pool(SCHEDULE, [ActionPool(final_size=pool_size), ActionPool(final_size=pool_size + 12)])
        self.set_tohpe_search(
            SCHEDULE,
            [
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=early_sparse, dense=early_dense, sparse_max_weight=4),
                    SourcePool(keep=early_keep, reserve=1),
                    z_choices=12,
                ),
                TohpeSearch(
                    SamplingBudget(one_hot=24, sparse=8, dense=6, sparse_max_weight=3),
                    SourcePool(keep=8, reserve=0),
                    z_choices=2,
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
                    SamplingBudget(one_hot=48, sparse=10, dense=tail_dense, sparse_max_weight=3),
                    SourcePool(keep=tail_keep, reserve=tail_reserve),
                    actions_per_bucket=tail_apb,
                    buckets=ZBucketSearch(min_buckets=tail_min, max_buckets=tail_cap, limit_bucket=tail_cap),
                ),
            ],
        )

    def __call__(self, params: Iterable[float]) -> float:
        return score_rank(self.run(params, SEEDS))


class Problem(ElementwiseProblem):
    def __init__(self, fun: Evaluator):
        n = len(fun.extract_active())
        super().__init__(n_var=n, n_obj=1, xl=np.full(n, -2.0), xu=np.full(n, 2.0))
        self.fun = fun

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.fun(np.asarray(x, dtype=float))


def optimize(fun: Evaluator, n_eval: int, seed: int = 23) -> np.ndarray:
    active = fun.extract_active()
    if len(active) == 0:
        fun.run([], SEEDS)
        return active
    algorithm = DE(pop_size=5, variant="DE/rand/1/bin", CR=0.76, F=0.5)
    result = minimize(Problem(fun), algorithm, termination=("n_eval", n_eval), seed=seed, verbose=False)
    return np.asarray(result.X if result.X is not None else fun.extract_active(), dtype=float)


def restart(fun: BaseEvaluator, rank_thr: int, xopt: np.ndarray | None = None) -> bool:
    return bool(fun.best_paths and fun.set_up_new_init(0, rank_thr=rank_thr, xopt=xopt) is not None)


def entrypoint():
    fun = Evaluator(path_name="init")
    xopt = optimize(fun, n_eval=64)
    mid = max(fun.best_rank + 1, (INITIAL_RANK + fun.best_rank) // 2)
    if restart(fun, mid, xopt=xopt):
        optimize(fun, n_eval=25, seed=24)
    return fun.get_best()
