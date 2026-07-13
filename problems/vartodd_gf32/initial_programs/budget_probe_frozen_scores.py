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
MID_RANK = 1500
TAIL_RANK = 1360
SEEDS = [41, 42]
SCHEDULE = [INITIAL_RANK, MID_RANK, TAIL_RANK]


def signed(x: float, scale: float = 2.6) -> float:
    return float(scale * np.tanh(float(x) / 1.6))


def unit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20.0, 20.0)))))


def score_rank(ranks: list[int]) -> float:
    return float(min(ranks)) if ranks else float(INITIAL_RANK)


class Evaluator(BaseEvaluator):
    """Reduced-budget finite TODD probe. The previous no-limit version was still
    running after the 10k-second target, so this keeps the same 20 active knobs
    but replaces unbounded bucket sweeps with small finite caps and fewer
    optimizer evaluations."""

    def policy_mapping(self):
        explore_w = [self.map_par(signed) for _ in range(5)]
        final_w = [self.map_par(signed) for _ in range(6)]
        dim_center = self.map_par(unit)
        bucket_center = self.map_par(unit)
        tohpe_center = self.map_par(unit)
        todd_min = self.map_par(lambda x: 32 + int(352 * unit(x)))
        todd_keep = self.map_par(lambda x: 4 + int(10 * unit(x)))
        todd_sparse = self.map_par(lambda x: int(6 * unit(x)))
        tohpe_keep = self.map_par(lambda x: 4 + int(14 * unit(x)))
        pool_size = self.map_par(lambda x: 10 + int(22 * unit(x)))
        temperature = self.map_par(lambda x: 0.08 + 0.34 * unit(x))
        todd_cap = 1024 + 16 * todd_min

        self.set_scores(
            PolicyScores(
                ExplorationScore(
                    explore_w,
                    centers=[0.0, dim_center, bucket_center, 0.0, 0.0],
                    pow=1,
                ),
                FinalizationScore(
                    final_w,
                    centers=[0.0, dim_center, bucket_center, 0.0, 0.0, tohpe_center],
                    pow=1,
                ),
            )
        )
        self.set_action_selection(
            SCHEDULE,
            [
                ActionSelection(beamwidth=1, mode="softmax", temperature=temperature),
                ActionSelection(beamwidth=1, mode="softmax", temperature=temperature),
                ActionSelection(beamwidth=2, mode="softmax", temperature=temperature),
            ],
        )
        self.set_action_pool(
            SCHEDULE,
            [ActionPool(final_size=pool_size), ActionPool(final_size=pool_size + 6), ActionPool(final_size=pool_size + 12)],
        )
        self.set_tohpe_search(
            SCHEDULE,
            [
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=16, dense=16, sparse_max_weight=3),
                    SourcePool(keep=tohpe_keep, reserve=0),
                    z_choices=5,
                ),
                TohpeSearch(
                    SamplingBudget(one_hot=32, sparse=8, dense=6, sparse_max_weight=3),
                    SourcePool(keep=max(3, tohpe_keep // 2), reserve=0),
                    z_choices=3,
                ),
                TohpeSearch(
                    SamplingBudget(one_hot=12, sparse=2, dense=2, sparse_max_weight=2),
                    SourcePool(keep=3, reserve=0),
                    z_choices=2,
                ),
            ],
        )
        self.set_todd_search(
            SCHEDULE,
            [
                ToddSearch(
                    SamplingBudget(one_hot=12, sparse=todd_sparse, dense=0, sparse_max_weight=2),
                    SourcePool(keep=todd_keep, reserve=1),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=todd_min, max_buckets=todd_cap, limit_bucket=todd_cap),
                ),
                ToddSearch(
                    SamplingBudget(one_hot=20, sparse=todd_sparse, dense=0, sparse_max_weight=2),
                    SourcePool(keep=todd_keep + 4, reserve=2),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=2 * todd_min, max_buckets=2 * todd_cap, limit_bucket=2 * todd_cap),
                ),
                ToddSearch(
                    SamplingBudget(one_hot=32, sparse=todd_sparse + 1, dense=0, sparse_max_weight=2),
                    SourcePool(keep=todd_keep + 6, reserve=2),
                    actions_per_bucket=2,
                    buckets=ZBucketSearch(min_buckets=3 * todd_min, max_buckets=3 * todd_cap, limit_bucket=3 * todd_cap),
                ),
            ],
        )

    def __call__(self, params: Iterable[float]) -> float:
        return score_rank(self.run(params, SEEDS))


class Problem(ElementwiseProblem):
    def __init__(self, fun: Evaluator):
        n = len(fun.extract_active())
        super().__init__(n_var=n, n_obj=1, xl=np.full(n, -1.8), xu=np.full(n, 1.8))
        self.fun = fun

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.fun(np.asarray(x, dtype=float))


def optimize(fun: Evaluator, n_eval: int, seed: int = 41) -> np.ndarray:
    active = fun.extract_active()
    if len(active) == 0:
        fun.run([], SEEDS)
        return active
    algorithm = PSO(pop_size=5, w=0.7, c1=0.45, c2=0.45, adaptive=False)
    result = minimize(Problem(fun), algorithm, termination=("n_eval", n_eval), seed=seed, verbose=False)
    return np.asarray(result.X if result.X is not None else fun.extract_active(), dtype=float)


def restart(fun: BaseEvaluator, rank_thr: int, xopt: np.ndarray | None = None) -> bool:
    return bool(fun.best_paths and fun.set_up_new_init(0, rank_thr=rank_thr, xopt=xopt) is not None)


def entrypoint():
    fun = Evaluator(path_name="init")
    xopt = optimize(fun, n_eval=10)
    if restart(fun, fun.best_rank + 30, xopt=xopt):
        optimize(fun, n_eval=3, seed=42)
    return fun.get_best()
