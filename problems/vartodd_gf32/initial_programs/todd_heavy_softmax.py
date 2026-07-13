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


INITIAL_RANK = 1701
HEAVY_RANK = 1450
TAIL_RANK = 1320
SEEDS = [27, 28]
SCHEDULE = [INITIAL_RANK, HEAVY_RANK, TAIL_RANK]


def signed(x: float, scale: float = 3.1) -> float:
    return float(scale * np.tanh(float(x) / 1.35))


def unit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20.0, 20.0)))))


def score_rank(ranks: list[int]) -> float:
    if not ranks:
        return float(INITIAL_RANK)
    values = np.asarray(ranks, dtype=float)
    return float(values.min() + 0.02 * values.std())


class Evaluator(BaseEvaluator):
    """Stochastic heavy TODD. Softmax selection and a narrow pool test whether
    late low-dimensional actions need stochastic tie breaking. The tail uses
    `limit_bucket=-1`, but the real timing CSV showed the old 100k-700k cap and
    42 evaluations were too slow. This version keeps the same mechanism with
    smaller max buckets, lower dense sampling, and fewer evaluations."""

    def policy_mapping(self):
        explore_w = [self.map_par(signed) for _ in range(5)]
        final_w = [self.map_par(signed) for _ in range(6)]
        dim_center = self.map_par(unit)
        y_center = self.map_par(unit)
        min_buckets = self.map_par(lambda x: 64 + int(512 * unit(x)))
        reserve_cap = self.map_par(lambda x: 24_000 + int(136_000 * unit(x)))
        todd_dense = self.map_par(lambda x: 4 + int(28 * unit(x)))
        todd_keep = self.map_par(lambda x: 6 + int(18 * unit(x)))
        reserve = self.map_par(lambda x: 1 + int(4 * unit(x)))
        pool_size = self.map_par(lambda x: 10 + int(18 * unit(x)))
        temperature = self.map_par(lambda x: 0.05 + 0.45 * unit(x))
        bucket_center = 0.0
        tohpe_center = 0.0
        tohpe_keep = 3

        self.set_scores(
            PolicyScores(
                ExplorationScore(
                    explore_w,
                    centers=[0.0, dim_center, bucket_center, y_center, y_center],
                    pow=1,
                ),
                FinalizationScore(
                    final_w,
                    centers=[0.0, dim_center, bucket_center, y_center, y_center, tohpe_center],
                    pow=1,
                ),
            )
        )
        self.set_action_selection(
            SCHEDULE,
            [
                ActionSelection(beamwidth=1, mode="softmax", temperature=temperature),
                ActionSelection(beamwidth=2, mode="softmax", temperature=temperature),
                ActionSelection(beamwidth=2, mode="softmax", temperature=temperature),
            ],
        )
        self.set_action_pool(
            SCHEDULE,
            [ActionPool(final_size=pool_size), ActionPool(final_size=pool_size + 6), ActionPool(final_size=pool_size + 10)],
        )
        self.set_tohpe_search(
            SCHEDULE,
            [
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=10, dense=10, sparse_max_weight=2),
                    SourcePool(keep=tohpe_keep + 4, reserve=0),
                    z_choices=3,
                ),
                TohpeSearch(
                    SamplingBudget(one_hot=24, sparse=4, dense=4, sparse_max_weight=2),
                    SourcePool(keep=tohpe_keep, reserve=0),
                    z_choices=2,
                ),
                TohpeSearch(
                    SamplingBudget(one_hot=8, sparse=1, dense=1, sparse_max_weight=2),
                    SourcePool(keep=max(1, tohpe_keep // 2), reserve=0),
                    z_choices=1,
                ),
            ],
        )
        self.set_todd_search(
            SCHEDULE,
            [
                ToddSearch(
                    SamplingBudget(one_hot=32, sparse=4, dense=6, sparse_max_weight=2),
                    SourcePool(keep=8, reserve=1),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=64, max_buckets=8192, limit_bucket=8192),
                ),
                ToddSearch(
                    SamplingBudget(one_hot=48, sparse=6, dense=todd_dense, sparse_max_weight=2),
                    SourcePool(keep=todd_keep, reserve=min(todd_keep, reserve)),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=min_buckets, max_buckets=reserve_cap, limit_bucket=-1),
                ),
                ToddSearch(
                    SamplingBudget(one_hot=64, sparse=4, dense=max(3, todd_dense // 2), sparse_max_weight=2),
                    SourcePool(keep=todd_keep + 4, reserve=min(todd_keep + 4, reserve + 1)),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=2 * min_buckets, max_buckets=reserve_cap, limit_bucket=-1),
                ),
            ],
        )

    def __call__(self, params: Iterable[float]) -> float:
        return score_rank(self.run(params, SEEDS))


class Problem(ElementwiseProblem):
    def __init__(self, fun: Evaluator):
        n = len(fun.extract_active())
        super().__init__(n_var=n, n_obj=1, xl=np.full(n, -2.1), xu=np.full(n, 2.1))
        self.fun = fun

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.fun(np.asarray(x, dtype=float))


def optimize(fun: Evaluator, n_eval: int, seed: int = 27) -> np.ndarray:
    active = fun.extract_active()
    if len(active) == 0:
        fun.run([], SEEDS)
        return active
    algorithm = DE(pop_size=5, variant="DE/rand/1/bin", CR=0.68, F=0.62)
    result = minimize(Problem(fun), algorithm, termination=("n_eval", n_eval), seed=seed, verbose=False)
    return np.asarray(result.X if result.X is not None else fun.extract_active(), dtype=float)


def restart(fun: BaseEvaluator, rank_thr: int, xopt: np.ndarray | None = None) -> bool:
    return bool(fun.best_paths and fun.set_up_new_init(0, rank_thr=rank_thr, xopt=xopt) is not None)


def entrypoint():
    fun = Evaluator(path_name="init")
    xopt = optimize(fun, n_eval=10)
    mid = max(fun.best_rank + 1, (INITIAL_RANK + fun.best_rank) // 2)
    if restart(fun, mid, xopt=xopt):
        optimize(fun, n_eval=3, seed=28)
    return fun.get_best()
