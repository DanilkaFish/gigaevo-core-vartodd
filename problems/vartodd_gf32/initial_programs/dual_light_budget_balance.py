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
MID_RANK = 1490
TAIL_RANK = 1340
SEEDS = [25, 26]
SCHEDULE = [INITIAL_RANK, MID_RANK, TAIL_RANK]


def signed(x: float, scale: float = 3.0) -> float:
    return float(scale * np.tanh(float(x) / 1.5))


def unit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20.0, 20.0)))))


def score_rank(ranks: list[int]) -> float:
    if not ranks:
        return float(INITIAL_RANK)
    values = np.asarray(ranks, dtype=float)
    return float(values.min() + 0.02 * values.std())


class Evaluator(BaseEvaluator):
    """Balanced finite-cap builder. Unlike the TOHPE scouts, both sources stay
    active from the first step. DE tunes all 11 score weights, two score powers,
    separate dim/y/TOHPE centers, and a moderate TODD cap ladder."""

    def policy_mapping(self):
        explore_w = [self.map_par(signed) for _ in range(5)]
        final_w = [self.map_par(signed) for _ in range(6)]
        dim_center = self.map_par(unit)
        y_center = self.map_par(unit)
        tohpe_center = self.map_par(unit)
        tohpe_sparse = self.map_par(lambda x: 10 + int(58 * unit(x)))
        tohpe_keep = self.map_par(lambda x: 10 + int(28 * unit(x)))
        todd_one_hot = self.map_par(lambda x: 12 + int(52 * unit(x)))
        todd_keep = self.map_par(lambda x: 8 + int(22 * unit(x)))
        todd_cap = self.map_par(lambda x: 4096 + int(61440 * unit(x)))
        tohpe_dense = 28
        tohpe_z = 7
        todd_sparse = 6
        pool_size = 48

        self.set_scores(
            PolicyScores(
                ExplorationScore(
                    explore_w,
                    centers=[0.0, dim_center, 0.0, y_center, y_center],
                    pow=1,
                ),
                FinalizationScore(
                    final_w,
                    centers=[0.0, dim_center, 0.0, y_center, y_center, tohpe_center],
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
            [ActionPool(final_size=pool_size), ActionPool(final_size=pool_size + 8), ActionPool(final_size=pool_size + 18)],
        )
        self.set_tohpe_search(
            SCHEDULE,
            [
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=tohpe_sparse, dense=tohpe_dense, sparse_max_weight=4),
                    SourcePool(keep=tohpe_keep, reserve=1),
                    z_choices=tohpe_z,
                ),
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=tohpe_sparse // 2, dense=tohpe_dense // 2, sparse_max_weight=4),
                    SourcePool(keep=max(8, tohpe_keep // 2), reserve=1),
                    z_choices=max(3, tohpe_z // 2),
                ),
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=12, dense=8, sparse_max_weight=3),
                    SourcePool(keep=8, reserve=0),
                    z_choices=3,
                ),
            ],
        )
        self.set_todd_search(
            SCHEDULE,
            [
                ToddSearch(
                    SamplingBudget(one_hot=max(4, todd_one_hot // 3), sparse=0, dense=0, sparse_max_weight=2),
                    SourcePool(keep=max(4, todd_keep // 2), reserve=1),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=64, max_buckets=max(2048, todd_cap // 4), limit_bucket=max(2048, todd_cap // 4)),
                ),
                ToddSearch(
                    SamplingBudget(one_hot=todd_one_hot, sparse=todd_sparse, dense=0, sparse_max_weight=3),
                    SourcePool(keep=todd_keep, reserve=max(2, todd_keep // 5)),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=128, max_buckets=todd_cap, limit_bucket=todd_cap),
                ),
                ToddSearch(
                    SamplingBudget(one_hot="all", sparse=todd_sparse + 4, dense=0, sparse_max_weight=3),
                    SourcePool(keep=todd_keep + 10, reserve=max(3, todd_keep // 4)),
                    actions_per_bucket=2,
                    buckets=ZBucketSearch(min_buckets=384, max_buckets=2 * todd_cap, limit_bucket=2 * todd_cap),
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


def optimize(fun: Evaluator, n_eval: int, seed: int = 25) -> np.ndarray:
    active = fun.extract_active()
    if len(active) == 0:
        fun.run([], SEEDS)
        return active
    algorithm = DE(pop_size=7, variant="DE/rand/1/bin", CR=0.72, F=0.52)
    result = minimize(Problem(fun), algorithm, termination=("n_eval", n_eval), seed=seed, verbose=False)
    return np.asarray(result.X if result.X is not None else fun.extract_active(), dtype=float)


def restart(fun: BaseEvaluator, rank_thr: int, xopt: np.ndarray | None = None) -> bool:
    return bool(fun.best_paths and fun.set_up_new_init(0, rank_thr=rank_thr, xopt=xopt) is not None)


def entrypoint():
    fun = Evaluator(path_name="init")
    xopt = optimize(fun, n_eval=30)
    if restart(fun, max(fun.best_rank + 25, 1420), xopt=xopt):
        xopt = optimize(fun, n_eval=10, seed=26)
    if restart(fun, fun.best_rank + 8, xopt=xopt):
        optimize(fun, n_eval=8, seed=27)
    return fun.get_best()
