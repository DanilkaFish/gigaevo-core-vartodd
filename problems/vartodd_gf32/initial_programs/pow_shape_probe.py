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
MID_RANK = 1495
TAIL_RANK = 1355
SEEDS = [35, 36]
SCHEDULE = [INITIAL_RANK, MID_RANK, TAIL_RANK]


def signed(x: float, scale: float = 3.0) -> float:
    return float(scale * np.tanh(float(x) / 1.3))


def unit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20.0, 20.0)))))


def pow_map(x: float) -> float:
    return float(0.45 + 2.8 * unit(x))


def score_rank(ranks: list[int]) -> float:
    if not ranks:
        return float(INITIAL_RANK)
    values = np.asarray(ranks, dtype=float)
    return float(values.min() + 0.03 * values.std())


class Evaluator(BaseEvaluator):
    """Ab-initio score-shape probe. It keeps the useful 20-D policy surface from
    the saved-path version (11 weights, three centers, both powers, and four
    TODD budget knobs), but starts from `init` and uses finite TODD caps so the
    full program stays inside the observed 10k-second wall budget."""

    def policy_mapping(self):
        explore_w = [self.map_par(signed) for _ in range(5)]
        final_w = [self.map_par(signed) for _ in range(6)]
        explore_centers = [
            0.0,
            self.map_par(unit),
            0.0,
            self.map_par(unit),
            self.map_par(unit),
        ]
        final_centers = [
            0.0,
            explore_centers[1],
            explore_centers[2],
            explore_centers[3],
            explore_centers[4],
            0.0,
        ]
        explore_pow = self.map_par(pow_map)
        final_pow = self.map_par(pow_map)
        reserve_cap = self.map_par(lambda x: 4096 + int(57_344 * unit(x)))
        min_buckets = self.map_par(lambda x: 96 + int(928 * unit(x)))
        todd_keep = self.map_par(lambda x: 6 + int(18 * unit(x)))
        todd_reserve = self.map_par(lambda x: 1 + int(5 * unit(x)))
        pool_size = 24
        temperature = 0.1

        self.set_scores(
            PolicyScores(
                ExplorationScore(explore_w, centers=explore_centers, pow=explore_pow),
                FinalizationScore(final_w, centers=final_centers, pow=final_pow),
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
            [ActionPool(final_size=pool_size), ActionPool(final_size=pool_size + 8), ActionPool(final_size=pool_size + 16)],
        )
        self.set_tohpe_search(
            SCHEDULE,
            [
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=20, dense=12, sparse_max_weight=3),
                    SourcePool(keep=8, reserve=0),
                    z_choices=4,
                ),
                TohpeSearch(
                    SamplingBudget(one_hot=24, sparse=4, dense=3, sparse_max_weight=2),
                    SourcePool(keep=5, reserve=0),
                    z_choices=2,
                ),
                TohpeSearch(
                    SamplingBudget(one_hot=8, sparse=1, dense=1, sparse_max_weight=2),
                    SourcePool(keep=3, reserve=0),
                    z_choices=1,
                ),
            ],
        )
        self.set_todd_search(
            SCHEDULE,
            [
                ToddSearch(
                    SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2),
                    SourcePool(keep=max(3, todd_keep // 2), reserve=1),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=64, max_buckets=2048, limit_bucket=2048),
                ),
                ToddSearch(
                    SamplingBudget(one_hot=16, sparse=4, dense=0, sparse_max_weight=2),
                    SourcePool(keep=todd_keep, reserve=min(todd_keep, todd_reserve)),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=min_buckets, max_buckets=reserve_cap, limit_bucket=reserve_cap),
                ),
                ToddSearch(
                    SamplingBudget(one_hot=32, sparse=6, dense=0, sparse_max_weight=2),
                    SourcePool(keep=todd_keep + 6, reserve=min(todd_keep + 6, todd_reserve + 2)),
                    actions_per_bucket=1,
                    buckets=ZBucketSearch(min_buckets=2 * min_buckets, max_buckets=2 * reserve_cap, limit_bucket=2 * reserve_cap),
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


def optimize(fun: Evaluator, n_eval: int, seed: int = 35) -> np.ndarray:
    active = fun.extract_active()
    if len(active) == 0:
        fun.run([], SEEDS)
        return active
    algorithm = PSO(pop_size=8, w=0.55, c1=0.8, c2=0.35, adaptive=False)
    result = minimize(Problem(fun), algorithm, termination=("n_eval", n_eval), seed=seed, verbose=False)
    return np.asarray(result.X if result.X is not None else fun.extract_active(), dtype=float)


def restart(fun: BaseEvaluator, rank_thr: int, xopt: np.ndarray | None = None) -> bool:
    return bool(fun.best_paths and fun.set_up_new_init(0, rank_thr=rank_thr, xopt=xopt) is not None)


def entrypoint():
    fun = Evaluator(path_name="init")
    xopt = optimize(fun, n_eval=12)
    if restart(fun, fun.best_rank + 24, xopt=xopt):
        optimize(fun, n_eval=4, seed=36)
    return fun.get_best()
