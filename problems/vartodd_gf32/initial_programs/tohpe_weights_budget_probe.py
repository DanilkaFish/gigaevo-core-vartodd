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
)


INITIAL_RANK = 1701
SEEDS = [21, 22]
RESTART_RANKS = [1510, 1390, 1280]


def signed(x: float, scale: float = 3.2) -> float:
    return float(scale * np.tanh(float(x) / 1.4))


def unit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(x, -20.0, 20.0)))))


def score_rank(ranks: list[int]) -> float:
    if not ranks:
        return float(INITIAL_RANK)
    values = np.asarray(ranks, dtype=float)
    return float(values.min() + 0.015 * values.std())


class Evaluator(BaseEvaluator):
    """Pure TOHPE builder. It deliberately has no TODD source, but it tunes the
    whole scoring surface: 5 exploration weights, 6 final weights, 3 exploration
    centers, 3 final centers, plus both score powers. The rest of the params
    control only TOHPE generation and pool pressure."""

    def policy_mapping(self):
        explore_w = [self.map_par(signed) for _ in range(5)]
        final_w = [self.map_par(signed) for _ in range(6)]
        explore_centers = [
            0.0,
            self.map_par(unit),
            0.0,
            self.map_par(unit),
            0.0,
        ]
        final_centers = [
            0.0,
            self.map_par(unit),
            0.0,
            explore_centers[3],
            0.0,
            0.0,
        ]
        sparse = self.map_par(lambda x: 20 + int(100 * unit(x)))
        dense = self.map_par(lambda x: 12 + int(92 * unit(x)))
        sparse_max_weight = self.map_par(lambda x: 2 + int(4 * unit(x)))
        keep = self.map_par(lambda x: 18 + int(54 * unit(x)))
        z_choices = self.map_par(lambda x: 4 + int(14 * unit(x)))
        pool_size = self.map_par(lambda x: 28 + int(56 * unit(x)))
        reserve = 1

        self.set_scores(
            PolicyScores(
                ExplorationScore(explore_w, centers=explore_centers, pow=1),
                FinalizationScore(final_w, centers=final_centers, pow=1),
            )
        )
        self.set_action_selection(ActionSelection(beamwidth=1, mode="best", temperature=0.0))
        self.set_action_pool(ActionPool(final_size=pool_size))
        self.set_tohpe_search(
            TohpeSearch(
                SamplingBudget(
                    one_hot="all",
                    sparse=sparse,
                    dense=dense,
                    sparse_max_weight=sparse_max_weight,
                ),
                SourcePool(keep=keep, reserve=min(keep, reserve)),
                z_choices=z_choices,
            )
        )
        self.set_todd_search(
            ToddSearch(
                SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=2),
                SourcePool(keep=0, reserve=0),
                actions_per_bucket=1,
            )
        )

    def __call__(self, params: Iterable[float]) -> float:
        return score_rank(self.run(params, SEEDS))


class Problem(ElementwiseProblem):
    def __init__(self, fun: Evaluator):
        n = len(fun.extract_active())
        super().__init__(n_var=n, n_obj=1, xl=np.full(n, -2.4), xu=np.full(n, 2.4))
        self.fun = fun

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.fun(np.asarray(x, dtype=float))


def optimize(fun: Evaluator, n_eval: int, seed: int = 21) -> np.ndarray:
    active = fun.extract_active()
    if len(active) == 0:
        fun.run([], SEEDS)
        return active
    algorithm = PSO(pop_size=9, w=0.62, c1=0.55, c2=0.55, adaptive=False)
    result = minimize(Problem(fun), algorithm, termination=("n_eval", n_eval), seed=seed, verbose=False)
    return np.asarray(result.X if result.X is not None else fun.extract_active(), dtype=float)


def restart(fun: BaseEvaluator, rank_thr: int, xopt: np.ndarray | None = None) -> bool:
    if not fun.best_paths:
        return False
    return fun.set_up_new_init(0, rank_thr=rank_thr, xopt=xopt) is not None


def entrypoint():
    fun = Evaluator(path_name="init")
    xopt = optimize(fun, n_eval=34, seed=21)
    for i, rank_thr in enumerate(RESTART_RANKS):
        if fun.best_rank >= rank_thr:
            continue
        if not restart(fun, rank_thr, xopt=xopt):
            continue
        xopt = optimize(fun, n_eval=8, seed=22 + i)
    return fun.get_best()
