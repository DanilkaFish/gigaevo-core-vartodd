"""Group-wise PSO and CMA-ES with deterministic beam-4 tail search."""

from collections.abc import Iterable

from helper import (
    INITIAL_RANK,
    TARGET_FINAL_RANK,
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
import numpy as np
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

OPTIMIZER_FAMILY = "pymoo_pso_cma_grouped_tail"
SEEDS = [21, 22, 23]
SCOUT_EVALS = 72
SCORE_REFINE_EVALS = 256
SEARCH_REFINE_EVALS = 32
LOWER_BOUND = -2.0
UPPER_BOUND = 2.0
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
MAX_DEPTH = max(500, RANK_SPAN + 64)


def rank_from_target(fraction: float) -> int:
    return min(
        INITIAL_RANK,
        TARGET_FINAL_RANK + max(1, round(RANK_SPAN * fraction)),
    )


def rank_margin(fraction: float) -> int:
    return max(1, round(RANK_SPAN * fraction))


TERMINAL_RANK = rank_from_target(0.12)
MARGIN = rank_margin(0.16)


class Evaluator(BaseEvaluator):
    """Optimize score and search groups separately around a TOHPE-first policy."""

    def float_range(self, low: float, high: float, *, group: str) -> float:
        return self.map_par(
            lambda x: low
            + (high - low) * (0.5 + 0.5 * np.tanh(float(x) / 2.0)),
            group=group,
        )

    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(
            lambda x: min(
                high,
                low
                + int(
                    (high - low + 1)
                    * (0.5 + 0.5 * np.tanh(float(x) / 2.0))
                ),
            ),
            group=group,
        )

    def policy_mapping(self):
        self.set_scores(
            PolicyScores(
                ExplorationScore(
                    [
                        self.float_range(-4.0, 4.0, group="scores")
                        for _ in range(5)
                    ],
                    centers=[0.0] * 5,
                    pow=1,
                ),
                FinalizationScore(
                    [
                        self.float_range(-4.0, 4.0, group="scores")
                        for _ in range(6)
                    ],
                    centers=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        self.float_range(0.0, 0.8, group="scores"),
                        self.float_range(-1.0, 0.5, group="scores"),
                    ],
                    pow=1,
                ),
            )
        )
        self.set_action_selection(ActionSelection(beamwidth=4, mode="best"))
        self.set_action_pool(
            ActionPool(
                final_size=self.int_range(24, 48, group="search")
            )
        )
        self.set_tohpe_search(
            TohpeSearch(
                SamplingBudget(
                    one_hot="all",
                    sparse=self.int_range(8, 24, group="search"),
                    dense=0,
                    sparse_max_weight=self.int_range(1, 3, group="search"),
                ),
                SourcePool(
                    keep=self.int_range(24, 48, group="search"),
                    reserve=self.int_range(6, 12, group="search"),
                ),
                z_choices=self.int_range(6, 16, group="search"),
            )
        )
        early_todd = ToddSearch(
            SamplingBudget(one_hot=3, sparse=0, dense=0, sparse_max_weight=1),
            SourcePool(keep=0, reserve=0),
            actions_per_bucket=0,
            buckets=ZBucketSearch(
                min_buckets=0, max_buckets=0, limit_bucket=0
            ),
        )
        terminal_todd = ToddSearch(
            SamplingBudget(one_hot=3, sparse=0, dense=0, sparse_max_weight=1),
            SourcePool(
                keep=self.int_range(6, 12, group="search"),
                reserve=self.int_range(2, 6, group="search"),
            ),
            actions_per_bucket=3,
            buckets=ZBucketSearch(
                min_buckets=self.int_range(50, 150, group="search"),
                max_buckets=self.int_range(800, 4000, group="search"),
                limit_bucket=self.int_range(2000, 12000, group="search"),
            ),
        )
        self.set_todd_search(
            ranks=[INITIAL_RANK, TERMINAL_RANK],
            values=[early_todd, terminal_todd],
        )

    def __call__(self, params: Iterable[float]) -> float:
        values = np.asarray(self.run(params, SEEDS), dtype=float)
        return float(values.min() + 0.015 * values.std())


class Problem(ElementwiseProblem):
    def __init__(self, evaluator: Evaluator):
        size = len(evaluator.extract_active())
        super().__init__(
            n_var=size,
            n_obj=1,
            xl=np.full(size, LOWER_BOUND),
            xu=np.full(size, UPPER_BOUND),
        )
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator(np.asarray(x, dtype=float))


def optimize_pso(
    evaluator: Evaluator, evaluations: int, seed: int
) -> np.ndarray:
    current = evaluator.extract_active()
    if len(current) == 0:
        evaluator([])
        return current
    result = minimize(
        Problem(evaluator),
        PSO(pop_size=16, w=0.85, c1=0.8, c2=0.8, adaptive=True),
        termination=("n_eval", evaluations),
        seed=seed,
        verbose=False,
    )
    best = np.asarray(
        result.X if result.X is not None else current, dtype=float
    )
    evaluator.insert(best)
    evaluator.reinit()
    return best


def optimize_cma(
    evaluator: Evaluator, evaluations: int, seed: int
) -> np.ndarray:
    current = evaluator.extract_active()
    if len(current) == 0:
        evaluator([])
        return current
    result = minimize(
        Problem(evaluator),
        CMAES(
            x0=np.clip(current, LOWER_BOUND, UPPER_BOUND),
            sigma=0.5,
            pop_size=8,
            restarts=1,
        ),
        termination=("n_eval", evaluations),
        seed=seed,
        verbose=False,
    )
    best = np.asarray(
        result.X if result.X is not None else current, dtype=float
    )
    evaluator.insert(best)
    evaluator.reinit()
    return best


def entrypoint():
    evaluator = Evaluator(path_name="init", margin=MARGIN, max_depth=MAX_DEPTH)
    evaluator.select_all_parameter_groups()
    optimize_pso(evaluator, SCOUT_EVALS, seed=21)
    evaluator.select_parameter_groups("scores")
    optimize_cma(evaluator, SCORE_REFINE_EVALS, seed=24)
    evaluator.select_parameter_groups("search")
    optimize_pso(evaluator, SEARCH_REFINE_EVALS, seed=25)
    return evaluator.get_best()


AUX_DESCRIPTION = """
GF16 source: f7aff6f1-3e02-4bf4-8f6c-f2aaea42b200.
Observed source result: rank 397, fitness 397.41939546599497, runtime
2370.990293 seconds, timeout_salvaged=0.
Pool role: deterministic beam 4 plus explicit parameter-group optimization.
It scouts all parameters, refines scores with CMA-ES, then polishes search
parameters with PSO.
Universal adaptation: terminal scheduling and margin are rank-span fractions;
the finite TODD search remains a tunable part of the search group.
Tuning note: the grouped vector changes size between stages, so every optimizer
constructs its Problem after selecting the intended active groups.
"""
