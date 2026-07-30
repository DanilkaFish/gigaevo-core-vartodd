"""Multi-restart PSO with a pure TOHPE policy and TODD disabled."""

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
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

OPTIMIZER_FAMILY = "pymoo_pso_restart_tohpe_only"
SEEDS = [23, 24, 25]
TOTAL_EVALS = 5000
RESTARTS = 4
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
MAX_DEPTH = max(500, RANK_SPAN + 64)


def rank_from_target(fraction: float) -> int:
    return min(
        INITIAL_RANK,
        TARGET_FINAL_RANK + max(1, round(RANK_SPAN * fraction)),
    )


def rank_margin(fraction: float) -> int:
    return max(1, round(RANK_SPAN * fraction))


MARGIN = rank_margin(0.43)


class Evaluator(BaseEvaluator):
    """Expose the cheap-source basin without any TODD contribution."""

    def float_range(
        self, low: float, high: float, *, group: str = "scores"
    ) -> float:
        return self.map_par(
            lambda x: low
            + (high - low)
            / (1.0 + np.exp(-np.clip(float(x), -8.0, 8.0) / 2.5)),
            group=group,
        )

    def int_range(
        self, low: int, high: int, *, group: str = "policy"
    ) -> int:
        return self.map_par(
            lambda x: min(
                high,
                low
                + int(
                    (high - low + 1)
                    / (1.0 + np.exp(-np.clip(float(x), -8.0, 8.0) / 2.5))
                ),
            ),
            group=group,
        )

    def policy_mapping(self):
        self.set_scores(
            PolicyScores(
                ExplorationScore(
                    [self.float_range(-4.0, 4.0) for _ in range(5)],
                    centers=[0.0] * 5,
                    pow=1,
                ),
                FinalizationScore(
                    [
                        self.float_range(-4.0, 4.0),
                        self.float_range(-4.0, 4.0),
                        self.float_range(-0.2, 0.5),
                        self.float_range(-4.0, 4.0),
                        self.float_range(-4.0, 4.0),
                        self.float_range(0.5, 4.0),
                    ],
                    centers=[0.0] * 6,
                    pow=1,
                ),
            )
        )
        self.set_todd_search(
            ToddSearch(
                SamplingBudget(
                    one_hot=0, sparse=0, dense=0, sparse_max_weight=0
                ),
                SourcePool(keep=0, reserve=0),
                actions_per_bucket=0,
                buckets=ZBucketSearch(
                    min_buckets=0, max_buckets=0, limit_bucket=0
                ),
            )
        )
        self.set_tohpe_search(
            TohpeSearch(
                SamplingBudget(
                    one_hot="all", sparse=0, dense=0, sparse_max_weight=2
                ),
                SourcePool(
                    keep=self.int_range(20, 40, group="tohpe"),
                    reserve=self.int_range(8, 14, group="tohpe"),
                ),
                z_choices=self.int_range(20, 40, group="tohpe"),
            )
        )
        self.set_action_selection(
            ActionSelection(
                beamwidth=self.int_range(4, 6, group="selection"),
                mode="softmax",
                temperature=0.35,
            )
        )
        self.set_action_pool(
            ActionPool(final_size=self.int_range(12, 20, group="pool"))
        )

    def __call__(self, params: Iterable[float]) -> float:
        values = np.asarray(self.run(params, SEEDS), dtype=float)
        return float(values.min() + 0.02 * values.std())


class Problem(ElementwiseProblem):
    def __init__(self, evaluator: Evaluator):
        size = len(evaluator.extract_active())
        super().__init__(
            n_var=size,
            n_obj=1,
            xl=np.full(size, -1.0),
            xu=np.full(size, 1.0),
        )
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator(np.asarray(x, dtype=float))


def optimize(
    evaluator: Evaluator, evaluations: int, base_seed: int
) -> np.ndarray:
    best_x = None
    best_f = float("inf")
    per_run = max(10, evaluations // RESTARTS)
    for index in range(RESTARTS):
        result = minimize(
            Problem(evaluator),
            PSO(pop_size=16, w=0.7, c1=0.9, c2=0.9, adaptive=True),
            termination=("n_eval", per_run),
            seed=(base_seed + index) % (2**31),
            verbose=False,
        )
        candidate = np.asarray(
            result.X
            if result.X is not None
            else evaluator.extract_active(),
            dtype=float,
        )
        fitness = float(evaluator(candidate))
        if fitness < best_f:
            best_f = fitness
            best_x = candidate
    if best_x is None:
        return np.asarray(evaluator.extract_active(), dtype=float)
    return best_x


def entrypoint():
    evaluator = Evaluator(path_name="init", margin=MARGIN, max_depth=MAX_DEPTH)
    optimize(evaluator, TOTAL_EVALS, base_seed=23)
    return evaluator.get_best()


AUX_DESCRIPTION = """
GF16 source: 67b2f195-25c2-4f82-b50a-ab70864fac20.
Observed source result: rank 387, fitness 387.4107450473729, runtime
2600.609607 seconds, timeout_salvaged=1.
Pool role: the pure-TOHPE control. It reached the same best observed rank with
TODD disabled and therefore preserves a source-diverse alternative to every
terminal-TODD program.
Universal adaptation: only its ab-initio margin and maximum depth depend on
INITIAL_RANK - TARGET_FINAL_RANK; the policy intentionally has no rank switch.
Tuning note: this archetype can be efficient while TOHPE still produces
positive actions, but it has no deep-source fallback after TOHPE starvation.
"""
