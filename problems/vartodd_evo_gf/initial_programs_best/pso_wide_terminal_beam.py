"""Multi-restart PSO with a beam widened only in the terminal band."""

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

OPTIMIZER_FAMILY = "pymoo_pso_wide_terminal_beam"
SEEDS = [23, 24, 25, 42]
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


TERMINAL_RANK = rank_from_target(0.10)
MARGIN = rank_margin(0.43)


def disabled_todd() -> ToddSearch:
    return ToddSearch(
        SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=0),
        SourcePool(keep=0, reserve=0),
        actions_per_bucket=0,
        buckets=ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0),
    )


class Evaluator(BaseEvaluator):
    """Widen terminal branching so rare TODD actions form more trajectories."""

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
                        self.float_range(-1.0, 1.0),
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
        terminal_todd = ToddSearch(
            SamplingBudget(one_hot=2, sparse=0, dense=0, sparse_max_weight=2),
            SourcePool(keep=12, reserve=4),
            actions_per_bucket=4,
            buckets=ZBucketSearch(
                min_buckets=12, max_buckets=60, limit_bucket=60
            ),
        )
        self.set_todd_search(
            ranks=[TERMINAL_RANK],
            values=[disabled_todd(), terminal_todd],
        )
        self.set_tohpe_search(
            TohpeSearch(
                SamplingBudget(
                    one_hot="all", sparse=0, dense=0, sparse_max_weight=2
                ),
                SourcePool(
                    keep=self.int_range(14, 26, group="tohpe"),
                    reserve=self.int_range(6, 10, group="tohpe"),
                ),
                z_choices=self.int_range(14, 24, group="tohpe"),
            )
        )
        self.set_action_selection(
            ranks=[TERMINAL_RANK],
            values=[
                ActionSelection(
                    beamwidth=self.int_range(4, 6, group="selection"),
                    mode="softmax",
                    temperature=0.35,
                ),
                ActionSelection(
                    beamwidth=self.int_range(6, 10, group="selection"),
                    mode="softmax",
                    temperature=0.35,
                ),
            ],
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
GF16 source: e350705c-c6f2-479f-b153-922fad712cf8.
Observed source result: rank 387, fitness 387.41978897502156, runtime
2600.165856 seconds, timeout_salvaged=1.
Pool role: the high-beam terminal archetype. It keeps beam 4-6 during the
TOHPE descent, then widens to 6-10 while enabling a small TODD pool.
Universal adaptation: the original rank-400 switch and margin 80 use fractions
of INITIAL_RANK - TARGET_FINAL_RANK.
Tuning note: the wider terminal beam increases trajectory diversity and the
cost of each policy evaluation, so it is most distinct when the tail is hard.
"""
