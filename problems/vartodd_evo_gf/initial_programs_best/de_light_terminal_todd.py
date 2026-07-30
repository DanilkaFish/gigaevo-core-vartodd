"""DE ab-initio search with TOHPE early and light TODD near the target."""

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
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

OPTIMIZER_FAMILY = "pymoo_de_light_terminal_todd"
SEEDS = [23, 24, 25]
TOTAL_EVALS = 8000
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
MAX_DEPTH = max(500, RANK_SPAN + 64)


def rank_from_target(fraction: float) -> int:
    return min(
        INITIAL_RANK,
        TARGET_FINAL_RANK + max(1, round(RANK_SPAN * fraction)),
    )


def rank_margin(fraction: float) -> int:
    return max(1, round(RANK_SPAN * fraction))


TERMINAL_RANK = rank_from_target(0.08)
MARGIN = rank_margin(0.43)


def disabled_todd() -> ToddSearch:
    return ToddSearch(
        SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=0),
        SourcePool(keep=0, reserve=0),
        actions_per_bucket=0,
        buckets=ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0),
    )


class Evaluator(BaseEvaluator):
    """Keep the cheap source dominant and add only a small terminal TODD."""

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
            SourcePool(
                keep=self.int_range(4, 8, group="todd"),
                reserve=1,
            ),
            actions_per_bucket=self.int_range(2, 4, group="todd"),
            buckets=ZBucketSearch(
                min_buckets=self.int_range(10, 20, group="todd"),
                max_buckets=self.int_range(50, 200, group="todd"),
                limit_bucket=self.int_range(30, 100, group="todd"),
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


def optimize(evaluator: Evaluator, evaluations: int, seed: int) -> np.ndarray:
    result = minimize(
        Problem(evaluator),
        DE(pop_size=25, CR=0.9, F=0.8),
        termination=("n_eval", evaluations),
        seed=seed,
        verbose=False,
    )
    return np.asarray(
        result.X if result.X is not None else evaluator.extract_active(),
        dtype=float,
    )


def entrypoint():
    evaluator = Evaluator(path_name="init", margin=MARGIN, max_depth=MAX_DEPTH)
    optimize(evaluator, TOTAL_EVALS, seed=23)
    return evaluator.get_best()


AUX_DESCRIPTION = """
GF16 source: a899151c-4745-4b00-8be9-e0ca0618391a.
Observed source result: rank 387, fitness 387.3896425495263, runtime
2600.98635 seconds, timeout_salvaged=1.
Pool role: the strongest single-DE archetype. It keeps TOHPE dominant through
the long ab-initio descent and introduces only a small terminal TODD search.
Universal adaptation: the original rank-394 switch and margin 80 are expressed
as fractions of INITIAL_RANK - TARGET_FINAL_RANK.
Tuning note: its 8000-evaluation DE budget is intentionally large and its
runtime will depend strongly on how often the terminal TODD band is reached.
"""
