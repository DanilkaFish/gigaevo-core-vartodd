"""Beam-2 PSO with three source bands and a wide mid-run restart."""

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

OPTIMIZER_FAMILY = "pymoo_pso_three_band_restart"
SEEDS = [23, 24, 25]
SCOUT_EVALS = 1200
MID_REFINE_EVALS = 800
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
MAX_DEPTH = max(500, RANK_SPAN + 64)


def rank_from_target(fraction: float) -> int:
    return min(
        INITIAL_RANK,
        TARGET_FINAL_RANK + max(1, round(RANK_SPAN * fraction)),
    )


def rank_margin(fraction: float) -> int:
    return max(1, round(RANK_SPAN * fraction))


MID_RANK = rank_from_target(0.66)
TERMINAL_RANK = rank_from_target(0.13)
MID_REOPEN_MARGIN = rank_margin(0.53)


class Evaluator(BaseEvaluator):
    """Use separate early, middle, and terminal TODD retention shapes."""

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
                        self.float_range(-0.5, 2.0),
                        self.float_range(-1.0, 4.0),
                        self.float_range(-0.5, 0.5),
                        self.float_range(-4.0, 4.0),
                        self.float_range(-4.0, 4.0),
                        self.float_range(-4.0, 4.0),
                    ],
                    centers=[0.0] * 6,
                    pow=1,
                ),
            )
        )
        shared_cap = self.int_range(512, 2000, group="todd")
        early_todd = ToddSearch(
            SamplingBudget(one_hot=6, sparse=1, dense=0, sparse_max_weight=2),
            SourcePool(
                keep=self.int_range(5, 10, group="todd"),
                reserve=self.int_range(2, 4, group="todd"),
            ),
            actions_per_bucket=3,
            buckets=ZBucketSearch(
                min_buckets=30,
                max_buckets=shared_cap,
                limit_bucket=shared_cap,
            ),
        )
        middle_todd = ToddSearch(
            SamplingBudget(one_hot=6, sparse=1, dense=0, sparse_max_weight=2),
            SourcePool(
                keep=self.int_range(4, 9, group="todd"),
                reserve=2,
            ),
            actions_per_bucket=4,
            buckets=ZBucketSearch(
                min_buckets=30,
                max_buckets=shared_cap,
                limit_bucket=shared_cap,
            ),
        )
        terminal_cap = self.int_range(200, 800, group="todd")
        terminal_todd = ToddSearch(
            SamplingBudget(one_hot=1, sparse=0, dense=0, sparse_max_weight=1),
            SourcePool(
                keep=self.int_range(3, 8, group="todd"),
                reserve=self.int_range(1, 3, group="todd"),
            ),
            actions_per_bucket=self.int_range(1, 3, group="todd"),
            buckets=ZBucketSearch(
                min_buckets=20,
                max_buckets=terminal_cap,
                limit_bucket=terminal_cap,
            ),
        )
        self.set_todd_search(
            ranks=[INITIAL_RANK, MID_RANK, TERMINAL_RANK],
            values=[early_todd, middle_todd, terminal_todd],
        )
        self.set_tohpe_search(
            TohpeSearch(
                SamplingBudget(
                    one_hot="all", sparse=0, dense=0, sparse_max_weight=2
                ),
                SourcePool(
                    keep=self.int_range(14, 24, group="tohpe"),
                    reserve=self.int_range(6, 12, group="tohpe"),
                ),
                z_choices=self.int_range(10, 18, group="tohpe"),
            )
        )
        self.set_action_selection(
            ActionSelection(beamwidth=2, mode="softmax", temperature=0.35)
        )
        self.set_action_pool(
            ActionPool(final_size=self.int_range(8, 16, group="pool"))
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
        PSO(pop_size=8, w=0.6, c1=1.2, c2=1.2, adaptive=False),
        termination=("n_eval", evaluations),
        seed=seed,
        verbose=False,
    )
    return np.asarray(
        result.X if result.X is not None else evaluator.extract_active(),
        dtype=float,
    )


def entrypoint():
    evaluator = Evaluator(path_name="init", max_depth=MAX_DEPTH)
    params = optimize(evaluator, SCOUT_EVALS, seed=23)
    params = evaluator.set_up_new_init(
        0,
        rank_thr=evaluator.best_rank + MID_REOPEN_MARGIN,
        xopt=params,
    )
    if params is not None:
        optimize(evaluator, MID_REFINE_EVALS, seed=26)
    return evaluator.get_best()


AUX_DESCRIPTION = """
GF16 source: 86ccecc3-e4f1-4fcc-b0ac-9cd4488069d0.
Observed source result: rank 399, fitness 399.3829365079365, runtime
2600.118248 seconds, timeout_salvaged=1.
Pool role: the low-beam, three-band, wide-restart archetype. Early and middle
bands retain more TODD candidates while the terminal band becomes narrower.
Universal adaptation: the source policy transitions are represented at 0.66
and 0.13 of the rank span, and its margin 100 becomes a 0.53 span fraction.
Tuning note: a wide restart deliberately revisits an earlier high-dimensional
region; it is useful only when a second trajectory can justify that work.
"""
