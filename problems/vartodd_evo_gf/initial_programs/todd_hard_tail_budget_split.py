"""Heavy TODD seed: rank-scheduled policy optimized by bounded NGOpt stages."""

from collections.abc import Iterable
from helper import (
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    ExplorationScore,
    FinalizationScore,
    INITIAL_RANK,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    TARGET_FINAL_RANK,
    ToddSearch,
    TohpePrefixSearch,
    TohpeSearch,
    ZBucketSearch,
)
import nevergrad as ng
import numpy as np

OPTIMIZER_FAMILY = "nevergrad_ngopt_with_restart"
SWITCH_RANK = TARGET_FINAL_RANK + 55
SCHEDULE = [INITIAL_RANK, SWITCH_RANK]
SEEDS = [23, 24]
FIRST_STAGE_BUDGET = 64
RESTART_BUDGET = 48
REOPEN_MARGIN = 55
TODD_ROLE = "heavy_tail_schedule"
LOWER_BOUND = -2.0
UPPER_BOUND = 2.0
EARLY_TODD_LIMIT = 512
TERMINAL_PREFIX_LIMIT = 24000


class Evaluator(BaseEvaluator):
    """Use finite TODD early and full z coverage only near the target."""

    def float_range(self, low: float, high: float) -> float:
        return self.map_par(lambda x: low + (high - low) * (0.5 + 0.5 * np.tanh(float(x) / 2.2)))

    def int_range(self, low: int, high: int) -> int:
        return self.map_par(lambda x: min(high, low + int((high - low + 1) * (0.5 + 0.5 * np.tanh(float(x) / 2.2)))))

    def policy_mapping(self):
        self.set_scores(
            PolicyScores(
                ExplorationScore(
                    [self.float_range(-3.5, 3.5) for _ in range(5)], centers=[0.0, 0.0, 0.0, 0.0, 0.0], pow=1
                ),
                FinalizationScore(
                    [self.float_range(-3.5, 3.5) for _ in range(6)],
                    centers=[
                        0.0,
                        0.0,
                        0.0,
                        self.float_range(0.0, 1.0),
                        self.float_range(0.0, 1.0),
                        0.0,
                    ],
                    pow=1,
                ),
            )
        )
        tail_keep = self.int_range(8, 18)
        tail_samples = SamplingBudget(
            one_hot="all", sparse=self.int_range(4, 16), dense=self.int_range(4, 20), sparse_max_weight=3
        )
        terminal_actions_per_bucket = self.int_range(2, 5)
        terminal_min_buckets = self.int_range(512, 2048)
        self.set_action_selection(
            SCHEDULE,
            [ActionSelection(beamwidth=2, mode="best"), ActionSelection(beamwidth=3, mode="softmax", temperature=0.14)],
        )
        self.set_action_pool(SCHEDULE, [ActionPool(final_size=24), ActionPool(final_size=self.int_range(28, 56))])
        self.set_tohpe_search(
            SCHEDULE,
            [
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=16, dense=12, sparse_max_weight=3),
                    SourcePool(keep=self.int_range(12, 32), reserve=1),
                    z_choices=self.int_range(4, 10),
                ),
                TohpeSearch(
                    SamplingBudget(one_hot=24, sparse=8, dense=6, sparse_max_weight=3),
                    SourcePool(keep=8, reserve=0),
                    z_choices=2,
                ),
            ],
        )
        light_prefix = TohpePrefixSearch(
            SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2),
            SourcePool(keep=2, reserve=0),
            actions_per_bucket=2,
            buckets=ZBucketSearch(min_buckets=8, max_buckets=128, limit_bucket=512),
        )
        heavy_prefix = TohpePrefixSearch(
            tail_samples,
            SourcePool(keep=tail_keep, reserve=2),
            actions_per_bucket=terminal_actions_per_bucket,
            buckets=ZBucketSearch(
                min_buckets=terminal_min_buckets, max_buckets=TERMINAL_PREFIX_LIMIT, limit_bucket=TERMINAL_PREFIX_LIMIT
            ),
        )
        self.set_tohpeprefix_search(SCHEDULE, [light_prefix, heavy_prefix])
        self.set_todd_search(
            SCHEDULE,
            [
                ToddSearch(
                    SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2),
                    SourcePool(keep=2, reserve=1),
                    actions_per_bucket=2,
                    buckets=ZBucketSearch(min_buckets=8, max_buckets=128, limit_bucket=EARLY_TODD_LIMIT),
                ),
                ToddSearch(
                    tail_samples,
                    SourcePool(keep=tail_keep, reserve=2),
                    actions_per_bucket=terminal_actions_per_bucket,
                    buckets=ZBucketSearch(min_buckets=terminal_min_buckets, max_buckets=100000, limit_bucket=-1),
                ),
            ],
        )

    def __call__(self, params: Iterable[float]) -> float:
        values = np.asarray(self.run(params, SEEDS), dtype=float)
        return float(values.min() + 0.015 * values.std())


def optimize_ngopt(evaluator: Evaluator, budget: int, seed: int) -> np.ndarray:
    current = evaluator.extract_active()
    dimensions = len(current)
    if dimensions == 0:
        evaluator([])
        return current
    parametrization = ng.p.Array(init=current).set_bounds(LOWER_BOUND, UPPER_BOUND)
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=budget, num_workers=1)
    optimizer.parametrization.random_state.seed(seed)
    best = current.copy()
    best_value = float("inf")
    for _ in range(budget):
        candidate = optimizer.ask()
        vector = np.asarray(candidate.value, dtype=float)
        value = evaluator(vector)
        optimizer.tell(candidate, value)
        if value < best_value:
            best_value = value
            best = vector.copy()
    evaluator.insert(best)
    evaluator.reinit()
    return best


def entrypoint():
    evaluator = Evaluator(path_name="init", max_depth=500)
    params = optimize_ngopt(evaluator, FIRST_STAGE_BUDGET, seed=23)
    restarted = evaluator.set_up_new_init(0, rank_thr=evaluator.best_rank + REOPEN_MARGIN, xopt=params)
    if restarted is not None:
        optimize_ngopt(evaluator, RESTART_BUDGET, seed=27)
    return evaluator.get_best()
