"""Low-discrepancy TOHPE policy search for very expensive evaluations."""

from collections.abc import Iterable

from helper import (
    INITIAL_RANK,
    TARGET_FINAL_RANK,
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
    policy,
)
import numpy as np


OPTIMIZER_FAMILY = "numpy_stratified_tohpe_scout"
SEARCH_SEEDS = (11,)
TARGET_TOTAL_EVALS = 100
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
MARGIN = max(1, round(0.40 * RANK_SPAN))
MAX_DEPTH = max(500, RANK_SPAN + 64)
TOHPE_FILTERS = ((1, 10), (4, 6), (3, 4))


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + fn.log(1.0 + k.nbucket) * p.w(2) + fn.exp(-k.nzw) * p.w(3) + k.nyw * p.w(4)


@policy.final
def final_score(k, p, fn):
    share = fn.clip(k.f_tohpe, 0.0, 1.0)
    band = fn.exp(-fn.abs(k.nzw - p.w(2)) * 4.0) * p.w(3)
    return k.nred * p.w(0) + k.ndim * p.w(1) + share * p.w(4) + band + k.ntohpe * p.w(5)


def rank_from_target(fraction: float) -> int:
    return min(INITIAL_RANK, TARGET_FINAL_RANK + max(1, round(fraction * RANK_SPAN)))


def rank_margin(fraction: float) -> int:
    return max(1, round(fraction * RANK_SPAN))


class Evaluator(BaseEvaluator):
    def float_range(self, low: float, high: float, *, group: str) -> float:
        return self.map_par(
            lambda x: low + (high - low) * np.clip((float(x) + 1.0) / 2.0, 0.0, 1.0),
            group=group,
        )

    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(
            lambda x: int(round(low + (high - low) * np.clip((float(x) + 1.0) / 2.0, 0.0, 1.0))),
            group=group,
        )

    def tohpe_filter(self) -> tuple[int, int]:
        return TOHPE_FILTERS[self.int_range(0, len(TOHPE_FILTERS) - 1, group="tohpe_filter")]

    def policy_mapping(self):
        target_min_red, target_max_red = self.tohpe_filter()
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind([self.float_range(-4.0, 4.0, group="scores") for _ in range(explore_score.n_params)]),
                final=final_score.bind([self.float_range(-4.0, 4.0, group="scores") for _ in range(final_score.n_params)]),
            )
        )
        self.set_tohpe_search(
            TohpeSearch(
                SamplingBudget(
                    one_hot="all",
                    sparse=self.int_range(0, 20, group="tohpe"),
                    dense=self.int_range(0, 24, group="tohpe"),
                    sparse_max_weight=2,
                ),
                SourcePool(
                    keep=self.int_range(8, 28, group="tohpe"),
                    reserve=self.int_range(1, 5, group="tohpe"),
                ),
                z_choices=self.int_range(4, 20, group="tohpe"),
                target_min_red=target_min_red,
                target_max_red=target_max_red,
            )
        )
        self.set_todd_search(
            ToddSearch(
                SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=0),
                SourcePool(keep=0, reserve=0),
                actions_per_bucket=0,
                buckets=ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0),
            )
        )
        self.set_action_selection(
            ActionSelection(beamwidth=2, mode="softmax", temperature=0.30)
        )
        self.set_action_pool(
            ActionPool(final_size=self.int_range(12, 32, group="pool"))
        )

    def __call__(self, params: Iterable[float]) -> float:
        return float(min(self.run(params, SEARCH_SEEDS)))


def stratified_search(evaluator: Evaluator, evaluations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dimensions = len(evaluator.extract_active())
    strata = (np.arange(evaluations)[:, None] + rng.random((evaluations, dimensions))) / evaluations
    candidates = 2.0 * strata - 1.0
    for column in range(dimensions):
        rng.shuffle(candidates[:, column])
    best_x = evaluator.extract_active()
    best_f = float("inf")
    for candidate in candidates:
        fitness = evaluator(candidate)
        if fitness < best_f:
            best_f = fitness
            best_x = np.asarray(candidate, dtype=float)
    return best_x


def entrypoint():
    evaluator = Evaluator(path_name="init", margin=MARGIN, max_depth=MAX_DEPTH)
    stratified_search(evaluator, TARGET_TOTAL_EVALS, seed=11)
    return evaluator.get_best()


AUX_DESCRIPTION = """
Pool role: the lowest-cost policy-diversity control for expensive TODD runs.
It spends one hundred single-seed evaluations on stratified TOHPE policies,
keeps TODD disabled, and relies on stochastic path generation for diversity.
"""
