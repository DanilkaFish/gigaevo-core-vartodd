"""One wide-beam descent followed by two path-relative restarts."""

from helper import (
    INITIAL_RANK,
    TARGET_FINAL_RANK,
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
    PolicyScores,
    policy,
)
import numpy as np

SEARCH_STRATEGY = "fixed_restart_ladder"
TARGET_POLICY_EVALUATIONS = 3
RANK_SPAN = INITIAL_RANK - TARGET_FINAL_RANK
SEARCH_DEPTH = max(1000, round(RANK_SPAN / 2))
SEEDS = (283, 307, 317)
RESTART_MARGINS = (300, 100)
FALLBACK_TODD_MIN_BUCKETS = 10
FALLBACK_TODD_MAX_BUCKETS = 200
FALLBACK_TODD_LIMIT_BUCKET = 1000
FALLBACK_TODD_KEEP = 3
FALLBACK_TODD_RESERVE = 1
EXPLORATION_RED_WEIGHT = 2.7
FINAL_RED_WEIGHT = 3.2
FINAL_TOHPE_WEIGHT = 1.0
TOHPE_FILTERS = ((1, 10), (4, 6), (3, 4))


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4) + k.ntohpe * p.w(5)


def rank_at_fraction(fraction: float) -> int:
    return TARGET_FINAL_RANK + round(RANK_SPAN * fraction)


def restart_rank(best_rank: int, margin: int) -> int:
    return min(INITIAL_RANK - 1, best_rank + margin)


def fallback_todd() -> ToddSearch:
    return ToddSearch(
        sampling=SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2),
        pool=SourcePool(keep=FALLBACK_TODD_KEEP, reserve=FALLBACK_TODD_RESERVE),
        actions_per_bucket=2,
        buckets=ZBucketSearch(
            min_buckets=FALLBACK_TODD_MIN_BUCKETS,
            max_buckets=FALLBACK_TODD_MAX_BUCKETS,
            limit_bucket=FALLBACK_TODD_LIMIT_BUCKET,
        ),
    )


class Evaluator(BaseEvaluator):
    def float_range(self, low: float, high: float, *, group: str) -> float:
        return self.map_par(lambda x: low + (high - low) * (0.5 + 0.5 * np.tanh(float(x) / 2.0)), group=group)

    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(lambda x: min(high, low + int((high - low + 1) * (0.5 + 0.5 * np.tanh(float(x) / 2.0)))), group=group)

    def tohpe_filter(self) -> tuple[int, int]:
        return TOHPE_FILTERS[self.int_range(0, 2, group="tohpe_filter")]

    def policy_mapping(self):
        target_min_red, target_max_red = self.tohpe_filter()
        terminal_maximum = min(self.buckets_space, 12000)
        terminal_limit = min(self.buckets_space, 48000)
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind([EXPLORATION_RED_WEIGHT, -0.5, 0.8, 0.4, 1.0]),
                final=final_score.bind([FINAL_RED_WEIGHT, -0.6, 0.9, 0.6, 1.4, FINAL_TOHPE_WEIGHT]),
            )
        )
        self.set_action_selection(ActionSelection(beamwidth=6, mode="best"))
        self.set_action_pool(ActionPool(final_size=64))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=SamplingBudget(
                    one_hot=5, sparse=3, dense=1, sparse_max_weight=3
                ),
                pool=SourcePool(keep=38, reserve=6),
                z_choices=18,
                target_min_red=target_min_red,
                target_max_red=target_max_red,
            )
        )
        terminal_todd = ToddSearch(
            sampling=SamplingBudget(one_hot=3, sparse=0, dense=0, sparse_max_weight=2),
            pool=SourcePool(keep=10, reserve=3),
            actions_per_bucket=5,
            buckets=ZBucketSearch(
                min_buckets=min(64, terminal_maximum),
                max_buckets=terminal_maximum,
                limit_bucket=terminal_limit,
            ),
        )
        self.set_todd_search(
            ranks=[rank_at_fraction(0.30)],
            values=[fallback_todd(), terminal_todd],
        )

    def run_fixed(self, seed: int) -> None:
        self.reinit()
        self.run(self.extract_active(), (seed,))


def entrypoint():
    evaluator = Evaluator(
        path_name="init",
        fin_rank=TARGET_FINAL_RANK,
        max_depth=SEARCH_DEPTH,
    )
    evaluator.run_fixed(SEEDS[0])
    for seed, margin in zip(SEEDS[1:], RESTART_MARGINS):
        params = evaluator.set_up_new_init(
            0,
            rank_thr=restart_rank(evaluator.best_rank, margin),
            xopt=None,
        )
        if params is None:
            break
        evaluator.run_fixed(seed)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: test wide-beam heavy-tail search with explicit path-relative restarts.
Cost: 1 full descent plus at most 2 single-seed restarts before the completed path's plateau.
Evidence: branches 300 and 100 ranks above the current best instead of requesting unreachable target-relative thresholds."""
