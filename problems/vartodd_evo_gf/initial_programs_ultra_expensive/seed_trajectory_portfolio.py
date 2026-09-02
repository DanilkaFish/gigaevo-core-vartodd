"""Eight independent seeds under one fixed balanced policy."""

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

SEARCH_STRATEGY = "fixed_seed_trajectory_portfolio"
TARGET_POLICY_EVALUATIONS = 2
RANK_SPAN = INITIAL_RANK - TARGET_FINAL_RANK
SEARCH_DEPTH = max(1000, round(RANK_SPAN / 2))
SEEDS = (331, 373)
FALLBACK_TODD_MIN_BUCKETS = 10
FALLBACK_TODD_MAX_BUCKETS = 200
FALLBACK_TODD_LIMIT_BUCKET = 600
FALLBACK_TODD_KEEP = 3
FALLBACK_TODD_RESERVE = 0
# RED_CENTER is an exact reduction count, not normalized. Its negative weight
# targets this count; positive TOHPE weight favors larger future TOHPE space.
RED_CENTER = 4.0
EXPLORATION_RED_WEIGHT = -4.0
FINAL_RED_WEIGHT = -4.0
FINAL_TOHPE_WEIGHT = 4.0
TOHPE_FILTERS = ((1, 10), (4, 6), (3, 4))


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    contested = k.nrank_red <= p.w(0)
    tight = k.nred * p.w(1) + k.ntohpe * p.w(2)
    spread = k.nred * p.w(3) - k.nrank_dim * fn.abs(p.w(4))
    return fn.where(contested, tight, spread) + k.f_todd * p.w(5)


def rank_at_fraction(fraction: float) -> int:
    return TARGET_FINAL_RANK + round(RANK_SPAN * fraction)


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
    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(lambda x: min(high, low + int((high - low + 1) * (0.5 + 0.5 * np.tanh(float(x) / 2.0)))), group=group)

    def tohpe_filter(self) -> tuple[int, int]:
        return TOHPE_FILTERS[self.int_range(0, 2, group="tohpe_filter")]

    def policy_mapping(self):
        target_min_red, target_max_red = self.tohpe_filter()
        terminal_maximum = min(self.buckets_space, 6000)
        terminal_limit = min(self.buckets_space, 24000)
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind([EXPLORATION_RED_WEIGHT, -0.4, 0.9, 0.3, 1.0]),
                final=final_score.bind([0.25, FINAL_RED_WEIGHT, 1.0, -0.5, 1.3, FINAL_TOHPE_WEIGHT]),
            )
        )
        self.set_action_selection(
            ActionSelection(beamwidth=5, mode="softmax", temperature=0.45)
        )
        self.set_action_pool(ActionPool(final_size=52))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=SamplingBudget(
                    one_hot=4, sparse=2, dense=1, sparse_max_weight=3
                ),
                pool=SourcePool(keep=32, reserve=5),
                z_choices=16,
                target_min_red=target_min_red,
                target_max_red=target_max_red,
            )
        )
        terminal_todd = ToddSearch(
            sampling=SamplingBudget(one_hot=2, sparse=0, dense=0, sparse_max_weight=2),
            pool=SourcePool(keep=8, reserve=2),
            actions_per_bucket=4,
            buckets=ZBucketSearch(
                min_buckets=min(48, terminal_maximum),
                max_buckets=terminal_maximum,
                limit_bucket=terminal_limit,
            ),
        )
        self.set_todd_search(
            ranks=[rank_at_fraction(0.25)],
            values=[fallback_todd(), terminal_todd],
        )

    def run_seed(self, seed: int) -> None:
        self.reinit()
        self.run(self.extract_active(), (seed,))


def entrypoint():
    evaluator = Evaluator(
        path_name="init",
        fin_rank=TARGET_FINAL_RANK,
        max_depth=SEARCH_DEPTH,
    )
    for seed in SEEDS:
        evaluator.run_seed(seed)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: stochastic-trajectory control under one unchanged balanced policy.
Cost: exactly 2 sequential single-seed policy evaluations when the deadline permits.
Evidence: compares search randomness under negative reduction weights and light initial TODD fallback."""
