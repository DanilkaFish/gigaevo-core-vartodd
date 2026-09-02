"""Five fixed terminal TODD breadth profiles for action-starved lower ranks."""

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

SEARCH_STRATEGY = "fixed_todd_bucket_breadth_portfolio"
TARGET_POLICY_EVALUATIONS = 2
RANK_SPAN = INITIAL_RANK - TARGET_FINAL_RANK
SEARCH_DEPTH = max(1000, round(RANK_SPAN / 2))
TODD_SWITCH_FRACTION = 0.28
SEEDS = (239, 263)
FALLBACK_TODD_MIN_BUCKETS = 10
FALLBACK_TODD_MAX_BUCKETS = 200
FALLBACK_TODD_LIMIT_BUCKET = 1000
FALLBACK_TODD_KEEP = 3
FALLBACK_TODD_RESERVE = 0
EXPLORATION_RED_WEIGHT = 2.4
FINAL_RED_WEIGHT = 2.8
FINAL_TOHPE_WEIGHT = 0.0
PROFILES = (
    dict(maximum=512, limit=2048, actions=2, keep=4, reserve=1),
    dict(maximum=1500, limit=6000, actions=3, keep=6, reserve=1),
    dict(maximum=4000, limit=16000, actions=4, keep=8, reserve=2),
    dict(maximum=10000, limit=40000, actions=6, keep=12, reserve=3),
    dict(maximum=24000, limit=96000, actions=8, keep=16, reserve=4),
)
TOHPE_FILTERS = ((1, 10), (4, 6), (3, 4))


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4) + k.ntohpe * p.w(5)


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
    def __init__(self, *args, **kwargs):
        self.profile_index = 0
        super().__init__(*args, **kwargs)

    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(lambda x: min(high, low + int((high - low + 1) * (0.5 + 0.5 * np.tanh(float(x) / 2.0)))), group=group)

    def tohpe_filter(self) -> tuple[int, int]:
        return TOHPE_FILTERS[self.int_range(0, 2, group="tohpe_filter")]

    def policy_mapping(self):
        profile = PROFILES[self.profile_index]
        target_min_red, target_max_red = self.tohpe_filter()
        maximum = min(self.buckets_space, profile["maximum"])
        limit = min(self.buckets_space, profile["limit"])
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind([EXPLORATION_RED_WEIGHT, -0.4, 1.2, 0.0, 1.2]),
                final=final_score.bind([FINAL_RED_WEIGHT, -0.5, 1.4, 0.3, 1.8, FINAL_TOHPE_WEIGHT]),
            )
        )
        self.set_action_selection(ActionSelection(beamwidth=5, mode="best"))
        self.set_action_pool(ActionPool(final_size=52))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=SamplingBudget(
                    one_hot=4, sparse=2, dense=0, sparse_max_weight=3
                ),
                pool=SourcePool(keep=28, reserve=4),
                z_choices=12,
                target_min_red=target_min_red,
                target_max_red=target_max_red,
            )
        )
        terminal_todd = ToddSearch(
            sampling=SamplingBudget(one_hot=3, sparse=0, dense=0, sparse_max_weight=2),
            pool=SourcePool(keep=profile["keep"], reserve=profile["reserve"]),
            actions_per_bucket=profile["actions"],
            buckets=ZBucketSearch(
                min_buckets=min(64, maximum),
                max_buckets=maximum,
                limit_bucket=limit,
            ),
        )
        self.set_todd_search(
            ranks=[rank_at_fraction(TODD_SWITCH_FRACTION)],
            values=[fallback_todd(), terminal_todd],
        )

    def run_profile(self, profile_index: int, seed: int) -> None:
        self.profile_index = profile_index
        self.reinit()
        self.run(self.extract_active(), (seed,))


def entrypoint():
    evaluator = Evaluator(
        path_name="init",
        fin_rank=TARGET_FINAL_RANK,
        max_depth=SEARCH_DEPTH,
    )
    for profile_index, seed in zip((0, 4), SEEDS):
        evaluator.run_profile(profile_index, seed)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: test terminal TODD action discovery under explicit finite z breadths.
Cost: at most 2 single-seed policy evaluations; the profiles compare the narrow and broad endpoints.
Evidence: separates researched-bucket breadth, hard limit, per-bucket diversity, and retained TODD actions."""
