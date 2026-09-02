"""One inexpensive schedule that introduces finite TODD in the lower band."""

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

SEARCH_STRATEGY = "fixed_late_todd_schedule_portfolio"
TARGET_POLICY_EVALUATIONS = 1
RANK_SPAN = INITIAL_RANK - TARGET_FINAL_RANK
SEARCH_DEPTH = max(1000, round(RANK_SPAN / 2))
SEEDS = (199,)
FALLBACK_TODD_MIN_BUCKETS = 10
FALLBACK_TODD_MAX_BUCKETS = 200
FALLBACK_TODD_LIMIT_BUCKET = 600
FALLBACK_TODD_KEEP = 3
FALLBACK_TODD_RESERVE = 0
# RED_CENTER is an exact reduction count, not normalized. Its negative weight
# targets this count; positive TOHPE weight favors larger future TOHPE space.
RED_CENTER = 3.0
EXPLORATION_RED_WEIGHT = -4.0
FINAL_RED_WEIGHT = -4.0
FINAL_TOHPE_WEIGHT = 4.0
PROFILES = (
    dict(switch=0.12, early=0, maximum=256, limit=1024, actions=2, keep=4, reserve=1),
    dict(switch=0.18, early=0, maximum=512, limit=2048, actions=2, keep=5, reserve=1),
    dict(switch=0.25, early=8, maximum=1000, limit=4000, actions=3, keep=6, reserve=2),
    dict(switch=0.32, early=16, maximum=2000, limit=8000, actions=4, keep=8, reserve=2),
    dict(
        switch=0.42, early=32, maximum=4000, limit=16000, actions=5, keep=10, reserve=3
    ),
    dict(
        switch=0.55, early=64, maximum=8000, limit=32000, actions=6, keep=12, reserve=4
    ),
)
TOHPE_FILTERS = ((1, 10), (4, 6), (3, 4))


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    blend = fn.clip((0.5 - k.nrank_red) * 4.0, 0.0, 1.0)
    tight = k.nred * p.w(0) + k.ntohpe * p.w(1) - k.nrank_dim * fn.abs(p.w(2))
    spread = k.nred * p.w(3) + k.nzw * p.w(4)
    return blend * tight + (1.0 - blend) * spread + k.ntohpe * p.w(5)


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


def make_todd(
    *, maximum: int, limit: int, actions: int, keep: int, reserve: int
) -> ToddSearch:
    if maximum == 0:
        return ToddSearch(
            sampling=SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=0),
            pool=SourcePool(keep=0, reserve=0),
            actions_per_bucket=0,
            buckets=ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0),
        )
    return ToddSearch(
        sampling=SamplingBudget(one_hot=4, sparse=1, dense=0, sparse_max_weight=2),
        pool=SourcePool(keep=keep, reserve=reserve),
        actions_per_bucket=actions,
        buckets=ZBucketSearch(
            min_buckets=min(32, maximum),
            max_buckets=maximum,
            limit_bucket=limit,
        ),
    )


class Evaluator(BaseEvaluator):
    def __init__(self, *args, **kwargs):
        self.profile_index = 0
        super().__init__(*args, **kwargs)

    def float_range(self, low: float, high: float, *, group: str) -> float:
        return self.map_par(lambda x: low + (high - low) * (0.5 + 0.5 * np.tanh(float(x) / 2.0)), group=group)

    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(lambda x: min(high, low + int((high - low + 1) * (0.5 + 0.5 * np.tanh(float(x) / 2.0)))), group=group)

    def tohpe_filter(self) -> tuple[int, int]:
        return TOHPE_FILTERS[self.int_range(0, 2, group="tohpe_filter")]

    def policy_mapping(self):
        target_min_red, target_max_red = self.tohpe_filter()
        profile = PROFILES[self.profile_index]
        available = self.buckets_space
        terminal_maximum = min(available, profile["maximum"])
        terminal_limit = min(available, profile["limit"])
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind([EXPLORATION_RED_WEIGHT, -0.5, 1.0, 0.2, 1.0]),
                final=final_score.bind([FINAL_RED_WEIGHT, -0.6, 1.0, 0.4, 1.4, FINAL_TOHPE_WEIGHT]),
            )
        )
        self.set_action_selection(ActionSelection(beamwidth=4, mode="best"))
        self.set_action_pool(ActionPool(final_size=44))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=SamplingBudget(
                    one_hot=5, sparse=2, dense=1, sparse_max_weight=3
                ),
                pool=SourcePool(keep=28, reserve=4),
                z_choices=14,
                target_min_red=target_min_red,
                target_max_red=target_max_red,
            )
        )
        terminal_todd = make_todd(
            maximum=terminal_maximum,
            limit=terminal_limit,
            actions=profile["actions"],
            keep=profile["keep"],
            reserve=profile["reserve"],
        )
        self.set_todd_search(
            ranks=[rank_at_fraction(profile["switch"])],
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
    for profile_index, seed in zip((0,), SEEDS):
        evaluator.run_profile(profile_index, seed)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: test one inexpensive late-TODD schedule in a long ab-initio descent.
Cost: one single-seed policy evaluation, with early TOHPE doing most work.
Evidence: uses the narrowest terminal schedule over an always-available 10..200 TODD fallback."""
