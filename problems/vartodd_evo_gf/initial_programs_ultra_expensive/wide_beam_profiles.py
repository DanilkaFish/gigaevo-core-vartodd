"""Four fixed wide-beam TOHPE profiles that emphasize trajectory coverage."""

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

SEARCH_STRATEGY = "fixed_wide_beam_portfolio"
TARGET_POLICY_EVALUATIONS = 1
RANK_SPAN = INITIAL_RANK - TARGET_FINAL_RANK
SEARCH_DEPTH = max(1000, round(RANK_SPAN / 2))
SEEDS = (269,)
FALLBACK_TODD_MIN_BUCKETS = 10
FALLBACK_TODD_MAX_BUCKETS = 200
FALLBACK_TODD_LIMIT_BUCKET = 400
FALLBACK_TODD_KEEP = 2
FALLBACK_TODD_RESERVE = 0
# RED_CENTER is an exact reduction count, not normalized. Its negative weight
# targets this count; positive TOHPE weight favors larger future TOHPE space.
RED_CENTER = 3.0
EXPLORATION_RED_WEIGHT = -4.0
FINAL_RED_WEIGHT = -4.0
FINAL_TOHPE_WEIGHT = 4.0
PROFILES = (
    dict(beam=8, final=40, keep=28, reserve=4, z=12),
    dict(beam=24, final=72, keep=48, reserve=8, z=18),
    dict(beam=50, final=120, keep=76, reserve=14, z=24),
    dict(beam=100, final=200, keep=120, reserve=24, z=32),
)
TOHPE_FILTERS = ((1, 10), (4, 6), (3, 4))


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4) + k.ntohpe * p.w(5)


def fallback_todd() -> ToddSearch:
    return ToddSearch(
        sampling=SamplingBudget(one_hot=6, sparse=2, dense=0, sparse_max_weight=2),
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
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind([EXPLORATION_RED_WEIGHT, -0.4, 0.8, 0.3, 1.0]),
                final=final_score.bind([FINAL_RED_WEIGHT, -0.5, 0.8, 0.5, 1.2, FINAL_TOHPE_WEIGHT]),
            )
        )
        self.set_action_selection(
            ActionSelection(beamwidth=profile["beam"], mode="best")
        )
        self.set_action_pool(ActionPool(final_size=profile["final"]))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=SamplingBudget(
                    one_hot=4, sparse=2, dense=1, sparse_max_weight=3
                ),
                pool=SourcePool(keep=profile["keep"], reserve=profile["reserve"]),
                z_choices=profile["z"],
                target_min_red=target_min_red,
                target_max_red=target_max_red,
            )
        )
        self.set_todd_search(fallback_todd())

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
    evaluator.run_profile(0, SEEDS[0])
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: measure whether broad trajectory enumeration beats policy fitting.
Cost: one single-seed evaluation at beam 8.
Evidence: tests broad trajectory retention with negative reduction weights and a light TODD fallback."""
