"""Four genuinely distinct score shapes under one generation policy."""

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

SEARCH_STRATEGY = "fixed_score_shape_portfolio"
TARGET_POLICY_EVALUATIONS = 4
RANK_SPAN = INITIAL_RANK - TARGET_FINAL_RANK
SEARCH_DEPTH = max(1000, round(RANK_SPAN / 2))
SEEDS = (163, 173, 181, 197)
FALLBACK_TODD_MIN_BUCKETS = 10
FALLBACK_TODD_MAX_BUCKETS = 200
FALLBACK_TODD_LIMIT_BUCKET = 1000
FALLBACK_TODD_KEEP = 3
FALLBACK_TODD_RESERVE = 0
PROFILES = (
    dict(
        e=(4.0, -0.4, 0.2, 0.0, 0.0),
        f=(4.0, -0.5, 0.2, 0.0, 0.0, 0.4),
        c=(0.0, 0.0),
    ),
    dict(
        e=(-1.5, 2.0, 0.3, 0.0, 0.4),
        f=(-1.0, 1.8, 0.3, 0.0, 0.4, 0.2),
        c=(0.0, 0.0),
    ),
    dict(
        e=(0.5, 0.4, 2.4, 1.8, 0.2),
        f=(0.8, 0.5, 2.2, 2.0, 0.3, 0.0),
        c=(0.45, 0.0),
    ),
    dict(
        e=(0.3, 0.8, 0.4, -0.6, 2.6),
        f=(0.5, 1.0, 0.5, -0.8, 2.5, -0.3),
        c=(0.0, 0.55),
    ),
)
PROFILE_INDICES = (0, 1, 2, 3)
TOHPE_FILTERS = ((1, 10), (4, 6), (3, 4))


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    contested = k.nrank_red <= 0.25
    tight = k.nred * p.w(0) + k.ntohpe * p.w(1)
    spread = k.nred * p.w(2) - k.nrank_dim * fn.abs(p.w(3)) + k.nyw * p.w(4) + k.nzw * p.w(5)
    return fn.where(contested, tight, spread)


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
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind(list(profile["e"])),
                final=final_score.bind(list(profile["f"])),
            )
        )
        self.set_action_selection(ActionSelection(beamwidth=3, mode="best"))
        self.set_action_pool(ActionPool(final_size=32))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=SamplingBudget(
                    one_hot=4, sparse=2, dense=1, sparse_max_weight=3
                ),
                pool=SourcePool(keep=22, reserve=3),
                z_choices=10,
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
    for profile_index, seed in zip(PROFILE_INDICES, SEEDS):
        evaluator.run_profile(profile_index, seed)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: isolate genuinely different fixed score shapes under one generation policy.
Cost: at most 4 single-seed policy evaluations, with no numerical optimizer.
Evidence: compares reduction-heavy, negative-reduction/dimension-preserving, bucket/y-heavy, and z-heavy hypotheses without overwriting profile weights."""
