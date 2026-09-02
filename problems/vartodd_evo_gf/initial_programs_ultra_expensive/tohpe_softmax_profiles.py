"""Rank-scheduled softmax trajectories for very costly full descents."""

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

SEARCH_STRATEGY = "fixed_tohpe_softmax_portfolio"
TARGET_POLICY_EVALUATIONS = 3
RANK_SPAN = INITIAL_RANK - TARGET_FINAL_RANK
SEARCH_DEPTH = max(1000, round(RANK_SPAN / 2))
SEEDS = (131, 139, 157)
FALLBACK_TODD_MIN_BUCKETS = 10
FALLBACK_TODD_MAX_BUCKETS = 200
FALLBACK_TODD_LIMIT_BUCKET = 1000
FALLBACK_TODD_KEEP = 2
FALLBACK_TODD_RESERVE = 0
# RED_CENTER is an exact reduction count, not normalized. Its negative weight
# targets this count; positive TOHPE weight favors larger future TOHPE space.
RED_CENTER = 5.0
EXPLORATION_RED_WEIGHT = -4.0
FINAL_RED_WEIGHT = -4.0
FINAL_TOHPE_WEIGHT = 4.0
PROFILES = (
    dict(
        beams=(2, 4, 6), temperatures=(1.40, 0.45, 0.12),
        final=24, keep=14, reserve=2, z=5,
    ),
    dict(
        beams=(3, 5, 6), temperatures=(0.30, 1.00, 0.25),
        final=32, keep=18, reserve=3, z=8,
    ),
    dict(
        beams=(4, 6, 8), temperatures=(1.80, 0.80, 0.40),
        final=40, keep=22, reserve=3, z=10,
    ),
)
TOHPE_FILTERS = ((1, 10), (4, 6), (3, 4))


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    contested = k.nrank_red <= p.w(0)
    tight = k.nred * p.w(1) + k.ntohpe * p.w(2)
    spread = k.nred * p.w(3) - k.nrank_dim * fn.abs(p.w(4))
    return fn.where(contested, tight, spread) + k.f_tohpe * p.w(5)


def rank_at_fraction(fraction: float) -> int:
    return TARGET_FINAL_RANK + round(RANK_SPAN * fraction)


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
                exploration=explore_score.bind([EXPLORATION_RED_WEIGHT, -0.4, 0.9, -0.3, 0.8]),
                final=final_score.bind([0.25, FINAL_RED_WEIGHT, 0.7, -0.4, 1.0, FINAL_TOHPE_WEIGHT]),
            )
        )
        self.set_action_selection(
            ranks=[rank_at_fraction(0.82), rank_at_fraction(0.55)],
            values=[
                ActionSelection(
                    beamwidth=profile["beams"][0],
                    mode="softmax",
                    temperature=profile["temperatures"][0],
                ),
                ActionSelection(
                    beamwidth=profile["beams"][1],
                    mode="softmax",
                    temperature=profile["temperatures"][1],
                ),
                ActionSelection(
                    beamwidth=profile["beams"][2],
                    mode="softmax",
                    temperature=profile["temperatures"][2],
                ),
            ],
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
    for profile_index, seed in zip((0, 1, 2), SEEDS):
        evaluator.run_profile(profile_index, seed)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: rank-scheduled stochastic-selection ab-initio experiment.
Cost: at most 3 single-seed policy evaluations, with no numerical optimizer.
Evidence: compares cooling, reheating, and broad stochastic schedules around the early plateau while retaining small TOHPE pools and a finite TODD fallback."""
