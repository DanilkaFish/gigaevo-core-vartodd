"""Six transparent greedy TOHPE baselines for very costly full descents."""

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

SEARCH_STRATEGY = "fixed_tohpe_greedy_portfolio"
TARGET_POLICY_EVALUATIONS = 3
RANK_SPAN = INITIAL_RANK - TARGET_FINAL_RANK
SEARCH_DEPTH = max(1000, round(RANK_SPAN / 2))
SEEDS = (101, 107, 127)
TOHPE_FILTERS = ((1, 10), (4, 6), (3, 4))
PROFILES = (
    dict(beam=1, final=12, keep=8, reserve=1, one_hot=4, sparse=0, dense=0, z=2),
    dict(beam=2, final=18, keep=12, reserve=2, one_hot=5, sparse=2, dense=0, z=3),
    dict(beam=2, final=24, keep=16, reserve=2, one_hot=6, sparse=4, dense=1, z=5),
    dict(beam=3, final=30, keep=20, reserve=3, one_hot=8, sparse=6, dense=2, z=8),
    dict(beam=3, final=38, keep=26, reserve=4, one_hot=10, sparse=8, dense=4, z=12),
    dict(beam=4, final=48, keep=32, reserve=5, one_hot=10, sparse=10, dense=6, z=18),
)


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4) + k.ntohpe * p.w(5)


def disabled_todd() -> ToddSearch:
    return ToddSearch(
        sampling=SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=0),
        pool=SourcePool(keep=0, reserve=0),
        actions_per_bucket=0,
        buckets=ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0),
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
                exploration=explore_score.bind([2.5, -0.5, 0.8, -0.4, 0.6]),
                final=final_score.bind([3.0, -0.7, 0.5, -0.5, 0.8, 0.2]),
            )
        )
        self.set_action_selection(
            ActionSelection(beamwidth=profile["beam"], mode="best")
        )
        self.set_action_pool(ActionPool(final_size=profile["final"]))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=SamplingBudget(
                    one_hot=profile["one_hot"],
                    sparse=profile["sparse"],
                    dense=profile["dense"],
                    sparse_max_weight=3,
                ),
                pool=SourcePool(keep=profile["keep"], reserve=profile["reserve"]),
                z_choices=profile["z"],
                target_min_red=target_min_red,
                target_max_red=target_max_red,
            )
        )
        self.set_todd_search(disabled_todd())

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
    for profile_index, seed in zip((0, 2, 5), SEEDS):
        evaluator.run_profile(profile_index, seed)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: deterministic ab-initio TOHPE baseline.
Cost: at most 3 single-seed policy evaluations, with no numerical optimizer.
Evidence: compares explicit greedy beam, retention, sampling, and z-choice profiles while TODD is disabled."""
