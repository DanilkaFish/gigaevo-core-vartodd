"""Frozen exhaustive terminal profile for an expensive descent."""

from helper import (
    INITIAL_RANK, TARGET_FINAL_RANK, ActionPool, ActionSelection,
    BaseEvaluator, PolicyScores, SamplingBudget, SourcePool, ToddSearch,
    TohpeSearch, ZBucketSearch, policy,
)

SEARCH_STRATEGY = "fixed_agnostic_exhaustive_terminal"
TARGET_POLICY_EVALUATIONS = 1
SEEDS = (51, 52, 53)
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
TAKEOVER_RANK = TARGET_FINAL_RANK + round(0.13 * RANK_SPAN)
MAX_DEPTH = max(500, RANK_SPAN + 64)


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    gate = fn.min(fn.max((0.5 - k.nrank_red) * 4.0, 0.0), 1.0)
    rich = k.nred * p.w(0) + k.ntohpe * p.w(1) - k.nrank_dim * fn.abs(p.w(2))
    lean = k.nred * p.w(3) + (k.red / fn.max(k.bucket, 1)) * p.w(4) + k.ndim * p.w(5)
    return gate * rich + (1.0 - gate) * lean + k.f_todd * p.w(6)


def disabled_todd():
    return ToddSearch(SamplingBudget(0, 0, 0, 0), SourcePool(0, 0), 0, ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0))


class Evaluator(BaseEvaluator):
    def policy_mapping(self):
        self.set_scores(PolicyScores(
            exploration=explore_score.bind([1.70, -1.34, -0.47, -1.78, 1.79]),
            final=final_score.bind([0.57, 1.80, 0.77, -0.85, 0.14, 1.52, -1.70]),
        ))
        early = TohpeSearch(SamplingBudget("all", 0, 8, 2), SourcePool(20, 2), 12, 4, 6)
        late = TohpeSearch(SamplingBudget("all", 8, 12, 3), SourcePool(10, 2), 6, 2, 6)
        todd = ToddSearch(SamplingBudget("all", 12, 12, 3), SourcePool(8, 2), 4, ZBucketSearch(min_buckets=14726, max_buckets=353442, limit_bucket=353442))
        self.set_action_pool(ActionPool(final_size=40))
        self.set_action_selection([INITIAL_RANK, TAKEOVER_RANK], [ActionSelection(beamwidth=2, mode="softmax", temperature=0.28), ActionSelection(beamwidth=4, mode="softmax", temperature=0.13)])
        self.set_tohpe_search([INITIAL_RANK, TAKEOVER_RANK], [early, late])
        self.set_todd_search([INITIAL_RANK, TAKEOVER_RANK], [disabled_todd(), todd])


def entrypoint():
    evaluator = Evaluator(path_name="init", fin_rank=TARGET_FINAL_RANK, max_depth=MAX_DEPTH)
    evaluator.run(evaluator.extract_active(), SEEDS)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: exhaustive terminal coverage. Cost: one three-seed trajectory evaluation with no optimizer. Strategy: gate rich and lean score terms, then spend the tail on a broad finite TODD search."""
