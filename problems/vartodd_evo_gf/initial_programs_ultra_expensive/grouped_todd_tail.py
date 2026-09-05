"""Frozen grouped-search tail policy for an expensive descent."""

from helper import (
    INITIAL_RANK, TARGET_FINAL_RANK, ActionPool, ActionSelection,
    BaseEvaluator, PolicyScores, SamplingBudget, SourcePool, ToddSearch,
    TohpeSearch, ZBucketSearch, policy,
)

SEARCH_STRATEGY = "fixed_agnostic_grouped_todd_tail"
TARGET_POLICY_EVALUATIONS = 1
SEEDS = (29, 31, 37)
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
TAKEOVER_RANK = TARGET_FINAL_RANK + round(0.12 * RANK_SPAN)
MAX_DEPTH = max(500, RANK_SPAN + 64)


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4) + k.ntohpe * p.w(5)


def disabled_todd():
    return ToddSearch(SamplingBudget(0, 0, 0, 0), SourcePool(0, 0), 0, ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0))


class Evaluator(BaseEvaluator):
    def policy_mapping(self):
        self.set_scores(PolicyScores(
            exploration=explore_score.bind([0.77, -0.18, -0.29, -0.52, 0.16]),
            final=final_score.bind([0.49, -0.62, -0.85, 1.54, -0.88, 0.15]),
        ))
        early = TohpeSearch(SamplingBudget("all", 0, 8, 2), SourcePool(21, 2), 8, 4, 6)
        late = TohpeSearch(SamplingBudget("all", 4, 8, 2), SourcePool(10, 2), 6, 2, 6)
        todd = ToddSearch(SamplingBudget("all", 2, 2, 2), SourcePool(5, 3), 2, ZBucketSearch(min_buckets=3122, max_buckets=49962, limit_bucket=49962))
        self.set_action_pool([INITIAL_RANK, TAKEOVER_RANK], [ActionPool(final_size=19), ActionPool(final_size=27)])
        self.set_action_selection(ActionSelection(beamwidth=2, mode="softmax", temperature=0.24))
        self.set_tohpe_search([INITIAL_RANK, TAKEOVER_RANK], [early, late])
        self.set_todd_search([INITIAL_RANK, TAKEOVER_RANK], [disabled_todd(), todd])


def entrypoint():
    evaluator = Evaluator(path_name="init", fin_rank=TARGET_FINAL_RANK, max_depth=MAX_DEPTH)
    evaluator.run(evaluator.extract_active(), SEEDS)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: grouped TODD tail. Cost: one three-seed trajectory evaluation with no optimizer. Strategy: keep a cheap TOHPE scout, then retain a small source pool for finite TODD."""
