"""Frozen TOHPE-only policy for an expensive descent."""

from helper import (
    INITIAL_RANK, TARGET_FINAL_RANK, ActionPool, ActionSelection,
    BaseEvaluator, PolicyScores, SamplingBudget, SourcePool, ToddSearch,
    TohpeSearch, ZBucketSearch, policy,
)

SEARCH_STRATEGY = "fixed_agnostic_tohpe_only"
TARGET_POLICY_EVALUATIONS = 1
SEEDS = (29, 31)
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
            exploration=explore_score.bind([1.35, 0.09, -0.95, -1.79, 1.84]),
            final=final_score.bind([-1.15, -0.30, -0.99, -0.36, 0.37, 0.40]),
        ))
        early = TohpeSearch(SamplingBudget("all", 0, 8, 2), SourcePool(23, 5), 11, 4, 6)
        late = TohpeSearch(SamplingBudget("all", 0, 8, 2), SourcePool(23, 5), 11, 4, 6)
        self.set_action_pool(ActionPool(final_size=24))
        self.set_action_selection(ActionSelection(beamwidth=2, mode="softmax", temperature=0.24))
        self.set_tohpe_search([INITIAL_RANK, TAKEOVER_RANK], [early, late])
        self.set_todd_search(disabled_todd())


def entrypoint():
    evaluator = Evaluator(path_name="init", fin_rank=TARGET_FINAL_RANK, max_depth=MAX_DEPTH)
    evaluator.run(evaluator.extract_active(), SEEDS)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: TOHPE-only control. Cost: one two-seed trajectory evaluation with no optimizer. Strategy: disable TODD completely to measure the contribution of TOHPE search alone."""
