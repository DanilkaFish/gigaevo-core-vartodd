"""Frozen PSO/pattern-derived terminal schedule for an expensive descent."""

from helper import (
    INITIAL_RANK, TARGET_FINAL_RANK, ActionPool, ActionSelection,
    BaseEvaluator, PolicyScores, SamplingBudget, SourcePool, ToddSearch,
    TohpeSearch, ZBucketSearch, policy,
)

SEARCH_STRATEGY = "fixed_agnostic_pso_pattern_terminal"
TARGET_POLICY_EVALUATIONS = 1
SEEDS = (37, 41, 53)
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
TAKEOVER_RANK = TARGET_FINAL_RANK + round(0.17 * RANK_SPAN)
MAX_DEPTH = max(500, RANK_SPAN + 64)


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.ntohpe * p.w(2) + k.f_todd * p.w(3)


def disabled_todd():
    return ToddSearch(SamplingBudget(0, 0, 0, 0), SourcePool(0, 0), 0, ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0))


class Evaluator(BaseEvaluator):
    def policy_mapping(self):
        self.set_scores(PolicyScores(
            exploration=explore_score.bind([1.8, 0.7, -0.8, 1.7, 0.5]),
            final=final_score.bind([1.9, -0.6, -0.4, -1.2]),
        ))
        early = TohpeSearch(SamplingBudget(42, 0, 4, 2), SourcePool(23, 7), 9, 4, 6)
        late = TohpeSearch(SamplingBudget("all", 6, 6, 3), SourcePool(23, 7), 9, 4, 6)
        todd = ToddSearch(SamplingBudget("all", 6, 6, 3), SourcePool(8, 3), 5, ZBucketSearch(min_buckets=64, max_buckets=24000, limit_bucket=24000))
        self.set_action_pool(ActionPool(final_size=19))
        self.set_action_selection(ActionSelection(beamwidth=3, mode="softmax", temperature=0.226))
        self.set_tohpe_search([INITIAL_RANK, TAKEOVER_RANK], [early, late])
        self.set_todd_search([INITIAL_RANK, TAKEOVER_RANK], [disabled_todd(), todd])


def entrypoint():
    evaluator = Evaluator(path_name="init", fin_rank=TARGET_FINAL_RANK, max_depth=MAX_DEPTH)
    evaluator.run(evaluator.extract_active(), SEEDS)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: broad-beam terminal schedule. Cost: one three-seed trajectory evaluation with no optimizer. Strategy: keep a moderate beam, then inject finite TODD in the contracted terminal band."""
