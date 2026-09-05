"""Frozen narrow-retention high-z policy for an expensive descent."""

from helper import (
    INITIAL_RANK, TARGET_FINAL_RANK, ActionPool, ActionSelection,
    BaseEvaluator, PolicyScores, SamplingBudget, SourcePool, ToddSearch,
    TohpeSearch, ZBucketSearch, policy,
)

SEARCH_STRATEGY = "fixed_agnostic_narrow_high_z_tail"
TARGET_POLICY_EVALUATIONS = 1
SEEDS = (37, 41, 53)
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
TAKEOVER_RANK = TARGET_FINAL_RANK + round(0.84 * RANK_SPAN)
MAX_DEPTH = max(500, RANK_SPAN + 64)


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    return fn.where(k.nrank_red <= -1.438, k.nred * 1.682 + k.ntohpe * -0.593, k.nred * 0.126 - k.nrank_dim * fn.abs(-1.77)) + k.f_todd * 1.698


def disabled_todd():
    return ToddSearch(SamplingBudget(0, 0, 0, 0), SourcePool(0, 0), 0, ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0))


class Evaluator(BaseEvaluator):
    def policy_mapping(self):
        self.set_scores(PolicyScores(
            exploration=explore_score.bind([1.85, 0.69, 0.12, -1.80, 1.83]),
            final=final_score.bind([]),
        ))
        scout = TohpeSearch(SamplingBudget("all", 0, 6, 2), SourcePool(20, 4), 8, 4, 6)
        tail = TohpeSearch(SamplingBudget("all", 6, 10, 3), SourcePool(8, 2), 5, 2, 6)
        todd = ToddSearch(SamplingBudget("all", 10, 10, 3), SourcePool(5, 1), 3, ZBucketSearch(min_buckets=12259, max_buckets=294221, limit_bucket=294221))
        self.set_action_pool(ActionPool(final_size=30))
        self.set_action_selection(ActionSelection(beamwidth=2, mode="softmax", temperature=0.14))
        self.set_tohpe_search([INITIAL_RANK, TAKEOVER_RANK], [scout, tail])
        self.set_todd_search([INITIAL_RANK, TAKEOVER_RANK], [disabled_todd(), todd])


def entrypoint():
    evaluator = Evaluator(path_name="init", fin_rank=TARGET_FINAL_RANK, max_depth=MAX_DEPTH)
    evaluator.run(evaluator.extract_active(), SEEDS)
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: narrow high-z tail. Cost: one three-seed trajectory evaluation with no optimizer. Strategy: use a low-beam scout and reserve the large bucket budget for terminal escape."""
