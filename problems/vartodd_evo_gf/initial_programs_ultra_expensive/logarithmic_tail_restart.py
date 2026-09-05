"""Frozen logarithmic terminal-tail policy for an expensive descent."""

from helper import (
    INITIAL_RANK, TARGET_FINAL_RANK, ActionPool, ActionSelection,
    BaseEvaluator, PolicyScores, SamplingBudget, SourcePool, ToddSearch,
    TohpeSearch, ZBucketSearch, policy,
)

SEARCH_STRATEGY = "fixed_agnostic_logarithmic_tail_restart"
TARGET_POLICY_EVALUATIONS = 2
SEEDS = (23, 51)
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
TAKEOVER_RANK = TARGET_FINAL_RANK + round(0.12 * RANK_SPAN)
MAX_DEPTH = max(500, RANK_SPAN + 64)


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    return fn.log(1.0 + k.nred) * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4) + k.ntohpe * p.w(5)


def disabled_todd():
    return ToddSearch(SamplingBudget(0, 0, 0, 0), SourcePool(0, 0), 0, ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0))


class Evaluator(BaseEvaluator):
    def __init__(self, *args, **kwargs):
        self.tail_mode = False
        super().__init__(*args, **kwargs)

    def policy_mapping(self):
        self.set_scores(PolicyScores(
            exploration=explore_score.bind([0.4, 1.8, -1.1, -0.7, 1.4]),
            final=final_score.bind([2.1, 1.8, 0.6, 0.0, -0.6, 1.4]),
        ))
        scout = TohpeSearch(SamplingBudget("all", 0, 4, 2), SourcePool(14, 2), 9, 4, 6)
        tail = TohpeSearch(SamplingBudget("all", 6, 10, 3), SourcePool(21, 4), 10, 2, 6)
        todd = ToddSearch(SamplingBudget("all", 6, 8, 3), SourcePool(8, 5), 6, ZBucketSearch(min_buckets=1665, max_buckets=13323, limit_bucket=13323))
        self.set_action_pool(ActionPool(final_size=23))
        self.set_action_selection(ActionSelection(beamwidth=4, mode="softmax", temperature=0.18))
        self.set_tohpe_search([INITIAL_RANK, TAKEOVER_RANK], [scout, tail])
        self.set_todd_search([INITIAL_RANK, TAKEOVER_RANK], [disabled_todd(), todd])


def entrypoint():
    evaluator = Evaluator(path_name="init", fin_rank=TARGET_FINAL_RANK, max_depth=MAX_DEPTH)
    evaluator.run(evaluator.extract_active(), SEEDS[:1])
    evaluator.run(evaluator.extract_active(), SEEDS[1:])
    return evaluator.get_best()


AUX_DESCRIPTION = """Role: logarithmic tail restart. Cost: two single-seed trajectory evaluations with no optimizer. Strategy: use a compressed reduction score and finite high-z TODD only in the reopened tail."""
