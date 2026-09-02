"""TOHPE scout followed by restarted score-only PatternSearch."""

from collections.abc import Iterable

from helper import (
    INITIAL_RANK,
    TARGET_FINAL_RANK,
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
    policy,
)
import numpy as np
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


OPTIMIZER_FAMILY = "pymoo_pso_then_score_pattern"
SEARCH_SEEDS = (17,)
SCOUT_EVALS = 60
TAIL_EVALS = 60
TARGET_TOTAL_EVALS = 120
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
MARGIN = max(1, round(0.40 * RANK_SPAN))
REOPEN_MARGIN = max(8, round(0.12 * RANK_SPAN))
MAX_DEPTH = max(500, RANK_SPAN + 64)
TOHPE_FILTERS = ((1, 10), (4, 6), (3, 4))


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4) + k.ntohpe * p.w(5)


def rank_from_target(fraction: float) -> int:
    return min(INITIAL_RANK, TARGET_FINAL_RANK + max(1, round(fraction * RANK_SPAN)))


def rank_margin(fraction: float) -> int:
    return max(1, round(fraction * RANK_SPAN))


class Evaluator(BaseEvaluator):
    def float_range(self, low: float, high: float, *, group: str) -> float:
        return self.map_par(
            lambda x: low + (high - low) / (1.0 + np.exp(-float(x))), group=group
        )

    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(
            lambda x: min(high, low + int((high - low + 1) / (1.0 + np.exp(-float(x))))),
            group=group,
        )

    def tohpe_filter(self) -> tuple[int, int]:
        return TOHPE_FILTERS[self.int_range(0, len(TOHPE_FILTERS) - 1, group="tohpe_filter")]

    def policy_mapping(self):
        target_min_red, target_max_red = self.tohpe_filter()
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind([self.float_range(-4.0, 4.0, group="scores") for _ in range(explore_score.n_params)]),
                final=final_score.bind([self.float_range(-4.0, 4.0, group="scores") for _ in range(final_score.n_params)]),
            )
        )
        self.set_tohpe_search(
            TohpeSearch(
                SamplingBudget(one_hot="all", sparse=0, dense=8, sparse_max_weight=2),
                SourcePool(
                    keep=self.int_range(10, 30, group="scout"),
                    reserve=self.int_range(1, 6, group="scout"),
                ),
                z_choices=self.int_range(4, 16, group="scout"),
                target_min_red=target_min_red,
                target_max_red=target_max_red,
            )
        )
        self.set_todd_search(
            ToddSearch(
                SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=0),
                SourcePool(keep=0, reserve=0),
                actions_per_bucket=0,
                buckets=ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0),
            )
        )
        self.set_action_selection(
            ActionSelection(beamwidth=2, mode="softmax", temperature=0.25)
        )
        self.set_action_pool(
            ActionPool(final_size=self.int_range(12, 28, group="scout"))
        )

    def __call__(self, params: Iterable[float]) -> float:
        return float(min(self.run(params, SEARCH_SEEDS)))


class Problem(ElementwiseProblem):
    def __init__(self, evaluator: Evaluator):
        super().__init__(
            n_var=len(evaluator.extract_active()),
            n_obj=1,
            xl=np.full(len(evaluator.extract_active()), -1.0),
            xu=np.full(len(evaluator.extract_active()), 1.0),
        )
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator(np.asarray(x, dtype=float))


def optimize(evaluator: Evaluator, algorithm, evaluations: int, seed: int) -> np.ndarray:
    current = evaluator.extract_active()
    result = minimize(
        Problem(evaluator), algorithm, termination=("n_eval", evaluations), seed=seed, verbose=False
    )
    return np.asarray(result.X if result.X is not None else current, dtype=float)


def entrypoint():
    evaluator = Evaluator(path_name="init", margin=MARGIN, max_depth=MAX_DEPTH)
    evaluator.select_parameter_groups("scores", "scout")
    params = optimize(
        evaluator,
        PSO(pop_size=10, w=0.7, c1=1.0, c2=1.0, adaptive=False),
        SCOUT_EVALS,
        seed=17,
    )
    restarted = evaluator.set_up_new_init(
        0, rank_thr=evaluator.best_rank + REOPEN_MARGIN, xopt=params
    )
    if restarted is not None:
        evaluator.select_parameter_groups("scores")
        optimize(
            evaluator,
            PatternSearch(x0=np.clip(evaluator.extract_active(), -1.0, 1.0), init_delta=0.3),
            TAIL_EVALS,
            seed=18,
        )
    return evaluator.get_best()


AUX_DESCRIPTION = """
Pool role: a cheap-source staged control. PSO identifies a useful TOHPE
descent, then a path restart spends the remaining budget only on score weights
instead of repeatedly paying to optimize action-generation dimensions.
"""
