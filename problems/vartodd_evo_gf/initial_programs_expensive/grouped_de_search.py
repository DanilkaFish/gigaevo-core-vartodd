"""Grouped DE search with cheap source discovery before finite TODD."""

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
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


OPTIMIZER_FAMILY = "pymoo_grouped_de_restart"
SEARCH_SEEDS = (29,)
SCOUT_EVALS = 80
TAIL_EVALS = 80
TARGET_TOTAL_EVALS = 160
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
    def __init__(self, *args, **kwargs):
        self.tail_mode = False
        super().__init__(*args, **kwargs)

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
        scout_tohpe = TohpeSearch(
            SamplingBudget(one_hot="all", sparse=0, dense=8, sparse_max_weight=2),
            SourcePool(
                keep=self.int_range(10, 28, group="scout"),
                reserve=self.int_range(1, 6, group="scout"),
            ),
                z_choices=self.int_range(4, 16, group="scout"),
                target_min_red=target_min_red,
                target_max_red=target_max_red,
        )
        tail_cap = max(128, round(self.buckets_space * 0.03))
        tail_todd = ToddSearch(
            SamplingBudget(one_hot=10, sparse=2, dense=2, sparse_max_weight=2),
            SourcePool(
                keep=self.int_range(3, 8, group="tail"),
                reserve=self.int_range(1, 3, group="tail"),
            ),
            actions_per_bucket=self.int_range(2, 3, group="tail"),
            buckets=ZBucketSearch(
                min_buckets=max(16, tail_cap // 16),
                max_buckets=tail_cap,
                limit_bucket=tail_cap,
            ),
        )
        scout_pool = ActionPool(
            final_size=self.int_range(14, 30, group="scout")
        )
        tail_pool = ActionPool(
            final_size=self.int_range(18, 36, group="tail")
        )
        if self.tail_mode:
            self.set_tohpe_search(
                TohpeSearch(
                    SamplingBudget(one_hot="all", sparse=4, dense=8, sparse_max_weight=2),
                    SourcePool(keep=10, reserve=2),
                    z_choices=6,
                    target_min_red=target_min_red,
                    target_max_red=target_max_red,
                )
            )
            self.set_todd_search(tail_todd)
            self.set_action_pool(tail_pool)
        else:
            self.set_tohpe_search(scout_tohpe)
            self.set_todd_search(
                ToddSearch(
                    SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=0),
                    SourcePool(keep=0, reserve=0),
                    actions_per_bucket=0,
                    buckets=ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0),
                )
            )
            self.set_action_pool(scout_pool)
        self.set_action_selection(
            ActionSelection(beamwidth=2, mode="softmax", temperature=0.24)
        )

    def __call__(self, params: Iterable[float]) -> float:
        return float(min(self.run(params, SEARCH_SEEDS)))


class Problem(ElementwiseProblem):
    def __init__(self, evaluator: Evaluator):
        size = len(evaluator.extract_active())
        super().__init__(n_var=size, n_obj=1, xl=np.full(size, -1.0), xu=np.full(size, 1.0))
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator(np.asarray(x, dtype=float))


def optimize(evaluator: Evaluator, evaluations: int, seed: int) -> np.ndarray:
    current = evaluator.extract_active()
    result = minimize(
        Problem(evaluator),
        DE(pop_size=8, variant="DE/rand/1/bin", CR=0.85, F=0.65),
        termination=("n_eval", evaluations),
        seed=seed,
        verbose=False,
    )
    return np.asarray(result.X if result.X is not None else current, dtype=float)


def entrypoint():
    evaluator = Evaluator(path_name="init", margin=MARGIN, max_depth=MAX_DEPTH)
    evaluator.select_parameter_groups("scores", "scout")
    params = optimize(evaluator, SCOUT_EVALS, seed=29)
    evaluator.tail_mode = True
    restarted = evaluator.set_up_new_init(
        0, rank_thr=evaluator.best_rank + REOPEN_MARGIN, xopt=params
    )
    if restarted is not None:
        evaluator.select_parameter_groups("scores", "tail")
        optimize(evaluator, TAIL_EVALS, seed=30)
    return evaluator.get_best()


AUX_DESCRIPTION = """
Pool role: a grouped differential-evolution strategy. It searches cheap TOHPE
generation first, then gives a restarted tail moderate matrix-scaled TODD
coverage without spending the initial descent on deep z research.
"""
