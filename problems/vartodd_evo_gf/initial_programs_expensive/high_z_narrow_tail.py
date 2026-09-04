"""High-z terminal coverage with deliberately narrow path retention."""

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
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


OPTIMIZER_FAMILY = "pymoo_de_high_z_narrow_tail"
SEARCH_SEEDS = (37,)
SCOUT_EVALS = 80
TAIL_EVALS = 120
TARGET_TOTAL_EVALS = 200
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
MARGIN = max(1, round(0.44 * RANK_SPAN))
REOPEN_MARGIN = max(8, round(0.10 * RANK_SPAN))
MAX_DEPTH = max(500, RANK_SPAN + 64)
TOHPE_FILTERS = ((1, 10), (4, 6), (3, 4))


@policy.exploration
def explore_score(k, p, fn):
    return k.nred * p.w(0) + k.ndim * p.w(1) + k.nbucket * p.w(2) + k.nyw * p.w(3) + k.nzw * p.w(4)


@policy.final
def final_score(k, p, fn):
    contested = k.nrank_red <= p.w(0)
    tight = k.nred * p.w(1) + k.ntohpe * p.w(2)
    spread = k.nred * p.w(3) - k.nrank_dim * fn.abs(p.w(4))
    return fn.where(contested, tight, spread) + k.f_todd * p.w(5)


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
        self.target_min_red, self.target_max_red = self.tohpe_filter()
        self.set_scores(
            PolicyScores(
                exploration=explore_score.bind([self.float_range(-4.0, 4.0, group="scores") for _ in range(explore_score.n_params)]),
                final=final_score.bind([self.float_range(-4.0, 4.0, group="scores") for _ in range(final_score.n_params)]),
            )
        )
        scout = self._scout_policy()
        tail = self._tail_policy()
        selection, pool, tohpe, todd = tail if self.tail_mode else scout
        self.set_action_selection(selection)
        self.set_action_pool(pool)
        self.set_tohpe_search(tohpe)
        self.set_todd_search(todd)

    def _scout_policy(self):
        return (
            ActionSelection(beamwidth=1, mode="softmax", temperature=0.30),
            ActionPool(final_size=self.int_range(12, 26, group="scout")),
            TohpeSearch(
                SamplingBudget(one_hot="all", sparse=0, dense=6, sparse_max_weight=2),
                SourcePool(
                    keep=self.int_range(8, 24, group="scout"),
                    reserve=self.int_range(1, 5, group="scout"),
                ),
                z_choices=self.int_range(4, 14, group="scout"),
                target_min_red=self.target_min_red,
                target_max_red=self.target_max_red,
            ),
            ToddSearch(
                SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=0),
                SourcePool(keep=0, reserve=0),
                actions_per_bucket=0,
                buckets=ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0),
            ),
        )

    def _tail_policy(self):
        cap = max(512, round(self.buckets_space * 0.18))
        return (
            ActionSelection(beamwidth=2, mode="softmax", temperature=0.14),
            ActionPool(final_size=self.int_range(20, 40, group="tail")),
            TohpeSearch(
                SamplingBudget(one_hot="all", sparse=6, dense=10, sparse_max_weight=3),
                SourcePool(keep=8, reserve=2),
                z_choices=5,
                target_min_red=self.target_min_red,
                target_max_red=self.target_max_red,
            ),
            ToddSearch(
                SamplingBudget(one_hot="all", sparse=10, dense=10, sparse_max_weight=3),
                SourcePool(keep=self.int_range(3, 6, group="tail"), reserve=1),
                actions_per_bucket=self.int_range(2, 3, group="tail"),
                buckets=ZBucketSearch(
                    min_buckets=max(128, cap // 24),
                    max_buckets=cap,
                    limit_bucket=cap,
                ),
            ),
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
        DE(pop_size=8, variant="DE/rand/1/bin", CR=0.85, F=0.65),
        SCOUT_EVALS,
        seed=37,
    )
    evaluator.tail_mode = True
    restarted = evaluator.set_up_new_init(
        0, rank_thr=evaluator.best_rank + REOPEN_MARGIN, xopt=params
    )
    if restarted is not None:
        evaluator.select_parameter_groups("scores", "tail")
        optimize(
            evaluator,
            PatternSearch(x0=np.clip(evaluator.extract_active(), -1.0, 1.0), init_delta=0.2),
            TAIL_EVALS,
            seed=38,
        )
    return evaluator.get_best()


AUX_DESCRIPTION = """
Pool role: the high-z, narrow-retention specialist. Its restarted tail spends
each policy evaluation on broad TODD bucket coverage while keeping beam and
retained TODD paths small enough to control tree expansion.
"""
