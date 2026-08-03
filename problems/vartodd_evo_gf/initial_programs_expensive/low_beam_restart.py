"""Two short low-beam PSO descents with a path restart between them."""

from collections.abc import Iterable

from helper import (
    INITIAL_RANK,
    TARGET_FINAL_RANK,
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    ExplorationScore,
    FinalizationScore,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpeSearch,
    ZBucketSearch,
)
import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


OPTIMIZER_FAMILY = "pymoo_low_beam_pso_restart"
SEARCH_SEEDS = (23,)
SCOUT_EVALS = 70
TAIL_EVALS = 70
TARGET_TOTAL_EVALS = 140
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
MARGIN = max(1, round(0.38 * RANK_SPAN))
REOPEN_MARGIN = max(8, round(0.14 * RANK_SPAN))
MAX_DEPTH = max(500, RANK_SPAN + 64)


def rank_from_target(fraction: float) -> int:
    return min(INITIAL_RANK, TARGET_FINAL_RANK + max(1, round(fraction * RANK_SPAN)))


def rank_margin(fraction: float) -> int:
    return max(1, round(fraction * RANK_SPAN))


def disabled_todd() -> ToddSearch:
    return ToddSearch(
        SamplingBudget(one_hot=0, sparse=0, dense=0, sparse_max_weight=0),
        SourcePool(keep=0, reserve=0),
        actions_per_bucket=0,
        buckets=ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0),
    )


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

    def policy_mapping(self):
        self.set_scores(
            PolicyScores(
                ExplorationScore(
                    [self.float_range(-4.0, 4.0, group="scores") for _ in range(5)],
                    centers=[0.0] * 5,
                    pow=1,
                ),
                FinalizationScore(
                    [self.float_range(-4.0, 4.0, group="scores") for _ in range(6)],
                    centers=[0.0] * 6,
                    pow=1,
                ),
            )
        )
        scout_tohpe = TohpeSearch(
            SamplingBudget(one_hot="all", sparse=0, dense=4, sparse_max_weight=2),
            SourcePool(
                keep=self.int_range(8, 20, group="scout"),
                reserve=self.int_range(1, 4, group="scout"),
            ),
            z_choices=self.int_range(4, 12, group="scout"),
        )
        tail_tohpe = TohpeSearch(
            SamplingBudget(one_hot="all", sparse=4, dense=8, sparse_max_weight=2),
            SourcePool(
                keep=self.int_range(10, 24, group="tail"),
                reserve=self.int_range(1, 5, group="tail"),
            ),
            z_choices=self.int_range(5, 14, group="tail"),
        )
        light_cap = max(32, min(512, round(self.buckets_space * 0.002)))
        light_todd = ToddSearch(
            SamplingBudget(one_hot=6, sparse=0, dense=0, sparse_max_weight=2),
            SourcePool(keep=2, reserve=1),
            actions_per_bucket=2,
            buckets=ZBucketSearch(
                min_buckets=max(8, light_cap // 8),
                max_buckets=light_cap,
                limit_bucket=light_cap,
            ),
        )
        scout_pool = ActionPool(
            final_size=self.int_range(10, 24, group="scout")
        )
        tail_pool = ActionPool(
            final_size=self.int_range(14, 30, group="tail")
        )
        if self.tail_mode:
            self.set_tohpe_search(tail_tohpe)
            self.set_todd_search(light_todd)
            self.set_action_selection(
                ActionSelection(beamwidth=2, mode="softmax", temperature=0.20)
            )
            self.set_action_pool(tail_pool)
        else:
            self.set_tohpe_search(scout_tohpe)
            self.set_todd_search(disabled_todd())
            self.set_action_selection(
                ActionSelection(beamwidth=1, mode="softmax", temperature=0.32)
            )
            self.set_action_pool(scout_pool)

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
        PSO(pop_size=10, w=0.75, c1=0.9, c2=0.9, adaptive=False),
        termination=("n_eval", evaluations),
        seed=seed,
        verbose=False,
    )
    return np.asarray(result.X if result.X is not None else current, dtype=float)


def entrypoint():
    evaluator = Evaluator(path_name="init", margin=MARGIN, max_depth=MAX_DEPTH)
    evaluator.select_parameter_groups("scores", "scout")
    params = optimize(evaluator, SCOUT_EVALS, seed=23)
    evaluator.tail_mode = True
    restarted = evaluator.set_up_new_init(
        0, rank_thr=evaluator.best_rank + REOPEN_MARGIN, xopt=params
    )
    if restarted is not None:
        evaluator.select_parameter_groups("scores", "tail")
        optimize(evaluator, TAIL_EVALS, seed=24)
    return evaluator.get_best()


AUX_DESCRIPTION = """
Pool role: a low-branching stochastic restart strategy. It preserves policy
trial count by keeping beam width at one or two and enables only a tightly
bounded TODD escape source after the best descent has been reopened.
"""
