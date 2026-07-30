"""Grouped PSO scout followed by heavy-TODD PatternSearch refinement."""

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
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

OPTIMIZER_FAMILY = "pymoo_pso_pattern_heavy_restart"
SEEDS = [40, 41]
SCOUT_PARTICLES = 16
SCOUT_EVALS = 64
TAIL_EVALS = 96
LOWER_BOUND = -1.0
UPPER_BOUND = 1.0
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
MAX_DEPTH = max(500, RANK_SPAN + 64)


def rank_from_target(fraction: float) -> int:
    return min(
        INITIAL_RANK,
        TARGET_FINAL_RANK + max(1, round(RANK_SPAN * fraction)),
    )


def rank_margin(fraction: float) -> int:
    return max(1, round(RANK_SPAN * fraction))


REOPEN_MARGIN = rank_margin(0.26)


class Evaluator(BaseEvaluator):
    """Declare both policy stages, then select one around the restart."""

    def __init__(self, *args, **kwargs):
        self.use_heavy_todd = False
        super().__init__(*args, **kwargs)

    def float_range(self, low: float, high: float, *, group: str) -> float:
        return self.map_par(
            lambda x: low
            + (high - low)
            / (1.0 + np.exp(-np.clip(float(x), -20.0, 20.0))),
            group=group,
        )

    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(
            lambda x: min(
                high,
                low
                + int(
                    (high - low + 1)
                    / (1.0 + np.exp(-np.clip(float(x), -20.0, 20.0)))
                ),
            ),
            group=group,
        )

    def policy_mapping(self):
        self.set_scores(
            PolicyScores(
                ExplorationScore(
                    [
                        self.float_range(-4.0, 4.0, group="scores")
                        for _ in range(5)
                    ],
                    centers=[0.0] * 5,
                    pow=1,
                ),
                FinalizationScore(
                    [
                        self.float_range(-4.0, 4.0, group="scores")
                        for _ in range(6)
                    ],
                    centers=[
                        0.0,
                        self.float_range(0.0, 1.0, group="scores"),
                        0.0,
                        self.float_range(0.0, 1.0, group="scores"),
                        0.0,
                        0.0,
                    ],
                    pow=1,
                ),
            )
        )
        scout_policy = self._build_light_scout_policy()
        tail_policy = self._build_heavy_tail_policy()
        self._install_policy(
            tail_policy if self.use_heavy_todd else scout_policy
        )

    def _build_light_scout_policy(self):
        group = "scout"
        tohpe_samples = SamplingBudget(
            one_hot="all", sparse=0, dense=16, sparse_max_weight=2
        )
        return (
            ActionSelection(beamwidth=2, mode="softmax", temperature=0.22),
            ActionPool(
                final_size=self.int_range(14, 28, group=group)
            ),
            TohpeSearch(
                tohpe_samples,
                SourcePool(
                    keep=self.int_range(3, 8, group=group), reserve=1
                ),
                z_choices=3,
            ),
            ToddSearch(
                SamplingBudget(
                    one_hot=8, sparse=2, dense=2, sparse_max_weight=2
                ),
                SourcePool(keep=3, reserve=1),
                actions_per_bucket=2,
                buckets=ZBucketSearch(
                    min_buckets=8,
                    max_buckets=self.int_range(512, 2000, group=group),
                    limit_bucket=2000,
                ),
            ),
        )

    def _build_heavy_tail_policy(self):
        group = "tail"
        samples = SamplingBudget(
            one_hot="all",
            sparse=self.int_range(4, 16, group=group),
            dense=self.int_range(4, 24, group=group),
            sparse_max_weight=3,
        )
        cap = self.int_range(8000, 36000, group=group)
        return (
            ActionSelection(beamwidth=3, mode="softmax", temperature=0.12),
            ActionPool(
                final_size=self.int_range(28, 52, group=group)
            ),
            TohpeSearch(
                samples, SourcePool(keep=8, reserve=2), z_choices=4
            ),
            ToddSearch(
                samples,
                SourcePool(
                    keep=self.int_range(6, 16, group=group), reserve=2
                ),
                actions_per_bucket=self.int_range(2, 4, group=group),
                buckets=ZBucketSearch(
                    min_buckets=self.int_range(512, 4096, group=group),
                    max_buckets=cap,
                    limit_bucket=cap,
                ),
            ),
        )

    def _install_policy(self, policy):
        selection, action_pool, tohpe, todd = policy
        self.set_action_selection(selection)
        self.set_action_pool(action_pool)
        self.set_tohpe_search(tohpe)
        self.set_todd_search(todd)

    def __call__(self, params: Iterable[float]) -> float:
        values = np.asarray(self.run(params, SEEDS), dtype=float)
        return float(values.min() + 0.02 * values.std())


class Problem(ElementwiseProblem):
    def __init__(self, evaluator: Evaluator):
        size = len(evaluator.extract_active())
        super().__init__(
            n_var=size,
            n_obj=1,
            xl=np.full(size, LOWER_BOUND),
            xu=np.full(size, UPPER_BOUND),
        )
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator(np.asarray(x, dtype=float))


def optimize_pso(
    evaluator: Evaluator, evaluations: int, seed: int
) -> np.ndarray:
    current = evaluator.extract_active()
    if len(current) == 0:
        evaluator([])
        return current
    result = minimize(
        Problem(evaluator),
        PSO(
            pop_size=SCOUT_PARTICLES,
            w=0.7,
            c1=1.4,
            c2=1.4,
            adaptive=False,
        ),
        termination=("n_eval", evaluations),
        seed=seed,
        verbose=False,
    )
    best = np.asarray(
        result.X if result.X is not None else current, dtype=float
    )
    evaluator.insert(best)
    evaluator.reinit()
    return best


def optimize_pattern_search(
    evaluator: Evaluator, evaluations: int, seed: int
) -> np.ndarray:
    current = evaluator.extract_active()
    if len(current) == 0:
        evaluator([])
        return current
    result = minimize(
        Problem(evaluator),
        PatternSearch(
            x0=np.clip(current, LOWER_BOUND, UPPER_BOUND),
            init_delta=0.25,
            init_rho=0.5,
        ),
        termination=("n_eval", evaluations),
        seed=seed,
        verbose=False,
    )
    best = np.asarray(
        result.X if result.X is not None else current, dtype=float
    )
    evaluator.insert(best)
    evaluator.reinit()
    return best


def entrypoint():
    evaluator = Evaluator(path_name="init", max_depth=MAX_DEPTH)
    evaluator.select_parameter_groups("scores", "scout")
    scout_params = optimize_pso(evaluator, SCOUT_EVALS, seed=40)
    evaluator.use_heavy_todd = True
    restarted = evaluator.set_up_new_init(
        0,
        rank_thr=evaluator.best_rank + REOPEN_MARGIN,
        xopt=scout_params,
    )
    if restarted is None:
        return evaluator.get_best()
    evaluator.select_parameter_groups("scores", "tail")
    optimize_pattern_search(evaluator, TAIL_EVALS, seed=44)
    return evaluator.get_best()


AUX_DESCRIPTION = """
GF16 source: 0d61fdac-8502-4e32-a416-4180b6a44691.
Observed source result: rank 399, fitness 399.4265873015873, runtime
658.697077 seconds, timeout_salvaged=0.
Pool role: the explicit heavy-after-restart archetype. A light grouped PSO
scout establishes a path, then PatternSearch tunes scores and heavy-tail knobs.
Universal adaptation: the source margin 48 becomes 0.26 of the rank span and
maximum depth follows the selected matrix and target.
Tuning note: its heavy phase can research 8000-36000 z buckets per evaluation;
the short refinement budget is essential to the source program's cost shape.
"""
