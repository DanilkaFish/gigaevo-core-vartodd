"""Heavy-after-restart seed: PSO scout followed by annealed tail search."""

from collections.abc import Iterable
from helper import (
    ActionPool,
    ActionSelection,
    BaseEvaluator,
    ExplorationScore,
    FinalizationScore,
    PolicyScores,
    SamplingBudget,
    SourcePool,
    ToddSearch,
    TohpePrefixSearch,
    TohpeSearch,
    ZBucketSearch,
)
import numpy as np
import pyswarms as ps
from scipy.optimize import dual_annealing

OPTIMIZER_FAMILY = "pyswarms_pso_then_scipy_dual_annealing"
SEEDS = [40, 41]
SCOUT_PARTICLES = 16
SCOUT_ITERS = 4
TAIL_EVALS = 96
REOPEN_MARGIN = 48
TODD_ROLE = "heavy_after_restart"
LOWER_BOUND = -1.0
UPPER_BOUND = 1.0


class Evaluator(BaseEvaluator):
    """Declare both stages permanently; the restart only selects the policy."""

    def __init__(self, *args, **kwargs):
        self.use_heavy_todd = False
        super().__init__(*args, **kwargs)

    def float_range(self, low: float, high: float, *, group: str) -> float:
        return self.map_par(lambda x: low + (high - low) / (1.0 + np.exp(-np.clip(float(x), -20.0, 20.0))), group=group)

    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(
            lambda x: min(high, low + int((high - low + 1) / (1.0 + np.exp(-np.clip(float(x), -20.0, 20.0))))),
            group=group,
        )

    def policy_mapping(self):
        self.set_scores(
            PolicyScores(
                ExplorationScore(
                    [self.float_range(-4, 4, group="scores") for _ in range(5)],
                    centers=[0.0, 0.5, 0.0, 0.5, 0.0],
                    pow=1,
                ),
                FinalizationScore(
                    [self.float_range(-4, 4, group="scores") for _ in range(6)],
                    centers=[0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
                    pow=1,
                ),
            )
        )
        scout_policy = self._build_light_scout_policy()
        tail_policy = self._build_heavy_tail_policy()
        if self.use_heavy_todd:
            self._install_policy(tail_policy)
        else:
            self._install_policy(scout_policy)

    def _build_light_scout_policy(self):
        group = "scout"
        samples = SamplingBudget(one_hot="all", sparse=0, dense=16, sparse_max_weight=2)
        cap = self.int_range(2000, 14000, group=group)
        return (
            ActionSelection(beamwidth=2, mode="softmax", temperature=0.22),
            ActionPool(final_size=self.int_range(14, 28, group=group)),
            TohpeSearch(samples, SourcePool(keep=self.int_range(3, 8, group=group), reserve=1), z_choices=3),
            TohpePrefixSearch(
                samples,
                SourcePool(keep=self.int_range(4, 10, group=group), reserve=1),
                actions_per_bucket=2,
                buckets=ZBucketSearch(min_buckets=64, max_buckets=cap, limit_bucket=cap),
            ),
            ToddSearch(
                SamplingBudget(one_hot=8, sparse=2, dense=2, sparse_max_weight=2),
                SourcePool(keep=3, reserve=1),
                actions_per_bucket=2,
                buckets=ZBucketSearch(
                    min_buckets=8, max_buckets=self.int_range(512, 2000, group=group), limit_bucket=2000
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
        prefix_keep = self.int_range(8, 20, group=group)
        todd_keep = self.int_range(6, 16, group=group)
        actions_per_bucket = self.int_range(2, 4, group=group)
        min_buckets = self.int_range(512, 4096, group=group)
        return (
            ActionSelection(beamwidth=3, mode="softmax", temperature=0.12),
            ActionPool(final_size=self.int_range(28, 52, group=group)),
            TohpeSearch(samples, SourcePool(keep=8, reserve=2), z_choices=4),
            TohpePrefixSearch(
                samples,
                SourcePool(keep=prefix_keep, reserve=2),
                actions_per_bucket=actions_per_bucket,
                buckets=ZBucketSearch(min_buckets=min_buckets, max_buckets=cap, limit_bucket=cap),
            ),
            ToddSearch(
                samples,
                SourcePool(keep=todd_keep, reserve=2),
                actions_per_bucket=actions_per_bucket,
                buckets=ZBucketSearch(min_buckets=min_buckets, max_buckets=cap, limit_bucket=cap),
            ),
        )

    def _install_policy(self, policy):
        selection, action_pool, tohpe, prefix, todd = policy
        self.set_action_selection(selection)
        self.set_action_pool(action_pool)
        self.set_tohpe_search(tohpe)
        self.set_tohpeprefix_search(prefix)
        self.set_todd_search(todd)

    def __call__(self, params: Iterable[float]) -> float:
        values = np.asarray(self.run(params, SEEDS), dtype=float)
        return float(values.min() + 0.02 * values.std())


def optimize_pso(evaluator: Evaluator, iterations: int, seed: int) -> np.ndarray:
    current = evaluator.extract_active()
    dimensions = len(current)
    if dimensions == 0:
        evaluator([])
        return current
    rng = np.random.default_rng(seed)
    positions = rng.uniform(LOWER_BOUND, UPPER_BOUND, size=(SCOUT_PARTICLES, dimensions))
    positions[0] = np.clip(current, LOWER_BOUND, UPPER_BOUND)
    optimizer = ps.single.GlobalBestPSO(
        n_particles=SCOUT_PARTICLES,
        dimensions=dimensions,
        options={"c1": 0.4, "c2": 0.4, "w": 0.7},
        bounds=(np.full(dimensions, LOWER_BOUND), np.full(dimensions, UPPER_BOUND)),
        init_pos=positions,
    )

    def objective(positions):
        return np.asarray([evaluator(position) for position in positions], dtype=float)

    _, best = optimizer.optimize(objective, iters=iterations, verbose=False)
    best = np.asarray(best, dtype=float)
    evaluator.insert(best)
    evaluator.reinit()
    return best


def optimize_annealing(evaluator: Evaluator, evaluations: int, seed: int) -> np.ndarray:
    current = evaluator.extract_active()
    dimensions = len(current)
    if dimensions == 0:
        evaluator([])
        return current
    result = dual_annealing(
        evaluator,
        bounds=[(LOWER_BOUND, UPPER_BOUND)] * dimensions,
        maxiter=evaluations,
        maxfun=evaluations,
        rng=np.random.default_rng(seed),
        no_local_search=True,
        x0=current,
    )
    best = np.asarray(result.x, dtype=float)
    evaluator.insert(best)
    evaluator.reinit()
    return best


def entrypoint():
    evaluator = Evaluator(path_name="init", max_depth=500)
    evaluator.select_parameter_groups("scores", "scout")
    scout_params = optimize_pso(evaluator, SCOUT_ITERS, seed=40)
    evaluator.use_heavy_todd = True
    restarted = evaluator.set_up_new_init(0, rank_thr=evaluator.best_rank + REOPEN_MARGIN, xopt=scout_params)
    if restarted is None:
        return evaluator.get_best()
    evaluator.select_parameter_groups("scores", "tail")
    optimize_annealing(evaluator, TAIL_EVALS, seed=44)
    return evaluator.get_best()
