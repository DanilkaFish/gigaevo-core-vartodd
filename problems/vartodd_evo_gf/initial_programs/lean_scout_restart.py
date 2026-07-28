"""Light TODD seed: low-discrepancy scout with local restart refinement."""

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
from scipy.optimize import Bounds, minimize
from scipy.stats import qmc

OPTIMIZER_FAMILY = "scipy_qmc_then_powell"
SEEDS = [29, 30]
SCOUT_POINTS = 160
REFINE_EVALS = 80
REOPEN_MARGINS = [80, 36]
LOWER_BOUND = -1.0
UPPER_BOUND = 1.0


def objective(ranks: list[int]) -> float:
    values = np.asarray(ranks, dtype=float)
    return float(values.min() + 0.02 * values.std())


class Evaluator(BaseEvaluator):
    def float_range(self, low: float, high: float) -> float:
        return self.map_par(lambda x: low + (high - low) * np.clip((float(x) + 1.0) / 2.0, 0.0, 1.0))

    def int_range(self, low: int, high: int) -> int:
        return self.map_par(lambda x: int(round(low + (high - low) * np.clip((float(x) + 1.0) / 2.0, 0.0, 1.0))))

    def policy_mapping(self):
        exploration = ExplorationScore(
            [self.float_range(-4, 4) for _ in range(5)], centers=[0.0, 0.0, 0.0, 0.0, 0.0], pow=1
        )
        finalization = FinalizationScore(
            [self.float_range(-4, 4) for _ in range(6)],
            centers=[0.0, 0.0, 0.0, self.float_range(0.0, 1.0), 0.0, 0.0],
            pow=1,
        )
        self.set_scores(PolicyScores(exploration=exploration, final=finalization))
        samples = SamplingBudget(
            one_hot=20, sparse=self.int_range(4, 20), dense=self.int_range(4, 20), sparse_max_weight=3
        )
        min_buckets = self.int_range(64, 512)
        bucket_cap = self.int_range(2000, 20000)
        self.set_action_selection(ActionSelection(beamwidth=2, mode="softmax", temperature=0.2))
        self.set_action_pool(ActionPool(final_size=self.int_range(16, 36)))
        self.set_tohpe_search(TohpeSearch(samples, SourcePool(keep=self.int_range(5, 14), reserve=1), z_choices=3))
        self.set_tohpeprefix_search(
            TohpePrefixSearch(
                samples,
                SourcePool(keep=self.int_range(4, 12), reserve=2),
                actions_per_bucket=2,
                buckets=ZBucketSearch(min_buckets=min_buckets, max_buckets=bucket_cap, limit_bucket=bucket_cap),
            )
        )
        self.set_todd_search(
            ToddSearch(
                SamplingBudget(one_hot=12, sparse=4, dense=4, sparse_max_weight=2),
                SourcePool(keep=3, reserve=1),
                actions_per_bucket=2,
                buckets=ZBucketSearch(min_buckets=64, max_buckets=4096, limit_bucket=4096),
            )
        )

    def __call__(self, params: Iterable[float]) -> float:
        return objective(self.run(params, SEEDS))


def scout_qmc(evaluator: Evaluator, points: int, seed: int) -> np.ndarray:
    current = evaluator.extract_active()
    dimensions = len(current)
    if dimensions == 0:
        evaluator([])
        return current
    sampler = qmc.LatinHypercube(d=dimensions, seed=seed)
    random_points = sampler.random(max(0, points - 1))
    candidates = np.vstack(
        [
            np.clip(current, LOWER_BOUND, UPPER_BOUND),
            qmc.scale(random_points, np.full(dimensions, LOWER_BOUND), np.full(dimensions, UPPER_BOUND)),
        ]
    )
    best = candidates[0].copy()
    best_value = float("inf")
    for candidate in candidates:
        value = evaluator(candidate)
        if value < best_value:
            best_value = value
            best = np.asarray(candidate, dtype=float)
    evaluator.insert(best)
    evaluator.reinit()
    return best


def refine_powell(evaluator: Evaluator, evaluations: int) -> np.ndarray:
    current = evaluator.extract_active()
    dimensions = len(current)
    if dimensions == 0:
        evaluator([])
        return current
    result = minimize(
        evaluator,
        current,
        method="Powell",
        bounds=Bounds(np.full(dimensions, LOWER_BOUND), np.full(dimensions, UPPER_BOUND)),
        options={"maxfev": evaluations, "maxiter": evaluations, "xtol": 0.08, "ftol": 0.0, "disp": False},
    )
    best = np.asarray(result.x, dtype=float)
    evaluator.insert(best)
    evaluator.reinit()
    return best


def entrypoint():
    evaluator = Evaluator(path_name="init", max_depth=500)
    params = scout_qmc(evaluator, SCOUT_POINTS, seed=29)
    for margin in REOPEN_MARGINS:
        restarted = evaluator.set_up_new_init(0, rank_thr=evaluator.best_rank + margin, xopt=params)
        if restarted is None:
            break
        params = refine_powell(evaluator, REFINE_EVALS)
    return evaluator.get_best()
