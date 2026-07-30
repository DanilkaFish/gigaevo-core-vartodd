"""Chunked beam-3 PSO with light TODD activated near the target."""

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

OPTIMIZER_FAMILY = "pymoo_chunked_pso_light_todd"
SEEDS = [23, 24, 25]
TOTAL_EVALS = 1000
CHUNK_EVALS = 600
PATIENCE_CHUNKS = 2
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
MAX_DEPTH = max(500, RANK_SPAN + 64)


def rank_from_target(fraction: float) -> int:
    return min(
        INITIAL_RANK,
        TARGET_FINAL_RANK + max(1, round(RANK_SPAN * fraction)),
    )


def rank_margin(fraction: float) -> int:
    return max(1, round(RANK_SPAN * fraction))


TERMINAL_RANK = rank_from_target(0.12)
MARGIN = rank_margin(0.43)


def disabled_todd() -> ToddSearch:
    return ToddSearch(
        SamplingBudget(one_hot=2, sparse=0, dense=0, sparse_max_weight=2),
        SourcePool(keep=0, reserve=0),
        actions_per_bucket=1,
        buckets=ZBucketSearch(min_buckets=0, max_buckets=0, limit_bucket=0),
    )


class Evaluator(BaseEvaluator):
    """Prefer reduction and give terminal ranks finite deep-source coverage."""

    def float_range(
        self, low: float, high: float, *, group: str = "scores"
    ) -> float:
        return self.map_par(
            lambda x: low
            + (high - low)
            / (1.0 + np.exp(-np.clip(float(x), -8.0, 8.0) / 2.5)),
            group=group,
        )

    def int_range(
        self, low: int, high: int, *, group: str = "policy"
    ) -> int:
        return self.map_par(
            lambda x: min(
                high,
                low
                + int(
                    (high - low + 1)
                    / (1.0 + np.exp(-np.clip(float(x), -8.0, 8.0) / 2.5))
                ),
            ),
            group=group,
        )

    def policy_mapping(self):
        self.set_scores(
            PolicyScores(
                ExplorationScore(
                    [self.float_range(-4.0, 4.0) for _ in range(5)],
                    centers=[0.0] * 5,
                    pow=1,
                ),
                FinalizationScore(
                    [
                        self.float_range(0.2, 2.0),
                        self.float_range(-4.0, 4.0),
                        self.float_range(-0.5, 1.0),
                        self.float_range(-4.0, 4.0),
                        self.float_range(-4.0, 4.0),
                        self.float_range(-4.0, 4.0),
                    ],
                    centers=[0.0] * 6,
                    pow=1,
                ),
            )
        )
        light_todd = ToddSearch(
            SamplingBudget(one_hot=2, sparse=0, dense=0, sparse_max_weight=2),
            SourcePool(keep=4, reserve=2),
            actions_per_bucket=4,
            buckets=ZBucketSearch(
                min_buckets=30, max_buckets=500, limit_bucket=500
            ),
        )
        self.set_todd_search(
            ranks=[TERMINAL_RANK],
            values=[disabled_todd(), light_todd],
        )
        self.set_tohpe_search(
            TohpeSearch(
                SamplingBudget(
                    one_hot="all", sparse=0, dense=0, sparse_max_weight=2
                ),
                SourcePool(keep=20, reserve=8),
                z_choices=12,
            )
        )
        self.set_action_selection(
            ActionSelection(beamwidth=3, mode="softmax", temperature=0.35)
        )
        self.set_action_pool(ActionPool(final_size=12))

    def __call__(self, params: Iterable[float]) -> float:
        values = np.asarray(self.run(params, SEEDS), dtype=float)
        return float(values.min() + 0.02 * values.std())


class Problem(ElementwiseProblem):
    def __init__(self, evaluator: Evaluator):
        size = len(evaluator.extract_active())
        super().__init__(
            n_var=size,
            n_obj=1,
            xl=np.full(size, -1.0),
            xu=np.full(size, 1.0),
        )
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator(np.asarray(x, dtype=float))


def optimize(evaluator: Evaluator, evaluations: int, seed: int) -> np.ndarray:
    consumed = 0
    best_before = evaluator.best_rank
    unimproved_chunks = 0
    result_x = None
    while consumed < evaluations:
        this_chunk = min(CHUNK_EVALS, evaluations - consumed)
        result = minimize(
            Problem(evaluator),
            PSO(pop_size=20, w=0.7, c1=0.6, c2=0.6, adaptive=True),
            termination=("n_eval", this_chunk),
            seed=seed + consumed,
            verbose=False,
        )
        consumed += this_chunk
        result_x = np.asarray(
            result.X
            if result.X is not None
            else evaluator.extract_active(),
            dtype=float,
        )
        if evaluator.best_rank < best_before:
            best_before = evaluator.best_rank
            unimproved_chunks = 0
        else:
            unimproved_chunks += 1
        if unimproved_chunks >= PATIENCE_CHUNKS:
            break
    if result_x is None:
        return np.asarray(evaluator.extract_active(), dtype=float)
    return result_x


def entrypoint():
    evaluator = Evaluator(path_name="init", margin=MARGIN, max_depth=MAX_DEPTH)
    optimize(evaluator, TOTAL_EVALS, seed=23)
    return evaluator.get_best()


AUX_DESCRIPTION = """
GF16 source: 4e37489e-8572-46ac-8303-e426d6b9ce8e.
Observed source result: rank 393, fitness 393.41518235793046, runtime
2600.173453 seconds, timeout_salvaged=1.
Pool role: the compact beam-3 and stagnation-aware optimizer archetype. PSO
runs in 600-evaluation chunks and stops after two unproductive chunks.
Universal adaptation: the original rank-402 switch and margin 80 are fractions
of INITIAL_RANK - TARGET_FINAL_RANK.
Tuning note: the patience rule can stop early on landscapes where improvements
arrive later than two chunks, but it avoids repeatedly searching a stale basin.
"""
