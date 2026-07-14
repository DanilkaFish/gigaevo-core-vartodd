import random
from typing import Iterable

import numpy as np
import pyswarms as ps

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
    TohpeSearch,
    ZBucketSearch,
)

np.random.seed(42)
random.seed(40)

LEGACY_ONE_HOT_CAP = 1 << 20
PSO_PARTICLES = 16
PSO_ITERS = 10
Z_RESERVE_CAP_MIN = 50_000
Z_RESERVE_CAP_SPAN = 150_000
Z_HARD_CAP_MIN = 100_000
Z_HARD_CAP_SPAN = 400_000
TODD_RESERVE_MAX = 3


def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))


def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0 / beta) * np.log(np.exp(-beta * (xs - m)).sum()))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def budget_int(budget: float, minimum: int, span: int) -> int:
    return int(minimum + int(span * float(budget)))


class Evaluator(BaseEvaluator):
    """Lean full-length stochastic descent, tuned by global PSO (pyswarms).

    One constant policy profile for the whole descent, pow=1: exhaustive
    one-hot y generation (`one_hot="all"`, legacy C++ gen_part=1.0 behavior)
    with brutal retention — TOHPE keep=2/z_choices=2, TODD keep=0 (reserve
    only), ActionPool final_size=12. beamwidth=2 softmax@0.2 everywhere
    corrects local scoring mistakes and decorrelates seeds, so every PSO
    particle evaluation is an independent sample of the path space. All
    parameters are active from eval 1 — no rank-schedule dead knobs."""

    seeds = [random.randint(1, 10000) for _ in range(2)]

    @staticmethod
    def _sample_caps(sample_count: int, one_hot_fraction: float):
        # path_store forced gen_part=1.0 in C++, so every one-hot basis vector
        # was tried in addition to num_samples random dense vectors. Keep
        # one_hot_fraction consumed for x0 compatibility, but do not let it
        # shrink the one-hot action set.
        _ = one_hot_fraction
        return [LEGACY_ONE_HOT_CAP, 0, max(0, int(sample_count))]

    def policy_mapping(self):
        ranks = [0]
        pool_score = ExplorationScore([self.map_par(_w_tanh) for _ in range(5)], pow=1)
        final_weights = [self.map_par(_w_tanh) for _ in range(6)]
        final_centers = [self.map_par(sigmoid) for _ in range(6)]
        final_score = FinalizationScore(final_weights, final_centers, pow=1)
        self.set_scores(ranks, [PolicyScores(exploration=pool_score, final=final_score)])

        z_budget = self.map_par(sigmoid)
        z_research_budget = self.map_par(sigmoid)
        todd_reserve_budget = self.map_par(sigmoid)
        sample_budget = self.map_par(sigmoid)
        one_hot_fraction = 0.1 + 0.5 * self.map_par(sigmoid)
        sample_count = 8 + int(48 * sample_budget)
        sample_caps = self._sample_caps(sample_count, one_hot_fraction)
        z_reserve_cap = budget_int(z_research_budget, Z_RESERVE_CAP_MIN, Z_RESERVE_CAP_SPAN)
        z_hard_cap = max(z_reserve_cap, budget_int(z_research_budget, Z_HARD_CAP_MIN, Z_HARD_CAP_SPAN))
        todd_reserve = min(TODD_RESERVE_MAX, int(TODD_RESERVE_MAX * todd_reserve_budget))

        sampling = SamplingBudget(one_hot="all", sparse=sample_caps[1], dense=sample_caps[2], sparse_max_weight=2)
        self.set_action_selection(ActionSelection(beamwidth=2, mode="softmax", temperature=0.2))
        self.set_action_pool(ActionPool(final_size=12))
        self.set_tohpe_search(TohpeSearch(sampling=sampling, pool=SourcePool(keep=2, reserve=0), z_choices=2))
        self.set_todd_search(
            ToddSearch(
                sampling=sampling,
                pool=SourcePool(keep=0, reserve=todd_reserve),
                actions_per_bucket=1,
                buckets=ZBucketSearch(
                    min_buckets=10 + int(250 * z_budget),
                    max_buckets=z_reserve_cap,
                    limit_bucket=z_hard_cap,
                ),
            )
        )

    def evaluate(self, params: Iterable[float], seeds: Iterable[int] | None = None) -> float:
        seeds = list(self.seeds if seeds is None else seeds)
        tcounts = self.run(params, seeds)
        bestish = softmin(tcounts, beta=6.0)
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return bestish + 0.02 * spread

    def __call__(self, params: Iterable):
        return self.evaluate(params, seeds=self.seeds)


def run_opt(fun: Evaluator, num_iters: int) -> np.ndarray:
    x = fun.extract_active()
    n_params = len(x)
    if n_params == 0:
        fun.run([], fun.seeds)
        return np.asarray(x, dtype=float)
    bounds = (np.full(n_params, -1.0), np.full(n_params, 1.0))
    options = {"c1": 0.4, "c2": 0.4, "w": 0.7}

    def objective(positions):
        return np.asarray([fun(np.asarray(pos, dtype=float)) for pos in positions], dtype=float)

    optimizer = ps.single.GlobalBestPSO(
        n_particles=PSO_PARTICLES,
        dimensions=n_params,
        options=options,
        bounds=bounds,
    )
    _, best_pos = optimizer.optimize(objective, iters=num_iters, verbose=False)
    return np.asarray(best_pos, dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=500)
    run_opt(fun, PSO_ITERS)
    return fun.get_best()
