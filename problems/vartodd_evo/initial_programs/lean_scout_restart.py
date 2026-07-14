from typing import Iterable

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

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

SEEDS = [29, 30]
PSO_POP = 12
STAGE1_EVALS = 128
RESTART_EVALS = 32
RESTART_MARGINS = [40, 16]


def _w_tanh(z: float, scale: float = 4.0, sharp: float = 1.5) -> float:
    return float(scale * np.tanh(z / sharp))


def softmin(xs, beta=6.0):
    xs = np.asarray(xs, dtype=float)
    m = xs.min()
    return float(m - (1.0 / beta) * np.log(np.exp(-beta * (xs - m)).sum()))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Evaluator(BaseEvaluator):
    """Lean beam-2 descent with a scout-to-refine restart ladder.

    Stage 1 tunes the constant lean profile (one_hot="all", TOHPE keep=2,
    TODD reserve-only, softmax@0.2) with 64 full-descent evals. Then two
    set_up_new_init restarts reopen the best path — first wide (+40 branch
    room, escapes a degenerate tail), then near-tail (+16 refinement) —
    continuing the same parameterization via xopt. Restart segments were
    where late gains appeared in prior runs (e.g. 1283->1259 after a
    reopen); this seed combines them with the lean descent shape."""

    def policy_mapping(self):
        pool_score = ExplorationScore([self.map_par(_w_tanh) for _ in range(5)], pow=1)
        final_weights = [self.map_par(_w_tanh) for _ in range(6)]
        final_centers = [self.map_par(sigmoid) for _ in range(6)]
        self.set_scores(PolicyScores(exploration=pool_score, final=FinalizationScore(final_weights, final_centers, pow=1)))

        z_budget = self.map_par(sigmoid)
        z_research_budget = self.map_par(sigmoid)
        todd_reserve = min(3, int(3 * self.map_par(sigmoid)))
        dense_samples = 8 + int(48 * self.map_par(sigmoid))
        z_reserve_cap = 50_000 + int(150_000 * z_research_budget)
        z_hard_cap = max(z_reserve_cap, 100_000 + int(400_000 * z_research_budget))

        sampling = SamplingBudget(one_hot="all", sparse=0, dense=dense_samples, sparse_max_weight=2)
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

    def __call__(self, params: Iterable[float]) -> float:
        tcounts = self.run(params, SEEDS)
        bestish = softmin(tcounts, beta=6.0)
        spread = float(np.std(tcounts)) if len(tcounts) > 1 else 0.0
        return bestish + 0.02 * spread


class Problem(ElementwiseProblem):
    def __init__(self, fun: Evaluator):
        n = len(fun.extract_active())
        super().__init__(n_var=n, n_obj=1, xl=np.full(n, -1.0), xu=np.full(n, 1.0))
        self.fun = fun

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.fun(np.asarray(x, dtype=float))


def optimize(fun: Evaluator, n_eval: int, seed: int) -> np.ndarray:
    active = fun.extract_active()
    if len(active) == 0:
        fun.run([], SEEDS)
        return np.asarray(active, dtype=float)
    algorithm = PSO(pop_size=PSO_POP, w=0.7, c1=0.4, c2=0.4, adaptive=False)
    result = minimize(Problem(fun), algorithm, termination=("n_eval", n_eval), seed=seed, verbose=False)
    return np.asarray(result.X if result.X is not None else fun.extract_active(), dtype=float)


def entrypoint():
    fun = Evaluator(path_name="init", max_depth=500)
    xopt = optimize(fun, STAGE1_EVALS, seed=29)
    for i, margin in enumerate(RESTART_MARGINS):
        params = fun.set_up_new_init(0, rank_thr=fun.best_rank + margin, xopt=xopt)
        if params is None:
            continue
        fun.run(params, SEEDS)
        xopt = optimize(fun, RESTART_EVALS, seed=31 + i)
    return fun.get_best()
