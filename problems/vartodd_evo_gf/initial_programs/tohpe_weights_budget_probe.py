"""Light TODD seed: search the joint policy, then refine score weights."""

from collections.abc import Iterable
import cma
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
import optuna

OPTIMIZER_FAMILY = "optuna_tpe_then_cma_es"
SEEDS = [21, 22]
JOINT_TRIALS = 192
SCORE_EVALS = 96
TODD_ROLE = "light"
LOWER_BOUND = -2.4
UPPER_BOUND = 2.4


def objective(ranks: list[int]) -> float:
    values = np.asarray(ranks, dtype=float)
    return float(values.min() + 0.015 * values.std())


class Evaluator(BaseEvaluator):
    """TOHPE supplies most candidates; prefix and TODD are cheap escape paths."""

    def float_range(self, low: float, high: float, *, group: str) -> float:
        return self.map_par(lambda x: low + (high - low) * (0.5 + 0.5 * np.tanh(float(x) / 1.6)), group=group)

    def int_range(self, low: int, high: int, *, group: str) -> int:
        return self.map_par(
            lambda x: min(high, low + int((high - low + 1) * (0.5 + 0.5 * np.tanh(float(x) / 1.6)))), group=group
        )

    def policy_mapping(self):
        explore = ExplorationScore(
            [self.float_range(-4.0, 4.0, group="scores") for _ in range(5)], centers=[0.0, 0.5, 0.0, 0.5, 0.0], pow=1
        )
        final = FinalizationScore(
            [self.float_range(-4.0, 4.0, group="scores") for _ in range(6)],
            centers=[0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
            pow=1,
        )
        self.set_scores(PolicyScores(exploration=explore, final=final))
        tohpe_sampling = SamplingBudget(
            one_hot="all",
            sparse=self.int_range(16, 96, group="search"),
            dense=self.int_range(8, 64, group="search"),
            sparse_max_weight=self.int_range(2, 5, group="search"),
        )
        self.set_action_selection(ActionSelection(beamwidth=2, mode="best"))
        self.set_action_pool(ActionPool(final_size=self.int_range(32, 72, group="search")))
        self.set_tohpe_search(
            TohpeSearch(
                sampling=tohpe_sampling,
                pool=SourcePool(keep=self.int_range(20, 64, group="search"), reserve=2),
                z_choices=self.int_range(3, 14, group="search"),
            )
        )
        self.set_tohpeprefix_search(
            TohpePrefixSearch(
                sampling=SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2),
                pool=SourcePool(keep=6, reserve=1),
                actions_per_bucket=2,
                buckets=ZBucketSearch(min_buckets=8, max_buckets=64, limit_bucket=512),
            )
        )
        self.set_todd_search(
            ToddSearch(
                sampling=SamplingBudget(one_hot=8, sparse=2, dense=0, sparse_max_weight=2),
                pool=SourcePool(keep=2, reserve=1),
                actions_per_bucket=2,
                buckets=ZBucketSearch(min_buckets=8, max_buckets=32, limit_bucket=256),
            )
        )

    def __call__(self, params: Iterable[float]) -> float:
        return objective(self.run(params, SEEDS))


def optimize_tpe(evaluator: Evaluator, trials: int, seed: int) -> np.ndarray:
    current = evaluator.extract_active()
    dimensions = len(current)
    if dimensions == 0:
        evaluator([])
        return current
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(24, max(8, trials // 5)))
    names = [f"x{index}" for index in range(dimensions)]

    def evaluate_trial(trial: optuna.Trial) -> float:
        candidate = np.asarray([trial.suggest_float(name, LOWER_BOUND, UPPER_BOUND) for name in names], dtype=float)
        return evaluator(candidate)

    previous_verbosity = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.enqueue_trial(dict(zip(names, current, strict=True)))
        study.optimize(evaluate_trial, n_trials=trials, show_progress_bar=False)
    finally:
        optuna.logging.set_verbosity(previous_verbosity)
    best = np.asarray([study.best_params[name] for name in names], dtype=float)
    evaluator.insert(best)
    evaluator.reinit()
    return best


def optimize_cma(evaluator: Evaluator, evaluations: int, seed: int) -> np.ndarray:
    current = evaluator.extract_active()
    dimensions = len(current)
    if dimensions == 0:
        evaluator([])
        return current
    population = min(8, evaluations)
    strategy = cma.CMAEvolutionStrategy(
        current.tolist(),
        0.65,
        {"bounds": [LOWER_BOUND, UPPER_BOUND], "popsize": population, "seed": seed, "verbose": -9, "verb_disp": 0},
    )
    best = current.copy()
    best_value = evaluator(best)
    calls = 1
    while calls + population <= evaluations and (not strategy.stop()):
        candidates = strategy.ask()
        values = [evaluator(candidate) for candidate in candidates]
        strategy.tell(candidates, values)
        calls += len(candidates)
        index = int(np.argmin(values))
        if values[index] < best_value:
            best_value = float(values[index])
            best = np.asarray(candidates[index], dtype=float)
    evaluator.insert(best)
    evaluator.reinit()
    return best


def entrypoint():
    evaluator = Evaluator(path_name="init", max_depth=500)
    evaluator.select_all_parameter_groups()
    optimize_tpe(evaluator, JOINT_TRIALS, seed=21)
    evaluator.select_parameter_groups("scores")
    optimize_cma(evaluator, SCORE_EVALS, seed=24)
    return evaluator.get_best()
