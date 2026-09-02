"""Archive selectors specific to the VarTODD experiments."""

from __future__ import annotations

import random

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata

from gigaevo.evolution.strategies.elite_selectors import (
    EliteSelector,
    WeightedEliteSelector,
)
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.evolution.strategies.utils import weighted_sample_without_replacement
from gigaevo.programs.program import Program


class RankAwareRetentionSelector(SumArchiveSelector):
    """Prefer productive, better-ranked searches within one archive cell.

    The normal :class:`SumArchiveSelector` compares only the primary fitness.
    This prevents an archive from retaining a path which failed to improve its
    starting/loaded rank, or replacing a wider-branching successful search
    with a much narrower one.  The selector uses the following strict ordering:

    1. A program that improved its applicable rank baseline wins over one that
       did not.
    2. Among programs with the same improvement outcome, use the configured
       fitness comparison (for VarTODD, lower final fitness wins).
    3. Only when fitness is exactly tied does the higher ``loaded_rank`` win.
       This preserves the longer useful descent without allowing an easier
       starting point to displace a genuinely better result.

    ``rank_improved`` is produced by the problem validator.  Treating a
    missing value as false keeps resumed archives conservative until their
    programs have been re-evaluated with the new validator.
    """

    def __init__(
        self,
        *args,
        improvement_key: str = "rank_improved",
        loaded_rank_key: str = "loaded_rank",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.improvement_key = improvement_key
        self.loaded_rank_key = loaded_rank_key

    @staticmethod
    def _metric(program: Program, key: str, default: float) -> float:
        value = program.metrics.get(key, default)
        return float(value) if isinstance(value, int | float) else default

    def __call__(self, new: Program, current: Program) -> bool:
        new_improved = self._metric(new, self.improvement_key, 0.0) > 0.0
        current_improved = self._metric(current, self.improvement_key, 0.0) > 0.0
        if new_improved != current_improved:
            return new_improved

        new_score = self.score(new)
        current_score = self.score(current)
        if new_score != current_score:
            return new_score > current_score

        new_loaded_rank = self._metric(new, self.loaded_rank_key, float("-inf"))
        current_loaded_rank = self._metric(
            current, self.loaded_rank_key, float("-inf")
        )
        return new_loaded_rank > current_loaded_rank


class RankWeightedEliteSelector(EliteSelector):
    """Sample by ordinal archive quality and lineage usage.

    Raw fitness gaps can vary greatly between runs.  Ranking makes selection
    pressure depend only on archive order, while average ranks keep equal
    fitness values equally likely before the lineage penalty is applied.
    """

    def __init__(
        self,
        fitness_key: str,
        fitness_key_higher_is_better: bool = True,
        rank_temperature: float = 8.0,
        child_penalty: float = 0.10,
        epsilon: float = 1e-8,
    ) -> None:
        self.fitness_key = fitness_key
        self.higher_is_better = fitness_key_higher_is_better
        self.rank_temperature = float(rank_temperature)
        self.child_penalty = float(child_penalty)
        self.epsilon = float(epsilon)
        if not np.isfinite(self.rank_temperature) or self.rank_temperature <= 0.0:
            raise ValueError("rank_temperature must be finite and positive")
        if not np.isfinite(self.child_penalty) or self.child_penalty < 0.0:
            raise ValueError("child_penalty must be finite and non-negative")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")

    def effective_fitness(self, program: Program) -> float:
        if self.fitness_key not in program.metrics:
            raise ValueError(
                f"Missing fitness key '{self.fitness_key}' in program {program.id}"
            )
        return float(program.metrics[self.fitness_key])

    def selection_weights(self, programs: list[Program]) -> list[float]:
        if not programs:
            return []

        fitnesses = np.asarray(
            [self.effective_fitness(program) for program in programs],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(fitnesses)):
            return [1.0] * len(programs)

        oriented = fitnesses if self.higher_is_better else -fitnesses
        archive_ranks = rankdata(-oriented, method="average") - 1.0
        quality_weights = np.exp(-archive_ranks / self.rank_temperature)
        child_counts = np.asarray(
            [program.lineage.child_count for program in programs],
            dtype=np.float64,
        )
        lineage_weights = 1.0 / (1.0 + self.child_penalty * child_counts)
        return np.maximum(
            quality_weights * lineage_weights,
            self.epsilon,
        ).tolist()

    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        if len(programs) <= total:
            return programs
        weights = self.selection_weights(programs)
        return weighted_sample_without_replacement(programs, weights, total)


class RankImprovementRankWeightedEliteSelector(RankWeightedEliteSelector):
    """Apply the route's no-improvement adjustment before ordinal ranking."""

    def __init__(
        self,
        *args,
        no_improvement_penalty: float,
        existing_fitness_penalty: float = 7.0,
        improvement_key: str = "rank_improved",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.no_improvement_penalty = self._validate_penalty(
            no_improvement_penalty,
            name="no_improvement_penalty",
        )
        self.existing_fitness_penalty = self._validate_penalty(
            existing_fitness_penalty,
            name="existing_fitness_penalty",
        )
        self.improvement_key = improvement_key

    @staticmethod
    def _validate_penalty(value: float, *, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    def effective_fitness(self, program: Program) -> float:
        stored = super().effective_fitness(program)
        improved = float(program.metrics.get(self.improvement_key, 0.0)) > 0.0
        if improved:
            return stored
        if self.higher_is_better:
            base = stored + self.existing_fitness_penalty
            return base - self.no_improvement_penalty
        base = stored - self.existing_fitness_penalty
        return base + self.no_improvement_penalty


class RankImprovementWeightedEliteSelector(WeightedEliteSelector):
    """Sample parents after replacing the validator's fixed branch penalty.

    Saved-path fitness already contains the legacy seven-rank penalty.  Island
    selection needs stronger route-specific pressure without rewriting stored
    metrics, including metrics from a resumed run.  This selector removes the
    existing shaping and applies its configured value only while computing
    parent-selection weights.
    """

    def __init__(
        self,
        *args,
        no_improvement_penalty: float,
        existing_fitness_penalty: float = 7.0,
        improvement_key: str = "rank_improved",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.no_improvement_penalty = self._validate_penalty(
            no_improvement_penalty,
            name="no_improvement_penalty",
        )
        self.existing_fitness_penalty = self._validate_penalty(
            existing_fitness_penalty,
            name="existing_fitness_penalty",
        )
        self.improvement_key = improvement_key

    @staticmethod
    def _validate_penalty(value: float, *, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    def effective_fitness(self, program: Program) -> float:
        if self.fitness_key not in program.metrics:
            raise ValueError(
                f"Missing fitness key '{self.fitness_key}' in program {program.id}"
            )
        stored = float(program.metrics[self.fitness_key])
        improved = float(program.metrics.get(self.improvement_key, 0.0)) > 0.0
        if improved:
            return stored
        if self.higher_is_better:
            base = stored + self.existing_fitness_penalty
            return base - self.no_improvement_penalty
        base = stored - self.existing_fitness_penalty
        return base + self.no_improvement_penalty

    def __call__(self, programs: list[Program], total: int) -> list[Program]:
        if len(programs) <= total:
            return programs

        fitnesses = [self.effective_fitness(program) for program in programs]
        arr = np.asarray(
            fitnesses if self.higher_is_better else [-value for value in fitnesses],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(arr)):
            return random.sample(programs, min(total, len(programs)))

        if self.normalize_fitness:
            fitness_range = float(np.ptp(arr))
            if fitness_range < 1e-10:
                arr = np.full_like(arr, 0.5)
            else:
                arr = (arr - float(arr.min())) / fitness_range

        median_fitness = float(np.median(arr))
        child_counts = np.asarray(
            [program.lineage.child_count for program in programs],
            dtype=np.float64,
        )
        fitness_weights = expit(self.lambda_ * (arr - median_fitness))
        lineage_weights = 1.0 / (1.0 + self.child_penalty * child_counts)
        weights = np.maximum(
            fitness_weights * lineage_weights,
            self.epsilon,
        ).tolist()
        return weighted_sample_without_replacement(programs, weights, total)


# Kept for existing GF32 configurations and runs.  GF16 uses the neutral name
# above, but both targets have identical behavior.
GF32RankRetentionSelector = RankAwareRetentionSelector
