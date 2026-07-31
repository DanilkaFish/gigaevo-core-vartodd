"""Archive selectors specific to the VarTODD experiments."""

from __future__ import annotations

from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.programs.program import Program


class RankAwareRetentionSelector(SumArchiveSelector):
    """Prefer productive, earlier-branching searches within one archive cell.

    The normal :class:`SumArchiveSelector` compares only the primary fitness.
    This prevents an archive from retaining a path which failed to improve its
    starting/loaded rank, or replacing a wider-branching successful search
    with a much narrower one.  The selector uses the following strict ordering:

    1. A program that improved its applicable rank baseline wins over one that
       did not.
    2. Among programs with the same improvement outcome, the higher
       ``loaded_rank`` wins.  A higher value means the child began its search
       earlier in the decomposition, so reaching the same cell demonstrates a
       larger useful reduction.
    3. If both values tie, use the configured fitness comparison (GF32: lower
       fitness wins).

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

        new_loaded_rank = self._metric(new, self.loaded_rank_key, float("-inf"))
        current_loaded_rank = self._metric(
            current, self.loaded_rank_key, float("-inf")
        )
        if new_loaded_rank != current_loaded_rank:
            return new_loaded_rank > current_loaded_rank

        return super().__call__(new, current)


# Kept for existing GF32 configurations and runs.  GF16 uses the neutral name
# above, but both targets have identical behavior.
GF32RankRetentionSelector = RankAwareRetentionSelector
