"""Tests for StrategyMetrics and EvolutionStrategy base class.

Covers computed fields, boundary conditions (zero populations, zero programs),
and the to_dict serialization.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from gigaevo.evolution.strategies.base import (
    EvolutionStrategy,
    MutationRoute,
    MutationSelection,
    StrategyMetrics,
)
from gigaevo.programs.program import Program


class StubStrategy(EvolutionStrategy):
    def __init__(self, parents: list[Program]):
        self.parents = parents
        self.selected_total: int | None = None

    async def add(self, program):
        return True

    async def select_elites(self, total):
        self.selected_total = total
        return self.parents

    async def get_program_ids(self):
        return [parent.id for parent in self.parents]


class TestStrategyMetrics:
    def test_default_values(self):
        m = StrategyMetrics()
        assert m.total_programs == 0
        assert m.active_populations == 0
        assert m.strategy_specific_metrics is None

    def test_programs_per_population(self):
        m = StrategyMetrics(total_programs=100, active_populations=4)
        assert m.programs_per_population == 25.0

    def test_programs_per_population_zero_populations(self):
        """Division by zero must return 0.0, not raise."""
        m = StrategyMetrics(total_programs=10, active_populations=0)
        assert m.programs_per_population == 0.0

    def test_has_programs_true(self):
        m = StrategyMetrics(total_programs=1)
        assert m.has_programs is True

    def test_has_programs_false(self):
        m = StrategyMetrics(total_programs=0)
        assert m.has_programs is False

    def test_to_dict_basic(self):
        m = StrategyMetrics(total_programs=10, active_populations=2)
        d = m.to_dict()
        assert d["total_programs"] == 10
        assert d["active_populations"] == 2
        assert d["programs_per_population"] == 5.0
        assert d["has_programs"] is True

    def test_to_dict_with_strategy_specific(self):
        m = StrategyMetrics(
            total_programs=5,
            active_populations=1,
            strategy_specific_metrics={"migration_count": 3, "stale_islands": 0},
        )
        d = m.to_dict()
        assert d["migration_count"] == 3
        assert d["stale_islands"] == 0

    def test_to_dict_without_strategy_specific(self):
        m = StrategyMetrics(total_programs=0, active_populations=0)
        d = m.to_dict()
        assert "migration_count" not in d

    def test_programs_per_population_rounding(self):
        """to_dict rounds programs_per_population to 2 decimal places."""
        m = StrategyMetrics(total_programs=10, active_populations=3)
        d = m.to_dict()
        assert d["programs_per_population"] == 3.33

    def test_negative_values_rejected(self):
        """Pydantic ge=0 constraint should reject negative values."""
        with pytest.raises(Exception):
            StrategyMetrics(total_programs=-1)

    def test_negative_populations_rejected(self):
        with pytest.raises(Exception):
            StrategyMetrics(active_populations=-1)


class TestEvolutionStrategyDefaults:
    """Test default implementations of optional methods on the ABC."""

    async def test_get_metrics_returns_none(self):
        class MinimalStrategy(EvolutionStrategy):
            async def add(self, program):
                return True

            async def select_elites(self, total):
                return []

            async def get_program_ids(self):
                return []

        s = MinimalStrategy()
        assert await s.get_metrics() is None

    async def test_remove_program_raises(self):
        class MinimalStrategy(EvolutionStrategy):
            async def add(self, program):
                return True

            async def select_elites(self, total):
                return []

            async def get_program_ids(self):
                return []

        s = MinimalStrategy()
        with pytest.raises(NotImplementedError):
            await s.remove_program_by_id("some-id")

    async def test_optional_methods_are_noop(self):
        """cleanup, pause, resume, restore_state, reindex_archive are all no-ops by default."""

        class MinimalStrategy(EvolutionStrategy):
            async def add(self, program):
                return True

            async def select_elites(self, total):
                return []

            async def get_program_ids(self):
                return []

        s = MinimalStrategy()
        # These should all complete without error
        await s.cleanup()
        await s.pause()
        await s.resume()
        await s.restore_state()
        await s.reindex_archive()


async def test_default_select_for_mutation_wraps_legacy_elites():
    parents = [Program(code="def solve(): return 1")]
    strategy = StubStrategy(parents)

    selection = await strategy.select_for_mutation(total=1)

    assert selection == MutationSelection(parents=parents, route=None)
    assert strategy.selected_total == 1


def test_mutation_route_is_immutable():
    route = MutationRoute(
        regime_id="ab_initio",
        island_id="ab_initio",
        guidance="Build from path_name='init'.",
    )

    with pytest.raises(FrozenInstanceError):
        route.island_id = "near_end"  # type: ignore[misc]


def test_mutation_route_carries_context_profile():
    route = MutationRoute(
        regime_id="near_end",
        island_id="near_end",
        guidance="Load a selectable path.",
        context_profile="path_refinement",
    )

    assert route.context_profile == "path_refinement"


def test_mutation_selection_roles_align_with_parents():
    parents = [
        Program(code="def solve(): return 1"),
        Program(code="def solve(): return 2"),
    ]

    selection = MutationSelection(
        parents=parents,
        route=None,
        parent_roles=("target_island", "mixed_donor"),
    )

    assert len(selection.parent_roles) == len(selection.parents)
