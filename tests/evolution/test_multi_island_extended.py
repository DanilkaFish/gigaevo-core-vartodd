"""Extended tests for MapElitesMultiIsland covering uncovered branches.

Targets: migration triggering, size-limit enforcement, remove_program_by_id,
restore_state, quota calculation, _perform_migration edge cases.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import ValidationError
import pytest

from gigaevo.evolution.strategies.base import MutationSelection
from gigaevo.evolution.strategies.models import MutationRouteConfig
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(island_id: str | None = None) -> Program:
    p = Program(code="def solve(): pass", state=ProgramState.DONE)
    if island_id is not None:
        p.metadata["current_island"] = island_id
    return p


def _mock_island(island_id: str, size: int = 0) -> MagicMock:
    m = MagicMock()
    m.config.island_id = island_id
    m.config.max_size = None
    m.add = AsyncMock(return_value=True)
    m.select_elites = AsyncMock(return_value=[])
    m.select_migrants = AsyncMock(return_value=[])
    m.get_elite_ids = AsyncMock(return_value=[])
    m.__len__ = AsyncMock(return_value=size)
    m._enforce_size_limit = AsyncMock()
    m.archive_storage.remove_elite_by_id = AsyncMock(return_value=True)
    m.state_manager.set_program_state = AsyncMock()
    m.state_manager.update_program = AsyncMock()
    return m


def _mock_storage() -> MagicMock:
    s = MagicMock()
    s.save_run_state = AsyncMock()
    s.load_run_state = AsyncMock(return_value=None)
    s.get = AsyncMock(return_value=None)
    return s


def _make_multi_island(
    n: int = 2,
    migration_interval: int = 50,
    enable_migration: bool = True,
    max_migrants_per_island: int = 5,
    mutation_routes: list[MutationRouteConfig | dict] | None = None,
    seed_island_map: dict[str, str] | None = None,
    initial_island_id: str | None = None,
    bootstrap_source_island: str | None = None,
    bootstrap_until_size: int = 0,
    bootstrap_mix_probability: float = 0.0,
    steady_mix_probability: float = 0.0,
    route_context_provider: MagicMock | None = None,
) -> tuple[MapElitesMultiIsland, dict[str, MagicMock], MagicMock]:
    """Return (multi, islands_dict, storage) with n mock islands."""
    storage = _mock_storage()
    mock_islands = {f"island_{i}": _mock_island(f"island_{i}") for i in range(n)}

    mock_configs = []
    for island_id in mock_islands:
        cfg = MagicMock()
        cfg.island_id = island_id
        cfg.max_size = None
        mock_configs.append(cfg)

    with patch(
        "gigaevo.evolution.strategies.multi_island.MapElitesIsland",
        side_effect=lambda cfg, s: mock_islands[cfg.island_id],
    ):
        multi = MapElitesMultiIsland(
            island_configs=mock_configs,
            program_storage=storage,
            migration_interval=migration_interval,
            enable_migration=enable_migration,
            max_migrants_per_island=max_migrants_per_island,
            mutation_routes=mutation_routes,
            seed_island_map=seed_island_map,
            initial_island_id=initial_island_id,
            bootstrap_source_island=bootstrap_source_island,
            bootstrap_until_size=bootstrap_until_size,
            bootstrap_mix_probability=bootstrap_mix_probability,
            steady_mix_probability=steady_mix_probability,
            route_context_provider=route_context_provider,
        )

    return multi, mock_islands, storage


def _route(
    regime_id: str,
    island_id: str,
    probability: float = 1.0,
    guidance: str | None = None,
) -> MutationRouteConfig:
    return MutationRouteConfig(
        regime_id=regime_id,
        island_id=island_id,
        probability=probability,
        guidance=guidance or f"Guidance for {regime_id}.",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestMultiIslandConstruction:
    def test_no_configs_raises_value_error(self):
        with pytest.raises(ValueError, match="At least one island"):
            with patch("gigaevo.evolution.strategies.multi_island.MapElitesIsland"):
                MapElitesMultiIsland(
                    island_configs=[],
                    program_storage=_mock_storage(),
                )

    def test_islands_dict_populated(self):
        multi, islands, _ = _make_multi_island(n=3)
        assert len(multi.islands) == 3

    def test_max_size_none_when_no_capped_islands(self):
        multi, _, _ = _make_multi_island(n=2)
        assert multi.max_size is None

    def test_routes_require_unique_regime_ids(self):
        with pytest.raises(ValueError, match="duplicate regime_id"):
            _make_multi_island(
                enable_migration=False,
                mutation_routes=[
                    _route("builder", "island_0"),
                    _route("builder", "island_1"),
                ],
            )

    def test_route_must_reference_existing_island(self):
        with pytest.raises(ValueError, match="unknown island"):
            _make_multi_island(
                enable_migration=False,
                mutation_routes=[_route("builder", "missing")],
            )

    def test_seed_map_must_reference_existing_island(self):
        with pytest.raises(ValueError, match="unknown island"):
            _make_multi_island(seed_island_map={"seed": "missing"})

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("initial_island_id", "missing", "unknown island"),
            ("bootstrap_source_island", "missing", "unknown island"),
            ("bootstrap_until_size", -1, "bootstrap_until_size"),
            ("bootstrap_mix_probability", -0.1, "probability"),
            ("bootstrap_mix_probability", 1.1, "probability"),
            ("steady_mix_probability", -0.1, "probability"),
            ("steady_mix_probability", 1.1, "probability"),
        ],
    )
    def test_rejects_invalid_bootstrap_configuration(self, field, value, match):
        kwargs = {field: value}
        with pytest.raises(ValueError, match=match):
            _make_multi_island(**kwargs)

    def test_rejects_ambiguous_initial_program_placement(self):
        with pytest.raises(ValueError, match="initial_island_id"):
            _make_multi_island(
                initial_island_id="island_0",
                seed_island_map={"seed": "island_1"},
            )

    def test_routed_topology_rejects_enabled_migration(self):
        with pytest.raises(ValueError, match="migration must be disabled"):
            _make_multi_island(
                mutation_routes=[_route("builder", "island_0")],
                enable_migration=True,
            )

    def test_route_mappings_are_normalized_to_config_models(self):
        multi, _, _ = _make_multi_island(
            n=1,
            enable_migration=False,
            mutation_routes=[
                {
                    "regime_id": "builder",
                    "island_id": "island_0",
                    "probability": 1.0,
                    "guidance": "Build from scratch.",
                }
            ],
        )

        assert multi.mutation_routes == (
            MutationRouteConfig(
                regime_id="builder",
                island_id="island_0",
                probability=1.0,
                guidance="Build from scratch.",
            ),
        )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("regime_id", ""),
            ("island_id", ""),
            ("probability", 0.0),
            ("guidance", "   "),
            ("context_profile", "   "),
        ],
    )
    def test_route_config_rejects_invalid_values(self, field, value):
        values = {
            "regime_id": "builder",
            "island_id": "island_0",
            "probability": 1.0,
            "guidance": "Build a path.",
        }
        values[field] = value
        with pytest.raises(ValidationError):
            MutationRouteConfig(**values)

    def test_route_context_profile_defaults_for_legacy_config(self):
        config = MutationRouteConfig(
            regime_id="builder",
            island_id="island_0",
            probability=1.0,
            guidance="Build a path.",
        )

        assert config.context_profile == "default"


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


class TestMultiIslandAdd:
    async def test_add_unknown_island_id_returns_false(self):
        multi, _, _ = _make_multi_island(n=2)
        prog = _prog()
        result = await multi.add(prog, island_id="nonexistent")
        assert result is False

    async def test_add_router_returns_none_returns_false(self):
        multi, _, _ = _make_multi_island(n=2)
        multi.mutant_router.route_mutant = AsyncMock(return_value=None)
        prog = _prog()
        result = await multi.add(prog)
        assert result is False

    async def test_add_to_known_island_id(self):
        multi, islands, _ = _make_multi_island(n=2)
        prog = _prog()
        islands["island_0"].add = AsyncMock(return_value=True)
        result = await multi.add(prog, island_id="island_0")
        assert result is True
        islands["island_0"].add.assert_called_once_with(prog)

    async def test_add_via_router(self):
        multi, islands, _ = _make_multi_island(n=2)
        prog = _prog()
        target_island = islands["island_1"]
        multi.mutant_router.route_mutant = AsyncMock(return_value=target_island)
        target_island.add = AsyncMock(return_value=True)
        result = await multi.add(prog)
        assert result is True

    async def test_add_uses_child_target_island_without_router(self):
        multi, islands, _ = _make_multi_island(
            n=3,
            enable_migration=False,
            mutation_routes=[
                _route("ab_initio", "island_0", 0.40),
                _route("mid_margin", "island_1", 0.35),
                _route("near_end", "island_2", 0.25),
            ],
        )
        child = _prog()
        child.metadata["target_island"] = "island_2"
        multi.mutant_router.route_mutant = AsyncMock()

        assert await multi.add(child) is True
        islands["island_2"].add.assert_awaited_once_with(child)
        multi.mutant_router.route_mutant.assert_not_awaited()

    async def test_unknown_child_target_is_rejected_without_random_fallback(self):
        multi, _, _ = _make_multi_island(n=2)
        child = _prog()
        child.metadata["target_island"] = "removed_island"
        multi.mutant_router.route_mutant = AsyncMock()

        assert await multi.add(child) is False
        multi.mutant_router.route_mutant.assert_not_awaited()

    async def test_initial_root_uses_seed_island_map(self):
        multi, islands, _ = _make_multi_island(
            n=3,
            seed_island_map={"pso_tohpe_only": "island_0"},
        )
        root = _prog()
        root.metadata.update(
            source="initial_program",
            strategy_name="pso_tohpe_only",
        )
        multi.mutant_router.route_mutant = AsyncMock()

        assert await multi.add(root) is True
        islands["island_0"].add.assert_awaited_once_with(root)
        multi.mutant_router.route_mutant.assert_not_awaited()

    async def test_unmapped_root_is_rejected_when_seed_map_is_configured(self):
        multi, _, _ = _make_multi_island(
            n=2,
            seed_island_map={"known_seed": "island_0"},
        )
        root = _prog()
        root.metadata.update(
            source="initial_program",
            strategy_name="unknown_seed",
        )
        multi.mutant_router.route_mutant = AsyncMock()

        assert await multi.add(root) is False
        multi.mutant_router.route_mutant.assert_not_awaited()

    async def test_all_initial_roots_use_initial_island(self):
        multi, islands, _ = _make_multi_island(
            n=3,
            initial_island_id="island_0",
        )
        root = _prog()
        root.metadata.update(
            source="initial_program",
            strategy_name="any_seed",
        )
        multi.mutant_router.route_mutant = AsyncMock()

        assert await multi.add(root) is True
        islands["island_0"].add.assert_awaited_once_with(root)
        multi.mutant_router.route_mutant.assert_not_awaited()


# ---------------------------------------------------------------------------
# select_elites() — quota, generation increment, sampling
# ---------------------------------------------------------------------------


class TestMultiIslandSelectElites:
    async def test_empty_when_total_zero(self):
        multi, _, _ = _make_multi_island(n=2)
        result = await multi.select_elites(total=0)
        assert result == []

    async def test_generation_increments_when_results_found(self):
        multi, islands, _ = _make_multi_island(n=2)
        prog = _prog()
        islands["island_0"].select_elites = AsyncMock(return_value=[prog])
        islands["island_1"].select_elites = AsyncMock(return_value=[])
        assert multi.generation == 0
        await multi.select_elites(total=2)
        assert multi.generation == 1

    async def test_generation_not_incremented_when_empty(self):
        multi, _, _ = _make_multi_island(n=2)
        await multi.select_elites(total=2)
        assert multi.generation == 0

    async def test_results_sampled_when_exceed_total(self):
        multi, islands, _ = _make_multi_island(n=2)
        progs = [_prog() for _ in range(10)]
        islands["island_0"].select_elites = AsyncMock(return_value=progs)
        islands["island_1"].select_elites = AsyncMock(return_value=[])
        result = await multi.select_elites(total=3)
        assert len(result) == 3

    async def test_migration_triggered_when_interval_reached(self):
        multi, _, _ = _make_multi_island(
            n=2, migration_interval=1, enable_migration=True
        )
        multi.generation = 1
        multi.last_migration = 0
        multi._perform_migration = AsyncMock()
        multi._enforce_all_island_size_limits = AsyncMock()
        await multi.select_elites(total=2)
        multi._perform_migration.assert_called_once()

    async def test_migration_not_triggered_when_disabled(self):
        multi, _, _ = _make_multi_island(
            n=2, migration_interval=1, enable_migration=False
        )
        multi.generation = 100
        multi._perform_migration = AsyncMock()
        await multi.select_elites(total=2)
        multi._perform_migration.assert_not_called()

    async def test_save_run_state_called_after_generation_increment(self):
        multi, islands, storage = _make_multi_island(n=2)
        prog = _prog()
        # Give both islands a result so that whichever gets quota>0 returns something
        islands["island_0"].select_elites = AsyncMock(return_value=[prog])
        islands["island_1"].select_elites = AsyncMock(return_value=[prog])
        await multi.select_elites(total=10)  # large quota ensures both islands called
        storage.save_run_state.assert_called()


class TestMultiIslandSelectForMutation:
    async def test_empty_target_uses_two_bootstrap_donors(self, monkeypatch):
        multi, islands, _ = _make_multi_island(
            n=3,
            enable_migration=False,
            mutation_routes=[_route("mid_margin", "island_1")],
            bootstrap_source_island="island_0",
            bootstrap_until_size=8,
            bootstrap_mix_probability=0.70,
            steady_mix_probability=0.10,
        )
        donors = [_prog("island_0"), _prog("island_0")]
        islands["island_0"].__len__ = AsyncMock(return_value=8)
        islands["island_1"].__len__ = AsyncMock(return_value=0)
        islands["island_0"].select_elites = AsyncMock(return_value=donors)
        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choices",
            lambda choices, weights, k: [choices[0]],
        )

        selection = await multi.select_for_mutation(2)

        assert selection.parents == donors
        assert selection.parent_roles == (
            "bootstrap_donor",
            "bootstrap_donor",
        )
        assert selection.route is not None
        assert selection.route.island_id == "island_1"

    async def test_small_target_forces_mix_when_only_one_local(self, monkeypatch):
        multi, islands, _ = _make_multi_island(
            n=3,
            enable_migration=False,
            mutation_routes=[_route("mid_margin", "island_1")],
            bootstrap_source_island="island_0",
            bootstrap_until_size=8,
            bootstrap_mix_probability=0.0,
        )
        local = _prog("island_1")
        donor = _prog("island_0")
        islands["island_0"].__len__ = AsyncMock(return_value=8)
        islands["island_1"].__len__ = AsyncMock(return_value=1)
        islands["island_1"].select_elites = AsyncMock(return_value=[local])
        islands["island_0"].select_elites = AsyncMock(return_value=[donor])
        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choices",
            lambda choices, weights, k: [choices[0]],
        )

        selection = await multi.select_for_mutation(2)

        assert selection.parents == [local, donor]
        assert selection.parent_roles == (
            "target_island",
            "bootstrap_donor",
        )

    @pytest.mark.parametrize(
        "target_size,random_value,expect_mix",
        [
            (7, 0.69, True),
            (7, 0.70, False),
            (8, 0.09, True),
            (8, 0.10, False),
        ],
    )
    async def test_mixing_probability_boundaries(
        self,
        monkeypatch,
        target_size,
        random_value,
        expect_mix,
    ):
        multi, islands, _ = _make_multi_island(
            n=3,
            enable_migration=False,
            mutation_routes=[_route("mid_margin", "island_1")],
            bootstrap_source_island="island_0",
            bootstrap_until_size=8,
            bootstrap_mix_probability=0.70,
            steady_mix_probability=0.10,
        )
        locals_ = [_prog("island_1"), _prog("island_1")]
        donor = _prog("island_0")
        islands["island_0"].__len__ = AsyncMock(return_value=8)
        islands["island_1"].__len__ = AsyncMock(return_value=target_size)
        islands["island_2"].__len__ = AsyncMock(return_value=0)
        islands["island_1"].select_elites = AsyncMock(return_value=locals_)
        islands["island_0"].select_elites = AsyncMock(return_value=[donor])
        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.random",
            lambda: random_value,
        )
        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choices",
            lambda choices, weights, k: [choices[0]],
        )
        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choice",
            lambda choices: choices[0],
        )

        selection = await multi.select_for_mutation(2)

        assert ("mixed_donor" in selection.parent_roles) is (
            expect_mix and target_size >= 8
        )
        assert ("bootstrap_donor" in selection.parent_roles) is (
            expect_mix and target_size < 8
        )

    async def test_established_target_replaces_at_most_one_parent(
        self, monkeypatch
    ):
        multi, islands, _ = _make_multi_island(
            n=3,
            enable_migration=False,
            mutation_routes=[_route("near_end", "island_2")],
            bootstrap_source_island="island_0",
            bootstrap_until_size=8,
            steady_mix_probability=1.0,
        )
        local = _prog("island_2")
        donor = _prog("island_1")
        islands["island_0"].__len__ = AsyncMock(return_value=8)
        islands["island_1"].__len__ = AsyncMock(return_value=8)
        islands["island_2"].__len__ = AsyncMock(return_value=8)
        islands["island_2"].select_elites = AsyncMock(return_value=[local])
        islands["island_1"].select_elites = AsyncMock(return_value=[donor])
        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choices",
            lambda choices, weights, k: [choices[0]],
        )
        chosen: list[tuple[str, ...]] = []

        def choose_donor(choices):
            chosen.append(tuple(choices))
            return "island_1"

        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choice",
            choose_donor,
        )

        selection = await multi.select_for_mutation(2)

        assert chosen == [("island_0", "island_1")]
        assert selection.parents == [local, donor]
        assert selection.parent_roles == ("target_island", "mixed_donor")

    async def test_context_provider_can_exclude_route(self, monkeypatch):
        provider = MagicMock()
        provider.route_is_available.side_effect = (
            lambda route: route.regime_id == "available"
        )
        multi, islands, _ = _make_multi_island(
            n=2,
            enable_migration=False,
            mutation_routes=[
                _route("blocked", "island_0", 100.0),
                _route("available", "island_1", 1.0),
            ],
            route_context_provider=provider,
        )
        parent = _prog("island_1")
        for island in islands.values():
            island.__len__ = AsyncMock(return_value=1)
        islands["island_1"].select_elites = AsyncMock(return_value=[parent])
        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choices",
            lambda choices, weights, k: [choices[0]],
        )

        selection = await multi.select_for_mutation(1)

        assert selection.route is not None
        assert selection.route.regime_id == "available"
        assert provider.route_is_available.call_count == 2

    async def test_duplicate_parent_ids_are_not_returned(self, monkeypatch):
        multi, islands, _ = _make_multi_island(
            n=2,
            enable_migration=False,
            mutation_routes=[_route("mid_margin", "island_1")],
            bootstrap_source_island="island_0",
            bootstrap_until_size=8,
        )
        duplicate = _prog("island_0")
        islands["island_0"].__len__ = AsyncMock(return_value=2)
        islands["island_1"].__len__ = AsyncMock(return_value=0)
        islands["island_0"].select_elites = AsyncMock(
            return_value=[duplicate, duplicate]
        )
        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choices",
            lambda choices, weights, k: [choices[0]],
        )

        selection = await multi.select_for_mutation(2)

        assert selection.parents == [duplicate]
        assert selection.parent_roles == ("bootstrap_donor",)

    async def test_uses_only_sampled_route_island(self, monkeypatch):
        routes = [
            _route("ab_initio", "island_0", 0.40),
            _route("mid_margin", "island_1", 0.35),
            _route("near_end", "island_2", 0.25),
        ]
        multi, islands, _ = _make_multi_island(
            n=3,
            enable_migration=False,
            mutation_routes=routes,
        )
        parent_a, parent_b = _prog("island_1"), _prog("island_1")
        for island in islands.values():
            island.__len__ = AsyncMock(return_value=2)
        islands["island_1"].select_elites = AsyncMock(
            return_value=[parent_a, parent_b]
        )
        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choices",
            lambda choices, weights, k: [choices[1]],
        )

        selection = await multi.select_for_mutation(total=2)

        assert selection.route is not None
        assert selection.route.regime_id == "mid_margin"
        assert selection.route.island_id == "island_1"
        assert selection.parents == [parent_a, parent_b]
        islands["island_0"].select_elites.assert_not_called()
        islands["island_2"].select_elites.assert_not_called()

    async def test_excludes_empty_route_islands_before_sampling(self, monkeypatch):
        multi, islands, _ = _make_multi_island(
            n=2,
            enable_migration=False,
            mutation_routes=[
                _route("empty", "island_0", 100.0),
                _route("available", "island_1", 1.0),
            ],
        )
        parent = _prog("island_1")
        islands["island_0"].__len__ = AsyncMock(return_value=0)
        islands["island_1"].__len__ = AsyncMock(return_value=1)
        islands["island_1"].select_elites = AsyncMock(return_value=[parent])
        sampled: list[str] = []

        def choose(choices, weights, k):
            sampled.extend(route.regime_id for route in choices)
            return [choices[0]]

        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choices",
            choose,
        )

        selection = await multi.select_for_mutation(total=2)

        assert sampled == ["available"]
        assert selection.parents == [parent]
        assert selection.route is not None
        assert selection.route.regime_id == "available"

    async def test_all_empty_routes_return_empty_selection(self):
        multi, _, _ = _make_multi_island(
            n=2,
            enable_migration=False,
            mutation_routes=[
                _route("a", "island_0"),
                _route("b", "island_1"),
            ],
        )
        assert await multi.select_for_mutation(total=2) == MutationSelection(
            parents=[],
            route=None,
        )

    async def test_does_not_borrow_missing_parent_from_another_island(
        self, monkeypatch
    ):
        multi, islands, _ = _make_multi_island(
            n=2,
            enable_migration=False,
            mutation_routes=[
                _route("a", "island_0"),
                _route("b", "island_1"),
            ],
        )
        parent = _prog("island_0")
        for island in islands.values():
            island.__len__ = AsyncMock(return_value=1)
        islands["island_0"].select_elites = AsyncMock(return_value=[parent])
        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choices",
            lambda choices, weights, k: [choices[0]],
        )

        selection = await multi.select_for_mutation(total=2)

        assert selection.parents == [parent]
        islands["island_1"].select_elites.assert_not_called()

    async def test_rejects_parent_owned_by_another_island(self, monkeypatch):
        multi, islands, _ = _make_multi_island(
            n=2,
            enable_migration=False,
            mutation_routes=[_route("a", "island_0")],
        )
        islands["island_0"].__len__ = AsyncMock(return_value=1)
        islands["island_0"].select_elites = AsyncMock(
            return_value=[_prog("island_1")]
        )
        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choices",
            lambda choices, weights, k: [choices[0]],
        )

        with pytest.raises(RuntimeError, match="route island"):
            await multi.select_for_mutation(total=1)

    async def test_without_routes_uses_legacy_selection(self):
        multi, islands, _ = _make_multi_island(n=2)
        parent = _prog()
        islands["island_0"].select_elites = AsyncMock(return_value=[parent])

        selection = await multi.select_for_mutation(total=2)

        assert selection.route is None
        assert selection.parents == [parent]

    async def test_persists_selected_route_counter(self, monkeypatch):
        multi, islands, storage = _make_multi_island(
            n=1,
            enable_migration=False,
            mutation_routes=[_route("mid_margin", "island_0")],
        )
        parent = _prog("island_0")
        islands["island_0"].__len__ = AsyncMock(return_value=1)
        islands["island_0"].select_elites = AsyncMock(return_value=[parent])
        monkeypatch.setattr(
            "gigaevo.evolution.strategies.multi_island.random.choices",
            lambda choices, weights, k: [choices[0]],
        )

        await multi.select_for_mutation(total=1)

        assert multi.route_selection_counts["mid_margin"] == 1
        storage.save_run_state.assert_any_await(
            "strategy:route_selections:mid_margin",
            1,
        )


# ---------------------------------------------------------------------------
# select_migrants()
# ---------------------------------------------------------------------------


class TestMultiIslandSelectMigrants:
    async def test_select_migrants_aggregates_all_islands(self):
        multi, islands, _ = _make_multi_island(n=2)
        p0, p1 = _prog(), _prog()
        islands["island_0"].select_migrants = AsyncMock(return_value=[p0])
        islands["island_1"].select_migrants = AsyncMock(return_value=[p1])
        result = await multi.select_migrants(count=5)
        assert p0 in result
        assert p1 in result

    async def test_select_migrants_empty_when_no_migrants(self):
        multi, _, _ = _make_multi_island(n=2)
        result = await multi.select_migrants(count=3)
        assert result == []


# ---------------------------------------------------------------------------
# get_global_archive_size()
# ---------------------------------------------------------------------------


class TestGetGlobalArchiveSize:
    async def test_sums_island_sizes(self):
        multi, islands, _ = _make_multi_island(n=3)
        islands["island_0"].__len__ = AsyncMock(return_value=5)
        islands["island_1"].__len__ = AsyncMock(return_value=3)
        islands["island_2"].__len__ = AsyncMock(return_value=7)
        total = await multi.get_global_archive_size()
        assert total == 15

    async def test_empty_islands_returns_zero(self):
        multi, _, _ = _make_multi_island(n=2)
        total = await multi.get_global_archive_size()
        assert total == 0


# ---------------------------------------------------------------------------
# remove_program_by_id()
# ---------------------------------------------------------------------------


class TestRemoveProgramById:
    async def test_returns_false_when_not_found(self):
        multi, islands, _ = _make_multi_island(n=2)
        for m in islands.values():
            m.archive_storage.remove_elite_by_id = AsyncMock(return_value=False)
        result = await multi.remove_program_by_id("does-not-exist")
        assert result is False

    async def test_returns_true_and_sets_discarded_when_found(self):
        multi, islands, storage = _make_multi_island(n=2)
        prog = _prog("island_0")
        program_id = prog.id  # auto-generated UUID — do not reassign
        islands["island_0"].archive_storage.remove_elite_by_id = AsyncMock(
            return_value=True
        )
        storage.get = AsyncMock(return_value=prog)
        islands["island_0"].state_manager.set_program_state = AsyncMock()
        result = await multi.remove_program_by_id(program_id)
        assert result is True
        islands["island_0"].state_manager.set_program_state.assert_called_once_with(
            prog, ProgramState.DISCARDED
        )

    async def test_returns_true_even_when_program_not_in_storage(self):
        multi, islands, storage = _make_multi_island(n=2)
        islands["island_0"].archive_storage.remove_elite_by_id = AsyncMock(
            return_value=True
        )
        storage.get = AsyncMock(return_value=None)
        result = await multi.remove_program_by_id("ghost-id")
        assert result is True


# ---------------------------------------------------------------------------
# restore_state()
# ---------------------------------------------------------------------------


class TestRestoreState:
    async def test_restores_generation_from_storage(self):
        multi, _, storage = _make_multi_island(n=2)
        storage.load_run_state = AsyncMock(side_effect=[42, 10])
        await multi.restore_state()
        assert multi.generation == 42
        assert multi.last_migration == 10

    async def test_no_op_when_storage_returns_none(self):
        multi, _, storage = _make_multi_island(n=2)
        storage.load_run_state = AsyncMock(return_value=None)
        await multi.restore_state()
        assert multi.generation == 0
        assert multi.last_migration == 0

    async def test_partial_restore_generation_only(self):
        multi, _, storage = _make_multi_island(n=2)
        storage.load_run_state = AsyncMock(side_effect=[7, None])
        await multi.restore_state()
        assert multi.generation == 7
        assert multi.last_migration == 0

    async def test_restores_route_selection_counters(self):
        multi, _, storage = _make_multi_island(
            n=1,
            enable_migration=False,
            mutation_routes=[_route("near_end", "island_0")],
        )
        storage.load_run_state = AsyncMock(side_effect=[12, 0, 7])

        await multi.restore_state()

        assert multi.generation == 12
        assert multi.route_selection_counts["near_end"] == 7


# ---------------------------------------------------------------------------
# _calculate_island_quotas()
# ---------------------------------------------------------------------------


class TestCalculateIslandQuotas:
    def test_zero_total_returns_empty(self):
        multi, _, _ = _make_multi_island(n=2)
        assert multi._calculate_island_quotas(0) == {}

    def test_even_distribution(self):
        multi, _, _ = _make_multi_island(n=2)
        quotas = multi._calculate_island_quotas(4)
        assert sum(quotas.values()) == 4
        assert all(v == 2 for v in quotas.values())

    def test_remainder_distributed(self):
        multi, _, _ = _make_multi_island(n=3)
        quotas = multi._calculate_island_quotas(7)
        assert sum(quotas.values()) == 7
        # With 3 islands and 7 total: base=2, rem=1 → one island gets 3
        counts = sorted(quotas.values())
        assert counts.count(3) == 1
        assert counts.count(2) == 2

    def test_single_island_gets_all(self):
        multi, _, _ = _make_multi_island(n=1)
        quotas = multi._calculate_island_quotas(10)
        assert sum(quotas.values()) == 10


# ---------------------------------------------------------------------------
# _perform_migration() edge cases
# ---------------------------------------------------------------------------


class TestPerformMigration:
    async def test_no_migrants_is_no_op(self):
        multi, islands, _ = _make_multi_island(n=2)
        for m in islands.values():
            m.select_migrants = AsyncMock(return_value=[])
        # Should not raise and should not call add
        await multi._perform_migration()
        for m in islands.values():
            m.add.assert_not_called()

    async def test_single_island_no_candidates(self):
        """With only 1 island, there are no candidate destinations → failed_migrations."""
        multi, islands, _ = _make_multi_island(n=1)
        prog = _prog("island_0")
        islands["island_0"].select_migrants = AsyncMock(return_value=[prog])
        multi.mutant_router.route_mutant = AsyncMock(return_value=None)
        # Should not raise even with no candidates
        await multi._perform_migration()
        islands["island_0"].add.assert_not_called()

    async def test_router_returns_none_for_migrant(self):
        multi, islands, _ = _make_multi_island(n=2)
        prog = _prog("island_0")
        islands["island_0"].select_migrants = AsyncMock(return_value=[prog])
        islands["island_1"].select_migrants = AsyncMock(return_value=[])
        multi.mutant_router.route_mutant = AsyncMock(return_value=None)
        await multi._perform_migration()
        islands["island_1"].add.assert_not_called()

    async def test_successful_migration_removes_from_source(self):
        multi, islands, _ = _make_multi_island(n=2)
        prog = _prog("island_0")
        migrant_id = prog.id  # auto-generated UUID
        islands["island_0"].select_migrants = AsyncMock(return_value=[prog])
        islands["island_1"].select_migrants = AsyncMock(return_value=[])
        multi.mutant_router.route_mutant = AsyncMock(return_value=islands["island_1"])
        islands["island_1"].add = AsyncMock(return_value=True)
        islands["island_0"].archive_storage.remove_elite_by_id = AsyncMock(
            return_value=True
        )
        await multi._perform_migration()
        islands["island_0"].archive_storage.remove_elite_by_id.assert_called_once_with(
            migrant_id
        )

    async def test_rollback_when_remove_from_source_fails(self):
        """If source remove fails, migrant must be removed from destination (rollback)."""
        multi, islands, _ = _make_multi_island(n=2)
        prog = _prog("island_0")
        migrant_id = prog.id  # auto-generated UUID
        islands["island_0"].select_migrants = AsyncMock(return_value=[prog])
        islands["island_1"].select_migrants = AsyncMock(return_value=[])
        multi.mutant_router.route_mutant = AsyncMock(return_value=islands["island_1"])
        islands["island_1"].add = AsyncMock(return_value=True)
        islands["island_0"].archive_storage.remove_elite_by_id = AsyncMock(
            return_value=False  # source remove fails
        )
        await multi._perform_migration()
        # Rollback: remove from destination
        islands["island_1"].archive_storage.remove_elite_by_id.assert_called_once_with(
            migrant_id
        )

    async def test_destination_rejects_migrant(self):
        multi, islands, _ = _make_multi_island(n=2)
        prog = _prog("island_0")
        islands["island_0"].select_migrants = AsyncMock(return_value=[prog])
        islands["island_1"].select_migrants = AsyncMock(return_value=[])
        multi.mutant_router.route_mutant = AsyncMock(return_value=islands["island_1"])
        islands["island_1"].add = AsyncMock(return_value=False)
        await multi._perform_migration()
        # No removal from source since add failed
        islands["island_0"].archive_storage.remove_elite_by_id.assert_not_called()

    async def test_migration_with_unknown_source_island(self):
        """Migrant whose source island_id is not in self.islands → one-way migration."""
        multi, islands, _ = _make_multi_island(n=2)
        prog = _prog(island_id=None)  # no current_island metadata
        islands["island_0"].select_migrants = AsyncMock(return_value=[prog])
        islands["island_1"].select_migrants = AsyncMock(return_value=[])
        multi.mutant_router.route_mutant = AsyncMock(return_value=islands["island_1"])
        islands["island_1"].add = AsyncMock(return_value=True)
        await multi._perform_migration()
        # No remove_elite_by_id called (source is None)
        islands["island_0"].archive_storage.remove_elite_by_id.assert_not_called()


# ---------------------------------------------------------------------------
# _enforce_all_island_size_limits()
# ---------------------------------------------------------------------------


class TestEnforceAllIslandSizeLimits:
    async def test_calls_enforce_on_each_island(self):
        multi, islands, _ = _make_multi_island(n=3)
        for m in islands.values():
            m._enforce_size_limit = AsyncMock()
        await multi._enforce_all_island_size_limits()
        for m in islands.values():
            m._enforce_size_limit.assert_called_once()


# ---------------------------------------------------------------------------
# get_metrics()
# ---------------------------------------------------------------------------


class TestGetMetrics:
    async def test_returns_strategy_metrics(self):
        multi, islands, _ = _make_multi_island(n=2)
        islands["island_0"].__len__ = AsyncMock(return_value=3)
        islands["island_1"].__len__ = AsyncMock(return_value=7)
        metrics = await multi.get_metrics()
        assert metrics.total_programs == 10
        assert metrics.active_populations == 2

    async def test_generation_in_strategy_specific_metrics(self):
        multi, islands, _ = _make_multi_island(n=2)
        multi.generation = 5
        metrics = await multi.get_metrics()
        assert metrics.strategy_specific_metrics["generation"] == 5

    async def test_route_counters_are_reported(self):
        multi, _, _ = _make_multi_island(
            n=1,
            enable_migration=False,
            mutation_routes=[_route("ab_initio", "island_0")],
        )
        multi.route_selection_counts["ab_initio"] = 4

        metrics = await multi.get_metrics()

        assert metrics.strategy_specific_metrics is not None
        assert metrics.strategy_specific_metrics["route_selections/ab_initio"] == 4
