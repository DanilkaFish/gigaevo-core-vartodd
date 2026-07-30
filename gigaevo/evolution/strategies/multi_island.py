from __future__ import annotations

import asyncio
import random

from loguru import logger

from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.evolution.mutation.constants import TARGET_ISLAND_METADATA_KEY
from gigaevo.evolution.strategies.base import (
    EvolutionStrategy,
    MutationRoute,
    MutationSelection,
    ParentRole,
    StrategyMetrics,
)
from gigaevo.evolution.strategies.island import (
    METADATA_KEY_CURRENT_ISLAND,
    IslandConfig,
    MapElitesIsland,
)
from gigaevo.evolution.strategies.island_selector import WeightedIslandSelector
from gigaevo.evolution.strategies.models import MutationRouteConfig
from gigaevo.evolution.strategies.mutant_router import RandomMutantRouter
from gigaevo.evolution.strategies.route_context import (
    MutationRouteContextProvider,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# Redis run-state field names (used for resume persistence)
_RUN_STATE_GENERATION = "strategy:generation"
_RUN_STATE_LAST_MIGRATION = "strategy:last_migration"


class MapElitesMultiIsland(EvolutionStrategy):
    """Multi-island MAP-Elites strategy (updated to new island API)."""

    def __init__(
        self,
        island_configs: list[IslandConfig],
        program_storage: RedisProgramStorage,
        migration_interval: int = 50,
        enable_migration: bool = True,
        max_migrants_per_island: int = 5,
        island_selector: WeightedIslandSelector | None = None,
        mutant_router: RandomMutantRouter | None = None,
        mutation_routes: list[MutationRouteConfig | dict] | None = None,
        seed_island_map: dict[str, str] | None = None,
        initial_island_id: str | None = None,
        bootstrap_source_island: str | None = None,
        bootstrap_until_size: int = 0,
        bootstrap_mix_probability: float = 0.0,
        steady_mix_probability: float = 0.0,
        route_context_provider: MutationRouteContextProvider | None = None,
    ):
        if not island_configs:
            raise ValueError("At least one island configuration is required")

        self.islands: dict[str, MapElitesIsland] = {
            cfg.island_id: MapElitesIsland(cfg, program_storage)
            for cfg in island_configs
        }

        self.program_storage = program_storage
        self.migration_interval = int(migration_interval)
        self.enable_migration = bool(enable_migration)
        self.max_migrants_per_island = int(max_migrants_per_island)
        self.mutation_routes = tuple(
            route
            if isinstance(route, MutationRouteConfig)
            else MutationRouteConfig.model_validate(route)
            for route in (mutation_routes or [])
        )
        self.seed_island_map = dict(seed_island_map or {})
        self.initial_island_id = initial_island_id
        self.bootstrap_source_island = bootstrap_source_island
        self.bootstrap_until_size = int(bootstrap_until_size)
        self.bootstrap_mix_probability = float(bootstrap_mix_probability)
        self.steady_mix_probability = float(steady_mix_probability)
        self.route_context_provider = route_context_provider
        self._validate_routed_topology()

        self.generation = 0
        self.last_migration = 0
        self._selection_counter_lock = asyncio.Lock()
        self.route_selection_counts = {
            route.regime_id: 0 for route in self.mutation_routes
        }

        # Pluggables (kept for API completeness)
        self.island_selector = island_selector or WeightedIslandSelector()
        self.mutant_router = mutant_router or RandomMutantRouter()

        capped = [cfg.max_size for cfg in island_configs if cfg.max_size is not None]
        self.max_size = sum(capped) if capped else None

        logger.info(
            "Initialized MAP-Elites with {} island(s), global max_size={}",
            len(self.islands),
            self.max_size,
        )

    # --------------------------- Public API ---------------------------

    async def add(self, program: Program, island_id: str | None = None) -> bool:
        """Add a program to a specific island or route it automatically."""
        if island_id is None:
            metadata_target = program.metadata.get(TARGET_ISLAND_METADATA_KEY)
            if isinstance(metadata_target, str):
                island_id = metadata_target
            elif (
                self.initial_island_id is not None
                and program.metadata.get("source") == "initial_program"
            ):
                island_id = self.initial_island_id
            elif (
                self.seed_island_map
                and program.metadata.get("source") == "initial_program"
            ):
                strategy_name = str(program.metadata.get("strategy_name") or "")
                island_id = self.seed_island_map.get(strategy_name)
                if island_id is None:
                    logger.error(
                        "MultiIsland: unmapped initial program '{}'", strategy_name
                    )
                    return False

        logger.debug(
            "MultiIsland: adding program {} (island_id={})",
            program.id,
            island_id or "auto-route",
        )

        if island_id is not None and island_id not in self.islands:
            logger.debug(
                "MultiIsland: program {} rejected (unknown island '{}')",
                program.id,
                island_id,
            )
            return False

        island = (
            self.islands[island_id]
            if island_id is not None
            else await self.mutant_router.route_mutant(
                program, list(self.islands.values())
            )
        )

        if island is None:
            logger.debug(
                "MultiIsland: program {} rejected (router returned None)",
                program.id,
            )
            return False

        logger.debug(
            "MultiIsland: routing program {} to island '{}'",
            program.id,
            island.config.island_id,
        )
        result = await island.add(program)

        if result:
            logger.debug(
                "MultiIsland: program {} successfully added to island '{}'",
                program.id,
                island.config.island_id,
            )
        else:
            logger.debug(
                "MultiIsland: program {} rejected by island '{}'",
                program.id,
                island.config.island_id,
            )

        return result

    async def select_for_mutation(self, total: int) -> MutationSelection:
        """Sample a route, then select local and optional donor parents."""
        if not self.mutation_routes:
            return await super().select_for_mutation(total)
        if total <= 0:
            return MutationSelection(parents=[], route=None)

        island_ids = list(self.islands)
        sizes = dict(
            zip(
                island_ids,
                await asyncio.gather(
                    *[self._island_size(island_id) for island_id in island_ids]
                ),
            )
        )
        route_pairs = [
            (config, self._build_route(config))
            for config in self.mutation_routes
        ]
        available_routes: list[MutationRouteConfig] = []
        materialized_routes: dict[str, MutationRoute] = {}
        for config, route in route_pairs:
            if (
                self.route_context_provider is not None
                and not self.route_context_provider.route_is_available(route)
            ):
                continue
            target_available = sizes[config.island_id] > 0
            bootstrap_available = (
                self.bootstrap_source_island is not None
                and self.bootstrap_until_size > 0
                and sizes[config.island_id] < self.bootstrap_until_size
                and sizes[self.bootstrap_source_island] > 0
            )
            if target_available or bootstrap_available:
                available_routes.append(config)
                materialized_routes[config.regime_id] = route
        if not available_routes:
            logger.debug("MultiIsland: no routed islands contain parents")
            return MutationSelection(parents=[], route=None)

        config = random.choices(
            available_routes,
            weights=[candidate.probability for candidate in available_routes],
            k=1,
        )[0]
        route = materialized_routes[config.regime_id]

        parents, roles = await self._select_route_parents(
            route.island_id,
            total,
            sizes,
        )

        if parents:
            await self._record_successful_selection(config.regime_id)
        return MutationSelection(
            parents=parents,
            route=route,
            parent_roles=roles,
        )

    async def select_elites(self, total: int = 10) -> list[Program]:
        """
        Sample elites from all islands (migration & enforcement on schedule).
        Returns up to `total` elite programs.
        """
        logger.debug(
            "MultiIsland: selecting elites (gen={}, total={}, islands={})",
            self.generation,
            total,
            len(self.islands),
        )

        # Check if migration is due
        if self.enable_migration:
            gens_since_migration = self.generation - self.last_migration
            logger.debug(
                "MultiIsland: migration check (gens_since_last={}, interval={})",
                gens_since_migration,
                self.migration_interval,
            )

            if gens_since_migration >= self.migration_interval:
                logger.info(
                    "MultiIsland: triggering migration (generation {}, last migration at {})",
                    self.generation,
                    self.last_migration,
                )
                await self._perform_migration()
                await self._enforce_all_island_size_limits()
                self.last_migration = self.generation
                await self.program_storage.save_run_state(
                    _RUN_STATE_LAST_MIGRATION, self.last_migration
                )

        # Calculate per-island quotas
        quotas = self._calculate_island_quotas(total)
        logger.debug(
            "MultiIsland: island quotas: {}",
            {k: v for k, v in quotas.items() if v > 0},
        )

        tasks = [
            asyncio.create_task(self.islands[island_id].select_elites(quota))
            for island_id, quota in quotas.items()
            if quota > 0
        ]
        if not tasks:
            logger.debug("MultiIsland: no elites to select (all islands empty)")
            return []

        selections = await asyncio.gather(*tasks)
        results = [p for group in selections for p in group]

        logger.debug(
            "MultiIsland: collected {} elites from {} islands",
            len(results),
            len(tasks),
        )

        # Shuffle and sample if needed
        random.shuffle(results)
        if len(results) > total:
            logger.debug(
                "MultiIsland: sampling {} from {} collected elites",
                total,
                len(results),
            )
            results = random.sample(results, total)

        if results:
            old_generation = self.generation
            await self._record_successful_selection()
            logger.debug(
                "MultiIsland: selected {} elites (generation {} -> {})",
                len(results),
                old_generation,
                self.generation,
            )

        return results

    async def select_migrants(self, count: int) -> list[Program]:
        """Select migrants across all islands (utility)."""
        tasks = [
            asyncio.create_task(island.select_migrants(count))
            for island in self.islands.values()
        ]
        groups = await asyncio.gather(*tasks)
        return [p for g in groups for p in g]

    async def get_program_ids(self) -> list[str]:
        tasks = [
            asyncio.create_task(island.get_elite_ids())
            for island in self.islands.values()
        ]
        groups = await asyncio.gather(*tasks)  # list[list[str]]
        ids: list[str] = [pid for group in groups for pid in group]
        return list(set(ids))

    async def get_global_archive_size(self) -> int:
        """Total elites across all islands (fast path via island counts)."""
        tasks = [
            asyncio.create_task(island.__len__()) for island in self.islands.values()
        ]
        sizes = await asyncio.gather(*tasks)
        return sum(int(s) for s in sizes)

    async def remove_program_by_id(self, program_id: str) -> bool:
        """Remove a program (by id) from whichever island holds it and transition to DISCARDED."""
        for island in self.islands.values():
            if await island.archive_storage.remove_elite_by_id(program_id):
                prog = await self.program_storage.get(program_id)
                if prog is not None:
                    if prog.metadata.get(METADATA_KEY_CURRENT_ISLAND):
                        prog.metadata[METADATA_KEY_CURRENT_ISLAND] = None
                        await island.state_manager.update_program(prog)
                    await island.state_manager.set_program_state(
                        prog, ProgramState.DISCARDED
                    )
                return True
        return False

    async def get_metrics(self) -> StrategyMetrics:
        # per-island counts concurrently
        island_ids = list(self.islands.keys())
        counts = await asyncio.gather(*[self.islands[i].__len__() for i in island_ids])
        population_sizes = {f"size/{i}": int(c) for i, c in zip(island_ids, counts)}
        total_programs = sum(population_sizes.values())

        return StrategyMetrics(
            total_programs=total_programs,
            active_populations=len(self.islands),
            strategy_specific_metrics={
                "generation": self.generation,
                "migration_enabled": self.enable_migration,
                "migration_interval": self.migration_interval,
                "max_migrants_per_island": self.max_migrants_per_island,
                "global_max_size": self.max_size,
                **{
                    f"route_selections/{regime_id}": count
                    for regime_id, count in self.route_selection_counts.items()
                },
                **population_sizes,
            },
        )

    async def restore_state(self) -> None:
        """Restore generation counters from Redis after a resume."""
        gen = await self.program_storage.load_run_state(_RUN_STATE_GENERATION)
        last_mig = await self.program_storage.load_run_state(_RUN_STATE_LAST_MIGRATION)
        if gen is not None:
            self.generation = gen
        if last_mig is not None:
            self.last_migration = last_mig
        restored_routes = False
        for regime_id in self.route_selection_counts:
            count = await self.program_storage.load_run_state(
                self._route_state_key(regime_id)
            )
            if count is not None:
                self.route_selection_counts[regime_id] = int(count)
                restored_routes = True
        if gen is not None or last_mig is not None or restored_routes:
            logger.info(
                "[MapElitesMultiIsland] Restored generation={}, "
                "last_migration={}, route_selections={}",
                self.generation,
                self.last_migration,
                self.route_selection_counts,
            )

    async def reindex_archive(self) -> None:
        """Re-evaluate archive placements on all islands using current metrics."""
        tasks = [island.reindex_archive() for island in self.islands.values()]
        if tasks:
            await asyncio.gather(*tasks)

    def _calculate_island_quotas(self, total: int) -> dict[str, int]:
        """Evenly distribute selection quotas across islands."""
        island_ids = list(self.islands.keys())
        if not island_ids or total <= 0:
            return {}
        base, rem = divmod(total, len(island_ids))
        random.shuffle(island_ids)
        return {
            island_id: base + (1 if i < rem else 0)
            for i, island_id in enumerate(island_ids)
        }

    def _validate_routed_topology(self) -> None:
        regime_ids = [route.regime_id for route in self.mutation_routes]
        if len(regime_ids) != len(set(regime_ids)):
            raise ValueError("mutation_routes contains duplicate regime_id values")
        unknown_route_islands = {
            route.island_id
            for route in self.mutation_routes
            if route.island_id not in self.islands
        }
        unknown_seed_islands = {
            island_id
            for island_id in self.seed_island_map.values()
            if island_id not in self.islands
        }
        configured_islands = {
            island_id
            for island_id in (
                self.initial_island_id,
                self.bootstrap_source_island,
            )
            if island_id is not None and island_id not in self.islands
        }
        unknown_islands = (
            unknown_route_islands | unknown_seed_islands | configured_islands
        )
        if unknown_islands:
            raise ValueError(
                "mutation routing references unknown island(s): "
                + ", ".join(sorted(unknown_islands))
            )
        if self.mutation_routes and self.enable_migration:
            raise ValueError("migration must be disabled when mutation_routes are set")
        if self.initial_island_id is not None and self.seed_island_map:
            raise ValueError(
                "initial_island_id and seed_island_map cannot both be configured"
            )
        if self.bootstrap_until_size < 0:
            raise ValueError("bootstrap_until_size must be non-negative")
        for name, probability in (
            ("bootstrap_mix_probability", self.bootstrap_mix_probability),
            ("steady_mix_probability", self.steady_mix_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} probability must be within [0, 1]")

    def _build_route(self, config: MutationRouteConfig) -> MutationRoute:
        return MutationRoute(
            regime_id=config.regime_id,
            island_id=config.island_id,
            guidance=config.guidance,
            context_profile=config.context_profile,
        )

    async def _island_size(self, island_id: str) -> int:
        return int(await self.islands[island_id].__len__())

    async def _select_unique(
        self,
        island_id: str,
        count: int,
        *,
        exclude_ids: set[str],
    ) -> list[Program]:
        if count <= 0:
            return []
        candidates = await self.islands[island_id].select_elites(
            count + len(exclude_ids)
        )
        result: list[Program] = []
        for candidate in candidates:
            parent_island = candidate.get_metadata(METADATA_KEY_CURRENT_ISLAND)
            if parent_island != island_id:
                raise RuntimeError(
                    "Selected parent belongs to "
                    f"{parent_island!r}, not route island {island_id!r}"
                )
            if candidate.id in exclude_ids or any(
                parent.id == candidate.id for parent in result
            ):
                continue
            result.append(candidate)
            if len(result) == count:
                break
        return result

    async def _select_route_parents(
        self,
        target_island: str,
        total: int,
        sizes: dict[str, int],
    ) -> tuple[list[Program], tuple[ParentRole, ...]]:
        source_island = self.bootstrap_source_island
        target_size = sizes[target_island]
        parents: list[Program] = []
        roles: list[ParentRole] = []
        exclude_ids: set[str] = set()

        if (
            target_size == 0
            and source_island is not None
            and source_island != target_island
            and self.bootstrap_until_size > 0
        ):
            donors = await self._select_unique(
                source_island,
                total,
                exclude_ids=exclude_ids,
            )
            return donors, tuple("bootstrap_donor" for _ in donors)

        donor_island: str | None = None
        donor_role: ParentRole | None = None
        donor_count = 0
        if (
            source_island is not None
            and source_island != target_island
            and 0 < target_size < self.bootstrap_until_size
        ):
            force_mix = target_size < total
            if force_mix or random.random() < self.bootstrap_mix_probability:
                donor_island = source_island
                donor_role = "bootstrap_donor"
                donor_count = min(1, total)
        elif target_size >= self.bootstrap_until_size:
            donor_islands = [
                island_id
                for island_id, size in sizes.items()
                if island_id != target_island and size > 0
            ]
            if (
                donor_islands
                and random.random() < self.steady_mix_probability
            ):
                donor_island = random.choice(donor_islands)
                donor_role = "mixed_donor"
                donor_count = min(1, total)

        local_count = max(0, total - donor_count)
        local = await self._select_unique(
            target_island,
            local_count,
            exclude_ids=exclude_ids,
        )
        parents.extend(local)
        roles.extend("target_island" for _ in local)
        exclude_ids.update(parent.id for parent in local)

        if donor_island is not None and donor_role is not None:
            missing = total - len(parents)
            donors = await self._select_unique(
                donor_island,
                min(donor_count, missing),
                exclude_ids=exclude_ids,
            )
            parents.extend(donors)
            roles.extend(donor_role for _ in donors)
            exclude_ids.update(parent.id for parent in donors)

        if len(parents) < total:
            local_fill = await self._select_unique(
                target_island,
                total - len(parents),
                exclude_ids=exclude_ids,
            )
            parents.extend(local_fill)
            roles.extend("target_island" for _ in local_fill)

        return parents, tuple(roles)

    @staticmethod
    def _route_state_key(regime_id: str) -> str:
        return f"strategy:route_selections:{regime_id}"

    async def _record_successful_selection(
        self, regime_id: str | None = None
    ) -> None:
        async with self._selection_counter_lock:
            self.generation += 1
            await self.program_storage.save_run_state(
                _RUN_STATE_GENERATION, self.generation
            )
            if regime_id is None:
                return
            self.route_selection_counts[regime_id] += 1
            await self.program_storage.save_run_state(
                self._route_state_key(regime_id),
                self.route_selection_counts[regime_id],
            )

    async def _perform_migration(self) -> None:
        """Migrate elites between islands to improve diversity."""
        logger.info(
            "MultiIsland: starting migration (max_migrants_per_island={})",
            self.max_migrants_per_island,
        )

        # Collect migrants from all islands
        tasks = [
            asyncio.create_task(island.select_migrants(self.max_migrants_per_island))
            for island in self.islands.values()
        ]
        groups = await asyncio.gather(*tasks)
        migrants = [p for g in groups for p in g]

        if not migrants:
            logger.info("MultiIsland: no migrants selected")
            return

        logger.info(
            "MultiIsland: collected {} migrants from {} islands",
            len(migrants),
            len(self.islands),
        )

        # Track migration statistics
        successful_migrations = 0
        failed_migrations = 0
        rollbacks = 0

        random.shuffle(migrants)
        for migrant in migrants:
            source_island_id = migrant.get_metadata("current_island")
            logger.debug(
                "MultiIsland: migrating program {} from island '{}'",
                migrant.id,
                source_island_id,
            )

            candidates = [
                i
                for i in self.islands.values()
                if i.config.island_id != source_island_id
            ]
            if not candidates:
                logger.debug(
                    "MultiIsland: no candidate islands for migrant {} (only 1 island?)",
                    migrant.id,
                )
                failed_migrations += 1
                continue

            destination = await self.mutant_router.route_mutant(migrant, candidates)
            if destination is None:
                logger.debug(
                    "MultiIsland: router returned None for migrant {}",
                    migrant.id,
                )
                failed_migrations += 1
                continue

            logger.debug(
                "MultiIsland: migrant {} routed to island '{}'",
                migrant.id,
                destination.config.island_id,
            )

            if await destination.add(migrant):
                # Successfully added to destination, remove from source
                if source_island_id is None or source_island_id not in self.islands:
                    # Source island unknown (program evicted or metadata cleared);
                    # nothing to remove — count as successful one-way migration.
                    successful_migrations += 1
                    continue
                removed = await self.islands[
                    source_island_id
                ].archive_storage.remove_elite_by_id(migrant.id)

                if not removed:
                    # Rollback: remove from destination to avoid duplicates
                    logger.warning(
                        "MultiIsland: migration rollback for {} (failed to remove from source '{}')",
                        migrant.id,
                        source_island_id,
                    )
                    await destination.archive_storage.remove_elite_by_id(migrant.id)
                    rollbacks += 1
                else:
                    logger.debug(
                        "MultiIsland: migrant {} successfully moved: '{}' -> '{}'",
                        migrant.id,
                        source_island_id,
                        destination.config.island_id,
                    )
                    successful_migrations += 1
            else:
                logger.debug(
                    "MultiIsland: migrant {} rejected by destination island '{}'",
                    migrant.id,
                    destination.config.island_id,
                )
                failed_migrations += 1

        logger.info(
            "MultiIsland: migration complete (success={}, failed={}, rollbacks={})",
            successful_migrations,
            failed_migrations,
            rollbacks,
        )

    async def _enforce_all_island_size_limits(self) -> None:
        """Enforce size limits on all capped islands."""
        tasks = [
            asyncio.create_task(self._enforce_one_island(island))
            for island in self.islands.values()
        ]
        if tasks:
            await asyncio.gather(*tasks)

    async def _enforce_one_island(self, island: MapElitesIsland) -> None:
        await island._enforce_size_limit()
