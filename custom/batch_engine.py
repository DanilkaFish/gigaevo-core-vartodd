from __future__ import annotations

import asyncio
import contextlib
import time
from typing import cast

from loguru import logger
from pydantic import Field

from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.ingestor import _run_bounded_post_step_hook
from gigaevo.evolution.engine.mutation import generate_one_mutation
from gigaevo.programs.program_state import ProgramState


class BatchEngineConfig(EngineConfig):
    """Generation-style engine config.

    A generation selects one elite pool, creates several unique parent groups
    from that pool via ``parent_selector``, evaluates the resulting children,
    ingests them, then refreshes the archive before the next generation.
    """

    max_mutations_per_generation: int = Field(default=16, gt=0)
    max_generations: int | None = Field(default=None, gt=0)


class BatchEvolutionEngine(EvolutionEngine):
    """Legacy-style batch generation/evaluation loop.

    This restores the old cadence without changing the core ``gigaevo``
    package:

    1. wait for current DAGs to finish;
    2. select ``max_elites_per_generation`` elites;
    3. build up to ``max_mutations_per_generation`` parent groups;
    4. persist child programs;
    5. wait for child DAGs, ingest, and refresh archive programs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        cfg = cast(BatchEngineConfig, self.config)
        if not isinstance(cfg, BatchEngineConfig):
            raise TypeError(
                f"BatchEvolutionEngine requires BatchEngineConfig, "
                f"got {type(self.config).__name__}"
            )
        self._batch_config = cfg

    async def run(self) -> None:
        logger.info(
            "[BatchEvolution] Start | elites={} mutations_per_generation={} stopper={}",
            self._batch_config.max_elites_per_generation,
            self._batch_config.max_mutations_per_generation,
            type(self._batch_config.stopper).__name__,
        )
        self._running = True
        self._run_start_time = time.monotonic()

        await self._write_snapshot(
            total_mutants=self.metrics.mutations_created,
            next_iteration=self.metrics.iteration,
            programs_processed=self.metrics.programs_processed,
        )

        try:
            await self._drain_and_ingest_initial_population()

            if self._pre_step_hook:
                await self._pre_step_hook()

            generation = 0
            while self._running:
                if self._reached_generation_cap(generation):
                    logger.info(
                        "[BatchEvolution] Stop: max_generations={}",
                        self._batch_config.max_generations,
                    )
                    break
                if self._reached_mutant_cap():
                    decision = self.stopper.should_stop(self.build_stop_context())
                    logger.info("[BatchEvolution] Stop: {}", decision.reason)
                    break

                created = await self._run_generation(generation)
                generation += 1
                if created == 0:
                    await asyncio.sleep(self._batch_config.loop_interval)

        except asyncio.CancelledError:
            logger.debug("[BatchEvolution] run() cancelled")
            raise
        finally:
            self._running = False
            with contextlib.suppress(Exception):
                await self._await_idle()
                await self._ingest_completed_programs()
                await self._write_snapshot(
                    total_mutants=self.metrics.mutations_created,
                    next_iteration=self.metrics.iteration,
                    programs_processed=self.metrics.programs_processed,
                )
            try:
                await self._post_run_hook.on_run_complete(self.storage)
            except Exception as exc:
                logger.error("[BatchEvolution] post-run hook failed: {}", exc)
            logger.info("[BatchEvolution] Stopped")

    async def _drain_and_ingest_initial_population(self) -> None:
        await self._await_idle()
        await self._ingest_completed_programs()
        self.storage.snapshot.bump(incremental=True)
        await self._write_snapshot(programs_processed=self.metrics.programs_processed)
        refreshed = await self._refresh_archive_programs()
        if refreshed:
            await self._await_idle()
            await self._ingest_completed_programs()

    async def _run_generation(self, generation: int) -> int:
        await self._await_idle()
        await self._ingest_completed_programs()

        elites = await self.strategy.select_elites(
            total=self._batch_config.max_elites_per_generation
        )
        self.metrics.elites_selected += len(elites)
        logger.info(
            "[BatchEvolution] generation={} selected {} elite(s)",
            generation,
            len(elites),
        )
        if not elites:
            return 0

        parent_groups = self._select_parent_groups(elites)
        if not parent_groups:
            logger.info("[BatchEvolution] generation={} no parent groups", generation)
            return 0

        created = 0
        for task_id, parents in enumerate(parent_groups):
            if self._reached_mutant_cap():
                break
            my_iteration = self.metrics.iteration
            self.metrics.iteration += 1
            new_id = await generate_one_mutation(
                parents,
                mutator=self.mutation_operator,
                storage=self.storage,
                state_manager=self.state,
                iteration=my_iteration,
                task_id=task_id,
            )
            if new_id is not None:
                created += 1
                self.metrics.mutations_created += 1
                await self._write_snapshot(
                    total_mutants=self.metrics.mutations_created,
                    next_iteration=self.metrics.iteration,
                )

        logger.info(
            "[BatchEvolution] generation={} created {} mutant(s)",
            generation,
            created,
        )

        await self._await_idle()
        before_added = self.metrics.added
        await self._ingest_completed_programs()
        self.storage.snapshot.bump(incremental=True)
        await self._write_snapshot(programs_processed=self.metrics.programs_processed)

        if self.metrics.added > before_added and self._post_step_hook is not None:
            await _run_bounded_post_step_hook(self)

        refreshed = await self._refresh_archive_programs()
        if refreshed:
            logger.info(
                "[BatchEvolution] generation={} refreshed {} archive program(s)",
                generation,
                refreshed,
            )
            await self._await_idle()

        return created

    def _select_parent_groups(self, elites) -> list:
        parent_iterator = self._batch_config.parent_selector.create_parent_iterator(
            elites
        )
        parent_groups = []
        for parents in parent_iterator:
            if len(parent_groups) >= self._batch_config.max_mutations_per_generation:
                break
            parent_groups.append(parents)
        logger.info(
            "[BatchEvolution] selected {} parent group(s) from {} elite(s)",
            len(parent_groups),
            len(elites),
        )
        return parent_groups

    async def _refresh_archive_programs(self) -> int:
        program_ids = await self.strategy.get_program_ids()
        if not program_ids:
            return 0

        programs = await self.storage.mget(program_ids)
        tasks = []
        for program in programs:
            if program is not None and program.state == ProgramState.DONE:
                tasks.append(
                    asyncio.create_task(
                        self.state.set_program_state(program, ProgramState.QUEUED)
                    )
                )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.metrics.submitted_for_refresh += len(tasks)
        return len(tasks)

    def _reached_generation_cap(self, generation: int) -> bool:
        cap = self._batch_config.max_generations
        return cap is not None and generation >= cap
