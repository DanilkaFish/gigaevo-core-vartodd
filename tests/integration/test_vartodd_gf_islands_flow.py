from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from custom.vartodd_islands_context import VartoddIslandsRouteContextProvider
from gigaevo.evolution.engine.mutation import generate_one_mutation
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.strategies.base import MutationRoute, ParentRole
from gigaevo.evolution.strategies.models import MutationRouteConfig
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from run_gf_islands import build_gf_islands_overrides

REPO_ROOT = Path(__file__).resolve().parents[2]


class _MemoryStorage:
    def __init__(self) -> None:
        self.programs: dict[str, Program] = {}
        self.run_state: dict[str, object] = {}

    async def add(self, program: Program) -> None:
        self.programs[program.id] = program

    async def get(self, program_id: str) -> Program | None:
        return self.programs.get(program_id)

    async def save_run_state(self, key: str, value: object) -> None:
        self.run_state[key] = value

    async def load_run_state(self, key: str) -> object | None:
        return self.run_state.get(key)


class _StateManager:
    async def update_program(self, program: Program) -> None:
        return None


class _FakeIsland:
    def __init__(self, island_id: str) -> None:
        self.config = SimpleNamespace(island_id=island_id, max_size=20)
        self.programs: list[Program] = []

    async def add(self, program: Program) -> bool:
        program.metadata["current_island"] = self.config.island_id
        if all(candidate.id != program.id for candidate in self.programs):
            self.programs.append(program)
        return True

    async def select_elites(self, total: int) -> list[Program]:
        return self.programs[:total]

    async def __len__(self) -> int:
        return len(self.programs)


class _CaptureRouteMutator(MutationOperator):
    def __init__(self, provider: VartoddIslandsRouteContextProvider) -> None:
        self.provider = provider
        self.prompt = ""

    async def mutate_single(
        self,
        selected_parents: list[Program],
        memory_instructions: str | None = None,
    ) -> MutationSpec | None:
        del selected_parents, memory_instructions
        raise AssertionError("routed flow must call mutate_with_route")

    async def mutate_with_route(
        self,
        selected_parents: list[Program],
        route: MutationRoute | None,
        parent_roles: tuple[ParentRole, ...] = (),
    ) -> MutationSpec:
        assert route is not None
        blocks = [
            self.provider.build_assignment(route, selected_parents, parent_roles)
        ]
        for parent, role in zip(selected_parents, parent_roles, strict=True):
            context = str(parent.metadata.get("mutation_context") or "")
            blocks.append(
                f"## Parent role: {role}\n"
                + self.provider.filter_parent_context(
                    route,
                    parent,
                    role,
                    context,
                )
            )
        blocks.extend(
            [
                self.provider.build_external_context(route),
                self.provider.build_route_guidance(route),
            ]
        )
        self.prompt = "\n\n".join(block for block in blocks if block)
        return MutationSpec(
            code="def entrypoint():\n    return 1\n",
            parents=selected_parents,
            name="mid-margin integration child",
        )


def _program(name: str) -> Program:
    return Program(
        code=f"def entrypoint():\n    return {name!r}\n",
        state=ProgramState.DONE,
        metadata={
            "source": "initial_program",
            "strategy_name": name,
            "mutation_context": (
                "## Parent program\n"
                "Use its policy evidence.\n"
                "## Program Aux Excerpt\n"
                "this path name: hidden-parent-path\n"
            ),
        },
    )


def _record(name: str) -> dict[str, object]:
    return {
        "name": name,
        "rank": 395,
        "depth": 40,
        "limit_buckets": 4096,
        "max_todd_z_researched": 3500,
        "init_rank": 435,
        "init_rank_thr": 420,
        "kind": "partial",
        "restart_band": "near_final",
        "used_count": 0,
        "improved_count": 1,
        "best_child_rank": 394,
        "child_improved_loaded": True,
        "is_stale_improved_child": False,
    }


def _island_config(island_id: str) -> SimpleNamespace:
    return SimpleNamespace(island_id=island_id, max_size=20)


async def _exercise_route_selection_prompt_metadata_and_destination_flow(
    monkeypatch,
    tmp_path: Path,
) -> None:
    overrides = build_gf_islands_overrides(
        ["matrix=16", "lb=380", "ub=421", "cache=false"],
        repository_root=REPO_ROOT,
        runtime_root=tmp_path,
    )
    overlay = Path(
        next(
            item.removeprefix("problem.dir=")
            for item in overrides
            if item.startswith("problem.dir=")
        )
    )
    monkeypatch.setenv("VARTODD_VARIANT_DIR", str(overlay))
    monkeypatch.setenv("VARTODD_REPOSITORY_ROOT", str(REPO_ROOT))

    provider = VartoddIslandsRouteContextProvider(
        problem_dir=overlay,
        root_dir=tmp_path / "cards",
    )
    store = provider._path_store()
    record = _record("f395_i435_card_lim4096_z3500of4096")
    monkeypatch.setattr(store, "iter_path_records", lambda: [record])
    store._write_evidence_card(
        str(record["name"]),
        {
            "version": 1,
            "path_stats": (
                "path_summary:\n"
                "  depth=40 init_rank=435 final_rank=395\n"
                "path_policy_groups:\n"
                "  g1 435->410\n"
                "  g2 410->395\n"
                "converged_policy_profiles:\n"
                "  P1 scores=final=w[red:+2]"
            ),
            "producer": {
                "runtime": 1400.0,
                "total_evals": 6000,
                "best_seen_times": 25,
                "last_improvement": "5100/395",
                "timeout_salvaged": False,
            },
        },
    )
    monkeypatch.setattr(provider, "_path_store", lambda: store)

    storage = _MemoryStorage()
    islands = {
        island_id: _FakeIsland(island_id)
        for island_id in ("ab_initio", "mid_margin", "near_end")
    }
    with patch(
        "gigaevo.evolution.strategies.multi_island.MapElitesIsland",
        side_effect=lambda config, _storage: islands[config.island_id],
    ):
        strategy = MapElitesMultiIsland(
            island_configs=[
                _island_config("ab_initio"),
                _island_config("mid_margin"),
                _island_config("near_end"),
            ],
            program_storage=storage,
            enable_migration=False,
            initial_island_id="ab_initio",
            bootstrap_source_island="ab_initio",
            bootstrap_until_size=8,
            bootstrap_mix_probability=0.70,
            steady_mix_probability=0.10,
            route_context_provider=provider,
            mutation_routes=[
                MutationRouteConfig(
                    regime_id="mid_margin",
                    island_id="mid_margin",
                    probability=1.0,
                    context_profile="path_refinement",
                )
            ],
        )

    roots = [_program("seed_a"), _program("seed_b")]
    for root in roots:
        await storage.add(root)
        assert await strategy.add(root) is True
        assert root.metadata["current_island"] == "ab_initio"

    selection = await strategy.select_for_mutation(2)
    assert selection.route is not None
    assert selection.route.regime_id == "mid_margin"
    assert selection.parent_roles == ("bootstrap_donor", "bootstrap_donor")

    mutator = _CaptureRouteMutator(provider)
    child_id = await generate_one_mutation(
        selection.parents,
        mutator=mutator,
        storage=storage,
        state_manager=_StateManager(),
        iteration=1,
        route=selection.route,
        parent_roles=selection.parent_roles,
    )
    assert child_id is not None
    child = storage.programs[child_id]
    assert child.metadata["mutation_regime"] == "mid_margin"
    assert child.metadata["target_island"] == "mid_margin"
    assert child.metadata["mutation_parent_roles"] == [
        "bootstrap_donor",
        "bootstrap_donor",
    ]
    assert "target_regime: mid_margin" in mutator.prompt
    assert mutator.prompt.count("Parent role: bootstrap_donor") == 2
    assert "## Selectable Shared Paths" in mutator.prompt
    assert str(record["name"]) in mutator.prompt
    assert mutator.prompt.count("## Required Island Regime:") == 1

    assert await strategy.add(child) is True
    assert child.metadata["current_island"] == "mid_margin"
    next_selection = await strategy.select_for_mutation(2)
    assert next_selection.parent_roles == (
        "target_island",
        "bootstrap_donor",
    )
    assert next_selection.parents[0].id == child.id
    assert next_selection.parents[1].id in {root.id for root in roots}


def test_route_selection_prompt_metadata_and_destination_flow(
    monkeypatch,
    tmp_path: Path,
) -> None:
    asyncio.run(
        _exercise_route_selection_prompt_metadata_and_destination_flow(
            monkeypatch,
            tmp_path,
        )
    )
