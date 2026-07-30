from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from custom.vartodd_islands_context import (
    IslandEvolutionaryStatisticsCollector,
    PathCardEnrichmentStage,
    PathCardEnrichmentInputs,
    VartoddIslandsRouteContextProvider,
)
from gigaevo.evolution.strategies.base import MutationRoute
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.common import Box


def _route(name: str, profile: str) -> MutationRoute:
    return MutationRoute(
        regime_id=name,
        island_id=name,
        guidance=f"## Required Island Regime\n{name}",
        context_profile=profile,
    )


def _metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="rank",
                is_primary=True,
                higher_is_better=False,
            ),
            "is_valid": MetricSpec(
                description="valid",
                higher_is_better=True,
            ),
        }
    )


def _program(
    fitness: float,
    island: str | None,
    *,
    source: str | None = None,
) -> Program:
    program = Program(code="def solve(): return 1", state=ProgramState.DONE)
    program.add_metrics({"fitness": fitness, "is_valid": 1.0})
    if island is not None:
        program.metadata["current_island"] = island
    if source is not None:
        program.metadata["source"] = source
    return program


def _aux() -> str:
    return """## Program execution aux info

path_summary:
  depth=15 init_rank=433 final_rank=395 total_reduction=38
path_policy_groups:
  g1 433->420 red=13 P1 s3 src=H:20/T:0 pool:20 z:10
  g2 420->410 red=10 P2 s4 src=H:9/T:1 pool:10 z:100
  g3 410->402 red=8 P3 s5 src=H:1/T:2 pool:3 z:900
  g4 402->398 red=4 P4 s5 src=H:0/T:2 pool:2 z:3000
  g5 398->395 red=3 P5 s6 src=H:0/T:1 pool:1 z:5000
converged_policy_profiles:
  P1 rank_region=433->420 scores=pool=w[red:+1]
  P2 rank_region=420->410 scores=pool=w[red:+2]
  P3 rank_region=410->402 scores=final=w[red:+3]
  P4 rank_region=402->398 scores=final=w[red:+4]
  P5 rank_region=398->395 scores=final=w[red:+5]
search_stat:
rank 0.9q=407.0
rank 0.1q=397.0
init=0/433: 39/399 -> 342/397
last_improvement=2034/395 total_evals=5535 best_seen_times=63
total_evals: 5535
best_seen_times: 63
loaded_path_improved: 1
loaded_rank: 433
loaded_path_name: f397_i463_parent_z512of1024
init_rank_thr: 432
this path name: f395_i433_child_z5000of8192"""


def test_provider_filters_context_by_route_and_shares_refinement_cards(
    tmp_path,
    monkeypatch,
) -> None:
    path_store = MagicMock()
    path_store.has_selectable_paths.return_value = True
    path_store.render_selectable_path_cards.return_value = (
        "## Selectable Shared Paths\n\n### path"
    )
    provider = VartoddIslandsRouteContextProvider(
        problem_dir=tmp_path,
        parent_aux_max_chars=12_000,
    )
    monkeypatch.setattr(provider, "_path_store", lambda: path_store)
    parent = _program(395.0, "ab_initio")
    parent.metadata["aux_info"] = _aux()
    stored_context = (
        "metrics\n\n## Execution Signal Digest\nold\n\n---\n\n"
        "## Program Aux Excerpt\nold excerpt"
    )
    ab_route = _route("ab_initio", "ab_initio")
    mid_route = _route("mid_margin", "path_refinement")
    near_route = _route("near_end", "path_refinement")

    ab_context = provider.filter_parent_context(
        ab_route,
        parent,
        "target_island",
        stored_context,
    )
    mid_context = provider.filter_parent_context(
        mid_route,
        parent,
        "target_island",
        stored_context,
    )
    near_context = provider.filter_parent_context(
        near_route,
        parent,
        "target_island",
        stored_context,
    )

    assert provider.route_is_available(ab_route) is True
    assert provider.route_is_available(mid_route) is True
    assert provider.build_external_context(ab_route) == ""
    assert (
        provider.build_external_context(mid_route)
        == provider.build_external_context(near_route)
    )
    assert "loaded_path_name" not in ab_context
    assert "this path name" not in ab_context
    assert "loaded_path_improved: 1" in mid_context
    assert "init_rank_thr: 432" in mid_context
    assert "g1 433->420" in mid_context
    assert "g1 433->420" not in near_context
    assert "g3 410->402" in near_context
    assert "g4 402->398" in near_context
    assert "g5 398->395" in near_context
    assert "P3 rank_region" in near_context
    assert "P5 rank_region" in near_context
    assert "last_improvement=2034/395" in near_context
    assert "pool:1 z:5000" in near_context
    assert len(ab_context) <= 12_000
    assert len(mid_context) <= 12_000
    assert len(near_context) <= 12_000


async def test_statistics_are_island_local_and_initial_roots_use_ab_initio():
    storage = AsyncMock()
    ab = _program(410.0, "ab_initio")
    mid = _program(400.0, "mid_margin")
    near = _program(390.0, "near_end")
    initial = _program(
        405.0,
        None,
        source="initial_program",
    )
    storage.snapshot.get_all.return_value = [ab, mid, near, initial]
    storage.mget.return_value = []
    collector = IslandEvolutionaryStatisticsCollector(
        storage=storage,
        metrics_context=_metrics_context(),
        timeout=5.0,
    )
    collector.attach_inputs({})

    result = await collector.execute(initial)

    assert result.output.archive_valid_fitnesses == (405.0, 410.0)
    storage.snapshot.get_all.assert_awaited_once_with(
        storage,
        exclude=EXCLUDE_STAGE_RESULTS,
    )


async def test_path_card_enrichment_records_exact_saved_path(
    tmp_path,
    monkeypatch,
) -> None:
    store = MagicMock()
    stage = PathCardEnrichmentStage(problem_dir=tmp_path, timeout=5.0)
    monkeypatch.setattr(stage, "_path_store", lambda: store)
    stage.attach_inputs(
        {
            "metrics": Box[dict[str, float]](
                data={"fitness": 395.2, "is_valid": 1.0}
            ),
            "non_metrics": Box[dict[str, str]](data={"aux info": _aux()}),
            "runtime": Box[dict[str, float]](data={"runtime": 651.0}),
        }
    )

    result = await stage.execute(_program(395.2, "near_end"))

    assert result.output.data == "f395_i433_child_z5000of8192"
    store.update_evidence_card.assert_called_once()
    name, producer = store.update_evidence_card.call_args.args
    assert name == "f395_i433_child_z5000of8192"
    assert producer["metrics"]["fitness"] == 395.2
    assert producer["runtime"] == 651.0
    assert producer["total_evals"] == 5535
    assert producer["best_seen_times"] == 63
    assert producer["last_improvement"] == "2034/395"
    assert producer["timeout_salvaged"] is False


def test_enrichment_input_contract_is_optional() -> None:
    params = PathCardEnrichmentInputs(
        metrics=None,
        non_metrics=None,
        runtime=None,
    )
    assert params.metrics is None

