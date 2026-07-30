from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE = REPO_ROOT / "config" / "pipeline" / "vartodd_islands_pipeline.yaml"


def test_island_pipeline_uses_local_statistics_and_enriches_path_cards() -> None:
    config = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    blueprint = config["dag_blueprint"]
    nodes = blueprint["nodes"]

    assert nodes["EvolutionaryStatisticsCollector"]["_target_"] == (
        "custom.vartodd_islands_context."
        "IslandEvolutionaryStatisticsCollector"
    )
    assert nodes["PathCardEnrichmentStage"]["_target_"] == (
        "custom.vartodd_islands_context.PathCardEnrichmentStage"
    )
    edges = {
        (
            edge["source_stage"],
            edge["destination_stage"],
            edge["input_name"],
        )
        for edge in blueprint["data_flow_edges"]
    }
    assert (
        "EnsureMetricsStage",
        "PathCardEnrichmentStage",
        "metrics",
    ) in edges
    assert (
        "EnsureNonMetricsStage",
        "PathCardEnrichmentStage",
        "non_metrics",
    ) in edges
    assert (
        "ComputeTimeStage",
        "PathCardEnrichmentStage",
        "runtime",
    ) in edges
    assert "MemoryContextStage" not in nodes
