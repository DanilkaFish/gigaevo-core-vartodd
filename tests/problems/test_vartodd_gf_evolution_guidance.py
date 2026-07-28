from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DESCRIPTION = REPO_ROOT / "problems" / "vartodd_evo_gf" / "task_description.txt"
INSIGHTS_SYSTEM = (
    REPO_ROOT / "problems" / "vartodd_evo_gf" / "prompts" / "insights" / "system.txt"
)
MUTATION_SYSTEM = (
    REPO_ROOT / "problems" / "vartodd_evo_gf" / "prompts" / "mutation" / "system.txt"
)
UPDATED_ALGORITHM = (
    REPO_ROOT / "config" / "algorithm" / "vartodd_diverse_gf16_tohpe_updated.yaml"
)


def test_regime_guidance_supports_capped_to_full_escalation() -> None:
    text = UPDATED_ALGORITHM.read_text(encoding="utf-8")

    assert "no more than 10_000" not in text
    assert "nom more than 10_000" not in text
    assert "limit_bucket=-1" in text
    assert "min_buckets" in text
    assert "max_buckets" in text
    assert "limit_bucket" in text


def test_task_context_separates_y_per_bucket_from_z_coverage() -> None:
    text = TASK_DESCRIPTION.read_text(encoding="utf-8")

    assert "for one researched z bucket" in text
    assert "does not bound how many z buckets" in text
    assert "Low `dim` alone" in text


def test_prompts_prioritize_scores_without_hiding_generation_failure() -> None:
    combined = (
        TASK_DESCRIPTION.read_text(encoding="utf-8")
        + INSIGHTS_SYSTEM.read_text(encoding="utf-8")
    )

    assert "most influential mapped policy parameters" in combined
    assert "cannot select an action that was never generated" in combined


def test_mutation_prompt_requires_one_primary_causal_experiment() -> None:
    text = MUTATION_SYSTEM.read_text(encoding="utf-8")

    assert "one primary causal hypothesis" in text
    assert "controls held fixed" in text
    assert "optimizer-specific evidence" in text
