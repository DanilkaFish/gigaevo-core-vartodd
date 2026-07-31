from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MUTATION_SYSTEM = (
    REPO_ROOT / "problems" / "vartodd_evo_gf" / "prompts" / "mutation" / "system.txt"
)
UPDATED_ALGORITHM = (
    REPO_ROOT / "config" / "algorithm" / "vartodd_diverse_gf16_tohpe_updated.yaml"
)
TASK_DESCRIPTION = (
    REPO_ROOT / "problems" / "vartodd_evo_gf" / "task_description.txt"
)


def test_regime_guidance_is_owned_by_three_parent_islands() -> None:
    text = UPDATED_ALGORITHM.read_text(encoding="utf-8")

    assert "regime_id: ab_initio" in text
    assert "regime_id: mid_margin" in text
    assert "regime_id: near_end" in text
    assert "enable_migration: false" in text
    assert "mutation_regime_guidance: []" in text
    assert "shared Live Path Store" in text


def test_mutation_prompt_requires_one_primary_causal_experiment() -> None:
    text = MUTATION_SYSTEM.read_text(encoding="utf-8")

    assert "one primary causal hypothesis" in text
    assert "controls held fixed" in text
    assert "optimizer-specific evidence" in text


def test_near_tail_regime_has_concrete_small_margin_plateau_escapes() -> None:
    text = UPDATED_ALGORITHM.read_text(encoding="utf-8")
    near_tail = text.split("regime_id: near_end", maxsplit=1)[1]

    assert "small margin" in near_tail.lower()
    assert "plateau" in near_tail.lower()
    assert "optimizer settings" in near_tail
    assert "broad beam search" in near_tail
    assert "TODD" in near_tail
    assert "rare and diverse actions" in near_tail
    assert "cognitive/social" not in near_tail
    assert "beamwidth 10..smth_big" not in near_tail


def test_shared_description_mentions_small_margin_plateaus_neutrally() -> None:
    text = TASK_DESCRIPTION.read_text(encoding="utf-8")

    assert "Small margins can also create broad optimization plateaus" in text
    assert "prefers the path with the highest initial rank" in text


def test_shared_description_does_not_explain_fitness_penalties() -> None:
    text = TASK_DESCRIPTION.read_text(encoding="utf-8")

    assert "Fitness shaping on top of rank" not in text
    assert "penalized" not in text
    assert "reward (up to" not in text


def test_shared_description_defines_z_saturation_as_research_starvation() -> None:
    text = TASK_DESCRIPTION.read_text(encoding="utf-8")
    compact = " ".join(text.split())
    compact_lower = compact.lower()

    assert "positive actions are rare" in compact_lower
    assert "finding more requires a broader z-bucket research budget" in compact_lower
    assert "increase `max_buckets`" in compact_lower
    assert "increase `limit_bucket`" in compact_lower
