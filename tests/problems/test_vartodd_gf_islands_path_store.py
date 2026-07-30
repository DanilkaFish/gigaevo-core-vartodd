from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEM_DIR = REPO_ROOT / "problems" / "vartodd_evo_gf_islands"
LEGACY_DIR = REPO_ROOT / "problems" / "vartodd_evo_gf"


def _load_path_store(monkeypatch, tmp_path: Path):
    variant_dir = tmp_path / "runtime" / "vartodd_evo_gf_islands16"
    variant_dir.mkdir(parents=True)
    (variant_dir / "metrics.yaml").write_text(
        "specs:\n  fitness:\n    lower_bound: 380\n",
        encoding="utf-8",
    )
    matrix_dir = tmp_path / "npy"
    matrix_dir.mkdir()
    (matrix_dir / "gf2^16_test.npy").write_bytes(b"matrix")
    monkeypatch.setenv("VARTODD_VARIANT_DIR", str(variant_dir))
    monkeypatch.setenv("VARTODD_REPOSITORY_ROOT", str(tmp_path))

    variant_spec = importlib.util.spec_from_file_location(
        "variant",
        PROBLEM_DIR / "variant.py",
    )
    assert variant_spec is not None and variant_spec.loader is not None
    variant = importlib.util.module_from_spec(variant_spec)
    sys.modules["variant"] = variant
    variant_spec.loader.exec_module(variant)

    spec = importlib.util.spec_from_file_location(
        "_test_vartodd_gf_islands_path_store",
        PROBLEM_DIR / "path_store.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path[:0] = [str(PROBLEM_DIR), str(LEGACY_DIR)]
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.path[:2]
    return module


def _record(
    name: str,
    rank: int,
    *,
    init_rank: int,
    used: int = 0,
    child_improved_loaded: bool | None = None,
) -> dict:
    return {
        "name": name,
        "rank": rank,
        "depth": init_rank - rank,
        "limit_buckets": 4096,
        "max_todd_z_researched": 3000,
        "init_rank": init_rank,
        "init_rank_thr": init_rank - 20,
        "kind": "partial",
        "restart_band": "near_final",
        "used_count": used,
        "improved_count": 1,
        "best_child_rank": rank - 1,
        "child_improved_loaded": child_improved_loaded,
        "is_stale_improved_child": False,
    }


def _write_card(root: Path, name: str, *, runtime: float = 651.0) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    card = {
        "version": 1,
        "path_stats": (
            "path_summary:\n"
            "  depth=15 init_rank=433 final_rank=395 total_reduction=38\n"
            "path_policy_groups:\n"
            "  g1 433->410 red=23 P1 s5\n"
            "  g2 410->395 red=15 P2 s7\n"
            "converged_policy_profiles:\n"
            "  P1 rank_region=433->410 scores=pool=w[red:+1]\n"
            "  P2 rank_region=410->395 scores=final=w[red:+2]"
        ),
        "producer": {
            "metrics": {"fitness": 395.2},
            "runtime": runtime,
            "total_evals": 5535,
            "best_seen_times": 63,
            "last_improvement": "2034/395",
            "timeout_salvaged": False,
        },
    }
    (directory / "evidence_card.json").write_text(
        json.dumps(card),
        encoding="utf-8",
    )


def test_selectable_cards_use_one_best_representative_per_rank(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    root = tmp_path / "cards"
    store = module.PathStore(root_dir=str(root))
    records = [
        _record("lower-init", 395, init_rank=500),
        _record("higher-init", 395, init_rank=540, used=2),
        _record("rank-397", 397, init_rank=530),
        _record(
            "nonimproved-child-name",
            394,
            init_rank=540,
            child_improved_loaded=False,
        ),
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)
    for record in records:
        _write_card(root, record["name"])

    cards = store.render_selectable_path_cards(top_k=8, max_chars=24_000)

    assert store.has_selectable_paths() is True
    assert "## Selectable Shared Paths" in cards
    assert "### higher-init" in cards
    assert "### lower-init" not in cards
    assert "### rank-397" in cards
    assert "nonimproved-child-name" not in cards
    assert "policy_bands:" in cards
    assert "policy_profiles:" in cards
    assert "producer_search:" in cards


def test_rank_tie_prefers_lower_reuse_after_initial_rank(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    records = [
        _record("used", 395, init_rank=540, used=3),
        _record("fresh", 395, init_rank=540, used=0),
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)

    assert [item["name"] for item in store.selectable_records()] == ["fresh"]


def test_missing_card_is_derived_and_complete_cards_obey_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    root = tmp_path / "cards"
    store = module.PathStore(root_dir=str(root))
    records = [
        _record("first", 395, init_rank=540),
        _record("second", 397, init_rank=530),
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)
    monkeypatch.setattr(
        store,
        "_derive_path_stats",
        lambda name: (
            "path_summary:\n  depth=1 init_rank=400 final_rank=399\n"
            "path_policy_groups:\n  g1 400->399\n"
            "converged_policy_profiles:\n  P1 scores=x"
        ),
    )

    one_card = store._render_record_card(records[0])
    rendered = store.render_selectable_path_cards(
        top_k=8,
        max_chars=len("## Selectable Shared Paths\n\n") + len(one_card),
    )

    assert "### first" in rendered
    assert "### second" not in rendered
    assert "selection:" in rendered
    assert not list(root.rglob("*.tmp"))


def test_update_evidence_card_is_atomic_and_preserves_path_stats(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    root = tmp_path / "cards"
    store = module.PathStore(root_dir=str(root))
    _write_card(root, "path")

    store.update_evidence_card(
        "path",
        {
            "runtime": 1200.0,
            "total_evals": 8000,
            "timeout_salvaged": True,
        },
    )

    card = json.loads((root / "path" / "evidence_card.json").read_text())
    assert card["path_stats"].startswith("path_summary:")
    assert card["producer"]["runtime"] == 1200.0
    assert card["producer"]["total_evals"] == 8000
    assert card["producer"]["timeout_salvaged"] is True
    assert not list((root / "path").glob("*.tmp"))
