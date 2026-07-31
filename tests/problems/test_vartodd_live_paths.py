from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_path_store(problem_name: str):
    problem_dir = REPO_ROOT / "problems" / problem_name
    module_path = problem_dir / "path_store.py"
    module_name = f"_test_{problem_name}_path_store"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(problem_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(problem_dir))
    return module.PathStore


def _record(
    name: str,
    *,
    rank: int,
    limit_buckets: int | None,
    child_improved_loaded: bool | None,
    used_count: int = 0,
    improved_count: int = 0,
    best_child_rank: int | None = None,
) -> dict:
    return {
        "name": name,
        "rank": rank,
        "depth": 2,
        "limit_buckets": limit_buckets,
        "init_rank": 20,
        "init_rank_thr": 15,
        "kind": "partial",
        "restart_band": "near_final",
        "used_count": used_count,
        "improved_count": improved_count,
        "best_child_rank": best_child_rank,
        "last_child_rank": None,
        "last_used_at": None,
        "last_improved_at": None,
        "parent_path_name": "parent" if child_improved_loaded is not None else None,
        "parent_loaded_rank": 10 if child_improved_loaded is not None else None,
        "parent_loaded_start_rank": 20 if child_improved_loaded is not None else None,
        "parent_init_rank_thr": 15 if child_improved_loaded is not None else None,
        "parent_best_child_rank": None,
        "child_improved_loaded": child_improved_loaded,
        "is_stale_improved_child": False,
    }


def _selectable_text(summary: str) -> str:
    before_evidence = summary.split("dead_end_paths:", 1)[0]
    return before_evidence.split("best_nonimproved_child_paths:", 1)[0]


@pytest.mark.parametrize("problem_name", ["vartodd_evo", "vartodd_gf32"])
def test_low_limit_nonimproving_child_is_selectable_only_near_tail(
    problem_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_store_type = _load_path_store(problem_name)
    store = path_store_type()
    records = [
        _record(
            "low_cap_nonimprover",
            rank=12,
            limit_buckets=500,
            child_improved_loaded=False,
        ),
        _record(
            "high_cap_nonimprover",
            rank=1,
            limit_buckets=100_000,
            child_improved_loaded=False,
        ),
        _record(
            "full_nonimprover",
            rank=2,
            limit_buckets=-1,
            child_improved_loaded=False,
        ),
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)

    summary = store.summarize(
        near_tail_top_k=10,
        wide_margin_top_k=10,
        near_tail_limit_cutoff=100_000,
    )
    selectable = _selectable_text(summary)

    assert "- low_cap_nonimprover " in selectable
    assert "high_cap_nonimprover" not in selectable
    assert "full_nonimprover" not in selectable
    assert "hidden(not selectable): nonimproved_child=2" in summary


@pytest.mark.parametrize("problem_name", ["vartodd_evo", "vartodd_gf32"])
def test_low_limit_nonimproving_child_still_retires_after_saturated_reuse(
    problem_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_store_type = _load_path_store(problem_name)
    store = path_store_type()
    records = [
        _record(
            "reused_low_cap_nonimprover",
            rank=12,
            limit_buckets=500,
            child_improved_loaded=False,
            used_count=11,
            improved_count=0,
            best_child_rank=12,
        )
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)

    summary = store.summarize(max_nonimproved_reuse=10)

    assert "reused_low_cap_nonimprover" not in _selectable_text(summary)
