from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(problem_name: str, filename: str):
    os.environ.setdefault("VARTODD_REPOSITORY_ROOT", str(REPO_ROOT))
    os.environ.setdefault(
        "VARTODD_VARIANT_DIR",
        str(
            REPO_ROOT
            / ".run_gf"
            / "overlays"
            / "gf16_lb380_ub420"
            / "vartodd_evo_gf16"
        ),
    )
    problem_dir = REPO_ROOT / "problems" / problem_name
    module_path = problem_dir / filename
    module_name = f"_test_{problem_name}_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(problem_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(problem_dir))
    return module


def _path_with_todd_limit(limit: int, researched_z: tuple[int, ...] = (64, 512, 128)):
    todd = SimpleNamespace(
        pool=SimpleNamespace(keep=4),
        buckets=SimpleNamespace(limit_bucket=limit),
    )
    dao = SimpleNamespace(mode=SimpleNamespace(todd=SimpleNamespace(points=[(567, todd)])))
    node = SimpleNamespace(
        state=SimpleNamespace(
            rows=567, to_numpy=lambda: np.zeros((567, 48), dtype=np.uint8)
        ),
        parent=None,
        incoming=None,
    )
    for index, total in enumerate(researched_z):
        rows = 500 - index * 40 if index + 1 < len(researched_z) else 411
        node = SimpleNamespace(
            state=SimpleNamespace(
                rows=rows,
                to_numpy=lambda rows=rows: np.zeros((rows, 48), dtype=np.uint8),
            ),
            parent=node,
            incoming=SimpleNamespace(
                total=total + 10_000,
                global_info=SimpleNamespace(
                    z_researched=total + 10_000,
                    z_researched_tohpeprefix=10_000,
                    z_researched_todd=total,
                ),
            ),
        )
    if not researched_z:
        node = SimpleNamespace(
            state=SimpleNamespace(
                rows=411,
                to_numpy=lambda: np.zeros((411, 48), dtype=np.uint8),
            ),
            parent=node,
            incoming=SimpleNamespace(),
        )
    return SimpleNamespace(final_node=node, daos=[dao])


def test_saved_path_name_embeds_todd_limit_and_observed_z(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_module("vartodd_evo_gf", "helper.py")
    monkeypatch.setenv("GIGAEVO_PROGRAM_ID", "12345678-abcdef")
    evaluator = SimpleNamespace(
        initial_rank=567,
        _path_root_rank=lambda path, fallback: 567,
        _path_name_mid_rank=lambda path: 490,
        _path_max_todd_z_researched=lambda path: 512,
    )

    name = helper.BaseEvaluator._auto_hashed_name(
        evaluator, "", _path_with_todd_limit(1024)
    )

    assert name == "f411_i490_12345678_z512of1024"


def test_observed_z_comes_from_selected_path_todd_stats_only() -> None:
    path_store = _load_module("vartodd_evo_gf", "path_store.py")

    assert (
        path_store.PathStore._path_max_todd_z_researched(
            _path_with_todd_limit(1024, (64, 512, 128))
        )
        == 512
    )


def test_legacy_aggregate_only_z_stats_are_not_reported_as_todd_zero() -> None:
    path_store = _load_module("vartodd_evo_gf", "path_store.py")
    path = _path_with_todd_limit(1024, (512,))
    path.final_node.incoming.global_info = SimpleNamespace(
        z_researched=512,
        z_researched_tohpeprefix=0,
        z_researched_todd=0,
    )

    assert path_store.PathStore._path_max_todd_z_researched(path) is None


def test_saved_path_name_marks_unknown_observed_z(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_module("vartodd_evo_gf", "helper.py")
    monkeypatch.setenv("GIGAEVO_PROGRAM_ID", "12345678-abcdef")
    evaluator = SimpleNamespace(
        initial_rank=567,
        _path_root_rank=lambda path, fallback: 567,
        _path_name_mid_rank=lambda path: 490,
        _path_max_todd_z_researched=lambda path: None,
    )

    name = helper.BaseEvaluator._auto_hashed_name(
        evaluator, "", _path_with_todd_limit(512, ())
    )

    assert name == "f411_i490_12345678_zunknownof512"


def test_live_path_summary_uses_metadata_in_name_not_duplicate_field() -> None:
    path_store = _load_module("vartodd_evo_gf", "path_store.py")
    record = {
        "name": "f411_i490_12345678_z384of512",
        "rank": 411,
        "depth": 40,
        "limit_buckets": 512,
        "max_todd_z_researched": 384,
        "used_count": 0,
        "improved_count": 0,
        "best_child_rank": None,
        "kind": "partial",
        "restart_band": "near_final",
        "parent_path_name": None,
    }

    summary = path_store.PathStore._format_summary_record(record)

    assert "_z384of512" in summary
    assert "limit_buckets=" not in summary
    assert "max_z_researched=" not in summary
    assert "max_todd_z_researched=" not in summary


def test_validation_reads_loaded_todd_limit_from_extended_path_name() -> None:
    validate = _load_module("vartodd_evo_gf", "validate.py")

    assert (
        validate._loaded_path_todd_limit(
            "loaded_path_name: f411_i490_12345678_lim512_z384"
        )
        == 512
    )


def test_live_path_store_defaults_to_problem_data_path() -> None:
    path_store = _load_module("vartodd_evo_gf", "path_store.py")

    assert path_store.PathStore().root_dir == path_store.DATA_PATH


def test_live_path_store_data_path_is_independent_of_worker_cwd() -> None:
    path_store = _load_module("vartodd_evo_gf", "path_store.py")

    # Evaluation workers may run from a Hydra output directory, not the repo root.
    # A relative store root silently yields the all-empty live-path fallback there.
    assert Path(path_store.DATA_PATH).is_absolute()


def test_live_path_cap_uses_current_dao_not_full_todd_parent() -> None:
    path_store = _load_module("vartodd_evo_gf", "path_store.py")

    def dao_with_limit(limit: int):
        todd = SimpleNamespace(
            pool=SimpleNamespace(keep=4),
            buckets=SimpleNamespace(limit_bucket=limit),
        )
        return SimpleNamespace(mode=SimpleNamespace(todd=SimpleNamespace(points=[(567, todd)])))

    # A saved-path branch retains its loaded DAO history.  The name and live
    # store must describe the child tail's cap, not a full-TODD parent.
    assert (
        path_store.PathStore._path_todd_limit_buckets(
            {}, [dao_with_limit(-1), dao_with_limit(1024)]
        )
        == 1024
    )


@pytest.mark.parametrize(
    "name",
    [
        "f411_i490_12345678_lim512",
        "f411_i490_12345678_lim512_z384",
        "f411_i490_12345678_z384of512",
    ],
)
def test_live_path_store_reads_legacy_and_extended_short_names(name: str) -> None:
    path_store = _load_module("vartodd_evo_gf", "path_store.py")
    with TemporaryDirectory() as temporary_dir:
        backup = Path(temporary_dir) / name
        backup.mkdir()
        (backup / "meta.json").write_text(
            json.dumps({"paths": [{"matrix_keys": ["p0_s0", "p0_s1"]}]}),
            encoding="utf-8",
        )
        np.savez(
            backup / "matrices.npz",
            p0_s0=np.zeros((567, 48), dtype=np.uint8),
            p0_s1=np.zeros((411, 48), dtype=np.uint8),
        )

        record = path_store.PathStore(root_dir=temporary_dir).iter_path_records()[0]

    assert record["init_rank"] == 567
    assert record["init_rank_thr"] == 490
    assert record["kind"] == "partial"


def test_select_top_prefers_highest_initial_rank_per_final_rank() -> None:
    path_store = _load_module("vartodd_evo_gf", "path_store.py")
    records = [
        {
            "name": "rank397_low_init",
            "rank": 397,
            "init_rank": 430,
            "limit_buckets": 1000,
        },
        {
            "name": "rank397_high_init",
            "rank": 397,
            "init_rank": 567,
            "limit_buckets": 2000,
        },
        {
            "name": "rank399",
            "rank": 399,
            "init_rank": 500,
            "limit_buckets": 1000,
        },
    ]
    for record in records:
        record.update(
            improved_count=0,
            init_rank_thr=450,
            used_count=0,
        )

    selected = path_store.PathStore._select_top(records, top_k=4)

    assert [record["name"] for record in selected] == [
        "rank397_high_init",
        "rank399",
    ]


def test_nonimproving_children_are_evidence_only_in_live_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_store = _load_module("vartodd_evo_gf", "path_store.py")
    store = path_store.PathStore()

    def record(name: str, *, rank: int, limit: int, improved: bool | None) -> dict:
        return {
            "name": name,
            "rank": rank,
            "depth": 2,
            "limit_buckets": limit,
            "init_rank": 567,
            "init_rank_thr": 450,
            "kind": "partial",
            "restart_band": "near_final",
            "used_count": 0,
            "improved_count": 0,
            "best_child_rank": None,
            "last_child_rank": None,
            "last_used_at": None,
            "last_improved_at": None,
            "parent_path_name": "parent" if improved is not None else None,
            "parent_loaded_rank": 410 if improved is not None else None,
            "parent_loaded_start_rank": 430 if improved is not None else None,
            "parent_init_rank_thr": 420 if improved is not None else None,
            "parent_best_child_rank": None,
            "child_improved_loaded": improved,
            "is_stale_improved_child": False,
        }

    monkeypatch.setattr(
        store,
        "iter_path_records",
        lambda: [
            record("near_nonimprover", rank=410, limit=1000, improved=False),
            record("wide_nonimprover", rank=409, limit=-1, improved=False),
            record("near_live", rank=411, limit=1000, improved=True),
            record("wide_live", rank=412, limit=-1, improved=True),
        ],
    )

    summary = store.summarize(
        near_tail_top_k=10,
        wide_margin_top_k=10,
        near_tail_limit_cutoff=100_000,
    )
    selectable = summary.split("best_nonimproved_child_paths:", 1)[0]

    assert "near_nonimprover" not in selectable
    assert "wide_nonimprover" not in selectable
    assert "- near_live " in selectable
    assert "- wide_live " in selectable
    assert "best_nonimproved_child_paths:" in summary
    assert "- near_nonimprover " in summary
    assert "- wide_nonimprover " in summary


def test_live_summary_never_repeats_rank_across_selectable_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_store = _load_module("vartodd_evo_gf", "path_store.py")
    store = path_store.PathStore()

    def record(name: str, rank: int, limit: int) -> dict:
        return {
            "name": name,
            "rank": rank,
            "depth": 2,
            "limit_buckets": limit,
            "init_rank": 567,
            "init_rank_thr": 450,
            "kind": "partial",
            "restart_band": "near_final",
            "used_count": 0,
            "improved_count": 0,
            "best_child_rank": None,
            "last_child_rank": None,
            "last_used_at": None,
            "last_improved_at": None,
            "parent_path_name": None,
            "parent_loaded_rank": None,
            "parent_loaded_start_rank": None,
            "parent_init_rank_thr": None,
            "parent_best_child_rank": None,
            "child_improved_loaded": None,
            "is_stale_improved_child": False,
        }

    monkeypatch.setattr(
        store,
        "iter_path_records",
        lambda: [
            record("near_397_a", 397, 1000),
            record("near_397_b", 397, 2000),
            record("wide_397", 397, -1),
            record("wide_399_a", 399, -1),
            record("wide_399_b", 399, -1),
        ],
    )

    summary = store.summarize(
        near_tail_top_k=4,
        wide_margin_top_k=4,
        near_tail_limit_cutoff=100_000,
    )
    selectable = summary.split("rule=", 1)[0]

    assert selectable.count(" rank=397 ") == 1
    assert selectable.count(" rank=399 ") == 1
