from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


def _record_with_route_failures(
    name: str,
    rank: int,
    failures: int,
    *,
    route_id: str,
    init_rank: int,
    best_child_rank: int | None = None,
) -> dict:
    record = _record(name, rank, init_rank=init_rank)
    record["route_usage"] = {
        route_id: {
            "used_count": failures,
            "improved_count": 0,
            "nonimproved_count": failures,
            "best_child_rank": best_child_rank,
        }
    }
    return record


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


def _improved_child(
    name: str,
    rank: int,
    *,
    parent: str,
    parent_rank: int,
    margin: int,
    route_failures: int = 0,
) -> dict:
    record = _record(
        name,
        rank,
        init_rank=540,
        child_improved_loaded=True,
    )
    record.update(
        {
            "created_by_route": "mid_margin",
            "parent_path_name": parent,
            "parent_loaded_rank": parent_rank,
            "parent_init_rank_thr": parent_rank + margin,
            "route_families": {"mid_margin": name},
            "route_usage": {
                "mid_margin": {
                    "used_count": route_failures,
                    "improved_count": 0,
                    "nonimproved_count": route_failures,
                }
            },
        }
    )
    return record


def _family_record(
    name: str,
    rank: int,
    *,
    init_rank: int,
    created_by_route: str,
    parent: str | None = None,
    family_root: str | None = None,
    route_id: str,
    used: int = 0,
    in_flight: int = 0,
    improved: bool = True,
) -> dict:
    record = _record(
        name,
        rank,
        init_rank=init_rank,
        child_improved_loaded=(None if parent is None else improved),
    )
    record.update(
        {
            "created_by_route": created_by_route,
            "parent_path_name": parent,
            "parent_loaded_rank": None if parent is None else rank + 1,
            "route_families": (
                {route_id: family_root} if family_root is not None else {}
            ),
            "route_usage": {
                route_id: {
                    "used_count": used,
                    "in_flight_count": in_flight,
                    "improved_count": int(improved and used > 0),
                    "nonimproved_count": int(not improved and used > 0),
                    "best_child_rank": rank - 1 if improved and used else None,
                    "processed_program_ids": [],
                }
            },
        }
    )
    return record


def test_mid_inventory_separates_roots_from_shared_refinement_families(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    limits = module.PathSelectionLimits(
        mid_root=6,
        mid_family=8,
        near_family=7,
        near_path=2,
    )
    records = [
        _family_record(
            "root-exhausted",
            390,
            init_rank=567,
            created_by_route="ab_initio",
            route_id="mid_margin",
            used=6,
        ),
        _family_record(
            "root-a",
            391,
            init_rank=567,
            created_by_route="ab_initio",
            route_id="mid_margin",
            used=2,
        ),
        _family_record(
            "root-b",
            392,
            init_rank=567,
            created_by_route="ab_initio",
            route_id="mid_margin",
        ),
        _family_record(
            "mid-a",
            387,
            init_rank=567,
            created_by_route="mid_margin",
            parent="root-a",
            family_root="mid-a",
            route_id="mid_margin",
            used=3,
        ),
        _family_record(
            "mid-a-best",
            384,
            init_rank=567,
            created_by_route="mid_margin",
            parent="mid-a",
            family_root="mid-a",
            route_id="mid_margin",
            used=4,
        ),
        _family_record(
            "near-child",
            383,
            init_rank=567,
            created_by_route="near_end",
            parent="mid-a-best",
            family_root="near-child",
            route_id="near_end",
        ),
        _family_record(
            "nonimprover",
            382,
            init_rank=567,
            created_by_route="mid_margin",
            parent="root-b",
            family_root="nonimprover",
            route_id="mid_margin",
            improved=False,
        ),
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)

    frontier, roots, families = store._mid_margin_inventory(
        limits=limits,
        root_top_k=4,
        family_top_k=4,
        max_per_rank=2,
    )

    assert frontier == 383
    assert [record["name"] for record in roots] == ["root-a", "root-b"]
    assert roots[0]["best_descendant_rank"] == 383
    assert [record["name"] for record in families] == ["mid-a-best"]
    assert families[0]["family_root"] == "mid-a"
    assert families[0]["family_uses"] == 7


def test_near_inventory_uses_best_live_representative_without_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    limits = module.PathSelectionLimits(6, 8, 7, 2)
    records = [
        _family_record(
            "family-a-root",
            390,
            init_rank=567,
            created_by_route="ab_initio",
            family_root="family-a-root",
            route_id="near_end",
            used=1,
        ),
        _family_record(
            "family-a-best",
            386,
            init_rank=567,
            created_by_route="near_end",
            parent="family-a-root",
            family_root="family-a-root",
            route_id="near_end",
            used=1,
        ),
        _family_record(
            "family-b-root",
            389,
            init_rank=567,
            created_by_route="ab_initio",
            family_root="family-b-root",
            route_id="near_end",
        ),
        _family_record(
            "family-b-exact-exhausted",
            385,
            init_rank=567,
            created_by_route="near_end",
            parent="family-b-root",
            family_root="family-b-root",
            route_id="near_end",
            used=2,
        ),
        _family_record(
            "family-c-exhausted",
            384,
            init_rank=567,
            created_by_route="mid_margin",
            family_root="family-c-exhausted",
            route_id="near_end",
            used=7,
        ),
        _family_record(
            "fresh-candidate",
            391,
            init_rank=567,
            created_by_route="mid_margin",
            route_id="mid_margin",
        ),
    ]
    records[3]["route_usage"]["near_end"].update(
        improved_count=0,
        nonimproved_count=2,
        best_child_rank=385,
    )
    monkeypatch.setattr(store, "iter_path_records", lambda: records)

    frontier, eligible = store._near_end_inventory(limits=limits)

    assert frontier == 384
    assert [record["name"] for record in eligible] == [
        "family-a-best",
        "fresh-candidate",
    ]
    assert eligible[0]["path_uses"] == 1
    assert eligible[0]["family_uses"] == 2
    assert eligible[1]["family_root"] == "fresh-candidate"


def test_productive_near_family_earns_bounded_reuse_credits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    limits = module.PathSelectionLimits(6, 8, 5, 2)
    root = _family_record(
        "productive-root",
        390,
        init_rank=567,
        created_by_route="mid_margin",
        family_root="productive-root",
        route_id="near_end",
        used=3,
    )
    frontier = _family_record(
        "productive-frontier",
        385,
        init_rank=567,
        created_by_route="near_end",
        parent="productive-root",
        family_root="productive-root",
        route_id="near_end",
        used=2,
    )
    frontier["route_usage"]["near_end"].update(
        improved_count=0,
        nonimproved_count=2,
        best_child_rank=385,
    )
    monkeypatch.setattr(store, "iter_path_records", lambda: [root, frontier])

    _frontier_rank, eligible = store._near_end_inventory(limits=limits)

    assert [record["name"] for record in eligible] == ["productive-frontier"]
    assert eligible[0]["path_limit"] == 3
    assert eligible[0]["family_limit"] == 6
    assert eligible[0]["family_improved_count"] == 1


def test_near_inventory_prefers_observed_yield_for_equal_rank_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    unproductive = _family_record(
        "a-unproductive",
        385,
        init_rank=567,
        created_by_route="mid_margin",
        family_root="a-unproductive",
        route_id="near_end",
        used=1,
        improved=False,
    )
    productive = _family_record(
        "z-productive",
        385,
        init_rank=567,
        created_by_route="mid_margin",
        family_root="z-productive",
        route_id="near_end",
        used=1,
    )
    monkeypatch.setattr(
        store,
        "iter_path_records",
        lambda: [unproductive, productive],
    )

    selected = store.selectable_records(
        route_id="near_end",
        top_k=1,
        max_per_rank=2,
    )

    assert [record["name"] for record in selected] == ["z-productive"]


def test_near_inventory_counts_stale_descendant_usage_like_reservation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    limits = module.PathSelectionLimits(6, 8, 2, 2)
    family_root = "family-root"
    records = [
        _family_record(
            family_root,
            390,
            init_rank=567,
            created_by_route="mid_margin",
            family_root=family_root,
            route_id="near_end",
        ),
        _family_record(
            "stale-child",
            389,
            init_rank=567,
            created_by_route="near_end",
            parent=family_root,
            family_root=family_root,
            route_id="near_end",
            used=2,
        ),
        _family_record(
            "best-child",
            385,
            init_rank=567,
            created_by_route="near_end",
            parent=family_root,
            family_root=family_root,
            route_id="near_end",
        ),
        _family_record(
            "other-family",
            384,
            init_rank=567,
            created_by_route="mid_margin",
            family_root="other-family",
            route_id="near_end",
        ),
    ]
    records[1]["is_stale_improved_child"] = True
    records[1]["route_usage"]["near_end"].update(
        improved_count=0,
        nonimproved_count=2,
        best_child_rank=389,
    )
    monkeypatch.setattr(store, "iter_path_records", lambda: records)
    with store._locked_usage_index() as index:
        stale = store._ensure_usage_record(index, "stale-child")
        stale["created_by_route"] = "near_end"
        stale["route_families"] = {"near_end": family_root}
        stale["route_usage"] = records[1]["route_usage"]

    _frontier, eligible = store._near_end_inventory(limits=limits)

    assert [record["name"] for record in eligible] == ["other-family"]
    assert not store.reserve_route_path(
        "best-child",
        route_id="near_end",
        program_id="rejected-after-stale-uses",
        limits=limits,
    )


def test_near_end_inventory_uses_inclusive_frontier_plus_100_window(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    root = tmp_path / "cards"
    store = module.PathStore(root_dir=str(root))
    records = [
        _family_record(
            "frontier",
            380,
            init_rank=567,
            created_by_route="ab_initio",
            route_id="near_end",
        ),
        _family_record(
            "boundary",
            480,
            init_rank=567,
            created_by_route="ab_initio",
            route_id="near_end",
        ),
        _family_record(
            "outside",
            481,
            init_rank=567,
            created_by_route="ab_initio",
            route_id="near_end",
        ),
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)
    for record in records:
        _write_card(root, record["name"])

    frontier, eligible = store._near_end_inventory(
        limits=module.PathSelectionLimits()
    )
    _mid_frontier, mid_roots, _mid_families = store._mid_margin_inventory(
        limits=module.PathSelectionLimits(),
        root_top_k=4,
        max_per_rank=2,
    )
    cards = store.render_selectable_path_cards(
        route_id="near_end",
        top_k=8,
        max_chars=24_000,
        limits=module.PathSelectionLimits(),
    )

    assert frontier == 380
    assert [record["name"] for record in eligible] == ["frontier", "boundary"]
    assert "outside" in {record["name"] for record in mid_roots}
    assert "rank_window=380..480" in cards
    assert "### outside" not in cards


def test_historical_near_children_inherit_the_selected_parent_family(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    root = _family_record(
        "mid-root",
        390,
        init_rank=567,
        created_by_route="mid_margin",
        route_id="mid_margin",
    )
    first = _family_record(
        "near-first",
        387,
        init_rank=567,
        created_by_route="near_end",
        parent="mid-root",
        route_id="near_end",
    )
    grandchild = _family_record(
        "near-grandchild",
        385,
        init_rank=567,
        created_by_route="near_end",
        parent="near-first",
        route_id="near_end",
    )
    by_name = {record["name"]: record for record in (root, first, grandchild)}

    assert store._family_root_for(
        first,
        route_id="near_end",
        by_name=by_name,
    ) == "mid-root"
    assert store._family_root_for(
        grandchild,
        route_id="near_end",
        by_name=by_name,
    ) == "mid-root"


def test_historical_provenance_is_persisted_before_reservation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    root = _record("root", 390, init_rank=567)
    root["route_usage"] = {
        "mid_margin": {
            "processed_program_ids": ["deadbeef-0000-0000-0000-000000000000"],
        }
    }
    child = _record(
        "f385_i500_deadbeef_z100of200",
        385,
        init_rank=500,
        child_improved_loaded=True,
    )
    child["parent_path_name"] = "root"
    monkeypatch.setattr(store, "iter_path_records", lambda: [root, child])

    _frontier, roots, families = store._mid_margin_inventory(
        limits=module.PathSelectionLimits(),
    )

    assert [record["name"] for record in roots] == ["root"]
    assert [record["name"] for record in families] == [child["name"]]
    usage = store.read_usage_index()["paths"]
    assert usage["root"]["created_by_route"] == "ab_initio"
    assert usage[child["name"]]["created_by_route"] == "mid_margin"
    assert usage[child["name"]]["route_families"]["mid_margin"] == child["name"]
    assert store.reserve_route_path(
        child["name"],
        route_id="mid_margin",
        program_id="new-program",
        limits=module.PathSelectionLimits(),
    )


def test_historical_near_family_persists_membership_on_its_selected_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    root = _family_record(
        "mid-root",
        390,
        init_rank=567,
        created_by_route="mid_margin",
        route_id="near_end",
        used=1,
    )
    child = _family_record(
        "near-child",
        387,
        init_rank=567,
        created_by_route="near_end",
        parent="mid-root",
        route_id="near_end",
        used=1,
    )
    root["route_families"] = {}
    child["route_families"] = {}
    store.register_created_path(
        "mid-root",
        route_id="mid_margin",
        parent_name=None,
        program_id="root-producer",
    )
    assert store.record_route_result(
        "mid-root",
        route_id="near_end",
        program_id="near-root-use",
        loaded_rank=390,
        child_rank=390,
    )
    store.register_created_path(
        "near-child",
        route_id="near_end",
        parent_name="mid-root",
        program_id="child-producer",
    )
    assert store.record_route_result(
        "near-child",
        route_id="near_end",
        program_id="near-child-use",
        loaded_rank=387,
        child_rank=387,
    )
    # Simulate historical records which lack the newer persisted family fields.
    with store._locked_usage_index() as index:
        index["paths"]["mid-root"]["route_families"] = {}
        index["paths"]["near-child"]["route_families"] = {}
    monkeypatch.setattr(store, "iter_path_records", lambda: [root, child])

    store._near_end_inventory(limits=module.PathSelectionLimits(6, 8, 2, 2))

    usage = store.read_usage_index()["paths"]
    assert usage["mid-root"]["route_families"]["near_end"] == "mid-root"
    assert usage["near-child"]["route_families"]["near_end"] == "mid-root"
    assert not store.reserve_route_path(
        "near-child",
        route_id="near_end",
        program_id="family-use-3",
        limits=module.PathSelectionLimits(6, 8, 2, 2),
    )


def test_mid_cards_name_restart_evidence_without_calling_it_initial_rank(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    record = _family_record(
        "root",
        390,
        init_rank=567,
        created_by_route="ab_initio",
        route_id="mid_margin",
    )
    monkeypatch.setattr(store, "iter_path_records", lambda: [record])

    cards = store.render_selectable_path_cards(
        route_id="mid_margin",
        top_k=8,
        max_chars=24_000,
        limits=module.PathSelectionLimits(6, 8, 7, 2),
    )

    assert "Ab-initio roots" in cards
    assert "Mid-margin family frontiers" in cards
    assert "producer_start_rank=" in cards
    assert " init_rank=" not in cards


def test_created_path_registration_preserves_route_family_across_promotions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))

    store.register_created_path(
        "mid-first",
        route_id="mid_margin",
        parent_name="root",
        program_id="program-1",
    )
    store.register_created_path(
        "mid-second",
        route_id="mid_margin",
        parent_name="mid-first",
        program_id="program-2",
    )
    store.register_created_path(
        "near-first",
        route_id="near_end",
        parent_name="mid-second",
        program_id="program-3",
    )

    paths = store.read_usage_index()["paths"]
    assert paths["mid-first"]["created_by_route"] == "mid_margin"
    assert paths["mid-first"]["route_families"]["mid_margin"] == "mid-first"
    assert paths["mid-second"]["route_families"]["mid_margin"] == "mid-first"
    assert paths["near-first"]["created_by_route"] == "near_end"
    assert paths["near-first"]["route_families"]["near_end"] == "mid-second"
    assert paths["mid-second"]["route_families"]["near_end"] == "mid-second"


def _consume_reservation(
    store,
    *,
    name: str,
    route_id: str,
    program_id: str,
    limits,
) -> bool:
    if not store.reserve_route_path(
        name,
        route_id=route_id,
        program_id=program_id,
        limits=limits,
    ):
        return False
    assert store.confirm_route_reservation(
        name,
        route_id=route_id,
        program_id=program_id,
    )
    assert store.finalize_route_reservation(program_id=program_id)
    return True


def test_mid_root_and_family_reservations_obey_independent_limits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    limits = module.PathSelectionLimits(6, 8, 7, 2)
    store.register_created_path(
        "root",
        route_id="ab_initio",
        parent_name=None,
        program_id="root-program",
    )
    store.register_created_path(
        "mid-first",
        route_id="mid_margin",
        parent_name="root",
        program_id="mid-first-program",
    )
    store.register_created_path(
        "mid-promoted",
        route_id="mid_margin",
        parent_name="mid-first",
        program_id="mid-promoted-program",
    )

    assert all(
        _consume_reservation(
            store,
            name="root",
            route_id="mid_margin",
            program_id=f"root-use-{index}",
            limits=limits,
        )
        for index in range(6)
    )
    assert not store.reserve_route_path(
        "root",
        route_id="mid_margin",
        program_id="root-use-7",
        limits=limits,
    )

    family_paths = ["mid-first"] * 4 + ["mid-promoted"] * 4
    assert all(
        _consume_reservation(
            store,
            name=name,
            route_id="mid_margin",
            program_id=f"family-use-{index}",
            limits=limits,
        )
        for index, name in enumerate(family_paths)
    )
    assert not store.reserve_route_path(
        "mid-promoted",
        route_id="mid_margin",
        program_id="family-use-9",
        limits=limits,
    )


def test_near_reservations_obey_path_two_and_family_seven(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    limits = module.PathSelectionLimits(6, 8, 7, 2)
    store.register_created_path(
        "candidate",
        route_id="mid_margin",
        parent_name="root",
        program_id="candidate-program",
    )

    for index in range(2):
        assert _consume_reservation(
            store,
            name="candidate",
            route_id="near_end",
            program_id=f"near-root-{index}",
            limits=limits,
        )
    assert not store.reserve_route_path(
        "candidate",
        route_id="near_end",
        program_id="near-root-3",
        limits=limits,
    )

    parent = "candidate"
    uses = 2
    for generation, count in enumerate((2, 2, 1), start=1):
        child = f"near-child-{generation}"
        store.register_created_path(
            child,
            route_id="near_end",
            parent_name=parent,
            program_id=f"producer-{generation}",
        )
        for local_use in range(count):
            assert _consume_reservation(
                store,
                name=child,
                route_id="near_end",
                program_id=f"near-{generation}-{local_use}",
                limits=limits,
            )
            uses += 1
        parent = child
    assert uses == 7
    assert not store.reserve_route_path(
        parent,
        route_id="near_end",
        program_id="near-family-8",
        limits=limits,
    )


def test_unconfirmed_reservation_is_released_without_consuming_use(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    limits = module.PathSelectionLimits(1, 8, 7, 2)
    store.register_created_path(
        "root",
        route_id="ab_initio",
        parent_name=None,
        program_id="root-program",
    )

    assert store.reserve_route_path(
        "root",
        route_id="mid_margin",
        program_id="failed-before-load",
        limits=limits,
    )
    assert store.finalize_route_reservation(program_id="failed-before-load")
    assert _consume_reservation(
        store,
        name="root",
        route_id="mid_margin",
        program_id="successful-load",
        limits=limits,
    )


def test_unconfirmed_program_can_retry_with_a_real_reservation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    limits = module.PathSelectionLimits(1, 8, 7, 2)
    store.register_created_path(
        "root",
        route_id="ab_initio",
        parent_name=None,
        program_id="root-program",
    )

    assert store.reserve_route_path(
        "root",
        route_id="mid_margin",
        program_id="retry-program",
        limits=limits,
    )
    assert store.finalize_route_reservation(program_id="retry-program")
    assert store.reserve_route_path(
        "root",
        route_id="mid_margin",
        program_id="retry-program",
        limits=limits,
    )
    stats = store.read_usage_index()["paths"]["root"]["route_usage"][
        "mid_margin"
    ]
    assert stats["in_flight_count"] == 1
    assert store.confirm_route_reservation(
        "root",
        route_id="mid_margin",
        program_id="retry-program",
    )
    assert store.finalize_route_reservation(program_id="retry-program")
    assert not store.reserve_route_path(
        "root",
        route_id="mid_margin",
        program_id="another-program",
        limits=limits,
    )

    stats = store.read_usage_index()["paths"]["root"]["route_usage"][
        "mid_margin"
    ]
    assert stats["used_count"] == 1
    assert stats["in_flight_count"] == 0


def test_concurrent_reservations_cannot_oversubscribe_root_limit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    limits = module.PathSelectionLimits(6, 8, 7, 2)
    store.register_created_path(
        "root",
        route_id="ab_initio",
        parent_name=None,
        program_id="root-program",
    )

    def reserve(index: int) -> bool:
        return store.reserve_route_path(
            "root",
            route_id="mid_margin",
            program_id=f"concurrent-{index}",
            limits=limits,
        )

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(reserve, range(12)))

    assert sum(results) == 6


def test_successful_record_load_confirms_program_reservation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    limits = module.PathSelectionLimits(6, 8, 7, 2)
    store.register_created_path(
        "root",
        route_id="ab_initio",
        parent_name=None,
        program_id="root-program",
    )
    assert store.reserve_route_path(
        "root",
        route_id="mid_margin",
        program_id="child-program",
        limits=limits,
    )
    monkeypatch.setenv("GIGAEVO_PROGRAM_ID", "child-program")
    monkeypatch.setenv("GIGAEVO_MUTATION_REGIME", "mid_margin")

    store.record_load("root", init_rank_thr=420)
    assert store.finalize_route_reservation(program_id="child-program")

    stats = store.read_usage_index()["paths"]["root"]["route_usage"][
        "mid_margin"
    ]
    assert stats["used_count"] == 1
    assert stats["in_flight_count"] == 0


def test_mid_siblings_start_independent_refinement_families(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    records = [
        _record("frontier-root", 380, init_rank=567),
        _improved_child(
            "same-rank-small-margin",
            390,
            parent="parent-a",
            parent_rank=395,
            margin=20,
        ),
        _improved_child(
            "same-rank-large-margin",
            390,
            parent="parent-a",
            parent_rank=395,
            margin=60,
        ),
        _improved_child(
            "better-rank-small-margin",
            388,
            parent="parent-b",
            parent_rank=395,
            margin=10,
        ),
        _improved_child(
            "worse-rank-large-margin",
            389,
            parent="parent-b",
            parent_rank=395,
            margin=80,
        ),
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)

    _frontier, _pinned, eligible = store._selection_inventory(
        route_id="mid_margin"
    )
    names = {record["name"] for record in eligible}

    assert "same-rank-large-margin" in names
    assert "same-rank-small-margin" in names
    assert "better-rank-small-margin" in names
    assert "worse-rank-large-margin" in names


def test_mid_family_limit_is_shared_by_promoted_child(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    records = [
        _family_record(
            "family-root",
            390,
            init_rank=567,
            created_by_route="mid_margin",
            parent="ab-root",
            family_root="family-root",
            route_id="mid_margin",
            used=4,
        ),
        _family_record(
            "promoted-child",
            388,
            init_rank=567,
            created_by_route="mid_margin",
            parent="family-root",
            family_root="family-root",
            route_id="mid_margin",
            used=4,
        ),
        _family_record(
            "independent-child",
            391,
            init_rank=567,
            created_by_route="mid_margin",
            parent="other-root",
            family_root="independent-child",
            route_id="mid_margin",
        ),
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)

    _frontier, _pinned, eligible = store._selection_inventory(
        route_id="mid_margin"
    )
    names = {record["name"] for record in eligible}

    assert "family-root" not in names
    assert "promoted-child" not in names
    assert "independent-child" in names


def test_selectable_cards_do_not_pin_frontier_and_obey_top_k(
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
        {
            **_record(
                "nonimproved-child-name",
                394,
                init_rank=540,
                child_improved_loaded=False,
            ),
            "parent_path_name": "loaded-parent",
            "parent_loaded_rank": 394,
            "parent_init_rank_thr": 420,
        },
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)
    for record in records:
        _write_card(root, record["name"])

    cards = store.render_selectable_path_cards(
        route_id="near_end",
        top_k=1,
        max_chars=24_000,
    )

    assert store.has_selectable_paths(route_id="near_end") is True
    assert "## Selectable Shared Paths" in cards
    assert "## Pinned Best Long-Descent Path" not in cards
    assert "### higher-init" in cards
    assert "### lower-init" not in cards
    assert "### rank-397" not in cards
    assert "nonimproved-child-name" not in cards
    assert "policy_bands:" in cards
    assert "policy_profiles:" in cards
    assert "producer_search:" in cards


def test_mid_roots_share_six_use_limit_at_every_rank(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    records = [
        _record_with_route_failures(
            "frontier-live", 385, 5, route_id="mid_margin", init_rank=567
        ),
        _record_with_route_failures(
            "frontier-retired", 385, 6, route_id="mid_margin", init_rank=560
        ),
        _record_with_route_failures(
            "worse-live", 397, 5, route_id="mid_margin", init_rank=550
        ),
        _record_with_route_failures(
            "worse-retired", 399, 6, route_id="mid_margin", init_rank=540
        ),
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)

    frontier, pinned, eligible = store._selection_inventory(
        route_id="mid_margin"
    )
    assert frontier == 385
    assert pinned is None
    assert {record["name"] for record in eligible} == {
        "frontier-live",
        "worse-live",
    }

    monkeypatch.setattr(store, "iter_path_records", lambda: records[1::2])
    assert store.has_selectable_paths(route_id="mid_margin") is False


def test_near_end_accepts_paths_more_than_four_ranks_above_frontier(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    records = [
        _record_with_route_failures(
            "frontier", 385, 0, route_id="near_end", init_rank=567
        ),
        _record_with_route_failures(
            "distant", 397, 0, route_id="near_end", init_rank=540
        ),
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)

    frontier, _pinned, eligible = store._selection_inventory(
        route_id="near_end"
    )

    assert frontier == 385
    assert {record["name"] for record in eligible} == {
        "frontier",
        "distant",
    }


def test_missing_card_is_derived_and_complete_cards_obey_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    root = tmp_path / "cards"
    store = module.PathStore(root_dir=str(root))
    records = [
        _record("first", 393, init_rank=550),
        _record("second", 395, init_rank=540),
        _record("third", 397, init_rank=530),
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

    _, inventory = store._near_end_inventory(
        limits=module.PathSelectionLimits(6, 8, 7, 2)
    )
    one_card = store._render_near_end_card(inventory[0], frontier_rank=393)
    header = (
        "## Selectable Shared Paths\n"
        "route=near_end frontier_rank=393 rank_window=393..493\n\n"
    )
    rendered = store.render_selectable_path_cards(
        route_id="near_end",
        top_k=8,
        max_chars=len(header) + len(one_card),
    )

    assert "### first" in rendered
    assert "### second" not in rendered
    assert "### third" not in rendered
    assert "selection:" in rendered
    assert not list(root.rglob("*.tmp"))


def test_legacy_evidence_card_refreshes_path_statistics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    root = tmp_path / "cards"
    store = module.PathStore(root_dir=str(root))
    record = _record("legacy", 395, init_rank=433)
    _write_card(root, "legacy", runtime=651.0)
    legacy_path = root / "legacy" / "evidence_card.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    legacy["path_stats"] = (
        "path_summary:\n  depth=1 init_rank=433 final_rank=395\n"
        "path_policy_groups:\n  g1 433->395 z:0\n"
        "converged_policy_profiles:\n  P1"
    )
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(
        store,
        "_derive_path_stats",
        lambda name: (
            "path_summary:\n  depth=1 init_rank=433 final_rank=395\n"
            "path_policy_groups:\n  g1 433->395 z:64\n"
            "converged_policy_profiles:\n  P1"
        ),
    )

    rendered = store._render_near_end_card(record, frontier_rank=395)

    refreshed = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert "z:64" in rendered
    assert "z:0" not in rendered
    assert refreshed["version"] == module.EVIDENCE_CARD_VERSION
    assert refreshed["producer"]["runtime"] == 651.0


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
