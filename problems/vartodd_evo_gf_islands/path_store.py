"""Evidence-rich saved paths for the isolated VarTODD island experiment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, NamedTuple

from variant import resolve_variant


def _load_legacy_path_store():
    repository_root = Path(
        os.environ.get(
            "VARTODD_REPOSITORY_ROOT",
            Path(__file__).resolve().parents[2],
        )
    ).absolute()
    legacy_dir = repository_root / "problems" / "vartodd_evo_gf"
    if not (legacy_dir / "path_store.py").exists():
        legacy_dir = Path(__file__).resolve().parent.parent / "vartodd_evo_gf"
    module_path = legacy_dir / "path_store.py"
    module_name = "_vartodd_evo_gf_islands_legacy_path_store"
    existing = sys.modules.get(module_name)
    if existing is not None and hasattr(existing, "X0_LENGTH"):
        return existing
    sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy PathStore from {module_path}")
    module = importlib.util.module_from_spec(spec)

    current_variant = sys.modules.get(resolve_variant.__module__)
    previous_variant = sys.modules.get("variant")
    sys.modules[module_name] = module
    if current_variant is not None:
        sys.modules["variant"] = current_variant
    sys.path.insert(0, str(legacy_dir))
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path.remove(str(legacy_dir))
        if previous_variant is None:
            sys.modules.pop("variant", None)
        else:
            sys.modules["variant"] = previous_variant
    return module


_legacy = _load_legacy_path_store()
X0_LENGTH = _legacy.X0_LENGTH
DATA_PATH = str(resolve_variant(__file__).data_path)
EVIDENCE_CARD_NAME = "evidence_card.json"
EVIDENCE_CARD_VERSION = 2
REFINEMENT_ROUTES = frozenset({"mid_margin", "near_end"})
NEAR_END_RANK_WINDOW = 100
SELECTION_ORDER_RULE = (
    "The FIRST entry is the least-used path (rank breaks usage ties). "
    "Remaining entries prioritize best descendant rank, then path rank, "
    "then usage. Start with the first entry for fresh exploration; choose "
    "another when its rank or card evidence is more useful. In mid-margin "
    "and near-end, a family frontier remains eligible while its own "
    "path_uses are below path_limit, even if family_uses has reached "
    "family_limit."
)


class PathSelectionLimits(NamedTuple):
    mid_root: int = 6
    mid_family: int = 8
    near_family: int = 7
    near_path: int = 2
    mid_path: int = 8


class PathStore(_legacy.PathStore):
    """Legacy-compatible store with unified selectable path evidence cards."""

    root_dir: str = DATA_PATH

    def record_load(self, name: str, *, init_rank_thr: int | None = None) -> None:
        super().record_load(name, init_rank_thr=init_rank_thr)
        program_id = os.environ.get("GIGAEVO_PROGRAM_ID")
        route_id = os.environ.get("GIGAEVO_MUTATION_REGIME")
        if program_id and route_id in REFINEMENT_ROUTES:
            self.confirm_route_reservation(
                name,
                route_id=route_id,
                program_id=program_id,
            )

    @staticmethod
    def _ensure_route_usage_record(
        usage_record: dict[str, Any],
        route_id: str,
    ) -> dict[str, Any]:
        route_usage = usage_record.setdefault("route_usage", {})
        if not isinstance(route_usage, dict):
            route_usage = {}
            usage_record["route_usage"] = route_usage
        stats = route_usage.setdefault(route_id, {})
        if not isinstance(stats, dict):
            stats = {}
            route_usage[route_id] = stats
        defaults: dict[str, Any] = {
            "used_count": 0,
            "in_flight_count": 0,
            "improved_count": 0,
            "nonimproved_count": 0,
            "best_child_rank": None,
            "last_child_rank": None,
            "last_used_at": None,
            "last_improved_at": None,
            "processed_program_ids": [],
            "finalized_program_ids": [],
        }
        for key, value in defaults.items():
            stats.setdefault(key, copy.deepcopy(value))
        if not isinstance(stats["processed_program_ids"], list):
            stats["processed_program_ids"] = []
        if not isinstance(stats["finalized_program_ids"], list):
            stats["finalized_program_ids"] = []
        return stats

    def record_route_result(
        self,
        name: str,
        *,
        route_id: str,
        program_id: str,
        loaded_rank: int,
        child_rank: int,
    ) -> bool:
        if route_id not in REFINEMENT_ROUTES:
            raise ValueError(f"unsupported refinement route: {route_id!r}")
        with self._locked_usage_index() as index:
            usage_record = self._ensure_usage_record(index, name)
            stats = self._ensure_route_usage_record(usage_record, route_id)
            processed = stats["processed_program_ids"]
            if program_id in processed:
                return False

            loaded_rank = int(loaded_rank)
            child_rank = int(child_rank)
            now = _legacy._utc_now()
            finalized = stats.get("finalized_program_ids", [])
            if program_id not in finalized:
                stats["used_count"] = int(stats.get("used_count") or 0) + 1
            stats["last_child_rank"] = child_rank
            stats["last_used_at"] = now
            best_child = stats.get("best_child_rank")
            if best_child is None or child_rank < int(best_child):
                stats["best_child_rank"] = child_rank
            if child_rank < loaded_rank:
                stats["improved_count"] = (
                    int(stats.get("improved_count") or 0) + 1
                )
                stats["last_improved_at"] = now
            else:
                stats["nonimproved_count"] = (
                    int(stats.get("nonimproved_count") or 0) + 1
                )
            processed.append(program_id)
        return True

    def register_created_path(
        self,
        name: str,
        *,
        route_id: str,
        parent_name: str | None,
        program_id: str,
    ) -> None:
        if route_id not in {"ab_initio", *REFINEMENT_ROUTES}:
            raise ValueError(f"unsupported path creation route: {route_id!r}")
        with self._locked_usage_index() as index:
            record = self._ensure_usage_record(index, name)
            record["created_by_route"] = route_id
            record["created_by_program_id"] = str(program_id)
            families = record.setdefault("route_families", {})
            if not isinstance(families, dict):
                families = {}
                record["route_families"] = families
            if route_id == "ab_initio" or not parent_name:
                return

            parent = self._ensure_usage_record(index, parent_name)
            parent_families = parent.setdefault("route_families", {})
            if not isinstance(parent_families, dict):
                parent_families = {}
                parent["route_families"] = parent_families
            if route_id == "mid_margin":
                parent_route = parent.get("created_by_route")
                family_root = (
                    parent_families.get("mid_margin")
                    if parent_route == "mid_margin"
                    else None
                )
                families["mid_margin"] = str(family_root or name)
            else:
                family_root = str(
                    parent_families.setdefault("near_end", parent_name)
                )
                families["near_end"] = family_root

    @staticmethod
    def _usage_total(stats: dict[str, Any]) -> int:
        return int(stats.get("used_count") or 0) + int(
            stats.get("in_flight_count") or 0
        )

    def _family_usage_total(
        self,
        index: dict[str, Any],
        *,
        route_id: str,
        family_root: str,
    ) -> int:
        total = 0
        for usage_record in index.get("paths", {}).values():
            if not isinstance(usage_record, dict):
                continue
            families = usage_record.get("route_families", {})
            if not isinstance(families, dict):
                continue
            if families.get(route_id) != family_root:
                continue
            stats = self._ensure_route_usage_record(usage_record, route_id)
            total += self._usage_total(stats)
        return total

    def _family_route_outcomes(
        self,
        index: dict[str, Any],
        *,
        route_id: str,
        family_root: str,
    ) -> dict[str, int | None]:
        """Aggregate completed child yield across one route-owned family."""
        outcomes: dict[str, int | None] = {
            "improved_count": 0,
            "nonimproved_count": 0,
            "best_child_rank": None,
        }
        for usage_record in index.get("paths", {}).values():
            if not isinstance(usage_record, dict):
                continue
            families = usage_record.get("route_families", {})
            if not isinstance(families, dict):
                continue
            if families.get(route_id) != family_root:
                continue
            stats = self._ensure_route_usage_record(usage_record, route_id)
            outcomes["improved_count"] = int(
                outcomes["improved_count"] or 0
            ) + int(stats.get("improved_count") or 0)
            outcomes["nonimproved_count"] = int(
                outcomes["nonimproved_count"] or 0
            ) + int(stats.get("nonimproved_count") or 0)
            child_rank = stats.get("best_child_rank")
            if child_rank is not None and (
                outcomes["best_child_rank"] is None
                or int(child_rank) < int(outcomes["best_child_rank"])
            ):
                outcomes["best_child_rank"] = int(child_rank)
        return outcomes

    def _indexed_route_outcomes(
        self,
        index: dict[str, Any],
        *,
        name: str,
        route_id: str,
    ) -> dict[str, int | None]:
        usage_record = index.get("paths", {}).get(name, {})
        if not isinstance(usage_record, dict):
            usage_record = {}
        stats = self._ensure_route_usage_record(usage_record, route_id)
        return {
            "improved_count": int(stats.get("improved_count") or 0),
            "nonimproved_count": int(stats.get("nonimproved_count") or 0),
            "best_child_rank": (
                int(stats["best_child_rank"])
                if stats.get("best_child_rank") is not None
                else None
            ),
        }

    def reserve_route_path(
        self,
        name: str,
        *,
        route_id: str,
        program_id: str,
        limits: PathSelectionLimits,
    ) -> bool:
        if route_id not in REFINEMENT_ROUTES:
            raise ValueError(f"unsupported refinement route: {route_id!r}")
        # Older saved paths predate explicit route provenance. Rendering and
        # reservation both backfill it from the saved parent/program lineage.
        lineage_records = self._normalized_lineage_records()
        mid_frontier_names = (
            self._mid_margin_frontier_names(lineage_records)
            if route_id == "mid_margin"
            else set()
        )
        near_frontier_names = (
            self._near_end_frontier_names(lineage_records)
            if route_id == "near_end"
            else set()
        )
        with self._locked_usage_index() as index:
            reservations = index.setdefault("route_reservations", {})
            existing = reservations.get(program_id)
            if isinstance(existing, dict):
                matches = (
                    existing.get("path_name") == name
                    and existing.get("route_id") == route_id
                )
                if not matches:
                    return False
                if existing.get("state") in {"reserved", "confirmed"}:
                    return True
                if existing.get("state") == "finalized" and existing.get(
                    "consumed"
                ):
                    return True
                # A finalized reservation which never reached PathStore.load
                # released its slot. The same program may be retried normally.
                reservations.pop(program_id, None)

            usage_record = self._ensure_usage_record(index, name)
            created_by_route = usage_record.get("created_by_route")
            families = usage_record.setdefault("route_families", {})
            if not isinstance(families, dict):
                families = {}
                usage_record["route_families"] = families
            stats = self._ensure_route_usage_record(usage_record, route_id)

            family_root: str | None = None
            if route_id == "mid_margin":
                if created_by_route == "ab_initio":
                    if self._usage_total(stats) >= int(limits.mid_root):
                        return False
                elif created_by_route == "mid_margin":
                    family_root = str(families.get("mid_margin") or name)
                    families["mid_margin"] = family_root
                    if self._usage_total(stats) >= int(limits.mid_path):
                        return False
                    if self._family_usage_total(
                        index,
                        route_id=route_id,
                        family_root=family_root,
                    ) >= int(limits.mid_family):
                        if name not in mid_frontier_names:
                            return False
                else:
                    return False
            else:
                family_root = str(families.get("near_end") or name)
                families["near_end"] = family_root
                outcomes = self._family_route_outcomes(
                    index,
                    route_id=route_id,
                    family_root=family_root,
                )
                improved_count = int(outcomes["improved_count"] or 0)
                effective_family_limit = (
                    int(limits.near_family) + improved_count
                )
                effective_path_limit = int(limits.near_path) + improved_count
                if self._usage_total(stats) >= effective_path_limit:
                    return False
                if self._family_usage_total(
                    index,
                    route_id=route_id,
                    family_root=family_root,
                ) >= effective_family_limit:
                    # A family quota must not hide its current frontier path
                    # while that path still has individual reuse capacity.
                    if name not in near_frontier_names:
                        return False

            stats["in_flight_count"] = int(stats.get("in_flight_count") or 0) + 1
            reservations[program_id] = {
                "path_name": name,
                "route_id": route_id,
                "family_root": family_root,
                "state": "reserved",
            }
        return True

    def confirm_route_reservation(
        self,
        name: str,
        *,
        route_id: str,
        program_id: str,
    ) -> bool:
        with self._locked_usage_index() as index:
            reservation = index.setdefault("route_reservations", {}).get(program_id)
            if not isinstance(reservation, dict):
                return False
            if (
                reservation.get("path_name") != name
                or reservation.get("route_id") != route_id
            ):
                return False
            if reservation.get("state") == "finalized":
                return True
            reservation["state"] = "confirmed"
        return True

    def finalize_route_reservation(self, *, program_id: str) -> bool:
        with self._locked_usage_index() as index:
            reservation = index.setdefault("route_reservations", {}).get(program_id)
            if not isinstance(reservation, dict):
                return False
            if reservation.get("state") == "finalized":
                return True
            name = str(reservation["path_name"])
            route_id = str(reservation["route_id"])
            usage_record = self._ensure_usage_record(index, name)
            stats = self._ensure_route_usage_record(usage_record, route_id)
            stats["in_flight_count"] = max(
                0,
                int(stats.get("in_flight_count") or 0) - 1,
            )
            if reservation.get("state") == "confirmed":
                stats["used_count"] = int(stats.get("used_count") or 0) + 1
                finalized = stats.setdefault("finalized_program_ids", [])
                if program_id not in finalized:
                    finalized.append(program_id)
            reservation["state"] = "finalized"
            reservation["consumed"] = program_id in stats.get(
                "finalized_program_ids", []
            )
        return True

    def iter_path_records(self) -> list[dict[str, Any]]:
        records = super().iter_path_records()
        usage = self.read_usage_index().get("paths", {})
        for record in records:
            usage_record = usage.get(str(record.get("name")), {})
            route_usage = (
                usage_record.get("route_usage", {})
                if isinstance(usage_record, dict)
                else {}
            )
            record["route_usage"] = (
                copy.deepcopy(route_usage)
                if isinstance(route_usage, dict)
                else {}
            )
            record["created_by_route"] = (
                usage_record.get("created_by_route")
                if isinstance(usage_record, dict)
                else None
            )
            route_families = (
                usage_record.get("route_families", {})
                if isinstance(usage_record, dict)
                else {}
            )
            record["route_families"] = (
                copy.deepcopy(route_families)
                if isinstance(route_families, dict)
                else {}
            )
        return records

    def save(
        self,
        name: str,
        paths: Sequence[Any],
        *,
        store_daos: bool = True,
        parent_info: dict[str, Any] | None = None,
    ) -> Path:
        base = super().save(
            name,
            paths,
            store_daos=store_daos,
            parent_info=parent_info,
        )
        existing = self._read_evidence_card(name)
        path_stats = (
            paths[0].format_path_stats()
            if paths and hasattr(paths[0], "format_path_stats")
            else "path_summary:\n  unavailable"
        )
        self._write_evidence_card(
            name,
            {
                "version": EVIDENCE_CARD_VERSION,
                "path_stats": path_stats,
                "producer": existing.get("producer", {}),
            },
        )
        return base

    def has_selectable_paths(
        self,
        *,
        route_id: str,
        limits: PathSelectionLimits = PathSelectionLimits(),
    ) -> bool:
        _frontier_rank, _pinned, records = self._selection_inventory(
            route_id=route_id,
            limits=limits,
        )
        return bool(records)

    def selectable_records(
        self,
        *,
        route_id: str,
        top_k: int = 6,
        max_per_rank: int = 2,
        nonimproved_limit: int = 4,
        limits: PathSelectionLimits = PathSelectionLimits(),
    ) -> list[dict[str, Any]]:
        del nonimproved_limit
        _frontier_rank, _pinned, records = self._selection_inventory(
            route_id=route_id,
            limits=limits,
        )
        return self._bounded_records(
            records,
            top_k=top_k,
            max_per_rank=max_per_rank,
        )

    @staticmethod
    def _bounded_records(
        records: Sequence[dict[str, Any]],
        *,
        top_k: int,
        max_per_rank: int,
    ) -> list[dict[str, Any]]:
        rank_counts: dict[int, int] = {}
        selected: list[dict[str, Any]] = []
        limit = max(0, int(top_k))
        per_rank = max(0, int(max_per_rank))
        if limit == 0 or per_rank == 0:
            return []
        for record in records:
            rank = int(record["rank"])
            if rank_counts.get(rank, 0) >= per_rank:
                continue
            selected.append(record)
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    def render_selectable_path_cards(
        self,
        *,
        route_id: str,
        top_k: int = 8,
        max_per_rank: int = 2,
        max_chars: int = 24_000,
        limits: PathSelectionLimits = PathSelectionLimits(),
    ) -> str:
        if route_id == "mid_margin":
            root_top_k = min(4, max(0, int(top_k)))
            family_top_k = min(4, max(0, int(top_k) - root_top_k))
            frontier_rank, roots, families = self._mid_margin_inventory(
                limits=limits,
                root_top_k=root_top_k,
                family_top_k=family_top_k,
                max_per_rank=max_per_rank,
            )
            frontier_value = (
                str(frontier_rank) if frontier_rank is not None else "none"
            )
            header = (
                "## Selectable Shared Paths\n"
                f"route={route_id} frontier_rank={frontier_value}\n"
                f"{SELECTION_ORDER_RULE}\n\n"
            )
            root_cards = [self._render_mid_margin_record(record) for record in roots]
            family_cards = [
                self._render_mid_margin_record(record) for record in families
            ]
            body = (
                "### Ab-initio roots\n"
                + ("\n".join(root_cards) if root_cards else "- none")
                + "\n\n### Mid-margin family frontiers\n"
                + ("\n".join(family_cards) if family_cards else "- none")
            )
            return (header + body)[: max(0, int(max_chars))]

        frontier_rank, records = self._near_end_inventory(limits=limits)
        frontier_value = (
            str(frontier_rank) if frontier_rank is not None else "none"
        )
        rank_window = (
            f"{frontier_rank}..{frontier_rank + NEAR_END_RANK_WINDOW}"
            if frontier_rank is not None
            else "none"
        )
        header = (
            "## Selectable Shared Paths\n"
            f"route={route_id} frontier_rank={frontier_value} "
            f"rank_window={rank_window}\n"
            f"{SELECTION_ORDER_RULE}\n\n"
        )
        cards: list[str] = []
        used_chars = len(header)
        selected = self._bounded_records(
            records,
            top_k=top_k,
            max_per_rank=max_per_rank,
        )
        for record in selected:
            assert frontier_rank is not None
            card = self._render_near_end_card(
                record,
                frontier_rank=frontier_rank,
            )
            separator = 2 if cards else 0
            if used_chars + separator + len(card) > max(0, int(max_chars)):
                continue
            cards.append(card)
            used_chars += separator + len(card)
        return header + ("\n\n".join(cards) if cards else "- none")

    def update_evidence_card(
        self,
        name: str,
        producer: dict[str, Any],
    ) -> None:
        card = self._read_evidence_card(name)
        if self._evidence_card_needs_refresh(card):
            existing_producer = card.get("producer", {})
            card = {
                "version": EVIDENCE_CARD_VERSION,
                "path_stats": self._derive_path_stats(name),
                "producer": (
                    existing_producer
                    if isinstance(existing_producer, dict)
                    else {}
                ),
            }
        current_producer = card.get("producer")
        if not isinstance(current_producer, dict):
            current_producer = {}
        current_producer.update(producer)
        card["version"] = EVIDENCE_CARD_VERSION
        card["producer"] = current_producer
        self._write_evidence_card(name, card)

    @staticmethod
    def _origin(record: dict[str, Any]) -> str | None:
        if record.get("parent_path_name") is None:
            return "ab_initio"
        if record.get("child_improved_loaded") is True:
            return "improved_child"
        return None

    @staticmethod
    def _route_stats(
        record: dict[str, Any],
        route_id: str,
    ) -> dict[str, Any]:
        route_usage = record.get("route_usage")
        if not isinstance(route_usage, dict):
            return {}
        stats = route_usage.get(route_id)
        return stats if isinstance(stats, dict) else {}

    @staticmethod
    def _path_init_rank(record: dict[str, Any]) -> int:
        """Return the restart rank encoded as ``i<rank>`` in path names."""
        for key in ("init_rank_thr", "init_rank"):
            try:
                value = record.get(key)
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue
        return -1

    @classmethod
    def _child_branch_margin(cls, record: dict[str, Any]) -> int:
        """Return the margin used when this child reopened its parent path.

        ``parent_init_rank_thr`` is the exact threshold derived from the
        child's ``Evaluator(..., margin=...)`` call.  The remaining fields are
        compatibility fallbacks for older saved records.
        """
        try:
            parent_rank = int(record.get("parent_loaded_rank"))
        except (TypeError, ValueError):
            return -1
        for key in (
            "parent_init_rank_thr",
            "parent_loaded_start_rank",
            "init_rank_thr",
        ):
            try:
                start_rank = record.get(key)
                if start_rank is not None:
                    return int(start_rank) - parent_rank
            except (TypeError, ValueError):
                continue
        return -1

    @classmethod
    def _canonicalize_sibling_paths(
        cls,
        records: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep at most one direct child for each saved parent path.

        Siblings are alternative attempts to refine the same parent, not
        independent path hypotheses.  The canonical child is the one with the
        lowest final rank; tied ranks prefer the larger reopening margin.
        Canonicalization happens before route-specific failure filtering so a
        retired canonical child cannot be replaced by a weaker sibling.
        Ab-initio roots have no saved parent and remain independent.
        """
        roots: list[dict[str, Any]] = []
        best_child_by_parent: dict[str, dict[str, Any]] = {}
        for record in records:
            parent_name = record.get("parent_path_name")
            if not isinstance(parent_name, str) or not parent_name:
                roots.append(record)
                continue
            current = best_child_by_parent.get(parent_name)
            candidate_key = (
                int(record["rank"]),
                -cls._child_branch_margin(record),
                str(record["name"]),
            )
            if current is None:
                best_child_by_parent[parent_name] = record
                continue
            current_key = (
                int(current["rank"]),
                -cls._child_branch_margin(current),
                str(current["name"]),
            )
            if candidate_key < current_key:
                best_child_by_parent[parent_name] = record
        return roots + list(best_child_by_parent.values())

    @staticmethod
    def _route_total_uses(record: dict[str, Any], route_id: str) -> int:
        stats = PathStore._route_stats(record, route_id)
        return int(stats.get("used_count") or 0) + int(
            stats.get("in_flight_count") or 0
        )

    def _usage_snapshot(
        self,
        records: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return persisted usage with record data as a legacy fallback.

        Persisted family members remain in this snapshot even when they are
        stale or otherwise excluded from the selectable lineage.
        """
        index = self.read_usage_index()
        paths = index.setdefault("paths", {})
        for record in records:
            name = str(record["name"])
            usage_record = paths.setdefault(name, {})
            if not isinstance(usage_record, dict):
                usage_record = {}
                paths[name] = usage_record

            persisted_families = usage_record.setdefault("route_families", {})
            if not isinstance(persisted_families, dict):
                persisted_families = {}
                usage_record["route_families"] = persisted_families
            record_families = record.get("route_families", {})
            if isinstance(record_families, dict):
                for route_id, family_root in record_families.items():
                    persisted_families.setdefault(route_id, family_root)

            persisted_usage = usage_record.setdefault("route_usage", {})
            if not isinstance(persisted_usage, dict):
                persisted_usage = {}
                usage_record["route_usage"] = persisted_usage
            record_usage = record.get("route_usage", {})
            if isinstance(record_usage, dict):
                for route_id, stats in record_usage.items():
                    if isinstance(stats, dict):
                        persisted_usage.setdefault(route_id, copy.deepcopy(stats))
        return index

    def _indexed_route_total(
        self,
        index: dict[str, Any],
        *,
        name: str,
        route_id: str,
    ) -> int:
        usage_record = index.get("paths", {}).get(name, {})
        if not isinstance(usage_record, dict):
            return 0
        stats = self._ensure_route_usage_record(usage_record, route_id)
        return self._usage_total(stats)

    @staticmethod
    def _producer_program_prefix(name: str) -> str | None:
        import re

        match = re.search(r"_([0-9a-fA-F]{8})_(?:z|lim)", name)
        return match.group(1).lower() if match is not None else None

    @classmethod
    def _infer_created_route(
        cls,
        record: dict[str, Any],
        by_name: dict[str, dict[str, Any]],
    ) -> str | None:
        explicit = record.get("created_by_route")
        if explicit in {"ab_initio", *REFINEMENT_ROUTES}:
            return str(explicit)
        parent_name = record.get("parent_path_name")
        if not isinstance(parent_name, str) or not parent_name:
            return "ab_initio"
        parent = by_name.get(parent_name)
        prefix = cls._producer_program_prefix(str(record.get("name") or ""))
        if parent is None or prefix is None:
            return None
        matches: list[str] = []
        for route_id in REFINEMENT_ROUTES:
            processed = cls._route_stats(parent, route_id).get(
                "processed_program_ids", []
            )
            if any(str(program_id).lower().startswith(prefix) for program_id in processed):
                matches.append(route_id)
        return matches[0] if len(matches) == 1 else None

    def _normalized_lineage_records(self) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for source_record in self.iter_path_records():
            try:
                rank = int(source_record["rank"])
            except (KeyError, TypeError, ValueError):
                continue
            record = dict(source_record)
            record["rank"] = rank
            normalized.append(record)
        by_name = {str(record["name"]): record for record in normalized}
        eligible: list[dict[str, Any]] = []
        for record in normalized:
            parent_name = record.get("parent_path_name")
            if parent_name and record.get("child_improved_loaded") is not True:
                continue
            if record.get("is_stale_improved_child"):
                continue
            record["created_by_route"] = self._infer_created_route(record, by_name)
            if record["created_by_route"] is None:
                continue
            record["origin"] = (
                "ab_initio"
                if record["created_by_route"] == "ab_initio"
                else "improved_child"
            )
            eligible.append(record)
        self._persist_lineage_metadata(eligible)
        return eligible

    def _persist_lineage_metadata(
        self,
        records: Sequence[dict[str, Any]],
    ) -> None:
        if not records:
            return
        by_name = {str(record["name"]): record for record in records}
        for record in records:
            families = record.get("route_families")
            if not isinstance(families, dict):
                families = {}
                record["route_families"] = families
            route_id = record.get("created_by_route")
            if route_id == "mid_margin":
                families["mid_margin"] = self._family_root_for(
                    record,
                    route_id="mid_margin",
                    by_name=by_name,
                )
            elif route_id == "near_end":
                families["near_end"] = self._family_root_for(
                    record,
                    route_id="near_end",
                    by_name=by_name,
                )
        for record in records:
            if record.get("created_by_route") != "near_end":
                continue
            family_root = record.get("route_families", {}).get("near_end")
            root = by_name.get(str(family_root))
            if root is None:
                continue
            root_families = root.get("route_families")
            if not isinstance(root_families, dict):
                root_families = {}
                root["route_families"] = root_families
            root_families["near_end"] = str(family_root)

        with self._locked_usage_index() as index:
            for record in records:
                usage_record = self._ensure_usage_record(
                    index,
                    str(record["name"]),
                )
                usage_record["created_by_route"] = record["created_by_route"]
                families = record.get("route_families", {})
                if isinstance(families, dict) and families:
                    persisted = usage_record.setdefault("route_families", {})
                    if not isinstance(persisted, dict):
                        persisted = {}
                        usage_record["route_families"] = persisted
                    persisted.update(families)

    @classmethod
    def _family_root_for(
        cls,
        record: dict[str, Any],
        *,
        route_id: str,
        by_name: dict[str, dict[str, Any]],
    ) -> str:
        families = record.get("route_families")
        if isinstance(families, dict):
            explicit = families.get(route_id)
            if isinstance(explicit, str) and explicit:
                return explicit
        name = str(record["name"])
        if record.get("created_by_route") != route_id:
            return name
        parent_name = record.get("parent_path_name")
        parent = by_name.get(parent_name) if isinstance(parent_name, str) else None
        if parent is None:
            return name
        if parent.get("created_by_route") != route_id:
            return str(parent["name"]) if route_id == "near_end" else name
        return cls._family_root_for(parent, route_id=route_id, by_name=by_name)

    @staticmethod
    def _record_order(record: dict[str, Any]) -> tuple[Any, ...]:
        return (
            int(record["rank"]),
            -int(record.get("init_rank") or -1),
            str(record["name"]),
        )

    @staticmethod
    def _smoothed_yield(record: dict[str, Any]) -> float:
        improved = int(record.get("yield_improved_count") or 0)
        nonimproved = int(record.get("yield_nonimproved_count") or 0)
        return (improved + 1.0) / (improved + nonimproved + 2.0)

    @staticmethod
    def _usage_key(record: dict[str, Any]) -> int:
        """How often this entry has already been handed out on its route.

        Families carry a shared counter, so a family representative is ordered
        by the family's total rather than by its own path count -- otherwise a
        fresh representative would keep re-surfacing an already exhausted
        family.
        """
        family_uses = record.get("family_uses")
        if family_uses is not None:
            return int(family_uses)
        return int(record.get("path_uses") or 0)

    @classmethod
    def _yield_order(cls, record: dict[str, Any]) -> tuple[Any, ...]:
        """Least-used first, then best descendant, then rank.

        Usage leads so the inventory spreads attempts across the store instead
        of replaying the few paths that were selected early; among equally
        untried entries the one with the strongest demonstrated descendant is
        offered first, and rank breaks the remaining ties.
        """
        return (
            cls._usage_key(record),
            int(record.get("best_descendant_rank") or record["rank"]),
            int(record["rank"]),
            -cls._smoothed_yield(record),
            -int(record.get("yield_improved_count") or 0),
            -int(record.get("init_rank") or -1),
            str(record["name"]),
        )

    @classmethod
    def _exploration_then_quality(
        cls,
        records: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep one least-used exploration slot, then rank by path quality.

        A pure usage-first ordering can hide the best frontier behind a long
        queue of untouched but weaker paths.  Reserve the first position for
        the least-used candidate (breaking that tie by rank), then order every
        remaining position by best descendant/rank before usage.  This keeps a
        single source of fresh exploration without sacrificing frontier
        refinement in the rest of the prompt.
        """
        if not records:
            return []

        first = min(
            records,
            key=lambda record: (
                cls._usage_key(record),
                int(record["rank"]),
                int(record.get("best_descendant_rank") or record["rank"]),
                -cls._smoothed_yield(record),
                -int(record.get("yield_improved_count") or 0),
                -int(record.get("init_rank") or -1),
                str(record["name"]),
            ),
        )
        remaining = [record for record in records if record is not first]
        remaining.sort(
            key=lambda record: (
                int(record.get("best_descendant_rank") or record["rank"]),
                int(record["rank"]),
                cls._usage_key(record),
                -cls._smoothed_yield(record),
                -int(record.get("yield_improved_count") or 0),
                -int(record.get("init_rank") or -1),
                str(record["name"]),
            )
        )
        return [first, *remaining]

    @classmethod
    def _mid_root_order(cls, record: dict[str, Any]) -> tuple[Any, ...]:
        """Least-used first, then best descendant, then rank."""
        return (
            cls._usage_key(record),
            int(record.get("best_descendant_rank") or record["rank"]),
            *cls._record_order(record),
        )

    @classmethod
    def _best_descendant_ranks(
        cls,
        records: Sequence[dict[str, Any]],
    ) -> dict[str, int]:
        children: dict[str, list[str]] = defaultdict(list)
        ranks = {str(record["name"]): int(record["rank"]) for record in records}
        for record in records:
            parent = record.get("parent_path_name")
            if isinstance(parent, str) and parent in ranks:
                children[parent].append(str(record["name"]))

        memo: dict[str, int] = {}

        def visit(name: str, active: set[str]) -> int:
            if name in memo:
                return memo[name]
            if name in active:
                return ranks[name]
            descendant_ranks = [
                visit(child, active | {name}) for child in children.get(name, [])
            ]
            memo[name] = min([ranks[name], *descendant_ranks])
            return memo[name]

        return {name: visit(name, set()) for name in ranks}

    def _mid_margin_inventory(
        self,
        *,
        limits: PathSelectionLimits,
        root_top_k: int = 4,
        family_top_k: int = 4,
        max_per_rank: int = 2,
    ) -> tuple[int | None, list[dict[str, Any]], list[dict[str, Any]]]:
        records = self._normalized_lineage_records()
        if not records:
            return None, [], []
        usage_index = self._usage_snapshot(records)
        frontier_rank = min(int(record["rank"]) for record in records)
        by_name = {str(record["name"]): record for record in records}
        best_descendant = self._best_descendant_ranks(records)

        roots: list[dict[str, Any]] = []
        for source in records:
            if source.get("created_by_route") != "ab_initio":
                continue
            path_uses = self._indexed_route_total(
                usage_index,
                name=str(source["name"]),
                route_id="mid_margin",
            )
            if path_uses >= int(limits.mid_root):
                continue
            record = dict(source)
            record["path_uses"] = path_uses
            record["path_limit"] = int(limits.mid_root)
            record["best_descendant_rank"] = best_descendant[str(source["name"])]
            outcomes = self._indexed_route_outcomes(
                usage_index,
                name=str(source["name"]),
                route_id="mid_margin",
            )
            record["yield_improved_count"] = outcomes["improved_count"]
            record["yield_nonimproved_count"] = outcomes[
                "nonimproved_count"
            ]
            roots.append(record)
        roots = self._exploration_then_quality(roots)
        roots = self._bounded_records(
            roots,
            top_k=root_top_k,
            max_per_rank=max_per_rank,
        )

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record.get("created_by_route") != "mid_margin":
                continue
            family_root = self._family_root_for(
                record,
                route_id="mid_margin",
                by_name=by_name,
            )
            grouped[family_root].append(record)

        representatives: list[dict[str, Any]] = []
        for family_root, members in grouped.items():
            family_uses = self._family_usage_total(
                usage_index,
                route_id="mid_margin",
                family_root=family_root,
            )
            representative = dict(min(members, key=self._record_order))
            path_uses = self._indexed_route_total(
                usage_index,
                name=str(representative["name"]),
                route_id="mid_margin",
            )
            if path_uses >= int(limits.mid_path):
                continue
            representative["family_root"] = family_root
            representative["family_uses"] = family_uses
            representative["family_limit"] = int(limits.mid_family)
            representative["path_uses"] = path_uses
            representative["path_limit"] = int(limits.mid_path)
            outcomes = self._family_route_outcomes(
                usage_index,
                route_id="mid_margin",
                family_root=family_root,
            )
            representative["yield_improved_count"] = outcomes[
                "improved_count"
            ]
            representative["yield_nonimproved_count"] = outcomes[
                "nonimproved_count"
            ]
            representative["best_descendant_rank"] = best_descendant[
                str(representative["name"])
            ]
            representatives.append(representative)
        representatives = self._exploration_then_quality(representatives)
        representatives = self._bounded_records(
            representatives,
            top_k=family_top_k,
            max_per_rank=max_per_rank,
        )
        return frontier_rank, roots, representatives

    def _mid_margin_frontier_names(
        self,
        records: Sequence[dict[str, Any]],
    ) -> set[str]:
        """Return the selected frontier representative for each mid family."""
        by_name = {str(record["name"]): record for record in records}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if record.get("created_by_route") != "mid_margin":
                continue
            family_root = self._family_root_for(
                record,
                route_id="mid_margin",
                by_name=by_name,
            )
            grouped[family_root].append(record)
        return {
            str(min(members, key=self._record_order)["name"])
            for members in grouped.values()
            if members
        }

    def _near_end_inventory(
        self,
        *,
        limits: PathSelectionLimits,
    ) -> tuple[int | None, list[dict[str, Any]]]:
        records = self._normalized_lineage_records()
        if not records:
            return None, []
        usage_index = self._usage_snapshot(records)
        frontier_rank = min(int(record["rank"]) for record in records)
        by_name = {str(record["name"]): record for record in records}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            family_root = self._family_root_for(
                record,
                route_id="near_end",
                by_name=by_name,
            )
            grouped[family_root].append(record)

        representatives: list[dict[str, Any]] = []
        for family_root, members in grouped.items():
            representative = min(members, key=self._record_order)
            family_uses = self._family_usage_total(
                usage_index,
                route_id="near_end",
                family_root=family_root,
            )
            path_uses = self._indexed_route_total(
                usage_index,
                name=str(representative["name"]),
                route_id="near_end",
            )
            outcomes = self._family_route_outcomes(
                usage_index,
                route_id="near_end",
                family_root=family_root,
            )
            improved_count = int(outcomes["improved_count"] or 0)
            effective_family_limit = int(limits.near_family) + improved_count
            effective_path_limit = int(limits.near_path) + improved_count
            if path_uses >= effective_path_limit:
                continue
            # ``representative`` is the current frontier for this family.
            # Keep it visible until its own path limit is reached, even when
            # descendants have already consumed the shared family allowance.
            selected = dict(representative)
            selected["family_root"] = family_root
            selected["family_uses"] = family_uses
            selected["family_limit"] = effective_family_limit
            selected["path_uses"] = path_uses
            selected["path_limit"] = effective_path_limit
            selected["yield_improved_count"] = improved_count
            selected["yield_nonimproved_count"] = int(
                outcomes["nonimproved_count"] or 0
            )
            selected["family_improved_count"] = improved_count
            representatives.append(selected)
        max_near_rank = frontier_rank + NEAR_END_RANK_WINDOW
        representatives = [
            record
            for record in representatives
            if int(record["rank"]) <= max_near_rank
        ]
        representatives = self._exploration_then_quality(representatives)
        return frontier_rank, representatives

    def _near_end_frontier_names(
        self,
        records: Sequence[dict[str, Any]],
    ) -> set[str]:
        """Return the selected frontier representative for each near family."""
        by_name = {str(record["name"]): record for record in records}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            family_root = self._family_root_for(
                record,
                route_id="near_end",
                by_name=by_name,
            )
            grouped[family_root].append(record)
        return {
            str(min(members, key=self._record_order)["name"])
            for members in grouped.values()
            if members
        }

    def _selection_inventory(
        self,
        *,
        route_id: str,
        nonimproved_limit: int | None = None,
        limits: PathSelectionLimits = PathSelectionLimits(),
    ) -> tuple[
        int | None,
        dict[str, Any] | None,
        list[dict[str, Any]],
    ]:
        del nonimproved_limit
        if route_id == "mid_margin":
            frontier, roots, families = self._mid_margin_inventory(limits=limits)
            return frontier, None, [*roots, *families]
        if route_id == "near_end":
            frontier, records = self._near_end_inventory(limits=limits)
            return frontier, None, records
        raise ValueError(f"unsupported refinement route: {route_id!r}")

    def _render_near_end_card(
        self,
        record: dict[str, Any],
        *,
        frontier_rank: int,
    ) -> str:
        card = self._read_evidence_card(str(record["name"]))
        if self._evidence_card_needs_refresh(card):
            existing_producer = card.get("producer", {})
            card = {
                "version": EVIDENCE_CARD_VERSION,
                "path_stats": self._derive_path_stats(str(record["name"])),
                "producer": (
                    existing_producer
                    if isinstance(existing_producer, dict)
                    else {}
                ),
            }
            self._write_evidence_card(str(record["name"]), card)
        stats = str(card.get("path_stats") or "").strip()
        producer_value = card.get("producer")
        producer: dict[str, Any] = (
            producer_value if isinstance(producer_value, dict) else {}
        )
        summary = self._section(
            stats,
            "path_summary:",
            ("path_policy_groups:", "converged_policy_profiles:"),
        )
        policy_bands = self._section(
            stats,
            "path_policy_groups:",
            ("converged_policy_profiles:",),
        )
        policy_profiles = self._section(
            stats,
            "converged_policy_profiles:",
            (),
        )
        route_stats = self._route_stats(record, "near_end")
        return "\n".join(
            [
                f"### {record['name']}",
                "selection:",
                (
                    f"  path_name={record['name']} rank={record['rank']} "
                    f"frontier_rank={frontier_rank} "
                    f"origin={record.get('origin') or self._origin(record)} "
                    f"producer_start_rank={self._path_init_rank(record)} "
                    f"depth={record.get('depth')} "
                    f"kind={record.get('kind', 'unknown')} "
                    f"band={record.get('restart_band', 'unknown')}"
                ),
                "route_usage:",
                (
                    "  near u/i/f="
                    f"{int(route_stats.get('used_count') or 0)}/"
                    f"{int(route_stats.get('improved_count') or 0)}/"
                    f"{int(route_stats.get('nonimproved_count') or 0)} "
                    "best_child_rank="
                    f"{self._value(route_stats.get('best_child_rank'))} "
                    f"path_uses={int(record.get('path_uses') or 0)}/"
                    f"{int(record.get('path_limit') or 0)} "
                    f"family={record.get('family_root')} "
                    f"family_uses={int(record.get('family_uses') or 0)}/"
                    f"{int(record.get('family_limit') or 0)} "
                    "family_yield="
                    f"{int(record.get('yield_improved_count') or 0)}/"
                    f"{int(record.get('yield_nonimproved_count') or 0)}"
                ),
                "path_summary:",
                self._indent(summary),
                "policy_bands:",
                self._indent(policy_bands),
                "policy_profiles:",
                self._indent(policy_profiles),
                "producer_search:",
                (
                    f"  runtime={self._value(producer.get('runtime'))} "
                    f"total_evals={self._value(producer.get('total_evals'))} "
                    "best_seen_times="
                    f"{self._value(producer.get('best_seen_times'))} "
                    "last_improvement="
                    f"{self._value(producer.get('last_improvement'))} "
                    "timeout_salvaged="
                    f"{self._bool_value(producer.get('timeout_salvaged'))}"
                ),
            ]
        )

    def _render_mid_margin_record(self, record: dict[str, Any]) -> str:
        parts = [
            f"path_name={record['name']} rank={record['rank']} "
            f"producer_start_rank={self._path_init_rank(record)} "
            f"origin={record.get('origin') or self._origin(record)}"
        ]
        if record.get("family_root") is not None:
            parts.append(
                f"family={record['family_root']} "
                f"family_uses={int(record.get('family_uses') or 0)}/"
                f"{int(record.get('family_limit') or 0)} "
                f"path_uses={int(record.get('path_uses') or 0)}/"
                f"{int(record.get('path_limit') or 0)}"
            )
        else:
            parts.append(
                f"uses={int(record.get('path_uses') or 0)}/"
                f"{int(record.get('path_limit') or 0)}"
            )
        parts.append(
            f"best_descendant_rank={self._value(record.get('best_descendant_rank'))}"
        )
        parts.append(
            "child_yield="
            f"{int(record.get('yield_improved_count') or 0)}/"
            f"{int(record.get('yield_nonimproved_count') or 0)}"
        )
        return " ".join(parts)

    def _derive_path_stats(self, name: str) -> str:
        try:
            paths = self.load(name)
            if paths and hasattr(paths[0], "format_path_stats"):
                return str(paths[0].format_path_stats())
        except Exception:
            pass
        return "path_summary:\n  unavailable"

    @staticmethod
    def _evidence_card_needs_refresh(card: dict[str, Any]) -> bool:
        try:
            return int(card.get("version", 0)) < EVIDENCE_CARD_VERSION
        except (TypeError, ValueError):
            return True

    def _read_evidence_card(self, name: str) -> dict[str, Any]:
        path = self._resolve_dir(name) / EVIDENCE_CARD_NAME
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_evidence_card(
        self,
        name: str,
        card: dict[str, Any],
    ) -> None:
        directory = self._resolve_dir(name)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / EVIDENCE_CARD_NAME
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=directory,
                prefix=".evidence_card.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(card, temporary, indent=2, sort_keys=True)
                temporary_name = temporary.name
            os.replace(temporary_name, destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    @staticmethod
    def _section(
        text: str,
        start: str,
        following: tuple[str, ...],
    ) -> str:
        if start not in text:
            return "  unavailable"
        body = text.split(start, 1)[1]
        stops = [body.find(marker) for marker in following if marker in body]
        if stops:
            body = body[: min(stop for stop in stops if stop >= 0)]
        return body.strip() or "unavailable"

    @staticmethod
    def _indent(text: str) -> str:
        return "\n".join(f"  {line}" for line in text.splitlines())

    @staticmethod
    def _value(value: Any) -> str:
        return "unknown" if value is None else str(value)

    @staticmethod
    def _bool_value(value: Any) -> str:
        if value is None:
            return "unknown"
        return "1" if bool(value) else "0"
