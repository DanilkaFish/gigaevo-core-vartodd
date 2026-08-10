"""Evidence-rich saved paths for the isolated VarTODD island experiment."""

from __future__ import annotations

from collections.abc import Sequence
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

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
REFINEMENT_ROUTES = frozenset({"mid_margin", "near_end"})


class PathStore(_legacy.PathStore):
    """Legacy-compatible store with unified selectable path evidence cards."""

    root_dir: str = DATA_PATH

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
            "improved_count": 0,
            "nonimproved_count": 0,
            "best_child_rank": None,
            "last_child_rank": None,
            "last_used_at": None,
            "last_improved_at": None,
            "processed_program_ids": [],
        }
        for key, value in defaults.items():
            stats.setdefault(key, copy.deepcopy(value))
        if not isinstance(stats["processed_program_ids"], list):
            stats["processed_program_ids"] = []
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
                "version": 1,
                "path_stats": path_stats,
                "producer": existing.get("producer", {}),
            },
        )
        return base

    def has_selectable_paths(self, *, route_id: str) -> bool:
        _frontier_rank, _pinned, records = self._selection_inventory(
            route_id=route_id,
        )
        return bool(records)

    def selectable_records(
        self,
        *,
        route_id: str,
        top_k: int = 6,
        max_per_rank: int = 2,
        nonimproved_limit: int = 4,
    ) -> list[dict[str, Any]]:
        _frontier_rank, _pinned, records = self._selection_inventory(
            route_id=route_id,
            nonimproved_limit=nonimproved_limit,
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
    ) -> str:
        frontier_rank, _pinned, records = self._selection_inventory(
            route_id=route_id,
        )
        frontier_value = (
            str(frontier_rank) if frontier_rank is not None else "none"
        )
        header = (
            "## Selectable Shared Paths\n"
            f"route={route_id} frontier_rank={frontier_value}\n\n"
        )
        cards: list[str] = []
        used_chars = len(header)
        selected = self._bounded_records(
            records,
            top_k=top_k,
            max_per_rank=max_per_rank,
        )
        for record in selected:
            if route_id == "near_end":
                assert frontier_rank is not None
                card = self._render_near_end_card(
                    record,
                    frontier_rank=frontier_rank,
                )
            else:
                card = self._render_mid_margin_record(record)
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
        if not card:
            card = {
                "version": 1,
                "path_stats": self._derive_path_stats(name),
                "producer": {},
            }
        current_producer = card.get("producer")
        if not isinstance(current_producer, dict):
            current_producer = {}
        current_producer.update(producer)
        card["version"] = 1
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

    def _selection_inventory(
        self,
        *,
        route_id: str,
        nonimproved_limit: int = 4,
    ) -> tuple[
        int | None,
        dict[str, Any] | None,
        list[dict[str, Any]],
    ]:
        if route_id not in REFINEMENT_ROUTES:
            raise ValueError(f"unsupported refinement route: {route_id!r}")
        normalized: list[dict[str, Any]] = []
        for source_record in self.iter_path_records():
            try:
                rank = int(source_record["rank"])
            except (KeyError, TypeError, ValueError):
                continue
            record = dict(source_record)
            record["rank"] = rank
            normalized.append(record)

        common: list[dict[str, Any]] = []
        for record in self._canonicalize_sibling_paths(normalized):
            origin = self._origin(record)
            if origin is None or record.get("is_stale_improved_child"):
                continue
            record["origin"] = origin
            common.append(record)
        if not common:
            return None, None, []

        frontier_rank = min(int(record["rank"]) for record in common)
        eligible: list[dict[str, Any]] = []
        for record in common:
            rank = int(record["rank"])
            stats = self._route_stats(record, route_id)
            failures = int(stats.get("nonimproved_count") or 0)
            if failures >= max(0, int(nonimproved_limit)):
                continue
            if route_id == "near_end":
                best_child = stats.get("best_child_rank")
                if (
                    rank > frontier_rank
                    and best_child is not None
                    and int(best_child) <= frontier_rank
                ):
                    continue
            eligible.append(record)

        eligible.sort(
            key=lambda record: (
                int(record["rank"]),
                int(
                    self._route_stats(record, route_id).get(
                        "nonimproved_count",
                        0,
                    )
                    or 0
                ),
                int(
                    self._route_stats(record, route_id).get("used_count", 0)
                    or 0
                ),
                -int(record.get("init_rank") or -1),
                str(record["name"]),
            )
        )
        return frontier_rank, None, eligible

    def _render_near_end_card(
        self,
        record: dict[str, Any],
        *,
        frontier_rank: int,
    ) -> str:
        card = self._read_evidence_card(str(record["name"]))
        if not card:
            card = {
                "version": 1,
                "path_stats": self._derive_path_stats(str(record["name"])),
                "producer": {},
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
                    f"init_rank={self._path_init_rank(record)} "
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
                    f"{self._value(route_stats.get('best_child_rank'))}"
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
        return (
            f"path_name={record['name']} rank={record['rank']} "
            f"init_rank={self._path_init_rank(record)} "
            f"origin={record.get('origin') or self._origin(record)}"
        )

    def _derive_path_stats(self, name: str) -> str:
        try:
            paths = self.load(name)
            if paths and hasattr(paths[0], "format_path_stats"):
                return str(paths[0].format_path_stats())
        except Exception:
            pass
        return "path_summary:\n  unavailable"

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
