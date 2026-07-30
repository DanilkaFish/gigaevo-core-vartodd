"""Evidence-rich saved paths for the isolated VarTODD island experiment."""

from __future__ import annotations

from collections.abc import Sequence
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


class PathStore(_legacy.PathStore):
    """Legacy-compatible store with unified selectable path evidence cards."""

    root_dir: str = DATA_PATH

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

    def has_selectable_paths(self) -> bool:
        return bool(self.selectable_records(top_k=1))

    def selectable_records(
        self,
        *,
        top_k: int = 8,
        max_nonimproved_reuse: int = 10,
    ) -> list[dict[str, Any]]:
        live = [
            record
            for record in self.iter_path_records()
            if self._is_live_record(
                record,
                max_nonimproved_reuse=max_nonimproved_reuse,
            )
            and record.get("child_improved_loaded") is not False
            and not record.get("is_stale_improved_child")
        ]
        by_rank: dict[int, list[dict[str, Any]]] = {}
        for record in live:
            by_rank.setdefault(int(record["rank"]), []).append(record)
        representatives = [
            min(
                records,
                key=lambda record: (
                    -int(record.get("init_rank") or -1),
                    int(record.get("used_count") or 0),
                    str(record["name"]),
                ),
            )
            for _rank, records in sorted(by_rank.items())
        ]
        representatives.sort(
            key=lambda record: (
                int(record["rank"]),
                -int(record.get("init_rank") or -1),
                int(record.get("used_count") or 0),
                str(record["name"]),
            )
        )
        return representatives[: max(0, int(top_k))]

    def render_selectable_path_cards(
        self,
        *,
        top_k: int = 8,
        max_chars: int = 24_000,
    ) -> str:
        header = "## Selectable Shared Paths\n\n"
        cards: list[str] = []
        used_chars = len(header)
        for record in self.selectable_records(top_k=top_k):
            card = self._render_record_card(record)
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

    def _render_record_card(self, record: dict[str, Any]) -> str:
        card = self._read_evidence_card(str(record["name"]))
        if not card:
            card = {
                "version": 1,
                "path_stats": self._derive_path_stats(str(record["name"])),
                "producer": {},
            }
            self._write_evidence_card(str(record["name"]), card)
        stats = str(card.get("path_stats") or "").strip()
        producer = (
            card.get("producer")
            if isinstance(card.get("producer"), dict)
            else {}
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
        return "\n".join(
            [
                f"### {record['name']}",
                "selection:",
                (
                    f"  rank={record['rank']} depth={record.get('depth')} "
                    f"u/i={record.get('used_count', 0)}/"
                    f"{record.get('improved_count', 0)} "
                    f"kind={record.get('kind', 'unknown')} "
                    f"band={record.get('restart_band', 'unknown')}"
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
