#!/usr/bin/env python3
"""Rebuild the GF32 MAP-Elites archive under the rank-aware retention rule.

This is a one-off migration for an existing ``vartodd_gf32`` Redis run.  It
does not execute programs or call an LLM.  Instead it derives ``rank_improved``
from the stored execution evidence, considers every historically evaluated
valid program, and writes a fresh 40-member archive.

The command is dry-run by default.  ``--apply`` takes the same Redis instance
lock as the evolution process, creates a local rollback snapshot, then updates
the archive hash, program states, and status sets in one Redis transaction.

Example:
    python tools/rebuild_gf32_archive.py
    python tools/rebuild_gf32_archive.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import sys
from typing import Iterable


# Permit ``python tools/rebuild_gf32_archive.py`` without package installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from custom.archive_selectors import GF32RankRetentionSelector
from gigaevo.config.helpers import build_behavior_space
from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.utils.json import dumps as json_dumps


DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_PROGRAM_PREFIX = "vartodd_gf32"
DEFAULT_ARCHIVE_PREFIX = "island_fitness_island"
DEFAULT_MAX_SIZE = 40

_EXPLICIT_IMPROVEMENT_RE = re.compile(
    r"\bloaded_path_improved\s*[:=]\s*([01])\b", re.IGNORECASE
)
_NO_IMPROVEMENT_RE = re.compile(
    r"\bNOTE:\s*no improvement over loaded path\b", re.IGNORECASE
)


def _first_int(text: str, field: str) -> int | None:
    match = re.search(rf"\b{re.escape(field)}\s*[:=]\s*(\d+)\b", text)
    return int(match.group(1)) if match else None


def _execution_evidence(program: Program) -> str:
    """Return all persisted execution text that can establish rank improvement."""
    metadata = program.metadata or {}
    return "\n".join(
        str(metadata.get(key) or "")
        for key in ("aux_info", "mutation_context")
    )


def infer_rank_improved(program: Program) -> float | None:
    """Infer whether a historical program beat its applicable rank baseline.

    ``loaded_path_improved`` is authoritative for saved-path refiners.  Fresh
    paths do not emit that marker, so their ``final_rank`` is compared with the
    starting ``loaded_rank`` (or ``init_rank`` / ``initial_rank`` fallback).
    ``None`` means the stored record cannot be classified safely.
    """
    existing = program.metrics.get("rank_improved")
    if isinstance(existing, (int, float)) and existing in (0, 1):
        return float(existing)

    if float(program.metrics.get("is_valid", 0.0)) < 1.0:
        return 0.0

    evidence = _execution_evidence(program)
    if match := _EXPLICIT_IMPROVEMENT_RE.search(evidence):
        return float(match.group(1))
    if _NO_IMPROVEMENT_RE.search(evidence):
        return 0.0

    final_rank = _first_int(evidence, "final_rank")
    baseline = (
        _first_int(evidence, "loaded_path_rank")
        or _first_int(evidence, "loaded_rank")
        or _first_int(evidence, "init_rank")
        or _first_int(evidence, "initial_rank")
    )
    if final_rank is None or baseline is None:
        return None
    return float(final_rank < baseline)


def build_gf32_behavior_space():
    """Return the behavior space configured in vartodd_diverse_gf32.yaml.

    This intentionally uses the *initial* dynamic bounds.  A resumed process
    reconstructs that same space from Hydra configuration; dynamic tightening
    is not persisted in Redis.
    """
    return build_behavior_space(
        keys=["fitness", "runtime", "loaded_rank", "is_valid"],
        bounds=[(1190.0, 1280.0), (0.0, 13200.0), (1200.0, 1701.0), (0.0, 1.0)],
        resolutions=[25, 4, 10, 2],
        binning_types=["linear", "linear", "linear", "linear"],
        dynamic=True,
        expansion_buffer_ratio=0.1,
    )


def _eligible(program: Program) -> bool:
    metrics = program.metrics
    return (
        float(metrics.get("is_valid", 0.0)) >= 1.0
        and all(key in metrics for key in ("fitness", "runtime", "loaded_rank"))
        and "rank_improved" in metrics
    )


def choose_elites(programs: Iterable[Program], max_size: int) -> tuple[list[Program], int]:
    """Select one rank-aware winner per cell, then apply the GF32 size cap."""
    behavior_space = build_gf32_behavior_space()
    selector = GF32RankRetentionSelector(
        fitness_keys=["fitness"], fitness_key_higher_is_better=[False]
    )
    by_cell: dict[tuple[int, ...], Program] = {}
    for program in sorted(programs, key=lambda item: item.id):
        cell = behavior_space.get_cell(program.metrics)
        current = by_cell.get(cell)
        if current is None or selector(program, current):
            by_cell[cell] = program

    cell_winners = list(by_cell.values())
    # GF32's configured FitnessArchiveRemover removes the highest numerical
    # primary fitness when the archive exceeds its 40-program cap.
    elites = sorted(
        cell_winners, key=lambda item: (item.metrics["fitness"], item.id)
    )[:max_size]
    return elites, len(cell_winners)


def _backup_payload(
    programs: list[Program], archive_mapping: dict[str, str], status_members: dict[str, list[str]]
) -> dict:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "archive_mapping": archive_mapping,
        "status_members": status_members,
        "program_state": {
            program.id: {
                "state": program.state.value,
                "metrics": dict(program.metrics),
                "home_island": program.metadata.get("home_island"),
                "current_island": program.metadata.get("current_island"),
            }
            for program in programs
        },
    }


async def _read_archive_state(
    storage: RedisProgramStorage, archive_prefix: str
) -> tuple[dict[str, str], dict[str, list[str]]]:
    async def read(redis):
        mapping = await redis.hgetall(f"{archive_prefix}:archive")
        statuses = {
            state.value: sorted(
                await redis.smembers(storage._keys.status_set(state.value))
            )
            for state in ProgramState
        }
        return mapping, statuses

    return await storage.with_redis("gf32_archive_rebuild:read", read)


async def _apply_rebuild(
    storage: RedisProgramStorage,
    programs: list[Program],
    elites: list[Program],
    archive_prefix: str,
) -> None:
    elite_ids = {program.id for program in elites}
    terminal_states = {ProgramState.DONE, ProgramState.DISCARDED}
    for program in programs:
        if program.id in elite_ids:
            program.state = ProgramState.DONE
            program.metadata.setdefault("home_island", "fitness_island")
            program.metadata["current_island"] = "fitness_island"
        elif program.state in terminal_states:
            program.state = ProgramState.DISCARDED
            program.metadata.pop("current_island", None)

    cells = {
        ",".join(map(str, build_gf32_behavior_space().get_cell(program.metrics))): program.id
        for program in elites
    }
    reverse_cells = {program_id: cell for cell, program_id in cells.items()}
    all_ids = [program.id for program in programs]
    by_state: dict[str, list[str]] = {state.value: [] for state in ProgramState}
    for program in programs:
        by_state[program.state.value].append(program.id)

    async def write(redis) -> None:
        end_counter = await redis.incrby(storage._keys.timestamp(), len(programs))
        start_counter = end_counter - len(programs) + 1
        pipe = redis.pipeline(transaction=True)
        for index, program in enumerate(programs):
            data = program.to_dict()
            data["atomic_counter"] = int(start_counter + index)
            pipe.set(storage._keys.program(program.id), json_dumps(data))

        archive_key = f"{archive_prefix}:archive"
        reverse_key = f"{archive_prefix}:archive:reverse"
        pipe.delete(archive_key, reverse_key)
        if cells:
            pipe.hset(archive_key, mapping=cells)
            pipe.hset(reverse_key, mapping=reverse_cells)

        # Remove every rebuilt program from every terminal/non-terminal status
        # set before adding its one canonical state.  This repairs the current
        # DONE/DISCARDED overlap as part of the migration.
        for state in ProgramState:
            pipe.srem(storage._keys.status_set(state.value), *all_ids)
        for state, ids in by_state.items():
            if ids:
                pipe.sadd(storage._keys.status_set(state), *ids)
        pipe.xadd(
            storage._keys.status_stream(),
            {
                "id": "gf32_archive_rebuild",
                "status": ProgramState.DONE.value,
                "event": "archive_rebuild_rank_aware",
            },
            maxlen=10_000,
            approximate=True,
        )
        await pipe.execute()

    await storage.with_redis("gf32_archive_rebuild:apply", write)


async def run(args: argparse.Namespace) -> int:
    if args.max_size < 1:
        raise ValueError("--max-size must be at least 1")

    config = RedisProgramStorageConfig(
        redis_url=args.redis_url,
        key_prefix=args.program_prefix,
        read_only=not args.apply,
    )
    async with RedisProgramStorage(config) as storage:
        programs = await storage.get_all()
        archive_mapping, status_members = await _read_archive_state(
            storage, args.archive_prefix
        )
        # Capture before adding the derived metric so the backup is a true
        # rollback point, not a snapshot of the pending migration.
        backup_payload = _backup_payload(programs, archive_mapping, status_members)

        unknown: list[Program] = []
        inferred = Counter()
        for program in programs:
            result = infer_rank_improved(program)
            if result is None:
                if float(program.metrics.get("is_valid", 0.0)) >= 1.0:
                    unknown.append(program)
                continue
            program.metrics["rank_improved"] = result
            inferred["improved" if result else "not_improved"] += 1

        if unknown:
            print(
                "Refusing rebuild: valid programs with no rank-improvement evidence: "
                + ", ".join(program.id[:8] for program in unknown),
                file=sys.stderr,
            )
            return 2

        candidates = [program for program in programs if _eligible(program)]
        elites, cell_winner_count = choose_elites(candidates, args.max_size)
        new_ids = {program.id for program in elites}
        old_ids = set(archive_mapping.values())

        print(f"Programs read: {len(programs)}")
        print(f"Valid candidates with rank evidence: {len(candidates)}")
        print(
            "Rank evidence: "
            f"improved={inferred['improved']}, not_improved={inferred['not_improved']}"
        )
        print(f"Unique occupied cells before global cap: {cell_winner_count}")
        print(f"Selected elites: {len(elites)} / cap {args.max_size}")
        print(
            "Archive delta: "
            f"keep={len(new_ids & old_ids)}, add={len(new_ids - old_ids)}, "
            f"remove={len(old_ids - new_ids)}"
        )
        print("Selected IDs:", " ".join(program.id[:8] for program in elites))

        if not args.apply:
            print("Dry run only. Re-run with --apply after stopping evolution.")
            return 0

        backup_dir = args.backup_dir
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / (
            "gf32_archive_rebuild_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json"
        )
        backup_path.write_text(
            json.dumps(
                backup_payload,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        await _apply_rebuild(storage, programs, elites, args.archive_prefix)
        print(f"Applied archive rebuild. Backup: {backup_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", default=DEFAULT_REDIS_URL)
    parser.add_argument("--program-prefix", default=DEFAULT_PROGRAM_PREFIX)
    parser.add_argument("--archive-prefix", default=DEFAULT_ARCHIVE_PREFIX)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "archive_rebuild_backups",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the rebuilt archive; without this flag the script is read-only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
