#!/usr/bin/env python3
"""Plot 24-hour GF16 best-so-far curves from archived Redis dumps."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from math import floor, isfinite
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPECTED_BACKUP_ORDER = (
    "gpt_deepseek_gf16_383_2806",
    "gf16_383_1507",
    "gf16_383_2207",
    "gf16_383_2307",
    "gpt_deepseek_gf16_385_2706",
    "gpt_gemini_gf16_385",
    "gf16_387_1707",
)


@dataclass(frozen=True)
class CurvePoint:
    program_id: str
    created_at: datetime
    elapsed_hours: float
    fitness: float
    t_count: int
    best_so_far: int


@dataclass(frozen=True)
class RunCurve:
    label: str
    backup_name: str
    points: list[CurvePoint]

    @property
    def program_count(self) -> int:
        return len(self.points)

    @property
    def best(self) -> int:
        if not self.points:
            raise ValueError(f"{self.label} has no points")
        return self.points[-1].best_so_far


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _numeric_fitness(record: dict[str, Any]) -> float | None:
    value = (record.get("metrics") or {}).get("fitness")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def build_curve(
    records: Iterable[dict[str, Any]],
    horizon_hours: float,
) -> list[CurvePoint]:
    evaluated: list[tuple[datetime, dict[str, Any], float]] = []
    for record in records:
        created_at = _parse_timestamp(record.get("created_at"))
        fitness = _numeric_fitness(record)
        if created_at is not None and fitness is not None:
            evaluated.append((created_at, record, fitness))
    if not evaluated:
        return []

    evaluated.sort(key=lambda item: item[0])
    started_at = evaluated[0][0]
    best: int | None = None
    points: list[CurvePoint] = []
    for created_at, record, fitness in evaluated:
        elapsed_hours = (created_at - started_at).total_seconds() / 3600.0
        metrics = record.get("metrics") or {}
        try:
            is_valid = float(metrics.get("is_valid")) > 0
        except (TypeError, ValueError):
            is_valid = False
        if not is_valid or elapsed_hours < 0 or elapsed_hours > float(horizon_hours):
            continue
        t_count = floor(fitness)
        best = t_count if best is None else min(best, t_count)
        points.append(
            CurvePoint(
                program_id=str(record.get("id") or ""),
                created_at=created_at,
                elapsed_hours=elapsed_hours,
                fitness=fitness,
                t_count=t_count,
                best_so_far=best,
            )
        )
    return points


def ordered_backup_dirs(
    root: Path,
    *,
    allow_extra: bool = False,
) -> list[Path]:
    discovered = {
        path.name: path
        for path in root.iterdir()
        if path.is_dir()
        and "gf16" in path.name.lower()
        and (path / "dump.rdb").is_file()
    }
    missing = [name for name in EXPECTED_BACKUP_ORDER if name not in discovered]
    extra = sorted(set(discovered) - set(EXPECTED_BACKUP_ORDER))
    if missing or (extra and not allow_extra):
        raise ValueError(
            "Expected exactly seven GF16 backups; "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )
    ordered = [discovered[name] for name in EXPECTED_BACKUP_ORDER]
    if allow_extra:
        ordered.extend(discovered[name] for name in extra)
    return ordered


def redis_server_command(
    redis_server: str,
    data_dir: Path,
    *,
    port: int,
) -> list[str]:
    return [
        redis_server,
        "--port",
        str(port),
        "--bind",
        "127.0.0.1",
        "--protected-mode",
        "yes",
        "--dir",
        str(data_dir),
        "--dbfilename",
        "dump.rdb",
        "--save",
        "",
        "--appendonly",
        "no",
        "--daemonize",
        "no",
    ]


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def load_programs_from_rdb(
    rdb_path: Path,
    *,
    redis_server: str = "redis-server",
    timeout: float = 15.0,
) -> list[dict[str, Any]]:
    """Load program JSON through a temporary Redis process.

    The source RDB is copied before Redis sees it. Persistence is disabled and
    the child is terminated after extraction, so neither the source backup nor
    the live Redis instance is touched.
    """
    executable = shutil.which(redis_server)
    if executable is None:
        raise FileNotFoundError(f"Redis executable not found: {redis_server}")
    if not rdb_path.is_file():
        raise FileNotFoundError(f"RDB file not found: {rdb_path}")

    import redis

    with tempfile.TemporaryDirectory(prefix="gf16-convergence-") as temp_name:
        temp_dir = Path(temp_name)
        shutil.copy2(rdb_path, temp_dir / "dump.rdb")
        port = _unused_local_port()
        process = subprocess.Popen(
            redis_server_command(executable, temp_dir, port=port),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        client = redis.Redis(
            host="127.0.0.1",
            port=port,
            db=0,
            decode_responses=True,
            socket_connect_timeout=0.2,
            socket_timeout=1.0,
        )
        try:
            deadline = time.monotonic() + float(timeout)
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Temporary Redis exited while loading {rdb_path}"
                    )
                try:
                    if client.ping():
                        break
                except redis.RedisError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out loading {rdb_path} after {timeout:g}s"
                    )
                time.sleep(0.05)

            keys = sorted(
                str(key) for key in client.scan_iter(match="*:program:*", count=1000)
            )
            programs: list[dict[str, Any]] = []
            for offset in range(0, len(keys), 256):
                values = client.mget(keys[offset : offset + 256])
                for value in values:
                    if not value:
                        continue
                    try:
                        record = json.loads(value)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        programs.append(record)
            return programs
        finally:
            try:
                client.close()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)


def _step_coordinates(
    points: Sequence[CurvePoint],
    *,
    horizon_hours: float,
) -> tuple[list[float], list[int]]:
    if not points:
        return [], []
    x_values = [point.elapsed_hours for point in points]
    y_values = [point.best_so_far for point in points]
    if x_values[-1] < float(horizon_hours):
        x_values.append(float(horizon_hours))
        y_values.append(y_values[-1])
    return x_values, y_values


def plot_curves(
    curves: Sequence[RunCurve],
    *,
    output_base: Path,
    horizon_hours: float,
) -> tuple[Path, Path, Path]:
    if not curves:
        raise ValueError("No run curves to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    csv_path = output_base.with_suffix(".csv")

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "run_label",
                "backup_name",
                "program_id",
                "created_at",
                "elapsed_hours",
                "fitness",
                "t_count",
                "best_so_far",
            ]
        )
        for curve in curves:
            for point in curve.points:
                writer.writerow(
                    [
                        curve.label,
                        curve.backup_name,
                        point.program_id,
                        point.created_at.isoformat(),
                        f"{point.elapsed_hours:.9f}",
                        f"{point.fitness:.12g}",
                        point.t_count,
                        point.best_so_far,
                    ]
                )

    styles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    colors = list(plt.get_cmap("tab10").colors)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 17,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for index, curve in enumerate(curves):
        x_values, y_values = _step_coordinates(
            curve.points,
            horizon_hours=horizon_hours,
        )
        ax.step(
            x_values,
            y_values,
            where="post",
            color=colors[index % len(colors)],
            linestyle=styles[index % len(styles)],
            linewidth=1.8,
            label=(
                f"{curve.label} ({curve.program_count} programs, "
                f"best {curve.best})"
            ),
        )

    ax.set_xlim(0, float(horizon_hours))
    ax.set_ylim(381, 405 )
    ax.set_xlabel("Elapsed time since evolution start (hours)")
    ax.set_ylabel("Program final T count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.16, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.07, linewidth=0.5)
    ax.legend(loc="upper right", frameon=True, fancybox=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, csv_path


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, default=Path("backup"))
    parser.add_argument("--hours", type=float, default=27.0)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("appendix/gf16_backup_convergence_24h"),
    )
    parser.add_argument("--redis-server", default="redis-server")
    parser.add_argument("--redis-timeout", type=float, default=15.0)
    parser.add_argument("--allow-extra", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.hours <= 0:
        raise ValueError("--hours must be positive")
    backup_dirs = ordered_backup_dirs(
        args.backup_root,
        allow_extra=args.allow_extra,
    )
    curves: list[RunCurve] = []
    for index, backup_dir in enumerate(backup_dirs):
        records = load_programs_from_rdb(
            backup_dir / "dump.rdb",
            redis_server=args.redis_server,
            timeout=args.redis_timeout,
        )
        points = build_curve(records, horizon_hours=args.hours)
        if not points:
            raise RuntimeError(
                f"No evaluated valid programs within {args.hours:g}h in "
                f"{backup_dir}"
            )
        curve = RunCurve(
            label=f"Run {chr(ord('A') + index)}",
            backup_name=backup_dir.name,
            points=points,
        )
        curves.append(curve)
        print(
            f"{curve.label}: {curve.backup_name}: "
            f"{curve.program_count} programs, best {curve.best}"
        )

    png_path, pdf_path, csv_path = plot_curves(
        curves,
        output_base=args.output_base,
        horizon_hours=args.hours,
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
