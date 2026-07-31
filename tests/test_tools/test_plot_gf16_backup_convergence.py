from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.plot_gf16_backup_convergence import (
    CurvePoint,
    EXPECTED_BACKUP_ORDER,
    RunCurve,
    _step_coordinates,
    build_curve,
    ordered_backup_dirs,
    plot_curves,
    redis_server_command,
)


def _program(
    program_id: str,
    hour: float,
    *,
    fitness: float | None,
    is_valid: float | None = 1.0,
) -> dict:
    created = datetime(
        2026, 1, 1, tzinfo=timezone.utc
    ).timestamp() + hour * 3600
    return {
        "id": program_id,
        "created_at": datetime.fromtimestamp(created, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "metrics": {
            "fitness": fitness,
            "is_valid": is_valid,
        },
    }


def test_build_curve_uses_evolution_start_and_valid_points_only() -> None:
    records = [
        _program("invalid-start", 0, fitness=410.2, is_valid=0),
        _program("first-valid", 1, fitness=407.9),
        _program("unevaluated", 2, fitness=None),
        _program("improvement", 3, fitness=403.2),
        _program("regression", 4, fitness=405.1),
        _program("after-horizon", 25, fitness=399.0),
    ]

    points = build_curve(records, horizon_hours=24)

    assert [point.program_id for point in points] == [
        "first-valid",
        "improvement",
        "regression",
    ]
    assert [point.elapsed_hours for point in points] == pytest.approx([1, 3, 4])
    assert [point.t_count for point in points] == [407, 403, 405]
    assert [point.best_so_far for point in points] == [407, 403, 403]


def test_build_curve_orders_records_by_creation_time() -> None:
    records = [
        _program("later", 2, fitness=401.8),
        _program("earlier", 1, fitness=405.9),
    ]

    points = build_curve(records, horizon_hours=24)

    assert [point.program_id for point in points] == ["earlier", "later"]
    assert [point.best_so_far for point in points] == [405, 401]


def test_ordered_backup_dirs_preserves_reference_run_mapping(
    tmp_path: Path,
) -> None:
    for name in reversed(EXPECTED_BACKUP_ORDER):
        backup_dir = tmp_path / name
        backup_dir.mkdir()
        (backup_dir / "dump.rdb").touch()
    (tmp_path / "not_gf16").mkdir()

    ordered = ordered_backup_dirs(tmp_path)

    assert [path.name for path in ordered] == list(EXPECTED_BACKUP_ORDER)


def test_ordered_backup_dirs_rejects_missing_expected_backup(
    tmp_path: Path,
) -> None:
    for name in EXPECTED_BACKUP_ORDER[:-1]:
        backup_dir = tmp_path / name
        backup_dir.mkdir()
        (backup_dir / "dump.rdb").touch()

    with pytest.raises(ValueError, match="Expected exactly seven GF16 backups"):
        ordered_backup_dirs(tmp_path)


def test_redis_server_command_is_local_and_disables_persistence(
    tmp_path: Path,
) -> None:
    command = redis_server_command("redis-server", tmp_path, port=6399)

    assert command[:3] == ["redis-server", "--port", "6399"]
    assert ["--bind", "127.0.0.1"] == command[3:5]
    assert command[command.index("--dir") + 1] == str(tmp_path)
    assert command[command.index("--save") + 1] == ""
    assert command[command.index("--appendonly") + 1] == "no"


def test_step_coordinates_extend_last_incumbent_to_horizon() -> None:
    points = [
        CurvePoint(
            program_id="a",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            elapsed_hours=1.0,
            fitness=405.2,
            t_count=405,
            best_so_far=405,
        ),
        CurvePoint(
            program_id="b",
            created_at=datetime(2026, 1, 1, 2, tzinfo=timezone.utc),
            elapsed_hours=2.0,
            fitness=401.2,
            t_count=401,
            best_so_far=401,
        ),
    ]

    x_values, y_values = _step_coordinates(points, horizon_hours=24)

    assert x_values == [1.0, 2.0, 24.0]
    assert y_values == [405, 401, 401]


def test_plot_curves_writes_png_pdf_and_point_csv(tmp_path: Path) -> None:
    points = build_curve(
        [
            _program("first", 0, fitness=405.8),
            _program("second", 2, fitness=401.2),
        ],
        horizon_hours=24,
    )
    curves = [RunCurve(label="Run A", backup_name="example_gf16", points=points)]

    png_path, pdf_path, csv_path = plot_curves(
        curves,
        output_base=tmp_path / "convergence",
        horizon_hours=24,
    )

    assert png_path.is_file() and png_path.stat().st_size > 0
    assert pdf_path.is_file() and pdf_path.stat().st_size > 0
    csv_text = csv_path.read_text(encoding="utf-8")
    assert (
        "run_label,backup_name,program_id,created_at,elapsed_hours,"
        "fitness,t_count,best_so_far"
    ) in csv_text
    assert "Run A,example_gf16,second" in csv_text
