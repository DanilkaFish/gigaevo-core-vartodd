#!/usr/bin/env python3
"""Launch a matrix-specific VarTODD GF evolution from shared source assets.

Usage:
    python run_gf.py experiment=<hydra-experiment> matrix=<degree|filename.npy> lb=<rank> ub=<rank> [run.py overrides...]
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
from typing import Iterable, Mapping

import numpy as np
import yaml


SOURCE_ASSETS = (
    "helper.py",
    "node.py",
    "mcts_dao.py",
    "todd.py",
    "full_pso.py",
    "path_store.py",
    "policy_expr",
    "validate.py",
    "task_description.txt",
    "prompts",
    "variant.py",
)
INITIAL_PROGRAM_POOLS = {
    "default": "initial_programs",
    "best": "initial_programs_best",
    "expensive": "initial_programs_expensive",
    "ultra_expensive": "initial_programs_ultra_expensive",
}
_RESERVED_OVERRIDES = {
    "problem.name",
    "problem.dir",
    "redis.prefix",
    "initial_exec_cache_dir",
}
DEFAULT_VARTODD_CALL_TIMEOUT = 3800
ULTRA_EXPENSIVE_CALL_TIMEOUT = 9000
MATRIX_MANIFEST_NAME = "matrix_manifest.yaml"


def default_call_timeout(initial_programs: str) -> int:
    """Return the default timeout for the selected initial-program pool."""
    if initial_programs == "ultra_expensive":
        return ULTRA_EXPENSIVE_CALL_TIMEOUT
    return DEFAULT_VARTODD_CALL_TIMEOUT


def _parse_positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name}=<positive integer> is required") from exc
    if parsed <= 0:
        raise ValueError(f"{name}=<positive integer> is required")
    return parsed


def _parse_matrix(value: str) -> str:
    try:
        return str(_parse_positive_int(value, name="matrix"))
    except ValueError:
        pass
    if (
        not value
        or Path(value).name != value
        or "\\" in value
        or not value.endswith(".npy")
    ):
        raise ValueError(
            "matrix must be a positive GF degree or exact .npy filename"
        )
    return value


def _parse_rank(value: str, *, name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name}=<integer> is required") from exc


def _parse_nonnegative_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name}=<non-negative integer> is required") from exc
    if parsed < 0:
        raise ValueError(f"{name}=<non-negative integer> is required")
    return parsed


def _split_launcher_args(
    argv: Iterable[str],
) -> tuple[str, int, int, bool, int | None, int | None, str, list[str]]:
    values: dict[str, str] = {}
    forwarded: list[str] = []
    for arg in argv:
        key, separator, value = arg.partition("=")
        if separator and key in {
            "matrix",
            "lb",
            "ub",
            "cache",
            "call_timeout",
            "soft_timeout_grace",
            "initial_programs",
        }:
            if key in values:
                raise ValueError(f"{key} was specified more than once")
            values[key] = value
        elif separator and key in _RESERVED_OVERRIDES:
            raise ValueError(
                f"{key} is managed by the GF launcher and cannot be overridden"
            )
        else:
            forwarded.append(arg)

    missing = [key for key in ("matrix", "lb", "ub") if key not in values]
    if missing:
        raise ValueError(f"missing launcher argument(s): {', '.join(missing)}")

    matrix = _parse_matrix(values["matrix"])
    lower_bound = _parse_rank(values["lb"], name="lb")
    upper_bound = _parse_rank(values["ub"], name="ub")
    if lower_bound >= upper_bound:
        raise ValueError("lb must be lower than ub")
    cache_value = values.get("cache", "true").lower()
    if cache_value not in {"true", "false"}:
        raise ValueError("cache must be true or false")
    initial_programs = values.get("initial_programs", "default").lower()
    if initial_programs not in INITIAL_PROGRAM_POOLS:
        supported = ", ".join(INITIAL_PROGRAM_POOLS)
        raise ValueError(f"initial_programs must be one of: {supported}")
    call_timeout = (
        _parse_positive_int(values["call_timeout"], name="call_timeout")
        if "call_timeout" in values
        else None
    )
    soft_timeout_grace = (
        _parse_nonnegative_int(
            values["soft_timeout_grace"], name="soft_timeout_grace"
        )
        if "soft_timeout_grace" in values
        else None
    )
    return (
        matrix,
        lower_bound,
        upper_bound,
        cache_value == "true",
        call_timeout,
        soft_timeout_grace,
        initial_programs,
        forwarded,
    )


def require_experiment(forwarded: list[str], required: str) -> list[str]:
    """Inject the required experiment or reject a conflicting selection."""
    experiment_overrides = [
        arg for arg in forwarded if arg.partition("=")[0] == "experiment"
    ]
    if not experiment_overrides:
        return [f"experiment={required}", *forwarded]
    if len(experiment_overrides) > 1:
        raise ValueError("experiment was specified more than once")
    if experiment_overrides[0] != f"experiment={required}":
        raise ValueError(f"experiment must be {required}")
    return forwarded


def _write_metrics(
    source_dir: Path,
    destination: Path,
    lower_bound: int,
    upper_bound: int,
    initial_rank: int,
    call_timeout: int,
) -> None:
    template_path = source_dir / "metrics_template.yaml"
    metrics = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    specs = metrics["specs"]
    specs["fitness"]["lower_bound"] = lower_bound
    specs["fitness"]["upper_bound"] = upper_bound
    specs["fitness"]["sentinel_value"] = upper_bound
    specs["loaded_rank"]["lower_bound"] = lower_bound
    specs["loaded_rank"]["upper_bound"] = initial_rank
    specs["loaded_rank"]["sentinel_value"] = initial_rank
    specs["runtime"]["upper_bound"] = call_timeout
    specs["runtime"]["sentinel_value"] = call_timeout
    destination.write_text(yaml.safe_dump(metrics, sort_keys=False), encoding="utf-8")


def _link_asset(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"overlay asset already exists: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def _link_overlay_assets(
    source_dir: Path,
    overlay: Path,
    *,
    initial_programs: str,
) -> None:
    for asset in SOURCE_ASSETS:
        source = source_dir / asset
        destination = overlay / asset
        _link_asset(source, destination)
    _link_asset(
        source_dir / INITIAL_PROGRAM_POOLS[initial_programs],
        overlay / "initial_programs",
    )


def _matrix_degree(matrix: str | int) -> int | None:
    text = str(matrix)
    if text.isdecimal() and int(text) > 0:
        return int(text)
    return None


def _resolve_matrix(repository_root: Path, matrix: str | int) -> Path:
    degree = _matrix_degree(matrix)
    if degree is None:
        matrix_path = repository_root / "npy" / str(matrix)
        if not matrix_path.is_file():
            raise FileNotFoundError(f"matrix does not exist: {matrix_path}")
        return matrix_path

    matches = list((repository_root / "npy").glob(f"gf2^{degree}_*.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one gf2^{degree}_*.npy matrix under "
            f"{repository_root / 'npy'}, found {len(matches)}"
        )
    return matches[0]


def _matrix_id(matrix: str | int, matrix_path: Path) -> str:
    degree = _matrix_degree(matrix)
    if degree is not None:
        return f"gf{degree}"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", matrix_path.stem).strip("_").lower()
    if not slug:
        raise ValueError(f"cannot derive a safe identifier from {matrix_path.name}")
    return slug


def _write_matrix_manifest(
    overlay: Path,
    *,
    matrix_path: Path,
    matrix_id: str,
    degree: int | None,
    data_dir: str,
) -> None:
    manifest_path = overlay / MATRIX_MANIFEST_NAME
    manifest = {
        "matrix_file": matrix_path.name,
        "matrix_id": matrix_id,
        "degree": degree,
        "data_dir": data_dir,
    }
    if manifest_path.exists():
        existing = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise FileExistsError(
                f"matrix namespace collision in existing overlay: {manifest_path}"
            )
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def build_gf_overrides(
    argv: list[str], *, repository_root: Path, runtime_root: Path
) -> list[str]:
    """Create one private metrics overlay and return final Hydra overrides."""
    (
        matrix,
        lower_bound,
        upper_bound,
        cache_enabled,
        call_timeout,
        soft_timeout_grace,
        initial_programs,
        forwarded,
    ) = _split_launcher_args(argv)
    repository_root = repository_root.resolve()
    source_dir = repository_root / "problems" / "vartodd_evo_gf"
    matrix_path = _resolve_matrix(repository_root, matrix)
    initial_rank = int(np.load(matrix_path, mmap_mode="r").shape[0])

    degree = _matrix_degree(matrix)
    matrix_id = _matrix_id(matrix, matrix_path)
    if degree is not None:
        variant_name = f"vartodd_evo_gf{degree}"
        data_dir = f"data_gf{degree}"
    else:
        variant_name = f"vartodd_evo_gf_{matrix_id}"
        data_dir = f"data_gf_{matrix_id}"
    overlay_key = f"{matrix_id}_lb{lower_bound}_ub{upper_bound}"
    if initial_programs != "default":
        overlay_key += f"_seeds_{initial_programs}"
    overlay = runtime_root.resolve() / overlay_key / variant_name
    overlay.mkdir(parents=True, exist_ok=True)
    effective_call_timeout = call_timeout or default_call_timeout(initial_programs)
    _write_metrics(
        source_dir,
        overlay / "metrics.yaml",
        lower_bound,
        upper_bound,
        initial_rank,
        effective_call_timeout,
    )
    _write_matrix_manifest(
        overlay,
        matrix_path=matrix_path,
        matrix_id=matrix_id,
        degree=degree,
        data_dir=data_dir,
    )
    _link_overlay_assets(
        source_dir,
        overlay,
        initial_programs=initial_programs,
    )
    (repository_root / data_dir / "path_backups").mkdir(parents=True, exist_ok=True)

    cache_dir = runtime_root.resolve() / "cache" / matrix_id
    if initial_programs != "default":
        cache_dir /= initial_programs
    timeout_overrides = []
    timeout_overrides.append(f"vartodd_call_timeout={effective_call_timeout}")
    if soft_timeout_grace is not None:
        timeout_overrides.append(
            f"vartodd_soft_timeout_grace_s={soft_timeout_grace}"
        )
    return [
        *forwarded,
        *timeout_overrides,
        f"problem.name={variant_name}",
        f"problem.dir={overlay}",
        f"redis.prefix={variant_name}",
        f"initial_exec_cache_dir={cache_dir if cache_enabled else 'null'}",
    ]


def build_gf_environment(
    base_environment: Mapping[str, str],
    *,
    lower_bound: int,
    repository_root: Path,
    variant_dir: Path,
) -> dict[str, str]:
    """Return the execution environment for one matrix-specific GF run."""
    environment = dict(base_environment)
    environment["VARTODD_VARIANT_DIR"] = str(variant_dir)
    environment["VARTODD_REPOSITORY_ROOT"] = str(repository_root)
    environment["VARTODD_TARGET_FINAL_RANK"] = str(lower_bound)
    return environment


def _usage() -> str:
    return (
        "Usage: python run_gf.py matrix=<GF degree|exact .npy filename> "
        "lb=<rank> ub=<rank> "
        "[cache=true|false] [call_timeout=<seconds>] "
        "[soft_timeout_grace=<seconds>] "
        "[initial_programs=default|best|expensive|ultra_expensive] "
        "[ordinary run.py Hydra overrides...]\n\n"
        "Example:\n"
        "  python run_gf.py experiment=vartodd_evo_tohpe_updated_steady "
        "matrix=16 lb=380 ub=420 call_timeout=3800 "
        "soft_timeout_grace=200 cache=false initial_programs=best redis.db=4\n"
        "  # For costly TODD evaluations, use initial_programs=expensive\n"
        "  # ultra_expensive defaults to call_timeout=9000 when omitted\n"
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        print(_usage())
        return

    repository_root = Path(__file__).resolve().parent
    runtime_root = repository_root / ".run_gf" / "overlays"
    overrides = build_gf_overrides(
        args, repository_root=repository_root, runtime_root=runtime_root
    )
    _, lower_bound, _, _, _, _, _, _ = _split_launcher_args(args)
    variant_dir = Path(next(item.removeprefix("problem.dir=") for item in overrides if item.startswith("problem.dir=")))
    environment = build_gf_environment(
        os.environ,
        lower_bound=lower_bound,
        repository_root=repository_root,
        variant_dir=variant_dir,
    )
    os.execvpe(
        sys.executable,
        [sys.executable, str(repository_root / "run.py"), *overrides],
        environment,
    )


if __name__ == "__main__":
    main()
