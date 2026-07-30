#!/usr/bin/env python3
"""Launch the dedicated regime-island VarTODD GF experiment."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

from run_gf import (
    DEFAULT_VARTODD_CALL_TIMEOUT,
    INITIAL_PROGRAM_POOLS,
    _link_asset,
    _resolve_matrix,
    _split_launcher_args,
    _write_metrics,
    build_gf_environment,
)

LEGACY_SOURCE_ASSETS = (
    "helper.py",
    "node.py",
    "mcts_dao.py",
    "todd.py",
    "full_pso.py",
    "validate.py",
)
ISLAND_SOURCE_ASSETS = (
    "variant.py",
    "path_store.py",
    "task_description.txt",
    "prompts",
)


def _link_island_overlay_assets(
    legacy_source: Path,
    island_source: Path,
    overlay: Path,
    *,
    initial_programs: str,
) -> None:
    for asset in LEGACY_SOURCE_ASSETS:
        _link_asset(legacy_source / asset, overlay / asset)
    for asset in ISLAND_SOURCE_ASSETS:
        _link_asset(island_source / asset, overlay / asset)
    _link_asset(
        legacy_source / INITIAL_PROGRAM_POOLS[initial_programs],
        overlay / "initial_programs",
    )


def build_gf_islands_overrides(
    argv: list[str],
    *,
    repository_root: Path,
    runtime_root: Path,
) -> list[str]:
    """Create an isolated matrix overlay and return its Hydra overrides."""
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
    legacy_source = repository_root / "problems" / "vartodd_evo_gf"
    island_source = repository_root / "problems" / "vartodd_evo_gf_islands"
    matrix_path = _resolve_matrix(repository_root, matrix)
    initial_rank = int(np.load(matrix_path, mmap_mode="r").shape[0])

    variant_name = f"vartodd_evo_gf_islands{matrix}"
    overlay_key = f"gf{matrix}_lb{lower_bound}_ub{upper_bound}"
    if initial_programs != "default":
        overlay_key += f"_seeds_{initial_programs}"
    overlay = (
        runtime_root.resolve()
        / "islands_overlays"
        / overlay_key
        / variant_name
    )
    overlay.mkdir(parents=True, exist_ok=True)

    effective_call_timeout = call_timeout or DEFAULT_VARTODD_CALL_TIMEOUT
    _write_metrics(
        legacy_source,
        overlay / "metrics.yaml",
        lower_bound,
        upper_bound,
        initial_rank,
        effective_call_timeout,
    )
    _link_island_overlay_assets(
        legacy_source,
        island_source,
        overlay,
        initial_programs=initial_programs,
    )
    (
        repository_root
        / f"data_gf_islands{matrix}"
        / "path_backups"
    ).mkdir(parents=True, exist_ok=True)

    cache_dir = runtime_root.resolve() / "islands_cache" / f"gf{matrix}"
    if initial_programs != "default":
        cache_dir /= initial_programs
    timeout_overrides = [f"vartodd_call_timeout={effective_call_timeout}"]
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


def _usage() -> str:
    return (
        "Usage: python run_gf_islands.py matrix=<positive integer> lb=<rank> "
        "ub=<rank> [cache=true|false] [call_timeout=<seconds>] "
        "[soft_timeout_grace=<seconds>] [initial_programs=default|best] "
        "[ordinary run.py Hydra overrides...]\n\n"
        "Start with a fresh Redis namespace:\n"
        "  python run_gf_islands.py "
        "experiment=vartodd_evo_gf_islands_steady matrix=16 lb=380 ub=421 "
        "cache=false max_concurrent_dags=12 max_in_flight=12 "
        "runner_config.prefetch_factor=1\n\n"
        "Resume the same three-island topology and Redis namespace:\n"
        "  python run_gf_islands.py "
        "experiment=vartodd_evo_gf_islands_steady matrix=16 lb=380 ub=421 "
        "cache=false redis.resume=true\n"
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        print(_usage())
        return

    repository_root = Path(__file__).resolve().parent
    runtime_root = repository_root / ".run_gf"
    overrides = build_gf_islands_overrides(
        args,
        repository_root=repository_root,
        runtime_root=runtime_root,
    )
    _, lower_bound, _, _, _, _, _, _ = _split_launcher_args(args)
    variant_dir = Path(
        next(
            item.removeprefix("problem.dir=")
            for item in overrides
            if item.startswith("problem.dir=")
        )
    )
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
