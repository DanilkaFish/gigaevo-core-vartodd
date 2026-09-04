#!/usr/bin/env python3
"""Launch the dedicated regime-island VarTODD GF experiment."""

from __future__ import annotations

from collections.abc import Iterable
import math
import os
from pathlib import Path
import sys
from typing import NamedTuple

import numpy as np

from run_gf import (
    INITIAL_PROGRAM_POOLS,
    _link_asset,
    _matrix_degree,
    _matrix_id,
    _resolve_matrix,
    _split_launcher_args,
    _write_matrix_manifest,
    _write_metrics,
    build_gf_environment,
    default_call_timeout,
    require_experiment,
)

LEGACY_SOURCE_ASSETS = (
    "helper.py",
    "node.py",
    "mcts_dao.py",
    "todd.py",
    "full_pso.py",
    "policy_expr",
    "validate.py",
)
ISLAND_SOURCE_ASSETS = (
    "variant.py",
    "path_store.py",
    "task_description.txt",
    "prompts",
)
ISLANDS_EXPERIMENT = "vartodd_evo_gf_islands_steady"
ISLAND_POLICY_ARGUMENTS = frozenset(
    {
        "mid_root_reuse_limit",
        "mid_family_reuse_limit",
        "mid_path_reuse_limit",
        "near_family_reuse_limit",
        "near_path_reuse_limit",
        "mid_no_improvement_penalty",
        "near_no_improvement_penalty",
    }
)


class IslandPathPolicyArgs(NamedTuple):
    mid_root_reuse_limit: int = 6
    mid_family_reuse_limit: int = 8
    mid_path_reuse_limit: int = 8
    near_family_reuse_limit: int = 7
    near_path_reuse_limit: int = 2
    mid_no_improvement_penalty: float = 12.0
    near_no_improvement_penalty: float = 16.0


def _positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name}=<positive integer> is required") from exc
    if parsed <= 0:
        raise ValueError(f"{name}=<positive integer> is required")
    return parsed


def _nonnegative_float(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name}=<finite non-negative number> is required") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name}=<finite non-negative number> is required")
    return parsed


def _parse_island_policy_values(values: dict[str, str]) -> IslandPathPolicyArgs:
    defaults = IslandPathPolicyArgs()
    return IslandPathPolicyArgs(
        mid_root_reuse_limit=_positive_int(
            values.get("mid_root_reuse_limit", str(defaults.mid_root_reuse_limit)),
            name="mid_root_reuse_limit",
        ),
        mid_family_reuse_limit=_positive_int(
            values.get(
                "mid_family_reuse_limit", str(defaults.mid_family_reuse_limit)
            ),
            name="mid_family_reuse_limit",
        ),
        mid_path_reuse_limit=_positive_int(
            values.get(
                "mid_path_reuse_limit", str(defaults.mid_path_reuse_limit)
            ),
            name="mid_path_reuse_limit",
        ),
        near_family_reuse_limit=_positive_int(
            values.get(
                "near_family_reuse_limit", str(defaults.near_family_reuse_limit)
            ),
            name="near_family_reuse_limit",
        ),
        near_path_reuse_limit=_positive_int(
            values.get("near_path_reuse_limit", str(defaults.near_path_reuse_limit)),
            name="near_path_reuse_limit",
        ),
        mid_no_improvement_penalty=_nonnegative_float(
            values.get(
                "mid_no_improvement_penalty",
                str(defaults.mid_no_improvement_penalty),
            ),
            name="mid_no_improvement_penalty",
        ),
        near_no_improvement_penalty=_nonnegative_float(
            values.get(
                "near_no_improvement_penalty",
                str(defaults.near_no_improvement_penalty),
            ),
            name="near_no_improvement_penalty",
        ),
    )


def _split_island_policy_args(
    argv: Iterable[str],
) -> tuple[IslandPathPolicyArgs, list[str]]:
    values: dict[str, str] = {}
    forwarded: list[str] = []
    for arg in argv:
        key, separator, value = arg.partition("=")
        if separator and key in ISLAND_POLICY_ARGUMENTS:
            if key in values:
                raise ValueError(f"{key} was specified more than once")
            values[key] = value
        else:
            forwarded.append(arg)
    return _parse_island_policy_values(values), forwarded


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
    path_policy, common_argv = _split_island_policy_args(argv)
    (
        matrix,
        lower_bound,
        upper_bound,
        cache_enabled,
        call_timeout,
        soft_timeout_grace,
        initial_programs,
        forwarded,
    ) = _split_launcher_args(common_argv)
    forwarded = require_experiment(forwarded, ISLANDS_EXPERIMENT)
    repository_root = repository_root.resolve()
    legacy_source = repository_root / "problems" / "vartodd_evo_gf"
    island_source = repository_root / "problems" / "vartodd_evo_gf_islands"
    matrix_path = _resolve_matrix(repository_root, matrix)
    initial_rank = int(np.load(matrix_path, mmap_mode="r").shape[0])

    degree = _matrix_degree(matrix)
    matrix_id = _matrix_id(matrix, matrix_path)
    if degree is not None:
        variant_name = f"vartodd_evo_gf_islands{degree}"
        data_dir = f"data_gf_islands{degree}"
    else:
        variant_name = f"vartodd_evo_gf_islands_{matrix_id}"
        data_dir = f"data_gf_islands_{matrix_id}"
    overlay_key = f"{matrix_id}_lb{lower_bound}_ub{upper_bound}"
    if initial_programs != "default":
        overlay_key += f"_seeds_{initial_programs}"
    overlay = (
        runtime_root.resolve()
        / "islands_overlays"
        / overlay_key
        / variant_name
    )
    overlay.mkdir(parents=True, exist_ok=True)

    effective_call_timeout = call_timeout or default_call_timeout(initial_programs)
    _write_metrics(
        legacy_source,
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
    _link_island_overlay_assets(
        legacy_source,
        island_source,
        overlay,
        initial_programs=initial_programs,
    )
    (
        repository_root
        / data_dir
        / "path_backups"
    ).mkdir(parents=True, exist_ok=True)

    cache_dir = runtime_root.resolve() / "islands_cache" / matrix_id
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
        f"mid_root_reuse_limit={path_policy.mid_root_reuse_limit}",
        f"mid_family_reuse_limit={path_policy.mid_family_reuse_limit}",
        f"mid_path_reuse_limit={path_policy.mid_path_reuse_limit}",
        f"near_family_reuse_limit={path_policy.near_family_reuse_limit}",
        f"near_path_reuse_limit={path_policy.near_path_reuse_limit}",
        (
            "mid_no_improvement_penalty="
            f"{path_policy.mid_no_improvement_penalty}"
        ),
        (
            "near_no_improvement_penalty="
            f"{path_policy.near_no_improvement_penalty}"
        ),
        f"problem.name={variant_name}",
        f"problem.dir={overlay}",
        f"redis.prefix={variant_name}",
        f"initial_exec_cache_dir={cache_dir if cache_enabled else 'null'}",
    ]


def _usage() -> str:
    return (
        "Usage: python run_gf_islands.py "
        "matrix=<GF degree|exact .npy filename> lb=<rank> ub=<rank> "
        "[cache=true|false] [call_timeout=<seconds>] "
        "[soft_timeout_grace=<seconds>] "
        "[initial_programs=default|best|expensive|ultra_expensive] "
        "[mid_root_reuse_limit=6] [mid_family_reuse_limit=8] "
        "[mid_path_reuse_limit=8] "
        "[near_family_reuse_limit=7] [near_path_reuse_limit=2] "
        "[mid_no_improvement_penalty=12] "
        "[near_no_improvement_penalty=16] "
        "[ordinary run.py Hydra overrides...]\n\n"
        "ultra_expensive defaults to call_timeout=9000 when omitted.\n\n"
        "Defaults to experiment=vartodd_evo_gf_islands_steady.\n\n"
        "Start with a fresh Redis namespace:\n"
        "  python run_gf_islands.py "
        "matrix=16 lb=380 ub=421 cache=false "
        "max_concurrent_dags=6 max_in_flight=6 "
        "runner_config.prefetch_factor=1\n\n"
        "Resume the same three-island topology and Redis namespace:\n"
        "  python run_gf_islands.py "
        "matrix=16 lb=380 ub=421 cache=false redis.resume=true\n\n"
        "Resume immediately, discarding interrupted evaluations:\n"
        "  python run_gf_islands.py "
        "matrix=16 lb=380 ub=421 cache=false redis.resume=true "
        "redis.resume_incomplete=discard\n"
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
