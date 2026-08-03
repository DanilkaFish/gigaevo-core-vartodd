"""Resolve matrix-specific assets for the isolated VarTODD island experiment."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

import yaml


@dataclass(frozen=True)
class VariantSpec:
    """The matrix, path store, and rank target owned by one island variant."""

    degree: int | None
    matrix_path: Path
    data_path: Path
    target_final_rank: int


def resolve_variant(anchor: str | Path) -> VariantSpec:
    """Resolve a degree-based or exact-file island runtime overlay."""
    logical_path = Path(os.environ.get("VARTODD_VARIANT_DIR", anchor)).absolute()
    variant_dir = logical_path if logical_path.is_dir() else logical_path.parent
    repository_root = Path(
        os.environ.get("VARTODD_REPOSITORY_ROOT", variant_dir.parents[1])
    ).absolute()

    manifest_path = variant_dir / "matrix_manifest.yaml"
    if manifest_path.is_file():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        try:
            matrix_file = str(manifest["matrix_file"])
            data_dir = str(manifest["data_dir"])
            degree_value = manifest.get("degree")
            degree = None if degree_value is None else int(degree_value)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid matrix manifest: {manifest_path}") from exc
        if Path(matrix_file).name != matrix_file or Path(data_dir).name != data_dir:
            raise ValueError(f"unsafe path in matrix manifest: {manifest_path}")
        matrix_path = repository_root / "npy" / matrix_file
        if not matrix_path.is_file():
            raise FileNotFoundError(f"matrix does not exist: {matrix_path}")
    else:
        match = re.fullmatch(
            r"vartodd_evo_gf_islands(?P<degree>\d+)",
            variant_dir.name,
        )
        if match is None:
            raise ValueError(
                "expected an island GF overlay with a matrix manifest or a "
                f"vartodd_evo_gf_islands<N> directory, got {variant_dir}"
            )
        degree = int(match.group("degree"))
        matrices = sorted((repository_root / "npy").glob(f"gf2^{degree}_*.npy"))
        if len(matrices) != 1:
            raise FileNotFoundError(
                f"expected one gf2^{degree}_*.npy matrix under "
                f"{repository_root / 'npy'}, found {len(matrices)}"
            )
        matrix_path = matrices[0]
        data_dir = f"data_gf_islands{degree}"

    metrics_path = variant_dir / "metrics.yaml"
    metrics = yaml.safe_load(metrics_path.read_text(encoding="utf-8"))
    specs = metrics.get("specs", metrics) if isinstance(metrics, dict) else None
    try:
        target_final_rank = int(specs["fitness"]["lower_bound"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"missing specs.fitness.lower_bound in {metrics_path}"
        ) from exc

    return VariantSpec(
        degree=degree,
        matrix_path=matrix_path,
        data_path=repository_root / data_dir / "path_backups",
        target_final_rank=target_final_rank,
    )
