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

    degree: int
    matrix_path: Path
    data_path: Path
    target_final_rank: int


def resolve_variant(anchor: str | Path) -> VariantSpec:
    """Resolve a ``vartodd_evo_gf_islands<N>`` runtime overlay."""
    logical_path = Path(os.environ.get("VARTODD_VARIANT_DIR", anchor)).absolute()
    variant_dir = logical_path if logical_path.is_dir() else logical_path.parent
    match = re.fullmatch(
        r"vartodd_evo_gf_islands(?P<degree>\d+)",
        variant_dir.name,
    )
    if match is None:
        raise ValueError(
            "expected an asset imported from a "
            "vartodd_evo_gf_islands<N> directory, "
            f"got {variant_dir}"
        )

    degree = int(match.group("degree"))
    repository_root = Path(
        os.environ.get("VARTODD_REPOSITORY_ROOT", variant_dir.parents[1])
    ).absolute()
    matrices = sorted((repository_root / "npy").glob(f"gf2^{degree}_*.npy"))
    if len(matrices) != 1:
        raise FileNotFoundError(
            f"expected one gf2^{degree}_*.npy matrix under "
            f"{repository_root / 'npy'}, found {len(matrices)}"
        )

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
        matrix_path=matrices[0],
        data_path=(
            repository_root
            / f"data_gf_islands{degree}"
            / "path_backups"
        ),
        target_final_rank=target_final_rank,
    )

