from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANT_SOURCE = (
    REPO_ROOT / "problems" / "vartodd_evo_gf_islands" / "variant.py"
)


def _load_variant():
    spec = importlib.util.spec_from_file_location(
        "_test_vartodd_gf_islands_variant",
        VARIANT_SOURCE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_variant_uses_dedicated_island_data_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    variant_dir = tmp_path / "runtime" / "vartodd_evo_gf_islands16"
    variant_dir.mkdir(parents=True)
    (variant_dir / "metrics.yaml").write_text(
        "specs:\n  fitness:\n    lower_bound: 380\n",
        encoding="utf-8",
    )
    matrix_dir = tmp_path / "npy"
    matrix_dir.mkdir()
    matrix = matrix_dir / "gf2^16_example.npy"
    matrix.write_bytes(b"matrix")
    monkeypatch.setenv("VARTODD_VARIANT_DIR", str(variant_dir))
    monkeypatch.setenv("VARTODD_REPOSITORY_ROOT", str(tmp_path))

    resolved = _load_variant().resolve_variant(variant_dir)

    assert resolved.degree == 16
    assert resolved.matrix_path == matrix
    assert resolved.data_path == (
        tmp_path / "data_gf_islands16" / "path_backups"
    )
    assert resolved.target_final_rank == 380

