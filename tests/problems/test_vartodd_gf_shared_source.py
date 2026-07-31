from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from run_gf import build_gf_overrides


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_DIR = REPO_ROOT / "problems" / "vartodd_evo_gf"
SHARED_ASSETS = (
    "helper.py",
    "node.py",
    "mcts_dao.py",
    "todd.py",
    "full_pso.py",
    "path_store.py",
    "validate.py",
    "task_description.txt",
    "prompts",
    "initial_programs",
    "variant.py",
)


def _load_shared_variant_module():
    module_path = SHARED_DIR / "variant.py"
    spec = importlib.util.spec_from_file_location("_test_vartodd_gf_variant", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _overlay(
    degree: int, lower_bound: int, upper_bound: int, tmp_path: Path
) -> Path:
    overrides = build_gf_overrides(
        [f"matrix={degree}", f"lb={lower_bound}", f"ub={upper_bound}"],
        repository_root=REPO_ROOT,
        runtime_root=tmp_path,
    )
    return Path(
        next(item.removeprefix("problem.dir=") for item in overrides if item.startswith("problem.dir="))
    )


@pytest.mark.parametrize(
    ("degree", "matrix_name", "data_name", "target"),
    [
        (15, "gf2^15_1510.npy", "data_gf15", 360),
        (16, "gf2^16_1612310.npy", "data_gf16", 380),
    ],
)
def test_resolve_variant_uses_launcher_overlay(
    degree: int,
    matrix_name: str,
    data_name: str,
    target: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    overlay = _overlay(degree, target, target + 40, tmp_path)
    monkeypatch.setenv("VARTODD_VARIANT_DIR", str(overlay))
    monkeypatch.setenv("VARTODD_REPOSITORY_ROOT", str(REPO_ROOT))

    spec = _load_shared_variant_module().resolve_variant(SHARED_DIR / "helper.py")

    assert spec.degree == degree
    assert spec.matrix_path.name == matrix_name
    assert spec.data_path == REPO_ROOT / data_name / "path_backups"
    assert spec.target_final_rank == target


def test_overlay_contains_only_generated_metrics_and_shared_links(tmp_path: Path) -> None:
    overlay = _overlay(16, 380, 420, tmp_path)

    assert (overlay / "metrics.yaml").is_file()
    assert not (overlay / "metrics.yaml").is_symlink()
    for asset_name in SHARED_ASSETS:
        asset = overlay / asset_name
        assert asset.is_symlink(), f"{asset} must be a shared-source symlink"
        assert asset.resolve().is_relative_to(SHARED_DIR)


def test_no_permanent_matrix_specific_problem_directories() -> None:
    for degree in (14, 15, 16):
        assert not (REPO_ROOT / "problems" / f"vartodd_evo_gf{degree}").exists()


def test_heavy_tail_seed_derives_its_schedule_from_shared_rank_constants() -> None:
    seed = (
        SHARED_DIR / "initial_programs" / "todd_hard_tail_budget_split.py"
    ).read_text(encoding="utf-8")

    assert "INITIAL_RANK," in seed
    assert "TARGET_FINAL_RANK," in seed
    assert "SWITCH_RANK = TARGET_FINAL_RANK + 55" in seed
    assert "INITIAL_RANK = 567" not in seed


def test_shared_bucket_api_omits_unsupported_exploration_controls() -> None:
    helper = (SHARED_DIR / "helper.py").read_text(encoding="utf-8")
    dao = (SHARED_DIR / "mcts_dao.py").read_text(encoding="utf-8")

    assert "random_fraction" not in helper
    assert "random_fraction" not in dao

    helper_bucket = helper.split("def _to_z_bucket_search", 1)[1].split(
        "def _to_tohpe_search", 1
    )[0]
    dao_bucket = dao.split("def _as_z_bucket_search", 1)[1].split(
        "def _as_tohpe_search", 1
    )[0]
    assert "temperature" not in helper_bucket
    assert "temperature" not in dao_bucket
