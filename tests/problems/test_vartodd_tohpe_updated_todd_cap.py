from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEM_DIR = REPO_ROOT / "problems" / "vartodd_evo_gf"


def _load_module(filename: str):
    os.environ.setdefault("VARTODD_REPOSITORY_ROOT", str(REPO_ROOT))
    os.environ.setdefault(
        "VARTODD_VARIANT_DIR",
        str(
            REPO_ROOT
            / ".run_gf"
            / "overlays"
            / "gf16_lb380_ub420"
            / "vartodd_evo_gf16"
        ),
    )
    module_path = PROBLEM_DIR / filename
    module_name = f"_test_tohpe_updated_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(PROBLEM_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(PROBLEM_DIR))
    return module


def test_validation_uses_only_todd_cap_from_three_source_profile() -> None:
    validate = _load_module("validate.py")
    profile = """
      P1 pool=final:48/tohpe:6/1/tohpeprefix:16/4/todd:5/2 \
samples=tohpe:oh:all/tohpeprefix:oh:all/todd:oh:all \
tohpe_z_choices:4 z_buckets=tohpeprefix:1024..4096/-1 todd:64..512/512
      P2 pool=final:48/tohpe:6/1/tohpeprefix:16/4/todd:0/0 \
samples=tohpe:oh:all/tohpeprefix:oh:all/todd:oh:all \
tohpe_z_choices:4 z_buckets=tohpeprefix:1024..4096/4096 todd:64..512/-1
    """

    assert validate._child_todd_limit_buckets(profile) == 512


def test_live_path_cap_uses_todd_not_tohpeprefix() -> None:
    path_store = _load_module("path_store.py")
    disabled_prefix_full = SimpleNamespace(
        pool=SimpleNamespace(keep=12),
        buckets=SimpleNamespace(limit_bucket=-1),
    )
    todd = SimpleNamespace(
        pool=SimpleNamespace(keep=4),
        buckets=SimpleNamespace(limit_bucket=512),
    )
    dao = SimpleNamespace(
        mode=SimpleNamespace(
            tohpeprefix=SimpleNamespace(points=[(567, disabled_prefix_full)]),
            todd=SimpleNamespace(points=[(567, todd)]),
        )
    )

    assert path_store.PathStore._path_todd_limit_buckets({}, [dao]) == 512


def _validated_fitness(
    validate,
    *,
    parent_name: str,
    child_name: str,
    found_rank: int = 400,
    timeout_salvaged: bool = False,
) -> float:
    import numpy as np

    result = np.zeros((found_rank, 1), dtype=np.uint8)
    validate.get_matrix = lambda: result
    validate.Tensor3D = np.asarray
    validate.Matrix = SimpleNamespace(from_numpy=lambda value: value)
    report = (
        "\nloaded_path_rank: 400"
        f"\nloaded_path_name: {parent_name}"
        + ("\ntimeout_salvaged: 1" if timeout_salvaged else "")
    )
    best_path = f"\nthis path name: {child_name}"
    return float(validate.validate((result, report, best_path))["fitness"])


def test_equal_rank_uses_reduced_penalty_for_smaller_todd_cap() -> None:
    validate = _load_module("validate.py")

    fitness = _validated_fitness(
        validate,
        parent_name="f400_i444_parent_z64of4096",
        child_name="f400_i420_child_z8192of64",
    )

    assert fitness == 401.0


def test_current_path_limit_is_parsed_from_todd_z_name() -> None:
    validate = _load_module("validate.py")
    text = (
        "loaded_path_name: f411_i490_12345678_z384of512\n"
        "this path name: f411_i490_87654321_z400of1024"
    )

    assert validate._loaded_path_todd_limit(text) == 512
    assert validate._child_path_todd_limit(text) == 1024


def test_equal_rank_uses_full_penalty_without_smaller_todd_cap() -> None:
    validate = _load_module("validate.py")

    for child_name in (
        "f400_i420_child_lim4096_z1",
        "f400_i420_child_lim8192_z1",
        "f400_i420_child_lim-1_z1",
    ):
        assert (
            _validated_fitness(
                validate,
                parent_name="f400_i444_parent_lim4096_z8192",
                child_name=child_name,
            )
            == 407.0
        )


def test_worse_rank_uses_full_penalty_even_with_smaller_todd_cap() -> None:
    validate = _load_module("validate.py")

    assert (
        _validated_fitness(
            validate,
            parent_name="f400_i444_parent_lim4096_z8192",
            child_name="f401_i420_child_lim64_z1",
            found_rank=401,
        )
        == 408.0
    )


def test_equal_rank_uses_full_penalty_for_unknown_parent_cap() -> None:
    validate = _load_module("validate.py")

    assert (
        _validated_fitness(
            validate,
            parent_name="f400_i444_parent_limunknown_z8192",
            child_name="f400_i420_child_lim64_z1",
        )
        == 407.0
    )


def test_timeout_salvage_does_not_change_fitness() -> None:
    validate = _load_module("validate.py")
    kwargs = {
        "parent_name": "f400_i444_parent_lim512_z512",
        "child_name": "f400_i420_child_lim512_z256",
    }

    assert _validated_fitness(validate, **kwargs, timeout_salvaged=True) == (
        _validated_fitness(validate, **kwargs, timeout_salvaged=False)
    )
