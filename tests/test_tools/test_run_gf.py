from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_launcher():
    path = ROOT / "run_gf.py"
    assert path.exists(), "run_gf.py launcher is missing"
    spec = importlib.util.spec_from_file_location("run_gf", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _value(overrides: list[str], key: str) -> str:
    prefix = f"{key}="
    return next(item.removeprefix(prefix) for item in overrides if item.startswith(prefix))


def test_build_gf_overrides_creates_hidden_matrix_overlay(tmp_path: Path) -> None:
    launcher = _load_launcher()

    overrides = launcher.build_gf_overrides(
        [
            "experiment=vartodd_evo_tohpe_updated_steady",
            "matrix=16",
            "lb=380",
            "ub=420",
            "redis.db=4",
        ],
        repository_root=ROOT,
        runtime_root=tmp_path,
    )

    overlay = Path(_value(overrides, "problem.dir"))
    metrics = yaml.safe_load((overlay / "metrics.yaml").read_text(encoding="utf-8"))

    assert overlay.name == "vartodd_evo_gf16"
    assert overlay.parent.name == "gf16_lb380_ub420"
    assert (overlay / "helper.py").is_symlink()
    assert (overlay / "initial_programs").is_symlink()
    assert (overlay / "initial_programs").resolve() == (
        ROOT / "problems" / "vartodd_evo_gf" / "initial_programs"
    ).resolve()
    assert (overlay / "prompts").is_symlink()
    assert metrics["specs"]["fitness"]["lower_bound"] == 380
    assert metrics["specs"]["fitness"]["upper_bound"] == 420
    assert metrics["specs"]["loaded_rank"]["lower_bound"] == 380
    assert metrics["specs"]["loaded_rank"]["upper_bound"] == 567
    assert metrics["specs"]["loaded_rank"]["sentinel_value"] == 567
    assert metrics["specs"]["runtime"]["upper_bound"] == 3800
    assert metrics["specs"]["runtime"]["sentinel_value"] == 3800
    assert _value(overrides, "vartodd_call_timeout") == "3800"
    assert _value(overrides, "problem.name") == "vartodd_evo_gf16"
    assert _value(overrides, "redis.prefix") == "vartodd_evo_gf16"
    assert _value(overrides, "initial_exec_cache_dir").endswith("cache/gf16")
    assert "redis.db=4" in overrides


@pytest.mark.parametrize(
    ("matrix", "lower_bound", "upper_bound", "initial_rank"),
    [
        (14, 323, 350, 492),
        (16, 380, 420, 567),
    ],
)
def test_loaded_rank_bounds_span_target_to_matrix_initial_rank(
    matrix: int,
    lower_bound: int,
    upper_bound: int,
    initial_rank: int,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    overrides = launcher.build_gf_overrides(
        [
            f"matrix={matrix}",
            f"lb={lower_bound}",
            f"ub={upper_bound}",
        ],
        repository_root=ROOT,
        runtime_root=tmp_path,
    )
    overlay = Path(_value(overrides, "problem.dir"))
    specs = yaml.safe_load(
        (overlay / "metrics.yaml").read_text(encoding="utf-8")
    )["specs"]

    assert specs["fitness"]["lower_bound"] == lower_bound
    assert specs["fitness"]["upper_bound"] == upper_bound
    assert specs["loaded_rank"]["lower_bound"] == lower_bound
    assert specs["loaded_rank"]["upper_bound"] == initial_rank
    assert specs["loaded_rank"]["sentinel_value"] == initial_rank


def test_build_gf_overrides_rejects_bad_launcher_arguments(tmp_path: Path) -> None:
    launcher = _load_launcher()

    try:
        launcher.build_gf_overrides(
            ["matrix=zero", "lb=380", "ub=420"],
            repository_root=ROOT,
            runtime_root=tmp_path,
        )
    except ValueError as exc:
        assert "positive GF degree or exact .npy filename" in str(exc)
    else:
        raise AssertionError("invalid matrix value must be rejected")


def test_build_gf_overrides_selects_best_initial_programs(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()

    overrides = launcher.build_gf_overrides(
        [
            "matrix=16",
            "lb=380",
            "ub=420",
            "initial_programs=best",
        ],
        repository_root=ROOT,
        runtime_root=tmp_path,
    )

    overlay = Path(_value(overrides, "problem.dir"))
    assert overlay.parent.name == "gf16_lb380_ub420_seeds_best"
    assert (overlay / "initial_programs").resolve() == (
        ROOT / "problems" / "vartodd_evo_gf" / "initial_programs_best"
    ).resolve()
    assert _value(overrides, "initial_exec_cache_dir").endswith(
        "cache/gf16/best"
    )
    assert "initial_programs=best" not in overrides


def test_build_gf_overrides_rejects_unknown_initial_program_pool(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()

    with pytest.raises(
        ValueError,
        match="initial_programs must be one of: default, best",
    ):
        launcher.build_gf_overrides(
            [
                "matrix=16",
                "lb=380",
                "ub=420",
                "initial_programs=other",
            ],
            repository_root=ROOT,
            runtime_root=tmp_path,
        )


@pytest.mark.parametrize(
    "override",
    [
        "problem.name=wrong",
        "problem.dir=/tmp/wrong",
        "redis.prefix=wrong",
        "initial_exec_cache_dir=/tmp/wrong-cache",
    ],
)
def test_build_gf_overrides_rejects_launcher_owned_overrides(
    override: str,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()

    with pytest.raises(ValueError, match="managed by the GF launcher"):
        launcher.build_gf_overrides(
            ["matrix=16", "lb=380", "ub=420", override],
            repository_root=ROOT,
            runtime_root=tmp_path,
        )


def test_build_gf_overrides_disables_initial_cache_when_requested(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()

    overrides = launcher.build_gf_overrides(
        [
            "matrix=16",
            "lb=380",
            "ub=420",
            "cache=false",
            "initial_programs=best",
        ],
        repository_root=ROOT,
        runtime_root=tmp_path,
    )

    assert _value(overrides, "initial_exec_cache_dir") == "null"
    assert "cache=false" not in overrides
    assert "initial_programs=best" not in overrides


def test_build_gf_overrides_maps_timeout_launcher_arguments(tmp_path: Path) -> None:
    launcher = _load_launcher()

    overrides = launcher.build_gf_overrides(
        [
            "matrix=16",
            "lb=380",
            "ub=420",
            "call_timeout=2700",
            "soft_timeout_grace=200",
        ],
        repository_root=ROOT,
        runtime_root=tmp_path,
    )

    assert _value(overrides, "vartodd_call_timeout") == "2700"
    assert _value(overrides, "vartodd_soft_timeout_grace_s") == "200"
    overlay = Path(_value(overrides, "problem.dir"))
    metrics = yaml.safe_load((overlay / "metrics.yaml").read_text(encoding="utf-8"))
    assert metrics["specs"]["runtime"]["upper_bound"] == 2700
    assert metrics["specs"]["runtime"]["sentinel_value"] == 2700
    assert "call_timeout=2700" not in overrides
    assert "soft_timeout_grace=200" not in overrides


def test_build_gf_environment_exports_lower_bound_as_target_rank() -> None:
    launcher = _load_launcher()

    environment = launcher.build_gf_environment(
        {"UNCHANGED": "value"},
        lower_bound=380,
        repository_root=ROOT,
        variant_dir=ROOT / ".run_gf" / "overlays" / "gf16" / "vartodd_evo_gf16",
    )

    assert environment["UNCHANGED"] == "value"
    assert environment["VARTODD_TARGET_FINAL_RANK"] == "380"


def test_redis_storage_uses_configured_prefix() -> None:
    redis_config = (ROOT / "config" / "redis" / "default.yaml").read_text(
        encoding="utf-8"
    )

    assert "key_prefix: ${redis.prefix}" in redis_config


def test_usage_documents_initial_program_selection() -> None:
    usage = _load_launcher()._usage()

    assert "[initial_programs=default|best|expensive]" in usage
    assert "initial_programs=best" in usage
