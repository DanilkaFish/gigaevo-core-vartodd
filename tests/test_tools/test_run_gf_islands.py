from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_launcher():
    path = ROOT / "run_gf_islands.py"
    spec = importlib.util.spec_from_file_location("run_gf_islands", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _value(overrides: list[str], key: str) -> str:
    prefix = f"{key}="
    return next(
        item.removeprefix(prefix)
        for item in overrides
        if item.startswith(prefix)
    )


def test_launcher_creates_isolated_overlay_store_and_redis_prefix(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()

    overrides = launcher.build_gf_islands_overrides(
        [
            "experiment=vartodd_evo_gf_islands_steady",
            "matrix=16",
            "lb=380",
            "ub=421",
            "initial_programs=best",
        ],
        repository_root=ROOT,
        runtime_root=tmp_path,
    )

    overlay = Path(_value(overrides, "problem.dir"))
    metrics = yaml.safe_load((overlay / "metrics.yaml").read_text())
    assert overlay.name == "vartodd_evo_gf_islands16"
    assert overlay.is_relative_to(tmp_path / "islands_overlays")
    assert _value(overrides, "experiment") == "vartodd_evo_gf_islands_steady"
    assert _value(overrides, "problem.name") == "vartodd_evo_gf_islands16"
    assert _value(overrides, "redis.prefix") == "vartodd_evo_gf_islands16"
    assert _value(overrides, "initial_exec_cache_dir").endswith(
        "islands_cache/gf16/best"
    )
    assert metrics["specs"]["fitness"]["lower_bound"] == 380
    assert metrics["specs"]["fitness"]["upper_bound"] == 421
    assert (overlay / "prompts").is_symlink()
    assert (
        overlay / "prompts" / "islands" / "ab_initio.txt"
    ).read_text(encoding="utf-8").strip()
    assert (
        overlay / "prompts" / "islands" / "mid_margin.txt"
    ).read_text(encoding="utf-8").strip()
    assert (
        overlay / "prompts" / "islands" / "near_end.txt"
    ).read_text(encoding="utf-8").strip()
    assert (ROOT / "data_gf_islands16" / "path_backups").is_dir()


def test_launcher_defaults_to_islands_experiment(tmp_path: Path) -> None:
    launcher = _load_launcher()

    overrides = launcher.build_gf_islands_overrides(
        ["matrix=16", "lb=380", "ub=421"],
        repository_root=ROOT,
        runtime_root=tmp_path,
    )

    assert _value(overrides, "experiment") == "vartodd_evo_gf_islands_steady"


def test_launcher_emits_default_path_policy_controls(tmp_path: Path) -> None:
    launcher = _load_launcher()

    overrides = launcher.build_gf_islands_overrides(
        ["matrix=16", "lb=380", "ub=421"],
        repository_root=ROOT,
        runtime_root=tmp_path,
    )

    assert _value(overrides, "mid_root_reuse_limit") == "6"
    assert _value(overrides, "mid_family_reuse_limit") == "8"
    assert _value(overrides, "near_family_reuse_limit") == "7"
    assert _value(overrides, "near_path_reuse_limit") == "2"
    assert _value(overrides, "mid_no_improvement_penalty") == "12.0"
    assert _value(overrides, "near_no_improvement_penalty") == "16.0"


def test_launcher_accepts_explicit_path_policy_controls(tmp_path: Path) -> None:
    launcher = _load_launcher()

    overrides = launcher.build_gf_islands_overrides(
        [
            "matrix=16",
            "lb=380",
            "ub=421",
            "mid_root_reuse_limit=5",
            "mid_family_reuse_limit=9",
            "near_family_reuse_limit=11",
            "near_path_reuse_limit=3",
            "mid_no_improvement_penalty=14.5",
            "near_no_improvement_penalty=19",
        ],
        repository_root=ROOT,
        runtime_root=tmp_path,
    )

    assert _value(overrides, "mid_root_reuse_limit") == "5"
    assert _value(overrides, "mid_family_reuse_limit") == "9"
    assert _value(overrides, "near_family_reuse_limit") == "11"
    assert _value(overrides, "near_path_reuse_limit") == "3"
    assert _value(overrides, "mid_no_improvement_penalty") == "14.5"
    assert _value(overrides, "near_no_improvement_penalty") == "19.0"


@pytest.mark.parametrize(
    "override",
    [
        "mid_root_reuse_limit=0",
        "mid_family_reuse_limit=-1",
        "near_family_reuse_limit=0",
        "near_path_reuse_limit=1.5",
        "mid_no_improvement_penalty=-1",
        "near_no_improvement_penalty=nan",
        "near_no_improvement_penalty=inf",
    ],
)
def test_launcher_rejects_invalid_path_policy_controls(
    override: str,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()

    with pytest.raises(ValueError):
        launcher.build_gf_islands_overrides(
            ["matrix=16", "lb=380", "ub=421", override],
            repository_root=ROOT,
            runtime_root=tmp_path,
        )


def test_launcher_rejects_duplicate_path_policy_control(tmp_path: Path) -> None:
    launcher = _load_launcher()

    with pytest.raises(ValueError, match="mid_root_reuse_limit"):
        launcher.build_gf_islands_overrides(
            [
                "matrix=16",
                "lb=380",
                "ub=421",
                "mid_root_reuse_limit=6",
                "mid_root_reuse_limit=7",
            ],
            repository_root=ROOT,
            runtime_root=tmp_path,
        )


def test_island_launcher_selects_ultra_expensive_pool_and_timeout(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()

    overrides = launcher.build_gf_islands_overrides(
        [
            "matrix=16",
            "lb=380",
            "ub=421",
            "initial_programs=ultra_expensive",
        ],
        repository_root=ROOT,
        runtime_root=tmp_path,
    )

    overlay = Path(_value(overrides, "problem.dir"))
    metrics = yaml.safe_load((overlay / "metrics.yaml").read_text())
    assert overlay.parent.name == "gf16_lb380_ub421_seeds_ultra_expensive"
    assert (overlay / "initial_programs").resolve() == (
        ROOT / "problems" / "vartodd_evo_gf" / "initial_programs_ultra_expensive"
    ).resolve()
    assert metrics["specs"]["runtime"]["upper_bound"] == 9000
    assert metrics["specs"]["runtime"]["sentinel_value"] == 9000
    assert _value(overrides, "vartodd_call_timeout") == "9000"
    assert _value(overrides, "initial_exec_cache_dir").endswith(
        "islands_cache/gf16/ultra_expensive"
    )


@pytest.mark.parametrize(
    "override",
    [
        "experiment=vartodd_evo_tohpe_updated_steady",
        "problem.name=wrong",
        "problem.dir=/tmp/wrong",
        "redis.prefix=wrong",
        "initial_exec_cache_dir=/tmp/wrong-cache",
    ],
)
def test_launcher_rejects_conflicting_managed_overrides(
    override: str,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()

    with pytest.raises(ValueError):
        launcher.build_gf_islands_overrides(
            ["matrix=16", "lb=380", "ub=421", override],
            repository_root=ROOT,
            runtime_root=tmp_path,
        )


def test_island_launcher_help_documents_fresh_and_resume_runs() -> None:
    usage = _load_launcher()._usage()

    assert "Defaults to experiment=vartodd_evo_gf_islands_steady" in usage
    assert "fresh Redis" in usage
    assert "runner_config.prefetch_factor=1" in usage
    assert "redis.resume=true" in usage
    assert "redis.resume_incomplete=discard" in usage
    assert "same three-island topology" in usage
    assert "[initial_programs=default|best|expensive|ultra_expensive]" in usage
    assert "9000" in usage
