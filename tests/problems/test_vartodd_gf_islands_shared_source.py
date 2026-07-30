from pathlib import Path

from run_gf_islands import build_gf_islands_overrides


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY = REPO_ROOT / "problems" / "vartodd_evo_gf"
ISLANDS = REPO_ROOT / "problems" / "vartodd_evo_gf_islands"


def _overlay(tmp_path: Path) -> Path:
    overrides = build_gf_islands_overrides(
        ["matrix=16", "lb=380", "ub=421"],
        repository_root=REPO_ROOT,
        runtime_root=tmp_path,
    )
    return Path(
        next(
            item.removeprefix("problem.dir=")
            for item in overrides
            if item.startswith("problem.dir=")
        )
    )


def test_overlay_links_runtime_and_initial_programs_from_legacy_source(
    tmp_path: Path,
) -> None:
    overlay = _overlay(tmp_path)

    for name in (
        "helper.py",
        "node.py",
        "mcts_dao.py",
        "todd.py",
        "full_pso.py",
        "validate.py",
        "initial_programs",
    ):
        assert (overlay / name).resolve().is_relative_to(LEGACY)


def test_overlay_links_island_owned_context_and_path_assets(
    tmp_path: Path,
) -> None:
    overlay = _overlay(tmp_path)

    for name in (
        "variant.py",
        "path_store.py",
        "task_description.txt",
        "prompts",
    ):
        assert (overlay / name).resolve().is_relative_to(ISLANDS)


def test_legacy_launcher_namespace_remains_unchanged(tmp_path: Path) -> None:
    from run_gf import build_gf_overrides

    overrides = build_gf_overrides(
        ["matrix=16", "lb=380", "ub=421"],
        repository_root=REPO_ROOT,
        runtime_root=tmp_path,
    )

    assert "problem.name=vartodd_evo_gf16" in overrides
    assert "redis.prefix=vartodd_evo_gf16" in overrides
    assert (REPO_ROOT / "data_gf16" / "path_backups").is_dir()
