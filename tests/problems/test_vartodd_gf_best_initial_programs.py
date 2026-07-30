from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRAM_DIR = (
    REPO_ROOT / "problems" / "vartodd_evo_gf" / "initial_programs_best"
)
CORE_FILES = {
    "de_light_terminal_todd.py",
    "pso_restart_terminal_todd.py",
    "pso_tohpe_only.py",
    "pso_wide_terminal_beam.py",
}


def _program_paths(names: set[str]) -> list[Path]:
    return [PROGRAM_DIR / name for name in sorted(names)]


def _function_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _last_assignment_name(tree: ast.Module) -> str | None:
    if not tree.body or not isinstance(tree.body[-1], ast.Assign):
        return None
    targets = tree.body[-1].targets
    if len(targets) != 1 or not isinstance(targets[0], ast.Name):
        return None
    return targets[0].id


def _map_par_calls_without_group(tree: ast.Module) -> list[int]:
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute) or function.attr != "map_par":
            continue
        if not any(keyword.arg == "group" for keyword in node.keywords):
            missing.append(node.lineno)
    return missing


def test_core_programs_exist() -> None:
    assert CORE_FILES <= {path.name for path in PROGRAM_DIR.glob("*.py")}


def test_core_programs_are_universal_ab_initio_programs() -> None:
    for path in _program_paths(CORE_FILES):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "INITIAL_RANK" in source
        assert "TARGET_FINAL_RANK" in source
        assert 'path_name="init"' in source
        assert "TohpePrefix" not in source
        assert _function_names(tree) >= {
            "rank_from_target",
            "rank_margin",
            "entrypoint",
        }
        assert _last_assignment_name(tree) == "AUX_DESCRIPTION"
        assert _map_par_calls_without_group(tree) == []
