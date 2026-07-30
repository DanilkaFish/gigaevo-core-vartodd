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
EXPECTED_FILES = CORE_FILES | {
    "pso_chunked_light_todd.py",
    "pso_cma_grouped_tail.py",
    "pso_three_band_restart.py",
    "pso_pattern_heavy_restart.py",
}
SOURCE_IDS = {
    "de_light_terminal_todd.py": "a899151c-4745-4b00-8be9-e0ca0618391a",
    "pso_restart_terminal_todd.py": "e8f52447-6132-4ecc-91f3-20630378c82a",
    "pso_tohpe_only.py": "67b2f195-25c2-4f82-b50a-ab70864fac20",
    "pso_wide_terminal_beam.py": "e350705c-c6f2-479f-b153-922fad712cf8",
    "pso_chunked_light_todd.py": "4e37489e-8572-46ac-8303-e426d6b9ce8e",
    "pso_cma_grouped_tail.py": "f7aff6f1-3e02-4bf4-8f6c-f2aaea42b200",
    "pso_three_band_restart.py": "86ccecc3-e4f1-4fcc-b0ac-9cd4488069d0",
    "pso_pattern_heavy_restart.py": "0d61fdac-8502-4e32-a416-4180b6a44691",
}
ARCHETYPE_MARKERS = {
    "de_light_terminal_todd.py": ("DE(", "TERMINAL_RANK"),
    "pso_restart_terminal_todd.py": ("RESTARTS = 3", "terminal_todd"),
    "pso_tohpe_only.py": ("RESTARTS = 4", "keep=0"),
    "pso_wide_terminal_beam.py": ("beamwidth=self.int_range(6, 10",),
    "pso_chunked_light_todd.py": (
        "while consumed < evaluations",
        "unimproved_chunks",
    ),
    "pso_cma_grouped_tail.py": (
        "CMAES",
        'select_parameter_groups("scores")',
    ),
    "pso_three_band_restart.py": ("MID_REFINE_EVALS", "set_up_new_init"),
    "pso_pattern_heavy_restart.py": ("PatternSearch", "use_heavy_todd"),
}
EXPECTED_EVALUATION_BUDGETS = {
    "de_light_terminal_todd.py": {"TOTAL_EVALS": 3800},
    "pso_restart_terminal_todd.py": {"TOTAL_EVALS": 1500},
    "pso_tohpe_only.py": {"TOTAL_EVALS": 2500},
    "pso_wide_terminal_beam.py": {"TOTAL_EVALS": 2500},
    "pso_chunked_light_todd.py": {"TOTAL_EVALS": 1000},
    "pso_cma_grouped_tail.py": {
        "SCOUT_EVALS": 72,
        "SCORE_REFINE_EVALS": 256,
        "SEARCH_REFINE_EVALS": 32,
    },
    "pso_three_band_restart.py": {
        "SCOUT_EVALS": 600,
        "MID_REFINE_EVALS": 400,
    },
    "pso_pattern_heavy_restart.py": {
        "SCOUT_EVALS": 64,
        "TAIL_EVALS": 96,
    },
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


def _integer_assignments(tree: ast.Module) -> dict[str, int]:
    assignments = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            assignments[node.targets[0].id] = node.value.value
    return assignments


def test_core_programs_exist() -> None:
    assert CORE_FILES <= {path.name for path in PROGRAM_DIR.glob("*.py")}


def test_programs_are_universal_ab_initio_programs() -> None:
    for path in _program_paths(EXPECTED_FILES):
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


def test_complete_bank_has_exactly_the_selected_programs() -> None:
    assert {path.name for path in PROGRAM_DIR.glob("*.py")} == EXPECTED_FILES


def test_programs_preserve_the_selected_archetypes() -> None:
    for path in _program_paths(EXPECTED_FILES):
        source = path.read_text(encoding="utf-8")
        assert all(
            marker in source for marker in ARCHETYPE_MARKERS[path.name]
        )


def test_aux_descriptions_record_source_provenance() -> None:
    for path in _program_paths(EXPECTED_FILES):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert _last_assignment_name(tree) == "AUX_DESCRIPTION"
        assert SOURCE_IDS[path.name] in source


def test_programs_do_not_embed_gf16_rank_schedule() -> None:
    forbidden = {
        "SWITCH_RANK = 394",
        "SWITCH_RANK = 398",
        "SWITCH_RANK = 402",
        "INITIAL_RANK = 567",
    }
    for path in _program_paths(EXPECTED_FILES):
        source = path.read_text(encoding="utf-8")
        assert forbidden.isdisjoint(source.splitlines())


def test_programs_use_only_pymoo_optimizer_imports() -> None:
    for path in _program_paths(EXPECTED_FILES):
        source = path.read_text(encoding="utf-8")
        assert "nevergrad" not in source
        assert "pyswarms" not in source
        assert "scipy.optimize" not in source


def test_launcher_registers_the_curated_best_pool() -> None:
    run_gf = (REPO_ROOT / "run_gf.py").read_text(encoding="utf-8")
    assert '"best": "initial_programs_best"' in run_gf


def test_programs_use_the_approved_evaluation_budgets() -> None:
    for path in _program_paths(EXPECTED_FILES):
        assignments = _integer_assignments(
            ast.parse(path.read_text(encoding="utf-8"))
        )
        expected = EXPECTED_EVALUATION_BUDGETS[path.name]
        assert {
            name: assignments[name] for name in expected
        } == expected
