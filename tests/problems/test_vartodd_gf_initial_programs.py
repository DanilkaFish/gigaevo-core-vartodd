from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRAM_DIR = REPO_ROOT / "problems" / "vartodd_evo_gf" / "initial_programs"
EXPECTED_PROGRAMS = {
    "beam3_temp_probe.py",
    "full_pso_pattern.py",
    "lean_beam_de.py",
    "lean_scout_restart.py",
    "todd_hard_tail_budget_split.py",
    "tohpe_weights_budget_probe.py",
}
EXPECTED_OPTIMIZERS = {
    "beam3_temp_probe.py": "pymoo_pso",
    "full_pso_pattern.py": "pymoo_pso_then_pattern_search",
    "lean_beam_de.py": "pymoo_de",
    "lean_scout_restart.py": "pymoo_cma_es_with_restarts",
    "todd_hard_tail_budget_split.py": "pymoo_de_then_pso_with_restart",
    "tohpe_weights_budget_probe.py": "pymoo_pso_then_cma_es",
}
EXPECTED_PARAMETER_COUNTS = {
    "beam3_temp_probe.py": 14,
    "full_pso_pattern.py": 23,
    "lean_beam_de.py": 15,
    "lean_scout_restart.py": 16,
    "todd_hard_tail_budget_split.py": 21,
    "tohpe_weights_budget_probe.py": 18,
}


def _program_sources() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(PROGRAM_DIR.glob("*.py"))
    }


def _program_trees() -> dict[str, ast.Module]:
    return {
        name: ast.parse(source, filename=name)
        for name, source in _program_sources().items()
    }


def _string_constant(tree: ast.Module, name: str) -> str | None:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    return None


def _method_calls(tree: ast.Module, method: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
    ]


def _score_centers(tree: ast.Module, score_type: str) -> list[ast.expr]:
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == score_type
    ]
    assert len(calls) == 1
    centers = next(
        keyword.value for keyword in calls[0].keywords if keyword.arg == "centers"
    )
    assert isinstance(centers, ast.List)
    return centers.elts


def _is_zero(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == 0.0


def _center_mapping_group(node: ast.expr) -> str | None:
    assert isinstance(node, ast.Call)
    assert isinstance(node.func, ast.Attribute)
    assert isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
    assert node.func.attr == "float_range"
    assert [ast.literal_eval(argument) for argument in node.args] == [0.0, 1.0]
    group = next(
        (keyword.value for keyword in node.keywords if keyword.arg == "group"),
        None,
    )
    if group is None:
        return None
    assert isinstance(group, ast.Constant) and isinstance(group.value, str)
    return group.value


def _literal_group_arguments(call: ast.Call) -> tuple[str, ...]:
    values: list[str] = []
    for argument in call.args:
        assert isinstance(argument, ast.Constant)
        assert isinstance(argument.value, str)
        values.append(argument.value)
    return tuple(values)


def _constant_range_size(comprehension: ast.comprehension) -> int:
    call = comprehension.iter
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name) and call.func.id == "range"
    assert not call.keywords
    values = [ast.literal_eval(argument) for argument in call.args]
    return len(range(*values))


def _mapped_calls(node: ast.AST, multiplier: int = 1) -> int:
    if isinstance(node, ast.ListComp):
        size = multiplier
        for generator in node.generators:
            size *= _constant_range_size(generator)
        return _mapped_calls(node.elt, size)

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr in {"float_range", "int_range"}
    ):
        return multiplier

    return sum(_mapped_calls(child, multiplier) for child in ast.iter_child_nodes(node))


def _mapped_parameter_count(tree: ast.Module) -> int:
    evaluator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Evaluator"
    )
    policy_methods = [
        node
        for node in evaluator.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (node.name == "policy_mapping" or node.name.startswith("_build_"))
    ]
    return sum(_mapped_calls(method) for method in policy_methods)


def test_initial_pool_has_six_distinct_optimizer_roles() -> None:
    trees = _program_trees()

    assert set(trees) == EXPECTED_PROGRAMS
    actual = {
        name: _string_constant(tree, "OPTIMIZER_FAMILY")
        for name, tree in trees.items()
    }
    assert actual == EXPECTED_OPTIMIZERS
    assert len(set(actual.values())) == 6


def test_initial_program_score_center_portfolio() -> None:
    expected_final_mappings = {
        "beam3_temp_probe.py": (),
        "full_pso_pattern.py": (1, 3),
        "lean_beam_de.py": (),
        "lean_scout_restart.py": (3,),
        "todd_hard_tail_budget_split.py": (3, 4),
        "tohpe_weights_budget_probe.py": (4,),
    }
    grouped_programs = {
        "full_pso_pattern.py",
        "tohpe_weights_budget_probe.py",
    }

    for name, tree in _program_trees().items():
        exploration = _score_centers(tree, "ExplorationScore")
        finalization = _score_centers(tree, "FinalizationScore")

        assert len(exploration) == 5
        assert all(_is_zero(center) for center in exploration), name
        assert len(finalization) == 6

        mapped_indices = tuple(
            index
            for index, center in enumerate(finalization)
            if not _is_zero(center)
        )
        assert mapped_indices == expected_final_mappings[name]
        for index in mapped_indices:
            expected_group = "scores" if name in grouped_programs else None
            assert _center_mapping_group(finalization[index]) == expected_group


def test_initial_programs_are_concise() -> None:
    line_counts = {
        name: len(source.splitlines())
        for name, source in _program_sources().items()
    }
    assert sum(line_counts.values()) <= 1_050, line_counts


def test_initial_programs_leave_tohpe_prefix_disabled(tmp_path: Path) -> None:
    variant_dir = tmp_path / "vartodd_evo_gf16"
    variant_dir.mkdir()
    (variant_dir / "metrics.yaml").write_text(
        "specs:\n  fitness:\n    lower_bound: 380\n",
        encoding="utf-8",
    )
    script = f"""
import importlib.util
from pathlib import Path

program_dir = Path({str(PROGRAM_DIR)!r})
for path in sorted(program_dir.glob("*.py")):
    spec = importlib.util.spec_from_file_location(f"initial_{{path.stem}}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator = module.Evaluator(path_name="init")
    prefix = evaluator.dao.mode.tohpeprefix.at(evaluator.loaded_rank)
    assert prefix.pool.keep == 0, (path.name, prefix.pool.keep)
    assert prefix.pool.reserve == 0, (path.name, prefix.pool.reserve)
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(PROGRAM_DIR.parent),
            "VARTODD_VARIANT_DIR": str(variant_dir),
            "VARTODD_REPOSITORY_ROOT": str(REPO_ROOT),
            "VARTODD_TARGET_FINAL_RANK": "380",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_exactly_two_programs_use_explicit_parameter_group_stages() -> None:
    trees = _program_trees()
    grouped = {
        name: [
            _literal_group_arguments(call)
            for call in _method_calls(tree, "select_parameter_groups")
        ]
        for name, tree in trees.items()
        if _method_calls(tree, "select_parameter_groups")
    }

    assert grouped == {
        "full_pso_pattern.py": [
            ("scores", "scout"),
            ("scores", "tail"),
        ],
        "tohpe_weights_budget_probe.py": [
            ("scores",),
        ],
    }


def test_initial_program_parameter_portfolio() -> None:
    actual_counts = {
        name: _mapped_parameter_count(tree)
        for name, tree in _program_trees().items()
    }

    assert actual_counts == EXPECTED_PARAMETER_COUNTS
    assert sum(value <= 18 for value in actual_counts.values()) == 4
    assert sum(19 <= value <= 24 for value in actual_counts.values()) == 2
    assert sum(value > 24 for value in actual_counts.values()) == 0


def test_grouped_tohpe_jointly_optimizes_before_score_refinement() -> None:
    source = _program_sources()["tohpe_weights_budget_probe.py"]

    joint = source.index("select_all_parameter_groups()")
    score = source.index('select_parameter_groups("scores")')
    assert joint < score
    assert "set_up_new_init(" not in source


@pytest.mark.parametrize(
    "filename",
    ["full_pso_pattern.py", "tohpe_weights_budget_probe.py"],
)
def test_grouped_range_calls_always_name_their_group(filename: str) -> None:
    tree = _program_trees()[filename]

    for method in ("float_range", "int_range"):
        for call in _method_calls(tree, method):
            assert any(keyword.arg == "group" for keyword in call.keywords), (
                f"{filename}:{call.lineno} calls {method} without group="
            )

    map_calls = _method_calls(tree, "map_par")
    assert map_calls
    assert all(
        any(
            keyword.arg == "group"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "group"
            for keyword in call.keywords
        )
        for call in map_calls
    )


def test_heavy_restart_declares_scout_and_tail_before_selecting_one() -> None:
    source = _program_sources()["full_pso_pattern.py"]

    scout = source.index("scout_policy = self._build_light_scout_policy()")
    tail = source.index("tail_policy = self._build_heavy_tail_policy()")
    selection = source.index("if self.use_heavy_todd:")
    assert scout < selection
    assert tail < selection


def test_all_bucket_retention_is_at_least_two() -> None:
    for filename, tree in _program_trees().items():
        assignments = {
            node.targets[0].id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.keyword) or node.arg != "actions_per_bucket":
                continue
            value = node.value
            if isinstance(value, ast.Name):
                value = assignments[value.id]
            if isinstance(value, ast.Constant):
                assert value.value >= 2, f"{filename}:{node.lineno}"
                continue
            assert isinstance(value, ast.Call), f"{filename}:{node.lineno}"
            assert isinstance(value.func, ast.Attribute)
            assert value.func.attr == "int_range"
            assert isinstance(value.args[0], ast.Constant)
            assert value.args[0].value >= 2, f"{filename}:{node.lineno}"


def test_hard_tail_escalates_from_finite_to_full_todd() -> None:
    source = _program_sources()["todd_hard_tail_budget_split.py"]

    assert "nevergrad" not in source
    assert 'OPTIMIZER_FAMILY = "pymoo_de_then_pso_with_restart"' in source
    assert "DE(pop_size=" in source
    assert "PSO(pop_size=" in source
    assert "EARLY_TODD_LIMIT = 512" in source
    assert "limit_bucket=EARLY_TODD_LIMIT" in source
    assert "max_buckets=100000" in source
    assert "limit_bucket=-1" in source
    assert "FIRST_STAGE_EVALS = 128" in source
    assert "RESTART_EVALS = 96" in source


def test_optimizer_dependencies_are_available() -> None:
    assert importlib.util.find_spec("pymoo") is not None


def test_initial_programs_use_only_pymoo_as_optimizer_library() -> None:
    allowed_roots = {"collections", "helper", "numpy", "pymoo"}
    for filename, tree in _program_trees().items():
        roots = {
            alias.name.split(".", maxsplit=1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in (
                node.names
                if isinstance(node, ast.Import)
                else [ast.alias(name=node.module or "")]
            )
        }
        assert roots <= allowed_roots, (filename, roots - allowed_roots)


def test_longest_seed_runs_pso_before_pattern_search() -> None:
    source = _program_sources()["full_pso_pattern.py"]

    assert 'OPTIMIZER_FAMILY = "pymoo_pso_then_pattern_search"' in source
    assert source.index("PSO(pop_size=") < source.index("PatternSearch(")
    assert source.index("optimize_pso(") < source.index("optimize_pattern_search(")


def test_initial_programs_do_not_use_nevergrad() -> None:
    for filename, source in _program_sources().items():
        assert "nevergrad" not in source, filename
        assert "NGOpt" not in source, filename
