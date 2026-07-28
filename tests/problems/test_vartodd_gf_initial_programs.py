from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRAM_DIR = REPO_ROOT / "problems" / "vartodd_evo_gf" / "initial_programs"
EXPECTED_PROGRAMS = {
    "beam3_temp_probe.py",
    "full_pso_pyswarms.py",
    "lean_beam_de.py",
    "lean_scout_restart.py",
    "todd_hard_tail_budget_split.py",
    "tohpe_weights_budget_probe.py",
}
EXPECTED_OPTIMIZERS = {
    "beam3_temp_probe.py": "pymoo_pso",
    "full_pso_pyswarms.py": "pyswarms_pso_then_scipy_dual_annealing",
    "lean_beam_de.py": "pymoo_de",
    "lean_scout_restart.py": "scipy_qmc_then_powell",
    "todd_hard_tail_budget_split.py": "nevergrad_ngopt_with_restart",
    "tohpe_weights_budget_probe.py": "optuna_tpe_then_cma_es",
}
EXPECTED_PARAMETER_COUNTS = {
    "beam3_temp_probe.py": 18,
    "full_pso_pyswarms.py": 24,
    "lean_beam_de.py": 18,
    "lean_scout_restart.py": 18,
    "todd_hard_tail_budget_split.py": 19,
    "tohpe_weights_budget_probe.py": 17,
}
EXPECTED_SEMANTIC_DIGESTS = {
    "beam3_temp_probe.py": "8934e663ea2d06df6a75bccf10a3ae63a6951c3e8bce9c155b35348af4d356c0",
    "full_pso_pyswarms.py": "5bab187efa5747d72e21284867a385e6c8bf39354c7549b346a77f2390827d92",
    "lean_beam_de.py": "08e98cd413da509f38d1111f1292bc602d0955ad4077d022351b115d7f80e02a",
    "lean_scout_restart.py": "2688902355981ee76966f839886a5f604a19210f9f5623ebf606f85af23c5988",
    "todd_hard_tail_budget_split.py": "59f87f6e353fea4647bcd6cd1c7ee714dfb39d08b64513d4d08e40a9e240a6b1",
    "tohpe_weights_budget_probe.py": "2a90b8c13bf796d34a2d8aae029f86c4ce4fcfd321d033ebe38e06ec694d2f8b",
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


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
        return node

    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


def _semantic_digest(source: str) -> str:
    tree = _StripDocstrings().visit(ast.parse(source))
    semantic_source = ast.dump(tree, include_attributes=False)
    return hashlib.sha256(semantic_source.encode()).hexdigest()


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


def test_initial_program_semantics_and_hyperparameters_are_frozen() -> None:
    actual = {
        name: _semantic_digest(source)
        for name, source in _program_sources().items()
    }
    assert actual == EXPECTED_SEMANTIC_DIGESTS


def test_initial_programs_are_concise() -> None:
    line_counts = {
        name: len(source.splitlines())
        for name, source in _program_sources().items()
    }
    assert sum(line_counts.values()) <= 1_050, line_counts


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
        "full_pso_pyswarms.py": [
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


def test_grouped_tohpe_jointly_optimizes_before_score_refinement() -> None:
    source = _program_sources()["tohpe_weights_budget_probe.py"]

    joint = source.index("select_all_parameter_groups()")
    score = source.index('select_parameter_groups("scores")')
    assert joint < score
    assert "set_up_new_init(" not in source


@pytest.mark.parametrize(
    "filename",
    ["full_pso_pyswarms.py", "tohpe_weights_budget_probe.py"],
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
    source = _program_sources()["full_pso_pyswarms.py"]

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

    assert "EARLY_TODD_LIMIT = 512" in source
    assert "limit_bucket=EARLY_TODD_LIMIT" in source
    assert "max_buckets=100000" in source
    assert "limit_bucket=-1" in source
    assert "FIRST_STAGE_BUDGET = 64" in source
    assert "RESTART_BUDGET = 48" in source


def test_optimizer_dependencies_are_available() -> None:
    for package in ("cma", "nevergrad", "optuna", "pymoo", "pyswarms", "scipy"):
        assert importlib.util.find_spec(package) is not None, package
