from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from run_gf import build_gf_overrides


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_helper(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    overrides = build_gf_overrides(
        ["matrix=16", "lb=380", "ub=420"],
        repository_root=REPO_ROOT,
        runtime_root=tmp_path,
    )
    overlay = Path(
        next(
            item.removeprefix("problem.dir=")
            for item in overrides
            if item.startswith("problem.dir=")
        )
    )
    monkeypatch.setenv("VARTODD_VARIANT_DIR", str(overlay))
    monkeypatch.setenv("VARTODD_REPOSITORY_ROOT", str(REPO_ROOT))
    module_path = overlay / "helper.py"
    module_name = f"_test_grouped_helper_{tmp_path.name}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(overlay))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(overlay))
    return module


def _grouped_evaluator(helper):
    class GroupedEvaluator(helper.BaseEvaluator):
        def __init__(self):
            self.x0 = [0.2, 1.1, -0.4, 3.0] + [0.0] * (helper.X0_LENGTH - 4)
            self.active_params = []
            self._selected_parameter_groups = None
            self._parameter_groups = {}
            self._parameter_group_layout = None
            self.reinit()

        def policy_mapping(self):
            self.score_a = self.map_par(lambda x: x, group="scores")
            self.search = self.map_par(
                lambda x, offset: x + offset,
                group="search",
                offset=2.0,
            )
            self.score_b = self.map_par(lambda x: x, group="scores")
            self.legacy = self.map_par(lambda x: x)

    return GroupedEvaluator()


def test_groups_are_declared_without_changing_default_active_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = _load_helper(monkeypatch, tmp_path)
    evaluator = _grouped_evaluator(helper)

    assert evaluator.parameter_group_names() == ("scores", "search", "default")
    assert evaluator.parameter_group_sizes() == {
        "scores": 2,
        "search": 1,
        "default": 1,
    }
    assert evaluator.parameter_group_size("scores") == 2
    assert evaluator.active_parameter_group_names() == (
        "scores",
        "search",
        "default",
    )
    np.testing.assert_allclose(
        evaluator.extract_parameter_group("scores"),
        [0.2, -0.4],
    )
    np.testing.assert_allclose(evaluator.extract_active(), [0.2, 1.1, -0.4, 3.0])
    assert evaluator.active_params == [0, 1, 2, 3]
    assert evaluator.search == pytest.approx(3.1)


def test_map_par_rejects_positional_group_and_invalid_group_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = _load_helper(monkeypatch, tmp_path)
    evaluator = _grouped_evaluator(helper)

    with pytest.raises(TypeError):
        evaluator.map_par(lambda x: x, "scores")
    with pytest.raises(ValueError, match="non-empty string"):
        evaluator.map_par(lambda x: x, group="")
    with pytest.raises(ValueError, match="non-empty string"):
        evaluator.map_par(lambda x: x, group=3)
    with pytest.raises(ValueError, match="unknown parameter group 'missing'"):
        evaluator.parameter_group_size("missing")
    with pytest.raises(ValueError, match="unknown parameter group 'missing'"):
        evaluator.extract_parameter_group("missing")


def test_selected_groups_form_a_compact_vector_and_freeze_other_x0_slots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = _load_helper(monkeypatch, tmp_path)
    evaluator = _grouped_evaluator(helper)

    selected = evaluator.select_parameter_groups("scores")

    np.testing.assert_allclose(selected, [0.2, -0.4])
    assert evaluator.active_params == [0, 2]
    assert evaluator.active_parameter_group_names() == ("scores",)

    assert evaluator.run([0.5, -0.8], seeds=[]) == []

    np.testing.assert_allclose(evaluator.x0[:4], [0.5, 1.1, -0.8, 3.0])
    np.testing.assert_allclose(evaluator.extract_active(), [0.5, -0.8])
    np.testing.assert_allclose(
        evaluator.extract_parameter_group("search"),
        [1.1],
    )


def test_replacing_full_x0_like_a_saved_path_preserves_group_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = _load_helper(monkeypatch, tmp_path)
    evaluator = _grouped_evaluator(helper)
    evaluator.select_parameter_groups("scores")

    evaluator.x0 = [10.0, 11.0, 12.0, 13.0] + evaluator.x0[4:]
    active = evaluator.reinit()

    np.testing.assert_allclose(active, [10.0, 12.0])
    np.testing.assert_allclose(
        evaluator.extract_parameter_group("search"),
        [11.0],
    )


def test_selection_argument_order_never_reorders_optimizer_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = _load_helper(monkeypatch, tmp_path)
    evaluator = _grouped_evaluator(helper)

    reverse_requested = evaluator.select_parameter_groups("search", "scores")
    declaration_requested = evaluator.select_parameter_groups("scores", "search")

    np.testing.assert_allclose(reverse_requested, [0.2, 1.1, -0.4])
    np.testing.assert_allclose(declaration_requested, [0.2, 1.1, -0.4])
    assert evaluator.active_parameter_group_names() == ("scores", "search")


def test_select_all_restores_the_original_vector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = _load_helper(monkeypatch, tmp_path)
    evaluator = _grouped_evaluator(helper)
    evaluator.select_parameter_groups("search")

    restored = evaluator.select_all_parameter_groups()

    np.testing.assert_allclose(restored, [0.2, 1.1, -0.4, 3.0])
    assert evaluator.active_params == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ((), "at least one parameter group"),
        (("scores", "scores"), "duplicate parameter groups"),
        (("missing",), "unknown parameter group 'missing'"),
    ],
)
def test_invalid_group_selections_fail_before_changing_the_layout(
    groups: tuple[str, ...],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helper = _load_helper(monkeypatch, tmp_path)
    evaluator = _grouped_evaluator(helper)
    before = evaluator.extract_active()

    with pytest.raises(ValueError, match=message):
        evaluator.select_parameter_groups(*groups)

    np.testing.assert_allclose(evaluator.extract_active(), before)
    assert evaluator.active_parameter_group_names() == (
        "scores",
        "search",
        "default",
    )


def test_reinit_rejects_parameter_group_membership_that_depends_on_x0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = _load_helper(monkeypatch, tmp_path)

    class ConditionalEvaluator(helper.BaseEvaluator):
        def __init__(self):
            self.x0 = [0.0] * helper.X0_LENGTH
            self.active_params = []
            self._selected_parameter_groups = None
            self._parameter_groups = {}
            self._parameter_group_layout = None
            self.reinit()

        def policy_mapping(self):
            switch = self.map_par(lambda x: x, group="switch")
            if switch > 0.0:
                self.map_par(lambda x: x, group="conditional")

    evaluator = ConditionalEvaluator()
    evaluator.insert([1.0])

    with pytest.raises(RuntimeError, match="parameter group layout changed"):
        evaluator.reinit()


def test_reinit_keeps_selected_groups_and_stable_slots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = _load_helper(monkeypatch, tmp_path)
    evaluator = _grouped_evaluator(helper)
    evaluator.select_parameter_groups("scores")

    first_indices = tuple(evaluator.active_params)
    second = evaluator.reinit()

    assert tuple(evaluator.active_params) == first_indices == (0, 2)
    np.testing.assert_allclose(second, [0.2, -0.4])


def test_full_pso_clone_does_not_share_parameter_group_lists() -> None:
    source = (
        REPO_ROOT / "problems" / "vartodd_evo_gf" / "full_pso.py"
    ).read_text(encoding="utf-8")

    assert "clone._parameter_groups = {" in source
    assert 'if hasattr(fun, "_parameter_groups"):' in source
    assert "name: list(indices)" in source
    assert "for name, indices in fun._parameter_groups.items()" in source
