from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from run_gf import build_gf_overrides


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEM_DIR = REPO_ROOT / "problems" / "vartodd_evo_gf"


def _load_problem_module(filename: str | Path, module_name: str):
    module_path = Path(filename)
    if not module_path.is_absolute():
        module_path = PROBLEM_DIR / module_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(module_path.parent))
    return module


@dataclass(frozen=True)
class FakeState:
    rows: int
    name: str


def _arrange_todd(monkeypatch: pytest.MonkeyPatch, *, depth: int):
    todd_module = _load_problem_module(
        "todd.py", f"_test_soft_timeout_todd_{depth}"
    )
    calls: list[int] = []

    def fake_policy_iteration(*, cur_mat, policy_cfg, seed, add_seed):
        calls.append(cur_mat.rows)
        children = [FakeState(cur_mat.rows - 1, f"{cur_mat.name}-a")]
        if cur_mat.rows == 10:
            children.append(FakeState(cur_mat.rows - 2, f"{cur_mat.name}-b"))
        candidates = [
            SimpleNamespace(final_score=float(child.rows)) for child in children
        ]
        return SimpleNamespace(
            chosen=candidates,
            states=children,
            stats=SimpleNamespace(),
        )

    monkeypatch.setattr(todd_module, "policy_iteration", fake_policy_iteration)
    root = todd_module.Node(FakeState(10, "root"))
    dao = SimpleNamespace(
        policy_config_at=lambda **_: SimpleNamespace(
            selection=SimpleNamespace(count=2)
        )
    )
    return (
        todd_module.Todd(dao, depth=depth),
        SimpleNamespace(final_node=root),
        calls,
    )


def test_todd_checks_deadline_after_recording_a_complete_beam_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    todd, path, calls = _arrange_todd(monkeypatch, depth=8)
    checkpoint_call_counts: list[int] = []

    def stop_requested() -> bool:
        checkpoint_call_counts.append(len(calls))
        return len(checkpoint_call_counts) == 2

    best, _counters = todd.run(
        path,
        with_report=True,
        stop_requested=stop_requested,
    )

    assert checkpoint_call_counts == [1, 3]
    assert len(calls) == 3
    assert best.state.rows == 7


def test_todd_without_stop_callback_keeps_existing_depth_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    todd, path, calls = _arrange_todd(monkeypatch, depth=2)

    best, counters = todd.run(path, with_report=True)

    assert len(calls) == 3
    assert best.state.rows == 7
    assert isinstance(counters, tuple)
    assert len(counters) == 2


def _load_helper_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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
    return _load_problem_module(
        overlay / "helper.py",
        f"_test_soft_timeout_helper_{tmp_path.name}",
    )


def test_worker_injects_soft_deadline_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helper = _load_helper_module(monkeypatch, tmp_path)
    captured = {}
    node = SimpleNamespace(state=SimpleNamespace(rows=9))

    class FakeTodd:
        def run(self, path, **kwargs):
            captured.update(kwargs)
            assert kwargs["stop_requested"]()
            return node, (4, 1), 12.5

    monkeypatch.setattr(helper, "_soft_deadline_reached", lambda: True)

    result = helper._worker_run_one_from_template(7, object(), FakeTodd())

    assert result == (7, node, (4, 1), 12.5)
    assert captured["stop_requested"] is helper._soft_deadline_reached


def _minimal_evaluator(helper, *, final_rank: int):
    improved_node = SimpleNamespace(state=SimpleNamespace(rows=final_rank))

    class FakePath:
        final_node = SimpleNamespace(state=SimpleNamespace(rows=10))

        def branch_path(self, node, dao, x0):
            return SimpleNamespace(final_node=node)

    class FakeTodd:
        def run(self, path, **kwargs):
            assert kwargs["stop_requested"]()
            return improved_node, (4, 1), time.perf_counter()

    evaluator = object.__new__(helper.BaseEvaluator)
    evaluator.active_params = []
    evaluator.insert = lambda params: None
    evaluator.reinit = lambda: None
    evaluator.current_path = FakePath()
    evaluator.todd = FakeTodd()
    evaluator.dao = SimpleNamespace()
    evaluator.x0 = []
    evaluator.total_eval = 0
    evaluator.tcount = []
    evaluator._best_rank = 10_000
    evaluator.best_paths = []
    evaluator.best_ranks = []
    evaluator.best_evals = []
    evaluator.best_eval = 0
    evaluator.best_seen = 0
    evaluator.best_seed = None
    evaluator._search_started_at = time.perf_counter()
    evaluator.time_to_final_rank_seconds = None
    evaluator._executor = None
    evaluator._executor_key = None
    return evaluator, improved_node


def test_single_worker_records_completed_level_before_graceful_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helper = _load_helper_module(monkeypatch, tmp_path)
    deadline_values = iter([False, False, True, True])
    monkeypatch.setattr(
        helper,
        "_soft_deadline_reached",
        lambda: next(deadline_values),
    )
    evaluator, improved_node = _minimal_evaluator(helper, final_rank=9)

    with pytest.raises(helper.GracefulEvaluationTimeout):
        evaluator.run([], seeds=[7], max_workers=1)

    assert evaluator.best_rank == 9
    assert evaluator.best_seed == 7
    assert evaluator.best_paths[0].final_node is improved_node
    assert evaluator.total_eval == 4
