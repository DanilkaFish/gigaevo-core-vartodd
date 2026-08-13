from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


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
