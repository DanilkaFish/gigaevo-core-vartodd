# TODD Beam-Level Soft-Timeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop a long Python `Todd.run()` descent after a completed beam level, record that level's best node, and activate the existing graceful-timeout salvage path.

**Architecture:** `BaseEvaluator` injects its existing `_soft_deadline_reached` function into `Todd.run()` as an optional callback. `Todd.run()` checks the callback only after merging and recording a complete beam level; after it returns, the single-worker evaluator records the seed result and raises `GracefulEvaluationTimeout` when the same deadline is active.

**Tech Stack:** Python 3.12, pytest, existing VarTODD Python wrapper and GigaEvo graceful-timeout hook.

## Global Constraints

- Do not modify native `pyvartodd` or call time checks from inside `policy_iteration()`.
- Preserve every completed beam level before checking the deadline.
- Preserve existing `Todd.run()` behavior when no callback is supplied.
- Preserve the existing `GracefulEvaluationTimeout` and `get_active_evaluator_best()` interfaces.
- Change only the shared `problems/vartodd_evo_gf` implementation and focused tests.

---

### Task 1: Add the completed-beam checkpoint to `Todd.run()`

**Files:**
- Modify: `problems/vartodd_evo_gf/todd.py:17-142`
- Create: `tests/problems/test_vartodd_gf_soft_timeout.py`

**Interfaces:**
- Consumes: existing `Todd.run(path, *, with_report, with_timing, seed)` behavior.
- Produces: `Todd.run(..., stop_requested: Callable[[], bool] | None = None)` with unchanged return values.

- [ ] **Step 1: Write the failing completed-level checkpoint test**

Create a focused module loader for `problems/vartodd_evo_gf/todd.py` and the
following shared test fixture. The fake root produces two children; both are
therefore evaluated before the second beam-level checkpoint.

```python
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBLEM_DIR = REPO_ROOT / "problems" / "vartodd_evo_gf"


def _load_problem_module(filename: str, module_name: str):
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
```

```python
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
```

This proves the callback runs after all nodes in each inner loop and that the second completed level updates `best_node` before stopping.

- [ ] **Step 2: Write the failing no-callback compatibility test**

Use the same deterministic fake with `depth=2`, omit `stop_requested`, and assert all expected policy calls occur and the existing `(best_node, counters)` report shape is unchanged.

```python
def test_todd_without_stop_callback_keeps_existing_depth_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    todd, path, calls = _arrange_todd(monkeypatch, depth=2)
    best, counters = todd.run(path, with_report=True)
    assert len(calls) == 3
    assert best.state.rows == 7
    assert isinstance(counters, tuple)
    assert len(counters) == 2
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest -q tests/problems/test_vartodd_gf_soft_timeout.py -k todd`

Expected: the checkpoint test fails because `Todd.run()` rejects the unknown `stop_requested` keyword.

- [ ] **Step 4: Implement the optional callback and checkpoint**

Import `Callable` and add the keyword-only argument:

```python
from collections.abc import Callable

def run(
    self,
    path: Path,
    *args: Any,
    with_report: bool = False,
    with_timing: bool = False,
    seed: int = 1,
    stop_requested: Callable[[], bool] | None = None,
    **kwargs: Any,
):
```

After `nodes = heapq.nlargest(...)`, check the callback:

```python
nodes = heapq.nlargest(next_width, new_nodes, self._beam_key)
if stop_requested is not None and stop_requested():
    break
```

Do not place the check before child deduplication, best-node updates, or next-beam selection.

- [ ] **Step 5: Run the focused tests**

Run: `pytest -q tests/problems/test_vartodd_gf_soft_timeout.py -k todd`

Expected: both tests pass.

- [ ] **Step 6: Commit the checkpoint**

```bash
git add problems/vartodd_evo_gf/todd.py tests/problems/test_vartodd_gf_soft_timeout.py
git commit -m "feat: stop TODD at completed beam deadline"
```

---

### Task 2: Record the stopped seed before graceful salvage

**Files:**
- Modify: `problems/vartodd_evo_gf/helper.py:127-140,1162-1219`
- Modify: `tests/problems/test_vartodd_gf_soft_timeout.py`

**Interfaces:**
- Consumes: `Todd.run(..., stop_requested: Callable[[], bool] | None = None)` from Task 1 and existing `_soft_deadline_reached()`.
- Produces: the single-worker `BaseEvaluator.run()` records the worker result before raising `GracefulEvaluationTimeout`.

- [ ] **Step 1: Write the failing worker-injection test**

Add an overlay-based helper loader so the shared helper resolves a small GF16
matrix during import:

```python
from run_gf import build_gf_overrides


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
        str(overlay / "helper.py"),
        f"_test_soft_timeout_helper_{tmp_path.name}",
    )
```

Then construct a fake `Todd` whose `run()` captures and invokes its
`stop_requested` argument. Call
`_worker_run_one_from_template()` directly and assert that the exact helper
deadline function is supplied without changing the worker result shape.

```python
def test_worker_injects_soft_deadline_callback(monkeypatch, tmp_path) -> None:
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
```

- [ ] **Step 2: Write the failing result-before-timeout test**

Create the following `BaseEvaluator` test double with no active parameters. Its
fake current path retains the exact improved node passed to `branch_path()`.

```python
def _minimal_evaluator(helper, *, final_rank: int):
    improved_node = SimpleNamespace(state=SimpleNamespace(rows=final_rank))

    class FakePath:
        final_node = SimpleNamespace(state=SimpleNamespace(rows=10))

        def branch_path(self, node, dao, x0):
            return SimpleNamespace(final_node=node)

    class FakeTodd:
        def run(self, path, **kwargs):
            assert kwargs["stop_requested"]()
            return improved_node, (4, 1), None

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
    evaluator._search_started_at = 0.0
    evaluator.time_to_final_rank_seconds = None
    evaluator._executor = None
    evaluator._executor_key = None
    return evaluator, improved_node
```

Script `_soft_deadline_reached()` to return false for the initial precheck and
true at the beam checkpoint and post-worker check.

```python
def test_single_worker_records_completed_level_before_graceful_timeout(
    monkeypatch, tmp_path,
) -> None:
    helper = _load_helper_module(monkeypatch, tmp_path)
    deadline_values = iter([False, True, True])
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
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `pytest -q tests/problems/test_vartodd_gf_soft_timeout.py -k "worker or graceful"`

Expected: the injection assertion fails and the evaluator returns normally without raising after its only seed.

- [ ] **Step 4: Inject the callback in the worker**

Update `_worker_run_one_from_template()`:

```python
node, counters, discovered_at = todd.run(
    path,
    with_report=True,
    with_timing=True,
    seed=seed,
    stop_requested=_soft_deadline_reached,
)
```

- [ ] **Step 5: Detect the deadline after the single worker returns**

In the `worker_count == 1` loop, append the completed result first, then set the existing `timed_out` flag and stop launching seeds:

```python
results.append(
    _worker_run_one_from_template(
        seed,
        self.current_path,
        self.todd,
    )
)
if _soft_deadline_reached():
    timed_out = True
    break
```

Leave the existing result sorting, `_record_run_result()` loop, and final `GracefulEvaluationTimeout` raise in place. This ordering is what preserves the returned path.

- [ ] **Step 6: Run all focused timeout tests**

Run: `pytest -q tests/problems/test_vartodd_gf_soft_timeout.py`

Expected: all tests pass.

- [ ] **Step 7: Run related shared-GF regression tests**

Run:

```bash
pytest -q \
  tests/problems/test_vartodd_gf_parameter_groups.py \
  tests/problems/test_vartodd_tohpe_updated_todd_cap.py \
  tests/problems/test_vartodd_gf_soft_timeout.py
```

Expected: all tests pass.

- [ ] **Step 8: Verify syntax and diff integrity**

Run:

```bash
python -m py_compile \
  problems/vartodd_evo_gf/todd.py \
  problems/vartodd_evo_gf/helper.py \
  tests/problems/test_vartodd_gf_soft_timeout.py
git diff --check
```

Expected: both commands exit successfully with no output from `git diff --check`.

- [ ] **Step 9: Commit the evaluator integration**

```bash
git add problems/vartodd_evo_gf/helper.py tests/problems/test_vartodd_gf_soft_timeout.py
git commit -m "fix: salvage TODD result at soft deadline"
```
