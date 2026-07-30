# `run_gf.py` Initial-Program Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `initial_programs=default|best` to `run_gf.py` with isolated best-pool overlays and caches, then reduce the approved best-pool evaluation budgets.

**Architecture:** The launcher maps a small public selector to one of two shared source directories but always exposes the result as `initial_programs/` inside the generated problem. Default paths remain byte-for-byte compatible; the best selector adds a path suffix and cache subdirectory. Budget changes are constant-only edits guarded by AST tests.

**Tech Stack:** Python, pathlib, NumPy, YAML, pytest, Python AST.

## Global Constraints

- Work only on `public-vartodd`.
- Preserve `initial_programs=default` as the omitted-argument behavior.
- Supported selectors are exactly `default` and `best`.
- Redis prefix, problem name, data paths, metrics, and environment variables do not change.
- Change only evaluation counts in `initial_programs_best/`; do not alter policies or optimizers.
- Preserve unrelated dirty-worktree changes.

---

### Task 1: Select and isolate the initial-program pool

**Files:**
- Modify: `tests/test_tools/test_run_gf.py`
- Modify: `run_gf.py`

**Interfaces:**
- Consumes: launcher arguments `initial_programs=default|best`.
- Produces: an overlay whose `initial_programs` symlink resolves to the selected source directory and an isolated cache path for `best`.

- [ ] **Step 1: Add failing launcher tests**

Add:

```python
def test_build_gf_overrides_selects_best_initial_programs(tmp_path: Path) -> None:
    launcher = _load_launcher()
    overrides = launcher.build_gf_overrides(
        ["matrix=16", "lb=380", "ub=420", "initial_programs=best"],
        repository_root=ROOT,
        runtime_root=tmp_path,
    )
    overlay = Path(_value(overrides, "problem.dir"))
    assert overlay.parent.name == "gf16_lb380_ub420_seeds_best"
    assert (overlay / "initial_programs").resolve() == (
        ROOT / "problems" / "vartodd_evo_gf" / "initial_programs_best"
    ).resolve()
    assert _value(overrides, "initial_exec_cache_dir").endswith("cache/gf16/best")
    assert "initial_programs=best" not in overrides


def test_build_gf_overrides_rejects_unknown_initial_program_pool(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="initial_programs must be one of: default, best"):
        _load_launcher().build_gf_overrides(
            ["matrix=16", "lb=380", "ub=420", "initial_programs=other"],
            repository_root=ROOT,
            runtime_root=tmp_path,
        )
```

Extend the existing cache-disabled test with `initial_programs=best`, and
assert `_usage()` contains `[initial_programs=default|best]` and
`initial_programs=best`.

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
../gigaevo-core-internal/.venv/bin/pytest \
  --confcutdir=tests/test_tools -o addopts='' \
  tests/test_tools/test_run_gf.py -v
```

Expected: the best-pool tests fail because the argument is currently forwarded
and the overlay still links the default pool.

- [ ] **Step 3: Implement selector parsing and pool-aware paths**

In `run_gf.py`:

```python
INITIAL_PROGRAM_POOLS = {
    "default": "initial_programs",
    "best": "initial_programs_best",
}
```

Consume `initial_programs` in `_split_launcher_args`, default it to `default`,
and reject values outside that mapping. Remove `"initial_programs"` from the
static `SOURCE_ASSETS`; make `_link_overlay_assets` accept the selected source
directory and link it explicitly to `overlay / "initial_programs"`.

For `best`, append `_seeds_best` to the overlay-key directory and `/best` to
the cache directory. Keep the existing paths for `default`. Update `main`
tuple unpacking and `_usage()`.

- [ ] **Step 4: Run launcher tests and verify GREEN**

Run the Task 1 test command. Expected: all launcher tests pass.

- [ ] **Step 5: Commit the launcher**

Stage only `run_gf.py` and `tests/test_tools/test_run_gf.py`, then commit:

```bash
git commit -m "feat: select vartodd initial program pool"
```

### Task 2: Cut best-pool evaluation budgets

**Files:**
- Modify: `tests/problems/test_vartodd_gf_best_initial_programs.py`
- Modify: seven files under `problems/vartodd_evo_gf/initial_programs_best/`

**Interfaces:**
- Consumes: the existing eight universal best programs.
- Produces: the exact approved evaluation-count portfolio without policy changes.

- [ ] **Step 1: Add a failing AST budget test**

Add an assignment reader and assert:

```python
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
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
../gigaevo-core-internal/.venv/bin/pytest \
  --confcutdir=tests/problems -o addopts='' \
  tests/problems/test_vartodd_gf_best_initial_programs.py -v
```

Expected: the budget test reports the old constants.

- [ ] **Step 3: Change only the approved constants**

Apply the exact assignments from `EXPECTED_EVALUATION_BUDGETS`. Do not change
any other code or auxiliary description.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Task 2 test command. Expected: all best-pool tests pass.

- [ ] **Step 5: Commit the budget cuts**

Stage only the focused test and seven changed program files, then commit:

```bash
git commit -m "tune: shorten universal best vartodd seeds"
```

### Task 3: Verify launcher and seed integration

**Files:**
- Verify: `run_gf.py`
- Verify: `tests/test_tools/test_run_gf.py`
- Verify: `tests/problems/test_vartodd_gf_best_initial_programs.py`
- Verify: `problems/vartodd_evo_gf/initial_programs_best/*.py`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: compilation and regression evidence for both selectable pools.

- [ ] **Step 1: Compile the launcher and all best programs**

```bash
../gigaevo-core-internal/.venv/bin/python -m py_compile \
  run_gf.py problems/vartodd_evo_gf/initial_programs_best/*.py
```

- [ ] **Step 2: Run focused and neighboring tests**

```bash
../gigaevo-core-internal/.venv/bin/pytest \
  --confcutdir=tests/problems -o addopts='' \
  tests/problems/test_vartodd_gf_best_initial_programs.py \
  tests/problems/test_vartodd_gf_initial_programs.py \
  tests/problems/test_vartodd_gf_shared_source.py -q

../gigaevo-core-internal/.venv/bin/pytest \
  --confcutdir=tests/test_tools -o addopts='' \
  tests/test_tools/test_run_gf.py -q
```

Expected: both commands pass.

- [ ] **Step 3: Inspect scoped status and whitespace**

```bash
git diff --check -- run_gf.py tests/test_tools/test_run_gf.py \
  tests/problems/test_vartodd_gf_best_initial_programs.py \
  problems/vartodd_evo_gf/initial_programs_best
git status --short
```

Leave every unrelated status entry unchanged.
