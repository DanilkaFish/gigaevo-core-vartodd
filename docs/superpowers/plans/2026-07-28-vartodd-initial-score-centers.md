# Vartodd Initial Score Centers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diversify finalization-score center parameterization across the six shared vartodd initial programs while zeroing every exploration center.

**Architecture:** Keep each seed self-contained and use its existing local `float_range` mapping. Add no shared abstraction: the center expression remains visible beside its score, and grouped seeds explicitly assign center parameters to `group="scores"`.

**Tech Stack:** Python, `ast`, pytest, existing vartodd evaluator API.

## Global Constraints

- Preserve optimizer settings, weights, score powers, sampling budgets, pools, schedules, action selection, restart behavior, and all non-center program logic.
- Exploration centers are `[0.0, 0.0, 0.0, 0.0, 0.0]` in every program.
- Finalization centers follow the exact six-program distribution in the approved design.
- Parameterized centers map to `[0.0, 1.0]`.
- Grouped programs use `group="scores"` for parameterized centers.

---

### Task 1: Specify the center portfolio with a failing source-level test

**Files:**
- Modify: `tests/problems/test_vartodd_gf_initial_programs.py`

**Interfaces:**
- Consumes: Python ASTs returned by `_program_trees()`.
- Produces: `test_initial_program_score_center_portfolio`, a regression contract for center positions and score grouping.

- [ ] **Step 1: Add AST helpers and the failing test**

Add a helper that finds `ExplorationScore` and `FinalizationScore` calls, reads their `centers` keyword, and classifies each element as fixed zero or a `self.float_range(0.0, 1.0, ...)` mapping. Assert this expected final-center index map:

```python
{
    "beam3_temp_probe.py": (),
    "lean_beam_de.py": (),
    "lean_scout_restart.py": (3,),
    "tohpe_weights_budget_probe.py": (4,),
    "full_pso_pyswarms.py": (1, 3),
    "todd_hard_tail_budget_split.py": (3, 4),
}
```

Also assert every exploration center is fixed zero and that mapped center calls in grouped programs carry `group="scores"`.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest -o addopts='' \
  --confcutdir=tests/problems \
  tests/problems/test_vartodd_gf_initial_programs.py::test_initial_program_score_center_portfolio -q
```

Expected: FAIL because every existing program still uses fixed `0.5` centers.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/problems/test_vartodd_gf_initial_programs.py
git commit -m "test: specify diverse initial score centers"
```

### Task 2: Implement the six center strategies

**Files:**
- Modify: `problems/vartodd_evo_gf/initial_programs/beam3_temp_probe.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/lean_beam_de.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/lean_scout_restart.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/tohpe_weights_budget_probe.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/full_pso_pyswarms.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/todd_hard_tail_budget_split.py`
- Modify: `tests/problems/test_vartodd_gf_initial_programs.py`

**Interfaces:**
- Consumes: each evaluator's existing `float_range(low, high[, group=...])`.
- Produces: the approved center portfolio and updated frozen parameter-count/semantic contracts.

- [ ] **Step 1: Replace only center expressions**

Use all-zero exploration centers. Use these final centers:

```python
# beam3_temp_probe.py and lean_beam_de.py
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# lean_scout_restart.py
[0.0, 0.0, 0.0, self.float_range(0.0, 1.0), 0.0, 0.0]

# tohpe_weights_budget_probe.py
[0.0, 0.0, 0.0, 0.0, self.float_range(0.0, 1.0, group="scores"), 0.0]

# full_pso_pyswarms.py
[
    0.0,
    self.float_range(0.0, 1.0, group="scores"),
    0.0,
    self.float_range(0.0, 1.0, group="scores"),
    0.0,
    0.0,
]

# todd_hard_tail_budget_split.py
[0.0, 0.0, 0.0, self.float_range(0.0, 1.0), self.float_range(0.0, 1.0), 0.0]
```

- [ ] **Step 2: Run the portfolio test and verify GREEN**

Run the command from Task 1. Expected: PASS.

- [ ] **Step 3: Update frozen parameter counts and semantic digests**

Set the expected parameter counts to:

```python
{
    "beam3_temp_probe.py": 18,
    "full_pso_pyswarms.py": 26,
    "lean_beam_de.py": 18,
    "lean_scout_restart.py": 19,
    "todd_hard_tail_budget_split.py": 21,
    "tohpe_weights_budget_probe.py": 18,
}
```

Recompute semantic digests from the resulting sources, preserving the exact current non-center contents of every file.

- [ ] **Step 4: Run the complete initial-program contract**

Run:

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest -o addopts='' \
  --confcutdir=tests/problems \
  tests/problems/test_vartodd_gf_initial_programs.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Verify focused diff and commit**

Run:

```bash
git diff --check
git diff -- problems/vartodd_evo_gf/initial_programs tests/problems/test_vartodd_gf_initial_programs.py
```

Confirm the program diff changes only score centers. Then commit:

```bash
git add problems/vartodd_evo_gf/initial_programs tests/problems/test_vartodd_gf_initial_programs.py
git commit -m "refactor: diversify initial score centers"
```

