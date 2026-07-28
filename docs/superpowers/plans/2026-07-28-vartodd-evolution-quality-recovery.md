# VarTODD Evolution Quality Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore evidence-driven capped-to-full TODD evolution, prioritize score weights, repair grouped optimization, and reduce the six initial programs to a 12--24 parameter portfolio.

**Architecture:** Prompt contracts live in the shared `vartodd_evo_gf` task description and stage prompts, while sampled search-regime guidance lives in the GF16 algorithm configs. Static AST tests protect the six executable seeds without importing their optional optimizer dependencies. The grouped TOHPE program first optimizes the complete policy jointly, then refines only score coordinates on the same evaluator state.

**Tech Stack:** Python 3.12, pytest, AST source inspection, Hydra YAML configuration, NumPy, pymoo, pyswarms, SciPy, Nevergrad, Optuna, CMA-ES.

## Global Constraints

- Work only on branch `public-vartodd`.
- Preserve the six existing optimizer-family labels and exactly two explicitly grouped programs.
- Four initial programs must expose 12--18 total mapped parameters.
- Two initial programs must expose 19--24 total mapped parameters.
- Score weights receive priority; fixed score centers do not consume dimensions.
- At least one seed uses finite early TODD and `limit_bucket=-1` terminal TODD.
- Do not extend legacy prompt tests for copied/obsolete problem directories.
- Preserve unrelated user changes in the dirty worktree.

---

### Task 1: Prompt and Search-Regime Contracts

**Files:**
- Create: `tests/problems/test_vartodd_gf_evolution_guidance.py`
- Modify: `config/algorithm/vartodd_diverse_gf16.yaml`
- Modify: `config/algorithm/vartodd_diverse_gf16_tohpe_updated.yaml`
- Modify: `problems/vartodd_evo_gf/task_description.txt`
- Modify: `problems/vartodd_evo_gf/prompts/insights/system.txt`
- Modify: `problems/vartodd_evo_gf/prompts/mutation/system.txt`

**Interfaces:**
- Consumes: shared task context injected through `{task_description}` and sampled `mutation_regime_guidance`.
- Produces: explicit capped-to-full escalation rules and score-priority diagnosis rules.

- [ ] **Step 1: Write failing guidance tests**

Add tests that load the files as text and assert:

```python
def test_regime_guidance_supports_capped_to_full_escalation():
    text = UPDATED_ALGORITHM.read_text()
    assert "no more than 10_000" not in text
    assert "limit_bucket=-1" in text
    assert "min_buckets" in text
    assert "max_buckets" in text
    assert "limit_bucket" in text


def test_task_context_separates_y_per_bucket_from_z_coverage():
    text = TASK_DESCRIPTION.read_text()
    assert "for one researched z bucket" in text
    assert "does not bound how many z buckets" in text
    assert "Low `dim` alone" in text


def test_prompts_prioritize_scores_without_hiding_generation_failure():
    combined = TASK_DESCRIPTION.read_text() + INSIGHTS_SYSTEM.read_text()
    assert "most influential mapped policy parameters" in combined
    assert "cannot select an action that was never generated" in combined


def test_mutation_prompt_requires_one_primary_causal_experiment():
    text = MUTATION_SYSTEM.read_text()
    assert "one primary causal hypothesis" in text
    assert "controls held fixed" in text
    assert "optimizer-specific evidence" in text
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest \
  tests/problems/test_vartodd_gf_evolution_guidance.py -q
```

Expected: failures for the missing escalation, y/z distinction, score-priority wording, and mutation-scope wording.

- [ ] **Step 3: Implement the prompt contracts**

Rewrite the algorithm guidance so that it:

```text
starts finite;
compares actual z research separately with min_buckets, max_buckets, and
limit_bucket;
raises a binding soft max separately from a binding hard limit;
permits limit_bucket=-1 for productive near-terminal capped paths;
controls full-search cost through schedules and the other policy/optimizer knobs.
```

Replace the mandatory “change both levels” framing with:

```text
Diagnose policy, optimizer, and interaction, then choose one primary causal
hypothesis. Change only the implicated level unless the hypothesis itself is
an interaction. State the important controls held fixed and one expected
next-run observation.
```

Add the exact y/z interpretation and score-priority rules from the approved design.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 1 pytest command. Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add tests/problems/test_vartodd_gf_evolution_guidance.py \
  config/algorithm/vartodd_diverse_gf16.yaml \
  config/algorithm/vartodd_diverse_gf16_tohpe_updated.yaml \
  problems/vartodd_evo_gf/task_description.txt \
  problems/vartodd_evo_gf/prompts/insights/system.txt \
  problems/vartodd_evo_gf/prompts/mutation/system.txt
git commit -m "fix: restore evidence-driven vartodd mutation guidance"
```

---

### Task 2: Initial-Program Dimension and Group-Flow Contracts

**Files:**
- Modify: `tests/problems/test_vartodd_gf_initial_programs.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/tohpe_weights_budget_probe.py`

**Interfaces:**
- Consumes: grouped `BaseEvaluator` methods `select_all_parameter_groups()` and `select_parameter_groups("scores")`.
- Produces: a 17-dimensional joint-TPE then score-CMA seed with no inter-stage branch.

- [ ] **Step 1: Add failing AST contracts**

Extend the AST helpers to expand constant `range(N)` comprehensions and count
`self.float_range(...)`/`self.int_range(...)` calls in all `Evaluator` policy
builder methods. Add:

```python
EXPECTED_PARAMETER_COUNTS = {
    "beam3_temp_probe.py": 18,
    "full_pso_pyswarms.py": 24,
    "lean_beam_de.py": 18,
    "lean_scout_restart.py": 18,
    "todd_hard_tail_budget_split.py": 19,
    "tohpe_weights_budget_probe.py": 17,
}


def test_initial_program_parameter_portfolio():
    assert actual_counts == EXPECTED_PARAMETER_COUNTS
    assert sum(value <= 18 for value in actual_counts.values()) == 4
    assert sum(19 <= value <= 24 for value in actual_counts.values()) == 2


def test_grouped_tohpe_jointly_optimizes_before_score_refinement():
    source = sources["tohpe_weights_budget_probe.py"]
    joint = source.index("select_all_parameter_groups()")
    score = source.index('select_parameter_groups("scores")')
    assert joint < score
    assert "set_up_new_init(" not in source
```

Update the exactly-two-grouped-program expectation so TOHPE contains only the
explicit score-subset selection; joint selection is checked separately.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest \
  tests/problems/test_vartodd_gf_initial_programs.py -q
```

Expected: dimension and TOHPE-flow failures.

- [ ] **Step 3: Repair grouped TOHPE**

Change score centers to fixed transparent literals, leaving 11 mapped score
weights. Keep the six search parameters. Rename `SEARCH_TRIALS` to
`JOINT_TRIALS=192`. Implement:

```python
evaluator.select_all_parameter_groups()
optimize_tpe(evaluator, JOINT_TRIALS, seed=21)
evaluator.select_parameter_groups("scores")
optimize_cma(evaluator, SCORE_EVALS, seed=24)
```

Do not call `set_up_new_init` between stages.

- [ ] **Step 4: Run focused tests**

Expected: TOHPE flow passes; other programs still fail their future exact dimension contracts.

- [ ] **Step 5: Commit Task 2**

```bash
git add tests/problems/test_vartodd_gf_initial_programs.py \
  problems/vartodd_evo_gf/initial_programs/tohpe_weights_budget_probe.py
git commit -m "fix: jointly optimize grouped tohpe seed"
```

---

### Task 3: Compact Four Light Initial Programs

**Files:**
- Modify: `problems/vartodd_evo_gf/initial_programs/beam3_temp_probe.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/lean_beam_de.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/lean_scout_restart.py`

**Interfaces:**
- Consumes: existing local `float_range`, `int_range`, optimizer flows, and three-source policy API.
- Produces: three additional compact 18-dimensional seeds; grouped TOHPE from Task 2 is the fourth compact seed.

- [ ] **Step 1: Confirm the exact-count tests remain RED**

Run the Task 2 pytest command and confirm failures for beam, lean DE, and lean scout.

- [ ] **Step 2: Simplify score mappings**

In all three files, retain exactly 11 mapped score weights and replace mapped
centers with fixed transparent centers. Preserve each optimizer and policy role.

- [ ] **Step 3: Remove redundant structural dimensions**

Keep only defining structural controls:

```text
beam3: 7 structural controls, total 18;
lean_beam_de: 7 structural controls, total 18;
lean_scout_restart: 7 structural controls, total 18.
```

Fix source reserves, inactive sampling budgets, and redundant coupled bucket
controls as literals. Preserve finite TODD caps for these light programs.

- [ ] **Step 4: Run focused tests**

Run the Task 2 pytest command. Expected: the four compact counts pass; medium programs still fail.

- [ ] **Step 5: Commit Task 3**

```bash
git add problems/vartodd_evo_gf/initial_programs/beam3_temp_probe.py \
  problems/vartodd_evo_gf/initial_programs/lean_beam_de.py \
  problems/vartodd_evo_gf/initial_programs/lean_scout_restart.py
git commit -m "refactor: compact vartodd light seed policies"
```

---

### Task 4: Medium and Full-Tail Initial Programs

**Files:**
- Modify: `problems/vartodd_evo_gf/initial_programs/full_pso_pyswarms.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/todd_hard_tail_budget_split.py`
- Modify: `tests/problems/test_vartodd_gf_initial_programs.py`

**Interfaces:**
- Consumes: rank schedules, parameter groups, `INITIAL_RANK`, and `TARGET_FINAL_RANK`.
- Produces: one 24-dimensional grouped staged seed and one 19-dimensional finite-early/full-terminal seed.

- [ ] **Step 1: Add/confirm the failing full-tail contract**

Add:

```python
def test_hard_tail_escalates_from_finite_to_full_todd():
    source = sources["todd_hard_tail_budget_split.py"]
    assert "EARLY_TODD_LIMIT = 512" in source
    assert "limit_bucket=EARLY_TODD_LIMIT" in source
    assert "limit_bucket=-1" in source
```

Run the focused tests. Expected: hard-tail escalation and exact dimensions fail.

- [ ] **Step 2: Simplify grouped full PSO**

Use 11 score weights, five scout-group controls, and eight tail-group controls,
for 24 total declared mapped parameters. Keep active stage sizes at 16 and 19.
Preserve PSO scout, restart, and annealing tail behavior.

- [ ] **Step 3: Build the 19-dimensional full-tail schedule**

Use 11 score weights and eight structural controls. Configure:

```python
EARLY_TODD_LIMIT = 512

early_todd = ToddSearch(
    ...,
    buckets=ZBucketSearch(
        min_buckets=8,
        max_buckets=128,
        limit_bucket=EARLY_TODD_LIMIT,
    ),
)

terminal_todd = ToddSearch(
    ...,
    buckets=ZBucketSearch(
        min_buckets=<one mapped control>,
        max_buckets=100_000,
        limit_bucket=-1,
    ),
)
```

Keep terminal TOHPEprefix finite. Raise Nevergrad budgets to
`FIRST_STAGE_BUDGET=64` and `RESTART_BUDGET=48` to match 19 dimensions.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 2 pytest command. Expected: all initial-program contracts pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add problems/vartodd_evo_gf/initial_programs/full_pso_pyswarms.py \
  problems/vartodd_evo_gf/initial_programs/todd_hard_tail_budget_split.py \
  tests/problems/test_vartodd_gf_initial_programs.py
git commit -m "refactor: add budget-aware vartodd tail seeds"
```

---

### Task 5: Verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes: all changes from Tasks 1--4.
- Produces: evidence that shared prompts and seeds satisfy current contracts.

- [ ] **Step 1: Run focused tests**

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest \
  tests/problems/test_vartodd_gf_evolution_guidance.py \
  tests/problems/test_vartodd_gf_initial_programs.py \
  tests/problems/test_vartodd_gf_parameter_groups.py \
  tests/problems/test_vartodd_gf_shared_source.py -q
```

- [ ] **Step 2: Run related VarTODD tests**

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest \
  tests/problems/test_vartodd_gf_path_names.py \
  tests/problems/test_vartodd_live_paths.py \
  tests/test_tools/test_run_gf.py -q
```

- [ ] **Step 3: Compile the six seeds**

```bash
../gigaevo-core-internal/.venv/bin/python -m compileall -q \
  problems/vartodd_evo_gf/initial_programs
```

- [ ] **Step 4: Check formatting and unintended changes**

```bash
git diff --check
git status --short
```

Confirm only scoped files and the pre-existing user changes are present.

- [ ] **Step 5: Final commit if verification required corrections**

Commit only scoped correction files with:

```bash
git commit -m "test: verify vartodd evolution recovery"
```

