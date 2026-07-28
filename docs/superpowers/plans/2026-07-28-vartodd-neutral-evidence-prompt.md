# VarTODD Neutral Evidence Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace four prescriptive interpretation sections in the shared GF task description with neutral descriptions of optimizer mechanics and reported evidence.

**Architecture:** Only prompt text and its text-regression tests change. Existing API rules, correctness constraints, score priority, factual cost relationships, and current user-owned prompt deletions remain untouched.

**Tech Stack:** Plain text, Python 3.12, pytest.

## Global Constraints

- Work only on `public-vartodd`; do not merge into `main`.
- Preserve current uncommitted edits in `problems/vartodd_evo_gf/task_description.txt`.
- Do not weaken action API, lifecycle validity, score-priority, path-loading, or correctness rules.
- Remove response prescriptions, not factual field and optimizer descriptions.

---

### Task 1: Neutralize Evidence and Optimizer Interpretation

**Files:**
- Modify: `tests/problems/test_vartodd_gf_evolution_guidance.py`
- Modify: `problems/vartodd_evo_gf/task_description.txt`

**Interfaces:**
- Consumes: `{task_description}` injected into insights and mutation prompts.
- Produces: neutral descriptions of optimizer families, optimizer evidence, stage lifecycle, and relative rank bands.

- [ ] **Step 1: Write failing neutrality tests**

Add:

```python
def test_task_context_describes_evidence_without_prescribing_response() -> None:
    text = TASK_DESCRIPTION.read_text(encoding="utf-8")

    for biased in (
        "appropriate only around a productive basin",
        "They are poor default scouts",
        "use global exploration, then refine",
        "another optimizer cannot repair",
        "supports changing the mechanism",
        "Do not call a second identical stage a restart",
        "Typical rank regimes:",
        "Cheap sources often",
        "useful changes can target",
    ):
        assert biased not in text

    for neutral in (
        "do not uniquely identify their cause",
        "program-defined labels",
        "Passing `xopt` preserves",
        "relative positions in the reached trajectory",
        "do not imply fixed properties",
    ):
        assert neutral in text
```

- [ ] **Step 2: Run the test and verify RED**

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest -o addopts='' \
  --confcutdir=tests/problems \
  tests/problems/test_vartodd_gf_evolution_guidance.py -q
```

Expected: the new neutrality test fails on the existing prescriptive phrases
and missing neutral wording.

- [ ] **Step 3: Replace optimizer-family verdicts**

Keep the available libraries, but describe their mechanisms without role
recommendations:

```text
Optimizer families expose different proposal mechanisms:
- population methods maintain multiple candidates and update them from
  inter-candidate information;
- evolution strategies update a sampling distribution and covariance;
- model/adaptive methods use completed trials or a strategy portfolio;
- stochastic global methods generate proposals through annealing, mutation,
  or basin transitions;
- low-discrepancy designs generate space-filling initial candidates;
- local derivative-free methods construct proposals from nearby objective
  values without gradients.
```

- [ ] **Step 4: Replace optimizer-evidence prescriptions**

Use:

```text
Quantile width, repeated profiles, and improvement timing describe observed
search outcomes but do not uniquely identify their cause. The same pattern can
result from the policy mapping, optimizer settings, path, seeds, evaluation
budget, or their interaction. Comparisons are easier to attribute when the
factors not under study are held fixed.
```

Then define quantile width, profile repetition, and improvement timing without
mapping them to required responses.

- [ ] **Step 5: Replace stage/restart judgments**

Use:

```text
`scout`, `refine`, and `restart` are program-defined labels; the evaluator
does not assign them fixed semantics. `set_up_new_init` branches evaluator
state and returns the active vector or `None`. Passing `xopt` preserves the
provided coordinates; omitting it uses the branched path's stored vector.
After a branch or active-group change, construct the next optimizer problem
from the current `extract_active()` layout.
```

- [ ] **Step 6: Replace fixed rank-regime advice**

Use:

```text
Early, middle, and terminal are relative positions in the reached trajectory.
They do not imply fixed properties of `dim`, source mix, pool fill, reduction,
or z research. Read those properties from each reported group and compare them
with the policy profile active in the same rank interval.
```

- [ ] **Step 7: Run focused and related tests**

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest -o addopts='' \
  --confcutdir=tests/problems \
  tests/problems/test_vartodd_gf_evolution_guidance.py \
  tests/problems/test_vartodd_gf_initial_programs.py \
  tests/problems/test_vartodd_gf_shared_source.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit only isolated task hunks**

Stage the test file normally. Stage only the four neutral prompt hunks from
`task_description.txt`, leaving its pre-existing uncommitted edits unstaged:

```bash
git add tests/problems/test_vartodd_gf_evolution_guidance.py
git add -p problems/vartodd_evo_gf/task_description.txt
git diff --cached --check
git commit -m "docs: neutralize vartodd evidence guidance"
```

---

### Task 2: Verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes: the neutral shared task description.
- Produces: evidence that prompt contracts and seed contracts remain valid.

- [ ] **Step 1: Run the current shared-source suite**

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest -o addopts='' \
  --confcutdir=tests/problems \
  tests/problems/test_vartodd_gf_evolution_guidance.py \
  tests/problems/test_vartodd_gf_initial_programs.py \
  tests/problems/test_vartodd_gf_parameter_groups.py \
  tests/problems/test_vartodd_gf_shared_source.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Check the scoped commit**

```bash
git diff --check 6229ccc4..HEAD
git status --short --branch
```

Expected: no whitespace errors in the task commit; the user's prior
`task_description.txt` edits remain in the worktree.
