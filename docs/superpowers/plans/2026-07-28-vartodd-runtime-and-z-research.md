# VarTODD Runtime and Z-Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make valid programs prefer productive runtime use, strongly penalize loaded-path non-improvement except for cheaper observed z-research experiments, and expose configured and observed z search separately in saved-path names.

**Architecture:** The shared `vartodd_evo_gf` problem remains the source of truth. `BaseEvaluator` derives observed z-research from the selected best path and writes it beside the configured TODD cap in the path name; validation compares child and parent name metadata. Metrics and prompts then express runtime and TODD trade-offs without prescribing a specific mechanism.

**Tech Stack:** Python, pytest, YAML/Hydra configuration, existing VarTODD `Path`, `PathStore`, and validation APIs.

## Global Constraints

- Work only on `public-vartodd`.
- Preserve unrelated dirty-worktree changes.
- Do not score `timeout_salvaged=1`.
- Non-improving loaded searches receive `+7`, or `+1` only when their maximum observed z-research is strictly smaller than the parent's.
- Compare observed z-research, never configured cap, for the reduced penalty.
- New path names are `f<final>_i<loaded>_<hash>_lim<cap>_z<max>`.
- Legacy paths remain readable but cannot establish the reduced penalty.

---

### Task 1: Observed Z-Research Path Metadata

**Files:**
- Modify: `tests/problems/test_vartodd_gf_path_names.py`
- Modify: `problems/vartodd_evo_gf/helper.py`
- Modify: `problems/vartodd_evo_gf/path_store.py`

**Interfaces:**
- Produces: `BaseEvaluator._path_max_z_researched(path) -> int | None`
- Produces path suffix: `_lim<cap>_z<max|unknown>`
- Stores optional `max_z_researched` in each path metadata record.

- [ ] **Step 1: Write failing path-name tests**

Add a fake selected path whose node chain has `incoming.total` values `64`,
`512`, and `128`. Assert `_auto_hashed_name(...)` ends in
`_lim1024_z512`, and assert missing incoming statistics produce `_zunknown`.
Also assert `_loaded_path_todd_limit` still reads the cap from the extended
name and Live Path Store parses both legacy and extended names.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/problems/test_vartodd_gf_path_names.py
```

Expected: failures because names currently end after `_lim<cap>`.

- [ ] **Step 3: Implement selected-path observed research**

Walk `path.final_node.parent` links and collect `node.incoming.total` only when
the attribute exists. Return the maximum integer or `None`. Append the value to
both program-ID and hash-based names. Save it as optional path metadata and
extend the short-name regex without making `_z` mandatory.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
pytest -q tests/problems/test_vartodd_gf_path_names.py
```

Expected: all tests pass.

### Task 2: Non-Improvement Fitness Shaping

**Files:**
- Modify: `tests/problems/test_vartodd_tohpe_updated_todd_cap.py`
- Modify: `problems/vartodd_evo_gf/validate.py`

**Interfaces:**
- Produces: `_loaded_path_max_z_researched(text: str) -> int | None`
- Produces: `_child_max_z_researched(text: str) -> int | None`
- Constants: `NO_IMPROVEMENT_PENALTY = 7.0`,
  `REDUCED_Z_RESEARCH_PENALTY = 1.0`.

- [ ] **Step 1: Write failing validation tests**

Use a valid matrix result and reports containing loaded and child path names.
Assert:

```text
parent z512, child z511, no improvement -> base + 1
parent z512, child z512, no improvement -> base + 7
parent z512, child z1024, no improvement -> base + 7
legacy/missing parent or child z -> base + 7
improved child -> base
timeout_salvaged=1 -> no additional change
```

Vary `_lim` independently in at least one case to prove cap does not control
the exception.

- [ ] **Step 2: Run the validation tests and verify RED**

Run:

```bash
pytest -q tests/problems/test_vartodd_tohpe_updated_todd_cap.py
```

Expected: failures showing the existing `+1` behavior and missing z parsers.

- [ ] **Step 3: Implement the minimal scoring rules**

Parse parent observed z only from `loaded_path_name:` and child observed z only
from `this path name:`. For a loaded, non-improving result, select `+1` only
when both values exist and `child_z < parent_z`; otherwise select `+7`.
Delete unused low-limit and over-limit shaping logic. Do not inspect or score
`timeout_salvaged`.

- [ ] **Step 4: Run the validation tests and verify GREEN**

Run:

```bash
pytest -q tests/problems/test_vartodd_tohpe_updated_todd_cap.py
```

Expected: all tests pass.

### Task 3: Runtime Metric and Neutral Guidance

**Files:**
- Modify: `tests/problems/test_vartodd_tohpe_updated_contract.py`
- Modify: `problems/vartodd_evo_gf/metrics_template.yaml`
- Modify: `problems/vartodd_evo_gf/task_description.txt`
- Modify: `config/algorithm/vartodd_diverse_gf16.yaml`
- Modify: `config/algorithm/vartodd_diverse_gf16_tohpe_updated.yaml`

**Interfaces:**
- Runtime remains non-primary with `higher_is_better: true`.
- Prompt names document `_lim` as configured cap and `_z` as observed maximum.

- [ ] **Step 1: Write failing contract assertions**

Assert the template marks runtime higher-is-better, prompt text contains the
extended name format and the `+7`/`+1` distinction, and algorithm guidance
does not claim a low-limit fitness reward.

- [ ] **Step 2: Run the contract tests and verify RED**

Run:

```bash
pytest -q tests/problems/test_vartodd_tohpe_updated_contract.py
```

Expected: failures on current runtime direction and obsolete guidance.

- [ ] **Step 3: Apply minimal metric and prompt changes**

Set runtime direction to higher-is-better. State that useful remaining time
can fund distinct evaluations or stages, while routine salvage indicates
imperfect budgeting but has no direct score. Explain the cap/recall/runtime
trade-off neutrally and remove the false reward claim.

- [ ] **Step 4: Run contract tests and verify GREEN**

Run:

```bash
pytest -q tests/problems/test_vartodd_tohpe_updated_contract.py
```

Expected: all tests pass.

### Task 4: Regression Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run all relevant problem tests**

```bash
pytest -q \
  tests/problems/test_vartodd_gf_path_names.py \
  tests/problems/test_vartodd_tohpe_updated_todd_cap.py \
  tests/problems/test_vartodd_tohpe_updated_contract.py \
  tests/problems/test_vartodd_gf_shared_source.py \
  tests/problems/test_vartodd_live_paths.py \
  tests/test_tools/test_run_gf.py
```

Expected: all pass.

- [ ] **Step 2: Run syntax and diff checks**

```bash
python -m py_compile \
  problems/vartodd_evo_gf/helper.py \
  problems/vartodd_evo_gf/path_store.py \
  problems/vartodd_evo_gf/validate.py
git diff --check
```

Expected: both commands exit zero.

- [ ] **Step 3: Review the final diff**

Confirm only the approved scoring, naming, metric, prompt, and test behavior
was changed, with all unrelated working-tree modifications preserved.
