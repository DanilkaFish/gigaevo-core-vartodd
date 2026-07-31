# VarTODD Runtime and Z-Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make valid programs prefer productive runtime use, expose full-TODD
configured and observed z search separately, and sample one selectable live
path per final rank.

**Architecture:** The shared `vartodd_evo_gf` problem remains the source of
truth. Native policy stats expose TOHPEprefix and TODD z research separately;
`BaseEvaluator` writes only TODD research and the TODD cap in the path name.
The Live Path Store groups candidates by final rank, samples one representative
per rank, then emits distinct ranks across both selectable sections.

**Tech Stack:** Python, pytest, YAML/Hydra configuration, existing VarTODD `Path`, `PathStore`, and validation APIs.

## Global Constraints

- Work only on `public-vartodd`.
- Preserve unrelated dirty-worktree changes.
- Do not score `timeout_salvaged=1`.
- Non-improving loaded searches receive `+7`, or `+1` only when final rank is
  equal and the configured TODD cap is strictly smaller than the parent's.
- New path names are
  `f<final>_i<loaded>_<hash>_z<max_todd_observed>of<max_todd_limit>`.
- Legacy paths remain readable but cannot establish the reduced penalty.

---

### Task 1: Observed Z-Research Path Metadata

**Files:**
- Modify: `tests/problems/test_vartodd_gf_path_names.py`
- Modify: `problems/vartodd_evo_gf/helper.py`
- Modify: `problems/vartodd_evo_gf/path_store.py`

**Interfaces:**
- Produces: `BaseEvaluator._path_max_todd_z_researched(path) -> int | None`
- Produces path suffix: `_z<max|unknown>of<limit|unknown>`
- Stores optional `max_todd_z_researched` in each path metadata record.

- [ ] **Step 1: Write failing path-name tests**

Add a fake selected path whose node stats have `z_researched_todd` values
`64`, `512`, and `128`, while aggregate/prefix values are much larger. Assert
`_auto_hashed_name(...)` ends in `_z512of1024`, and assert missing
source-specific statistics produce `_zunknownof1024`.
Also assert `_loaded_path_todd_limit` still reads the cap from the extended
name and Live Path Store parses both legacy and extended names.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/problems/test_vartodd_gf_path_names.py
```

Expected: failures because names currently end after `_lim<cap>`.

- [ ] **Step 3: Implement selected-path observed research**

Walk `path.final_node.parent` links and collect only
`node.incoming.global_info.z_researched_todd`. Return the maximum integer or
`None`; never substitute aggregate research. Append it to both name branches,
save source-specific metadata, and parse both current and legacy formats.

### Task 1B: Distinct-Rank Live Path Sampling

**Files:**
- Modify: `tests/problems/test_vartodd_gf_path_names.py`
- Modify: `problems/vartodd_evo_gf/path_store.py`

- [ ] **Step 1: Write failing tests**

Assert `_select_top` samples one member from tied ranks and that a summary
never repeats a final rank across near-tail and wide-margin sections.

- [ ] **Step 2: Implement rank-group sampling**

Group eligible records by `rank`, use `random.choice` once per group, sort the
representatives, and exclude ranks selected by near-tail from wide-margin.

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
- Produces: `_child_path_todd_limit(text: str) -> int | None`
- Constants: `NO_IMPROVEMENT_PENALTY = 7.0`,
  `REDUCED_CAP_PENALTY = 1.0`.

- [ ] **Step 1: Write failing validation tests**

Use a valid matrix result and reports containing loaded and child path names.
Assert:

```text
equal rank, parent lim512, child lim511 -> base + 1
equal rank, equal/larger/unrestricted child cap -> base + 7
worse rank with smaller cap -> base + 7
missing parent or child cap -> base + 7
improved child -> base
timeout_salvaged=1 -> no additional change
```

Vary `_z` independently in at least one case to prove observed research does
not control the exception.

- [ ] **Step 2: Run the validation tests and verify RED**

Run:

```bash
pytest -q tests/problems/test_vartodd_tohpe_updated_todd_cap.py
```

Expected: failures showing the existing `+1` behavior and missing z parsers.

- [ ] **Step 3: Implement the minimal scoring rules**

Parse parent cap only from `loaded_path_name:` and child cap only from
`this path name:`. For a loaded, non-improving result, select `+1` only when
the final rank is exactly equal and the child cap is strictly smaller;
otherwise select `+7`. Delete unused shaping logic. Do not inspect or score
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
