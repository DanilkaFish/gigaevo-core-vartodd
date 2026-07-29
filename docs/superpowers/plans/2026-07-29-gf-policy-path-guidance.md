# GF Policy and Path Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make shared GF evolution guidance favor evidence-derived rank schedules, useful path branching, and broader/diverse lower-region TODD action discovery.

**Architecture:** Prompt contracts define the required language before production text changes. The shared task description explains the policy mechanics, while the updated TOHPE algorithm YAML controls how often wide-margin versus saved-tail mutation regimes are presented.

**Tech Stack:** Python/pytest prompt-contract tests, plain-text task description, Hydra YAML configuration.

## Global Constraints

- Work only on branch `public-vartodd`.
- Do not name optimizer-family libraries or make numerical optimization primary.
- Preserve the complete action-policy and explicit `map_par(..., group=...)` APIs.
- Keep source-specific `z=P.../T.../A...` semantics.
- Do not introduce TOHPEprefix breadth restrictions or fixed beam/temperature guidance.

---

### Task 1: Shared Task-Description Guidance

**Files:**
- Modify: `tests/problems/test_vartodd_tohpe_updated_contract.py`
- Modify: `problems/vartodd_evo_gf/task_description.txt`

**Interfaces:**
- Consumes: execution fields `src=H/P/T`, `z=P.../T.../A...`, and `ZBucketSearch.max_buckets/limit_bucket`.
- Produces: prompt requirements for evidence-derived switches and diverse lower-region TODD action discovery.

- [ ] **Step 1: Write the failing prompt-contract assertions**

Add assertions requiring:

```python
assert "Do not use `TARGET_FINAL_RANK + 55` as a universal switch point" in description
assert "TODD may become the only source of positive actions" in description
assert "Raising TODD `max_buckets` searches more distinct z buckets" in description
assert "diversity of actions available to scoring and beam selection" in description
assert "`limit_bucket` must be large enough not to truncate that search" in description
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
../gigaevo-core-internal/.venv/bin/pytest -o addopts='' \
  --confcutdir=tests/problems -q \
  tests/problems/test_vartodd_tohpe_updated_contract.py
```

Expected: the new task-description assertions fail because the wording is absent.

- [ ] **Step 3: Implement the task-description guidance**

Replace the generic cap trade-off paragraph with the approved lower-region
TODD explanation. Replace the universal schedule example with:

```python
LOWER_REGION_OFFSET = 30  # chosen from the parent's observed rank-band boundary
LOWER_REGION_START = TARGET_FINAL_RANK + LOWER_REGION_OFFSET

self.set_todd_search(
    ranks=[INITIAL_RANK, LOWER_REGION_START],
    values=[early_todd, late_todd],
)
```

State explicitly that the offset is illustrative and must be chosen from
same-band evidence, not copied universally.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command.

Expected: all tests in `test_vartodd_tohpe_updated_contract.py` pass.

---

### Task 2: Updated TOHPE Mutation-Regime Guidance

**Files:**
- Modify: `tests/problems/test_vartodd_tohpe_updated_contract.py`
- Modify: `config/algorithm/vartodd_diverse_gf16_tohpe_updated.yaml`

**Interfaces:**
- Consumes: Hydra `mutation_operator.mutation_regime_guidance`.
- Produces: wide-margin probability `0.6`, exploitation probability `0.4`, wide margins `70..140`, and near-tail margins `15..40`.

- [ ] **Step 1: Write the failing YAML-contract assertions**

Add `import re`, parse the two `probability:` values, and assert:

```python
probabilities = re.findall(r"^    - probability: ([0-9.]+)$", updated_algorithm, re.M)
assert probabilities == ["0.6", "0.4"]
assert "margin 70..140" in updated_algorithm
assert "near_tail_paths (margin 15..40)" in updated_algorithm
assert "Raising TODD max_buckets searches more distinct z buckets" in updated_algorithm
assert "diversity of actions available to scoring and beam selection" in updated_algorithm
```

- [ ] **Step 2: Run the focused test and verify RED**

Run the Task 1 Step 2 command.

Expected: YAML probability, margin, and TODD-diversity assertions fail.

- [ ] **Step 3: Implement the YAML guidance**

Change regime probabilities from `0.4/0.6` to `0.6/0.4`, wide margins from
`50..140` to `70..140`, near-tail margins from `10..30` to `15..40`, and the
wide-margin-path range from `30..100` to `70..140`. Add the approved
lower-region TODD `max_buckets`/`limit_bucket` action-discovery and diversity
paragraph without prescribing TOHPEprefix, beamwidth, or temperature.

- [ ] **Step 4: Run relevant regression tests**

Run:

```bash
../gigaevo-core-internal/.venv/bin/pytest -o addopts='' \
  --confcutdir=tests/problems -q \
  tests/problems/test_vartodd_tohpe_updated_contract.py \
  tests/problems/test_vartodd_gf_shared_source.py \
  tests/problems/test_vartodd_gf_evolution_guidance.py \
  tests/problems/test_vartodd_gf_parameter_groups.py
git diff --check
```

Expected: all tests pass and `git diff --check` emits no errors.
