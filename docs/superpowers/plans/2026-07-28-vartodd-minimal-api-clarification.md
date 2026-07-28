# Vartodd Minimal API Clarification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add minimal failure-driven API clarification to the shared vartodd task description.

**Architecture:** Extend the existing grouped-parameter and rank-schedule paragraphs in place. Protect the compact contract with source-level prompt tests; do not change evaluator code or reorganize the prompt.

**Tech Stack:** Plain-text task prompt, Python, pytest.

## Global Constraints

- Require explicit `group=` on every `map_par` call.
- Require stable group names, declaration order, and complete layout across reinitialization and restarts.
- Keep schedules in `set_*` methods and configurations in constructors.
- Show one valid schedule and two invalid schedule forms.
- Preserve every unrelated worktree change to `task_description.txt`.

---

### Task 1: Add a failing API-clarity contract

**Files:**
- Modify: `tests/problems/test_vartodd_gf_evolution_guidance.py`

**Interfaces:**
- Consumes: `TASK_DESCRIPTION`.
- Produces: `test_task_context_clarifies_groups_and_schedule_setters`.

- [ ] **Step 1: Write the focused failing test**

Assert the prompt contains these rules and examples:

```python
assert "Every `map_par` call must pass `group=` explicitly" in text
assert "must already have been declared" in text
assert "complete parameter layout must remain identical" in text
assert "constructors describe one configuration" in text
assert "Never mix a positional configuration with `ranks=` or `values=`" in text
assert "ActionPool(ranks=" in text
assert "set_action_pool(early_pool, ranks=" in text
assert text.count("### Rank schedules") == 1
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest -o addopts='' \
  --confcutdir=tests/problems \
  tests/problems/test_vartodd_gf_evolution_guidance.py::test_task_context_clarifies_groups_and_schedule_setters -q
```

Expected: FAIL because the explicit contract is absent.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/problems/test_vartodd_gf_evolution_guidance.py
git commit -m "test: specify vartodd API clarification"
```

### Task 2: Clarify the prompt in place

**Files:**
- Modify: `problems/vartodd_evo_gf/task_description.txt`

**Interfaces:**
- Consumes: existing grouped-parameter and rank-schedule documentation.
- Produces: compact explicit rules and valid/invalid schedule examples.

- [ ] **Step 1: Tighten the grouped-parameter paragraph**

Replace the implicit-default wording with an explicit rule:

```text
Every `map_par` call must pass `group=` explicitly, including
`group="default"` when that name is intentional. A name passed to
`select_parameter_groups(...)` must already have been declared by at least one
`map_par(..., group="<name>")` call.
```

Add one sentence stating that group names, parameter declaration order, and
the complete parameter layout must remain identical across every `reinit()`
and restart; policy branches must not conditionally add or remove `map_par`
calls.

- [ ] **Step 2: Add the minimal schedule contract**

Immediately before the existing schedule explanation, state that constructors
describe one configuration and only setters apply schedules. Keep the existing
valid `set_todd_search(ranks=..., values=...)` example and add:

```python
# Invalid: constructors do not accept schedules.
ActionPool(ranks=[INITIAL_RANK, SWITCH_RANK], values=[early_pool, late_pool])

# Invalid: do not mix setter forms.
self.set_action_pool(early_pool, ranks=[SWITCH_RANK], values=[late_pool])
```

State: `Never mix a positional configuration with ranks= or values=`.

- [ ] **Step 3: Run focused and shared prompt tests**

Run:

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest -o addopts='' \
  --confcutdir=tests/problems \
  tests/problems/test_vartodd_gf_evolution_guidance.py \
  tests/problems/test_vartodd_gf_shared_source.py -q
```

Expected: all tests PASS.

- [ ] **Step 4: Isolate prompt hunks and commit**

Use interactive staging for `task_description.txt`. Confirm the staged prompt
diff contains only the grouped-parameter and schedule clarification, then run:

```bash
git diff --cached --check
git commit -m "docs: clarify vartodd grouped and scheduled APIs"
```

