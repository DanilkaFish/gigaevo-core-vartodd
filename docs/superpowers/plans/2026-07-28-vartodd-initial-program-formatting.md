# VarTODD Initial Program Formatting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce redundant comments, docstrings, blank lines, and physical line breaks in the six shared initial programs without changing their executable AST or search hyperparameters.

**Architecture:** The programs remain standalone and retain their current policy and optimizer structure. A semantic AST fingerprint strips docstrings before hashing, so comments, docstrings, and formatting may change while every executable statement and literal remains frozen.

**Tech Stack:** Python 3.12, `ast`, `hashlib`, pytest, compileall.

## Global Constraints

- Work only on `public-vartodd`; do not merge into `main`.
- Do not change any numerical or categorical hyperparameter.
- Do not change optimizer families, stages, objectives, schedules, groups, mapped parameter counts, or policy construction.
- Do not introduce shared optimizer or policy utilities.
- Preserve standalone readability; compact related expressions but keep distinct policy sections separated.
- Preserve unrelated user changes in the dirty worktree.

---

### Task 1: Add Behavior-Preservation and Source-Size Contracts

**Files:**
- Modify: `tests/problems/test_vartodd_gf_initial_programs.py`

**Interfaces:**
- Consumes: the six files under `problems/vartodd_evo_gf/initial_programs`.
- Produces: `_semantic_digest(source: str) -> str`, a complete executable-AST fingerprint that ignores docstrings and formatting.

- [ ] **Step 1: Add the semantic fingerprints**

Add `hashlib` and this characterization data:

```python
EXPECTED_SEMANTIC_DIGESTS = {
    "beam3_temp_probe.py": "8934e663ea2d06df6a75bccf10a3ae63a6951c3e8bce9c155b35348af4d356c0",
    "full_pso_pyswarms.py": "5bab187efa5747d72e21284867a385e6c8bf39354c7549b346a77f2390827d92",
    "lean_beam_de.py": "08e98cd413da509f38d1111f1292bc602d0955ad4077d022351b115d7f80e02a",
    "lean_scout_restart.py": "2688902355981ee76966f839886a5f604a19210f9f5623ebf606f85af23c5988",
    "todd_hard_tail_budget_split.py": "59f87f6e353fea4647bcd6cd1c7ee714dfb39d08b64513d4d08e40a9e240a6b1",
    "tohpe_weights_budget_probe.py": "2a90b8c13bf796d34a2d8aae029f86c4ce4fcfd321d033ebe38e06ec694d2f8b",
}


class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
        return node

    visit_Module = _strip
    visit_ClassDef = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip


def _semantic_digest(source: str) -> str:
    tree = _StripDocstrings().visit(ast.parse(source))
    semantic_source = ast.dump(tree, include_attributes=False)
    return hashlib.sha256(semantic_source.encode()).hexdigest()
```

Add:

```python
def test_initial_program_semantics_and_hyperparameters_are_frozen() -> None:
    actual = {
        name: _semantic_digest(source)
        for name, source in _program_sources().items()
    }
    assert actual == EXPECTED_SEMANTIC_DIGESTS


def test_initial_programs_are_concise() -> None:
    line_counts = {
        name: len(source.splitlines())
        for name, source in _program_sources().items()
    }
    assert sum(line_counts.values()) <= 1_050, line_counts
```

- [ ] **Step 2: Run the characterization tests**

Run:

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest -o addopts='' \
  --confcutdir=tests/problems \
  tests/problems/test_vartodd_gf_initial_programs.py -q
```

Expected: the semantic fingerprint test passes and the concise-source test
fails with a total of `1_287` lines.

---

### Task 2: Format the Six Initial Programs In Place

**Files:**
- Modify: `problems/vartodd_evo_gf/initial_programs/beam3_temp_probe.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/full_pso_pyswarms.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/lean_beam_de.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/lean_scout_restart.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/todd_hard_tail_budget_split.py`
- Modify: `problems/vartodd_evo_gf/initial_programs/tohpe_weights_budget_probe.py`
- Test: `tests/problems/test_vartodd_gf_initial_programs.py`

**Interfaces:**
- Consumes: the exact semantic fingerprints from Task 1.
- Produces: the same six executable ASTs in at most 1,050 physical lines.

- [ ] **Step 1: Format the three smaller light programs**

In `beam3_temp_probe.py`, `lean_beam_de.py`, and `lean_scout_restart.py`:

- remove module, class, range-helper, and optimizer docstrings that restate code;
- delete redundant comments and blank lines;
- keep short signatures and constructor calls on one line;
- compact small score centers, sampling budgets, bounds, and optimizer options;
- do not rename, add, remove, or reorder executable statements.

- [ ] **Step 2: Verify the light-program ASTs remain unchanged**

Run the Task 1 pytest command. Expected: only the total-line-count assertion
may still fail; the semantic fingerprint and all existing structure tests pass.

- [ ] **Step 3: Format the three staged programs**

Apply the same rules to `full_pso_pyswarms.py`,
`todd_hard_tail_budget_split.py`, and `tohpe_weights_budget_probe.py`.
Retain the few comments that explain group-stage transitions or finite-to-full
TODD escalation. Keep optimizer stages and policy sections visually separated.

- [ ] **Step 4: Run the focused tests**

Run the Task 1 pytest command. Expected: all tests pass and the reported total
line count is at most 1,050.

- [ ] **Step 5: Compile all seeds**

Run:

```bash
../gigaevo-core-internal/.venv/bin/python -m compileall -q \
  problems/vartodd_evo_gf/initial_programs
```

Expected: exit code 0 with no output.

- [ ] **Step 6: Commit the formatting**

```bash
git add tests/problems/test_vartodd_gf_initial_programs.py \
  problems/vartodd_evo_gf/initial_programs/beam3_temp_probe.py \
  problems/vartodd_evo_gf/initial_programs/full_pso_pyswarms.py \
  problems/vartodd_evo_gf/initial_programs/lean_beam_de.py \
  problems/vartodd_evo_gf/initial_programs/lean_scout_restart.py \
  problems/vartodd_evo_gf/initial_programs/todd_hard_tail_budget_split.py \
  problems/vartodd_evo_gf/initial_programs/tohpe_weights_budget_probe.py
git commit -m "refactor: simplify vartodd initial programs"
```

---

### Task 3: Final Verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes: the formatted seeds and their characterization tests.
- Produces: evidence of unchanged semantics and reduced source size.

- [ ] **Step 1: Run all current shared-source contracts**

```bash
../gigaevo-core-internal/.venv/bin/python -m pytest -o addopts='' \
  --confcutdir=tests/problems \
  tests/problems/test_vartodd_gf_evolution_guidance.py \
  tests/problems/test_vartodd_gf_initial_programs.py \
  tests/problems/test_vartodd_gf_parameter_groups.py \
  tests/problems/test_vartodd_gf_shared_source.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Report exact source-size reduction**

```bash
wc -l problems/vartodd_evo_gf/initial_programs/*.py
```

Expected: total at most 1,050, down from 1,287.

- [ ] **Step 3: Check the scoped committed diff**

```bash
git diff --check 2bc2e9af..HEAD
git status --short --branch
```

Expected: no whitespace errors in the task commits; unrelated pre-existing
worktree changes remain untouched.
