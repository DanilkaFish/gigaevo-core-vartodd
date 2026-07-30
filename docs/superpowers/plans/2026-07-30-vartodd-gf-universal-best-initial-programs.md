# Universal Best VarTODD GF Initial Programs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add eight universal, diverse ab-initio VarTODD seed programs adapted from the strongest logical archetypes in `vartodd_gf16.csv`.

**Architecture:** A new passive seed bank lives under `problems/vartodd_evo_gf/initial_programs_best/`; neither the active seed directory nor `run_gf.py` changes. Each self-contained program derives rank schedules from `INITIAL_RANK` and `TARGET_FINAL_RANK`, preserves one observed search archetype, and ends with machine-readable source provenance.

**Tech Stack:** Python, NumPy, pymoo, existing VarTODD helper API, pytest, Python AST.

## Global Constraints

- Work only on the current `public-vartodd` branch.
- Do not modify `problems/vartodd_evo_gf/initial_programs/` or `run_gf.py`.
- All programs use `path_name="init"` and never load saved paths.
- Every `map_par` call supplies an explicit `group=`.
- Only pymoo optimizer implementations are used.
- TOHPEprefix is absent.
- Existing unrelated dirty-worktree changes remain untouched.

---

### Task 1: Establish the universal seed-bank contract

**Files:**
- Create: `tests/problems/test_vartodd_gf_best_initial_programs.py`
- Create: `problems/vartodd_evo_gf/initial_programs_best/de_light_terminal_todd.py`
- Create: `problems/vartodd_evo_gf/initial_programs_best/pso_restart_terminal_todd.py`
- Create: `problems/vartodd_evo_gf/initial_programs_best/pso_tohpe_only.py`
- Create: `problems/vartodd_evo_gf/initial_programs_best/pso_wide_terminal_beam.py`

**Interfaces:**
- Consumes: `helper.INITIAL_RANK`, `helper.TARGET_FINAL_RANK`, and the existing evaluator policy API.
- Produces: Four importable ab-initio programs with `entrypoint()`, `rank_from_target(float) -> int`, `rank_margin(float) -> int`, and terminal `AUX_DESCRIPTION`.

- [ ] **Step 1: Write the failing bank and universality tests**

Create a test that declares the first four expected filenames, parses each
module with `ast`, and asserts:

```python
CORE_FILES = {
    "de_light_terminal_todd.py",
    "pso_restart_terminal_todd.py",
    "pso_tohpe_only.py",
    "pso_wide_terminal_beam.py",
}


def test_core_programs_exist() -> None:
    assert CORE_FILES <= {path.name for path in PROGRAM_DIR.glob("*.py")}


def test_core_programs_are_universal_ab_initio_programs() -> None:
    for path in _program_paths(CORE_FILES):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "INITIAL_RANK" in source
        assert "TARGET_FINAL_RANK" in source
        assert 'path_name="init"' in source
        assert "TohpePrefix" not in source
        assert _function_names(tree) >= {"rank_from_target", "rank_margin", "entrypoint"}
        assert _last_assignment_name(tree) == "AUX_DESCRIPTION"
        assert _map_par_calls_without_group(tree) == []
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
.venv/bin/pytest tests/problems/test_vartodd_gf_best_initial_programs.py -v
```

Expected: FAIL because `initial_programs_best/` and the four programs do not
exist.

- [ ] **Step 3: Implement the four quality-frontier programs**

Adapt the CSV sources as follows:

- `de_light_terminal_todd.py`: source `a899151c`; single DE run, beam 5,
  TOHPE-heavy policy, TODD disabled above `rank_from_target(0.08)` and light
  below it.
- `pso_restart_terminal_todd.py`: source `e8f52447`; multi-restart PSO,
  beam 5, moderate terminal TODD below `rank_from_target(0.08)`.
- `pso_tohpe_only.py`: source `67b2f195`; multi-restart PSO and TODD disabled
  in every rank band.
- `pso_wide_terminal_beam.py`: source `e350705c`; beam 5 above
  `rank_from_target(0.10)`, beam 8 and light TODD below it.

Each program defines:

```python
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)
MAX_DEPTH = max(500, RANK_SPAN + 64)


def rank_from_target(fraction: float) -> int:
    return min(
        INITIAL_RANK,
        TARGET_FINAL_RANK + max(1, round(RANK_SPAN * fraction)),
    )


def rank_margin(fraction: float) -> int:
    return max(1, round(RANK_SPAN * fraction))
```

End each file with its complete GF16 source UUID, observed metrics, logical
role, universal adaptation, and limitation inside `AUX_DESCRIPTION`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/problems/test_vartodd_gf_best_initial_programs.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the first four programs**

```bash
git add tests/problems/test_vartodd_gf_best_initial_programs.py \
  problems/vartodd_evo_gf/initial_programs_best
git commit -m "feat: add universal best vartodd core seeds"
```

### Task 2: Add the structurally diverse programs

**Files:**
- Modify: `tests/problems/test_vartodd_gf_best_initial_programs.py`
- Create: `problems/vartodd_evo_gf/initial_programs_best/pso_chunked_light_todd.py`
- Create: `problems/vartodd_evo_gf/initial_programs_best/pso_cma_grouped_tail.py`
- Create: `problems/vartodd_evo_gf/initial_programs_best/pso_three_band_restart.py`
- Create: `problems/vartodd_evo_gf/initial_programs_best/pso_pattern_heavy_restart.py`

**Interfaces:**
- Consumes: The universal rank helper convention established in Task 1.
- Produces: Four additional programs covering chunked search, explicit parameter-group refinement, a three-band policy, and a heavy-after-restart policy.

- [ ] **Step 1: Extend the test with the final file set and archetype markers**

Set the exact expected bank to:

```python
EXPECTED_FILES = CORE_FILES | {
    "pso_chunked_light_todd.py",
    "pso_cma_grouped_tail.py",
    "pso_three_band_restart.py",
    "pso_pattern_heavy_restart.py",
}
```

Add structural assertions:

```python
ARCHETYPE_MARKERS = {
    "pso_chunked_light_todd.py": ("while consumed < evaluations", "unimproved_chunks"),
    "pso_cma_grouped_tail.py": ("CMAES", 'select_parameter_groups("scores")'),
    "pso_three_band_restart.py": ("MID_REFINE_EVALS", "set_up_new_init"),
    "pso_pattern_heavy_restart.py": ("PatternSearch", "use_heavy_todd"),
}
```

Assert the directory contains exactly `EXPECTED_FILES`, and each file contains
all of its markers. Also add the fixed-GF16-rank, optimizer-library, and
`run_gf.py` non-rewiring assertions shown in Task 3.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
.venv/bin/pytest tests/problems/test_vartodd_gf_best_initial_programs.py -v
```

Expected: FAIL because the four structural programs are missing.

- [ ] **Step 3: Implement the four structural programs**

Adapt:

- `pso_chunked_light_todd.py`: source `4e37489e`; beam 3, light TODD below
  `rank_from_target(0.12)`, chunked PSO, and stagnation-aware chunk handling.
- `pso_cma_grouped_tail.py`: source `f7aff6f1`; deterministic beam 4,
  TODD disabled above `rank_from_target(0.12)`, heavy finite TODD below it,
  and PSO all-groups → CMA-ES scores → PSO search-groups.
- `pso_three_band_restart.py`: source `86ccecc3`; policy transitions at
  `rank_from_target(0.66)` and `rank_from_target(0.13)`, beam 2, and a restart
  margin derived with `rank_margin(0.53)`.
- `pso_pattern_heavy_restart.py`: source `0d61fdac`; light grouped PSO scout,
  restart margin from `rank_margin(0.26)`, then heavy grouped TODD with
  PatternSearch refinement.

Use the same universal helper functions and terminal auxiliary-description
format as Task 1.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/problems/test_vartodd_gf_best_initial_programs.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the diverse programs**

```bash
git add tests/problems/test_vartodd_gf_best_initial_programs.py \
  problems/vartodd_evo_gf/initial_programs_best
git commit -m "feat: complete diverse universal vartodd seed bank"
```

### Task 3: Validate source safety and integration boundaries

**Files:**
- Verify: `tests/problems/test_vartodd_gf_best_initial_programs.py`
- Verify: `problems/vartodd_evo_gf/initial_programs_best/*.py`

**Interfaces:**
- Consumes: All eight completed programs.
- Produces: Verification evidence that the new directory is passive, safe,
  and independent of GF16 fixed ranks.

- [ ] **Step 1: Confirm the focused test contains the cross-cutting assertions**

Inspect the completed focused test and confirm it contains:

```python
def test_programs_do_not_embed_gf16_rank_schedule() -> None:
    forbidden = {"SWITCH_RANK = 394", "SWITCH_RANK = 398", "SWITCH_RANK = 402", "INITIAL_RANK = 567"}
    for path in _program_paths(EXPECTED_FILES):
        source = path.read_text(encoding="utf-8")
        assert forbidden.isdisjoint(source.splitlines())


def test_programs_use_only_pymoo_optimizer_imports() -> None:
    for path in _program_paths(EXPECTED_FILES):
        source = path.read_text(encoding="utf-8")
        assert "nevergrad" not in source
        assert "pyswarms" not in source
        assert "scipy.optimize" not in source


def test_active_initial_pool_is_not_rewired() -> None:
    assert '"initial_programs_best"' not in (REPO_ROOT / "run_gf.py").read_text(encoding="utf-8")
```

- [ ] **Step 2: Compile every program**

Run:

```bash
.venv/bin/python -m py_compile problems/vartodd_evo_gf/initial_programs_best/*.py
```

No source rewrite is expected unless compilation exposes an actual defect.

- [ ] **Step 3: Run focused and neighboring regression tests**

Run:

```bash
.venv/bin/pytest \
  tests/problems/test_vartodd_gf_best_initial_programs.py \
  tests/problems/test_vartodd_gf_initial_programs.py \
  tests/problems/test_vartodd_gf_shared_source.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Inspect the scoped diff**

```bash
git diff --check -- \
  tests/problems/test_vartodd_gf_best_initial_programs.py \
  problems/vartodd_evo_gf/initial_programs_best
git status --short
```

Confirm only the focused test and eight seed files belong to this
implementation; leave all unrelated entries unchanged.
