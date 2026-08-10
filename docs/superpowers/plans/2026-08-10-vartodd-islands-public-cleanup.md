# VarTODD Islands Public Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run_gf_islands.py` the only documented VarTODD workflow and reduce `problems/` to the shared GF and island-specific source directories.

**Architecture:** Keep `problems/vartodd_evo_gf/` as the shared evaluator/seed implementation and `problems/vartodd_evo_gf_islands/` as the island-owned overlay. `run_gf.py` continues to own common launcher mechanics; `run_gf_islands.py` adds a strict islands experiment contract and remains the public CLI documented by a rewritten README.

**Tech Stack:** Python 3.12, Hydra/OmegaConf YAML configuration, pytest, Redis, native `pyvartodd`, Markdown.

## Global Constraints

- The immediate directory set under `problems/` must be exactly `vartodd_evo_gf` and `vartodd_evo_gf_islands`.
- Do not merge the two retained problem source directories.
- Document only `run_gf_islands.py`; do not advertise legacy `run_gf.py`, `vartodd_evo`, `vartodd_gf32`, or batch presets.
- Preserve the three regimes `ab_initio`, `mid_margin`, and `near_end`; do not alter search, prompt, archive, or native VarTODD logic.
- Default `call_timeout` remains `3800`; default `cache` remains `true`; default seed pool remains `default`.
- The supported experiment is exactly `vartodd_evo_gf_islands_steady`.
- Do not modify curated matrices or `evolution_results/`.
- Preserve unrelated pre-existing worktree changes.

---

### Task 1: Enforce the Islands Launcher Contract

**Files:**
- Modify: `tests/test_tools/test_run_gf.py`
- Modify: `tests/test_tools/test_run_gf_islands.py`
- Modify: `run_gf.py`
- Modify: `run_gf_islands.py`

**Interfaces:**
- Consumes: `_split_launcher_args(argv)` and existing matrix/overlay helpers from `run_gf.py`.
- Produces: `require_experiment(forwarded: list[str], required: str) -> list[str]` and islands overrides containing exactly one valid experiment.

- [ ] **Step 1: Add failing tests for default experiment injection**

Add this assertion to the existing island overlay test:

```python
assert _value(overrides, "experiment") == "vartodd_evo_gf_islands_steady"
```

Add a no-experiment case:

```python
def test_launcher_injects_islands_experiment_when_omitted(tmp_path: Path) -> None:
    launcher = _load_launcher()
    overrides = launcher.build_gf_islands_overrides(
        ["matrix=16", "lb=380", "ub=421"],
        repository_root=ROOT,
        runtime_root=tmp_path,
    )
    assert _value(overrides, "experiment") == "vartodd_evo_gf_islands_steady"
```

- [ ] **Step 2: Add failing tests for conflicting experiment and owned overrides**

Import `pytest` and add:

```python
@pytest.mark.parametrize(
    "argument",
    [
        "experiment=vartodd_evo_tohpe_updated_steady",
        "problem.name=wrong",
        "problem.dir=/tmp/wrong",
        "redis.prefix=wrong",
        "initial_exec_cache_dir=/tmp/wrong-cache",
    ],
)
def test_launcher_rejects_conflicting_owned_overrides(
    argument: str,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    with pytest.raises(ValueError):
        launcher.build_gf_islands_overrides(
            ["matrix=16", "lb=380", "ub=421", argument],
            repository_root=ROOT,
            runtime_root=tmp_path,
        )
```

Also add direct shared-parser coverage to `tests/test_tools/test_run_gf.py`:

```python
@pytest.mark.parametrize(
    "argument",
    ["problem.name=wrong", "redis.prefix=wrong"],
)
def test_build_gf_overrides_rejects_launcher_owned_overrides(
    argument: str,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    with pytest.raises(ValueError, match="managed by the GF launcher"):
        launcher.build_gf_overrides(
            ["matrix=16", "lb=380", "ub=420", argument],
            repository_root=ROOT,
            runtime_root=tmp_path,
        )
```

- [ ] **Step 3: Run focused tests and confirm the new cases fail**

Run:

```bash
direnv exec . python -m pytest tests/test_tools/test_run_gf.py tests/test_tools/test_run_gf_islands.py -q
```

Expected: failures for missing experiment injection and accepted conflicting/owned overrides.

- [ ] **Step 4: Implement strict shared argument handling**

Change `_split_launcher_args()` in `run_gf.py` so launcher-owned keys raise:

```python
elif separator and key in _RESERVED_OVERRIDES:
    raise ValueError(f"{key} is managed by the GF launcher and cannot be overridden")
else:
    forwarded.append(arg)
```

Add:

```python
def require_experiment(forwarded: list[str], required: str) -> list[str]:
    matches = [arg for arg in forwarded if arg.partition("=")[0] == "experiment"]
    if not matches:
        return [f"experiment={required}", *forwarded]
    if len(matches) > 1:
        raise ValueError("experiment was specified more than once")
    if matches[0] != f"experiment={required}":
        raise ValueError(f"experiment must be {required}")
    return forwarded
```

- [ ] **Step 5: Require the islands experiment in `run_gf_islands.py`**

Import `require_experiment`, define:

```python
ISLANDS_EXPERIMENT = "vartodd_evo_gf_islands_steady"
```

and normalize `forwarded` immediately after `_split_launcher_args()`:

```python
forwarded = require_experiment(forwarded, ISLANDS_EXPERIMENT)
```

Update `_usage()` so the main command omits the optional experiment, states the default, and retains the resume example.

- [ ] **Step 6: Run focused launcher tests**

Run the Step 3 command again. Expected: all launcher tests pass.

- [ ] **Step 7: Commit the launcher contract**

```bash
git add run_gf.py run_gf_islands.py tests/test_tools/test_run_gf.py tests/test_tools/test_run_gf_islands.py
git commit -m "feat: make islands the safe GF launcher"
```

---

### Task 2: Rewrite README Around The Islands API

**Files:**
- Create: `tests/test_readme_vartodd_islands.py`
- Replace: `README.md`

**Interfaces:**
- Consumes: the actual launcher contract and `vartodd_evo_gf_islands_steady` defaults.
- Produces: the sole public installation, API, operation, and troubleshooting guide.

- [ ] **Step 1: Add a failing documentation contract test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_readme_documents_only_the_islands_launcher() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    required = [
        "run_gf_islands.py", "vartodd_evo_gf_islands_steady",
        "matrix=", "lb=", "ub=", "cache=", "call_timeout=",
        "soft_timeout_grace=", "initial_programs=", "redis.resume=true",
        "ab_initio", "mid_margin", "near_end",
    ]
    for value in required:
        assert value in text
    forbidden = [
        "python run_gf.py", "problem.name=vartodd_evo ",
        "problem.name=vartodd_gf32", "vartodd_evo_gf16_batch",
        "vartodd_gf32_batch",
    ]
    for value in forbidden:
        assert value not in text

def test_readme_launcher_options_match_source() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    launcher = (ROOT / "run_gf_islands.py").read_text(encoding="utf-8")
    for option in (
        "matrix", "lb", "ub", "cache", "call_timeout",
        "soft_timeout_grace", "initial_programs",
    ):
        assert f"{option}=" in readme
        assert option in launcher
```

- [ ] **Step 2: Run the test and confirm the old README fails**

```bash
direnv exec . python -m pytest tests/test_readme_vartodd_islands.py -q
```

Expected: failure because old batch problems are advertised and the islands API is incomplete.

- [ ] **Step 3: Replace `README.md` with the approved structure**

Use these sections:

```markdown
# GigaEvo VarTODD Islands
## How The Three-Island Search Works
## Repository Layout
## Requirements And Installation
## Build pyvartodd
## Configure OpenRouter
## Start Redis
## Launcher API
## Run A Fresh Evolution
## Resume Or Isolate A Run
## Matrix Selection Examples
## Runtime Files And Curated Results
## Troubleshooting
```

The API table must give required/default/meaning information for all seven options. Examples must cover numeric degree, exact filename, both alternate seed pools, cache disable, custom timeouts, six-worker concurrency, one-mutant smoke run, resume, and alternate Redis DB.

- [ ] **Step 4: Run docs tests and inspect help**

```bash
direnv exec . python -m pytest tests/test_readme_vartodd_islands.py -q
direnv exec . python run_gf_islands.py --help
```

Expected: tests pass and help matches the documented launcher options/default experiment.

- [ ] **Step 5: Commit the README rewrite**

```bash
git add README.md tests/test_readme_vartodd_islands.py
git commit -m "docs: document the VarTODD islands workflow"
```

---

### Task 3: Reduce problems/ To The Two Retained Sources

**Files:**
- Create: `tests/problems/test_vartodd_problem_layout.py`
- Modify: `tests/problems/test_vartodd_live_paths.py`
- Delete: every immediate `problems/` subdirectory except the two retained directories.

**Interfaces:**
- Consumes: the approved exact directory layout.
- Produces: a filesystem invariant independent of git-ignore behavior.

- [ ] **Step 1: Add the failing layout invariant**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_only_supported_vartodd_problem_directories_remain() -> None:
    problem_dirs = {
        path.name for path in (ROOT / "problems").iterdir() if path.is_dir()
    }
    assert problem_dirs == {"vartodd_evo_gf", "vartodd_evo_gf_islands"}
```

- [ ] **Step 2: Run the invariant and confirm it reports 24 extra directories**

```bash
direnv exec . python -m pytest tests/problems/test_vartodd_problem_layout.py -q
```

- [ ] **Step 3: Retarget the old live-path regression test**

Replace both parameter lists in `tests/problems/test_vartodd_live_paths.py` with:

```python
@pytest.mark.parametrize("problem_name", ["vartodd_evo_gf"])
```

The island wrapper delegates to this implementation and already has dedicated path-store coverage.

- [ ] **Step 4: Delete the exact out-of-scope directories**

Delete only:

```text
problems/adversarial
problems/algotune
problems/alphaevolve
problems/chains
problems/dashboard
problems/heilbron
problems/hexagon_improver
problems/hexagon_pack
problems/kissing_number_11d
problems/kissing_number_12d
problems/prompt_evolution
problems/prompt_evolution_hover
problems/prompts
problems/santa2025_n100
problems/santa2025_tree_packing
problems/spherical_codes_baseline
problems/spherical_codes_improver
problems/spherical_codes_improver_v2
problems/tabular_regression
problems/tabular_regression_optuna
problems/toy_example
problems/toy_kadane
problems/vartodd_evo
problems/vartodd_gf32
```

- [ ] **Step 5: Run retained-source and layout tests**

```bash
direnv exec . python -m pytest \
  tests/problems/test_vartodd_problem_layout.py \
  tests/problems/test_vartodd_live_paths.py \
  tests/problems/test_vartodd_gf_shared_source.py \
  tests/problems/test_vartodd_gf_islands_shared_source.py \
  tests/problems/test_vartodd_gf_islands_variant.py \
  tests/problems/test_vartodd_gf_islands_path_store.py -q
```

Expected: all retained-source and layout tests pass.

- [ ] **Step 6: Commit the problem cleanup**

```bash
git add -A problems tests/problems/test_vartodd_live_paths.py tests/problems/test_vartodd_problem_layout.py
git commit -m "chore: remove unsupported problem payloads"
```

---

### Task 4: Compose And Verify The Supported Workflow

**Files:**
- Read: `tests/test_config.py`
- Read: `tests/integration/test_vartodd_gf_islands_flow.py`
- Read: `config/experiment/vartodd_evo_gf_islands_steady.yaml`
- Read: `config/pipeline/vartodd_islands_pipeline.yaml`
- Read: `config/algorithm/vartodd_diverse_gf_islands.yaml`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: fresh evidence that the public command composes and the retained workflow is consistent.

- [ ] **Step 1: Run the focused complete suite**

```bash
direnv exec . python -m pytest \
  tests/test_tools/test_run_gf.py \
  tests/test_tools/test_run_gf_islands.py \
  tests/test_readme_vartodd_islands.py \
  tests/problems/test_vartodd_problem_layout.py \
  tests/problems/test_vartodd_live_paths.py \
  tests/problems/test_vartodd_gf_shared_source.py \
  tests/problems/test_vartodd_gf_islands_shared_source.py \
  tests/problems/test_vartodd_gf_islands_variant.py \
  tests/problems/test_vartodd_gf_islands_path_store.py \
  tests/integration/test_vartodd_gf_islands_flow.py \
  tests/test_config.py -q
```

Expected: all selected tests pass. Use the configured project environment if a declared pytest plugin is unavailable.

- [ ] **Step 2: Verify help, layout, stale docs, and diff hygiene**

```bash
direnv exec . python run_gf_islands.py --help
test "$(find problems -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 2
! rg -n "python run_gf.py|problem.name=vartodd_evo |problem.name=vartodd_gf32|vartodd_evo_gf16_batch|vartodd_gf32_batch" README.md
git diff --check
git status --short
```

Expected: islands-only help, two problem directories, no stale public commands, no whitespace errors, and only approved changes plus pre-existing worktree edits. If a command fails, return to the task that owns the failing behavior and repeat that task's test/change cycle; Task 4 itself makes no file changes and creates no commit.
