# VarTODD Island Path-Family Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each VarTODD island the approved route-specific path inventory, lineage-shared reuse limits, global-rank mid-margin guidance, and tunable non-improvement selection pressure.

**Architecture:** Extend the dedicated island `PathStore` with explicit route provenance, family derivation, atomic path reservations, and separate mid/near inventory renderers. A dedicated execution-stage subclass reserves the literal saved path before evaluation; the existing successful `record_load` call confirms it, and the parent stage finalizes or releases it. Route-aware elite selection replaces the validator's fixed seven-rank penalty with configurable effective penalties without changing stored metrics.

**Tech Stack:** Python 3.11+, pytest, Hydra/OmegaConf YAML, filesystem JSON ledger protected by `fcntl`, Python `ast`, NumPy/SciPy selection utilities.

## Global Constraints

- Work only on branch `public-vartodd` and only in the dedicated `vartodd_evo_gf_islands` pipeline except for a backward-compatible environment hook in the shared call stage.
- Ab-initio receives no saved path inventory.
- Mid-margin shows four ab-initio roots with reuse limit 6 and four mid-family representatives with family limit 8.
- Mid-margin uses uppercase matrix-wide `INITIAL_RANK`, never path-card `i<rank>` or eventual `loaded_rank`, to define its target margin span.
- Near-end exact-path reuse limit is 2 and near-family reuse limit is 7.
- Only improving children replace family representatives; non-improving children are never selectable.
- Mid-margin and near-end non-improvement selection penalties default to 12 and 16 rank units respectively.
- `timeout_salvaged` has no independent penalty.
- Reuse-limit and penalty defaults are controllable through `run_gf_islands.py` arguments.
- Preserve unrelated dirty-worktree changes and do not commit them.

---

### Task 1: Launcher options and island configuration

**Files:**
- Modify: `run_gf_islands.py`
- Modify: `config/algorithm/vartodd_diverse_gf_islands.yaml`
- Modify: `config/pipeline/vartodd_islands_pipeline.yaml`
- Test: `tests/test_tools/test_run_gf_islands.py`
- Test: `tests/integration/test_vartodd_gf_islands_flow.py`

**Interfaces:**
- Produces: `IslandPathPolicyArgs` with six integer/float fields.
- Produces Hydra keys: `mid_root_reuse_limit`, `mid_family_reuse_limit`, `near_family_reuse_limit`, `near_path_reuse_limit`, `mid_no_improvement_penalty`, and `near_no_improvement_penalty`.
- Later tasks consume those keys in the route-context provider, execution stage, and elite selectors.

- [ ] **Step 1: Write failing launcher tests**

Add behavior tests that call `build_gf_islands_overrides` with no island options and with explicit values. Assert these literal defaults/overrides are emitted:

```python
assert _value(overrides, "mid_root_reuse_limit") == "6"
assert _value(overrides, "mid_family_reuse_limit") == "8"
assert _value(overrides, "near_family_reuse_limit") == "7"
assert _value(overrides, "near_path_reuse_limit") == "2"
assert _value(overrides, "mid_no_improvement_penalty") == "12.0"
assert _value(overrides, "near_no_improvement_penalty") == "16.0"
```

Add parameterized rejection tests for zero/negative reuse limits, negative/non-finite penalties, and duplicate island launcher arguments.

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
pytest -q tests/test_tools/test_run_gf_islands.py tests/integration/test_vartodd_gf_islands_flow.py
```

Expected: failures because the six options are currently forwarded as unknown Hydra values or absent from generated overrides.

- [ ] **Step 3: Implement island-only argument parsing**

Add a frozen dataclass and parser in `run_gf_islands.py`:

```python
@dataclass(frozen=True)
class IslandPathPolicyArgs:
    mid_root_reuse_limit: int = 6
    mid_family_reuse_limit: int = 8
    near_family_reuse_limit: int = 7
    near_path_reuse_limit: int = 2
    mid_no_improvement_penalty: float = 12.0
    near_no_improvement_penalty: float = 16.0


def _split_island_policy_args(
    argv: Iterable[str],
) -> tuple[IslandPathPolicyArgs, list[str]]:
    values: dict[str, str] = {}
    forwarded: list[str] = []
    for arg in argv:
        key, separator, value = arg.partition("=")
        if separator and key in ISLAND_POLICY_ARGUMENTS:
            if key in values:
                raise ValueError(f"{key} was specified more than once")
            values[key] = value
        else:
            forwarded.append(arg)
    return _parse_island_policy_values(values), forwarded
```

The parser removes its six managed arguments before delegating remaining launcher arguments to `run_gf._split_launcher_args`, rejects duplicates, requires strictly positive integer limits, and requires finite non-negative penalties. `build_gf_islands_overrides` emits the six normalized Hydra overrides, and `_usage()` documents them.

- [ ] **Step 4: Wire Hydra defaults to consumers**

Declare the six defaults at the top of `vartodd_diverse_gf_islands.yaml`. Pass four limits to `vartodd_islands_route_context`, use the two penalties in the mid and near elite-selector configurations, and pass all four limits to the island call stage in `vartodd_islands_pipeline.yaml` once Task 4 defines it.

- [ ] **Step 5: Run launcher and composition tests and verify GREEN**

Run:

```bash
pytest -q tests/test_tools/test_run_gf_islands.py tests/integration/test_vartodd_gf_islands_flow.py tests/test_config.py
```

Expected: all selected tests pass and Hydra composes the dedicated experiment with explicit default and overridden values.

---

### Task 2: Route-aware non-improvement selection penalty

**Files:**
- Modify: `custom/archive_selectors.py`
- Modify: `config/algorithm/vartodd_diverse_gf_islands.yaml`
- Create: `tests/evolution/test_vartodd_island_penalty_selector.py`

**Interfaces:**
- Produces: `RankImprovementWeightedEliteSelector(fitness_key: str, fitness_key_higher_is_better: bool, lambda_: float, epsilon: float, normalize_fitness: bool, child_penalty: float, no_improvement_penalty: float, existing_fitness_penalty: float = 7.0)`.
- Consumes program metrics `fitness` and `rank_improved`; uses `Program.lineage.child_count` exactly like `WeightedEliteSelector`.

- [ ] **Step 1: Write failing effective-fitness tests**

Construct real `Program` values and assert the selector's public `effective_fitness(program)` returns these hand-derived values for lower-is-better fitness:

```python
improved fitness 390.25 -> 390.25
non-improved stored fitness 397.25 (base 390.25 + existing 7) with penalty 12 -> 402.25
non-improved stored fitness 397.25 with penalty 16 -> 406.25
```

Also assert negative and non-finite penalties raise `ValueError`, and that a seeded selection run gives an improving program a higher empirical selection count than an otherwise identical non-improver.

- [ ] **Step 2: Run the selector tests and verify RED**

Run:

```bash
pytest -q tests/evolution/test_vartodd_island_penalty_selector.py
```

Expected: import failure because `RankImprovementWeightedEliteSelector` does not exist.

- [ ] **Step 3: Implement the selector**

Subclass or factor the weighted selector without mutating program metrics. For a non-improver, replace the existing validator shaping with the route value:

```python
base = stored_fitness - existing_fitness_penalty
effective = base + no_improvement_penalty
```

For higher-is-better metrics, reverse the penalty direction. Preserve normalization, sigmoid weighting, lineage child penalty, epsilon, finite-value fallback, and sampling-without-replacement semantics.

- [ ] **Step 4: Configure route defaults and verify GREEN**

Use the ordinary `WeightedEliteSelector` for ab-initio and the new selector for mid-margin and near-end. Run:

```bash
pytest -q tests/evolution/test_vartodd_island_penalty_selector.py tests/evolution/test_elite_selectors.py tests/test_config.py
```

Expected: all tests pass; stored `fitness` remains unchanged and `timeout_salvaged` is not read by the selector.

---

### Task 3: Provenance, family derivation, and route inventories

**Files:**
- Modify: `problems/vartodd_evo_gf_islands/path_store.py`
- Modify: `custom/vartodd_islands_context.py`
- Test: `tests/problems/test_vartodd_gf_islands_path_store.py`
- Test: `tests/custom/test_vartodd_islands_context.py`

**Interfaces:**
- Produces: `PathSelectionLimits(mid_root=6, mid_family=8, near_family=7, near_path=2)`.
- Produces: `PathStore.register_created_path(name, *, route_id, parent_name, program_id)`.
- Produces route inventory records with `created_by_route`, `family_root`, `path_uses`, `family_uses`, `best_descendant_rank`, and `producer_start_rank`.
- `render_selectable_path_cards` consumes the limits and renders two mid sections or near family representatives.

- [ ] **Step 1: Write failing provenance and mid-inventory tests**

Build literal parent graphs containing ab-initio roots, improving mid children/grandchildren, near children, and non-improving children. Assert:

```python
mid root names == four lowest-rank ab-initio roots under uses/6
mid family names == four lowest-rank representatives under family uses/8
an improved ab-initio root remains in the root section
best_descendant_rank spans all generations
near-created and non-improving paths are absent from mid families
the rendered card contains producer_start_rank, not init_rank
```

Add a historical-provenance fixture where the child's eight-character program id matches a full id under the parent's `route_usage[route].processed_program_ids`; assert its creation route is recovered.

- [ ] **Step 2: Write failing near-family tests**

Assert that a near child and grandchild collapse to the best representative, exact path usage 2 retires a representative without falling back to an ancestor, family usage 7 retires every descendant, and a non-improving child never appears.

- [ ] **Step 3: Run path-store tests and verify RED**

Run:

```bash
pytest -q tests/problems/test_vartodd_gf_islands_path_store.py tests/custom/test_vartodd_islands_context.py
```

Expected: failures because current selection is path-local, has one common inventory, and renders the restart marker as `init_rank`.

- [ ] **Step 4: Implement provenance and family helpers**

Extend usage records with `created_by_route` and `route_families`. `register_created_path` records the producer route and assigns:

```text
mid child of non-mid parent -> mid family rooted at child
mid child of mid family member -> inherit mid family root
near child of near family member -> inherit near family root
near child without an existing near family -> family rooted at its loaded parent
```

Use parent links plus explicit provenance to derive descendants. For historical records only, infer a route when the producer id prefix in the child name uniquely matches a processed program id on the exact parent. Leave ambiguous records unknown.

- [ ] **Step 5: Implement separate mid and near inventories**

Replace the common `_selection_inventory` path with route-specific builders:

```python
def _mid_margin_inventory(limits) -> tuple[int | None, list[dict], list[dict]]
def _near_end_inventory(limits) -> tuple[int | None, list[dict]]
```

Mid roots retain four rank-sorted eligible ab-initio roots and separately retain four rank-sorted mid-family representatives. Near-end groups explicit near families, treats an unclaimed candidate as a prospective independent family, and keeps one best representative per family under both limits.

- [ ] **Step 6: Render evidence and record creation route**

`PathCardEnrichmentStage` passes the program's `mutation_regime`/target island and loaded parent name to `register_created_path`. Mid cards print two headings, route usage, family usage, best descendant rank, and `producer_start_rank`. Near cards retain detailed path evidence plus `path uses/2` and `family uses/7`.

- [ ] **Step 7: Run inventory tests and verify GREEN**

Run:

```bash
pytest -q tests/problems/test_vartodd_gf_islands_path_store.py tests/custom/test_vartodd_islands_context.py
```

Expected: all tests pass with deterministic family representatives and no stale/non-improving paths in either inventory.

---

### Task 4: Atomic reservation around program execution

**Files:**
- Modify: `custom/additional_stages.py`
- Modify: `custom/vartodd_islands_context.py`
- Modify: `problems/vartodd_evo_gf_islands/path_store.py`
- Modify: `config/pipeline/vartodd_islands_pipeline.yaml`
- Test: `tests/custom/test_vartodd_islands_context.py`
- Test: `tests/problems/test_vartodd_gf_islands_path_store.py`

**Interfaces:**
- Produces: `extract_literal_evaluator_path_name(code: str) -> str | None`.
- Produces: `PathStore.reserve_route_path(name: str, *, route_id: str, program_id: str, limits: PathSelectionLimits) -> bool`, `confirm_route_reservation(name: str, *, route_id: str, program_id: str) -> bool`, and `finalize_route_reservation(*, program_id: str) -> bool`.
- Produces: `IslandPathAwareCallProgramFunction`, a subclass of `CachedCallProgramFunction`.
- Adds backward-compatible `CachedCallProgramFunction._program_env_updates(program) -> dict[str, Any]` hook.

- [ ] **Step 1: Write failing AST extraction tests**

Assert literal extraction handles both forms and rejects ambiguous/dynamic choices:

```python
Evaluator(path_name="f390_i420_deadbeef_z10of20")
PATH = "f390_i420_deadbeef_z10of20"; Evaluator(path_name=PATH)
```

Multiple distinct evaluator paths, a computed string, or a refinement program using `"init"` must produce a pre-execution failure.

- [ ] **Step 2: Write failing reservation tests**

Using a real temporary `PathStore`, assert atomically that:

```text
six root reservations succeed and the seventh fails;
eight reservations shared across promoted mid descendants succeed and the ninth fails;
two reservations of one near path succeed and the third fails;
seven reservations distributed across near descendants succeed and the eighth fails;
an unconfirmed reservation is released;
a confirmed reservation is converted from in-flight to completed;
duplicate reserve/confirm/finalize calls for one program id are idempotent.
```

Use concurrent workers against the filesystem ledger and assert successful reservations never exceed the configured limit.

- [ ] **Step 3: Run reservation tests and verify RED**

Run:

```bash
pytest -q tests/problems/test_vartodd_gf_islands_path_store.py -k reservation
pytest -q tests/custom/test_vartodd_islands_context.py -k 'literal or reservation or call_program'
```

Expected: failures because reservation APIs and execution wrapper do not exist.

- [ ] **Step 4: Implement atomic reservation lifecycle**

Under `_locked_usage_index`, `reserve_route_path` validates route provenance and current path/family totals, increments `in_flight_count`, and stores a program-id reservation. `confirm_route_reservation` marks the reservation after successful `record_load`. `finalize_route_reservation` decrements in-flight and increments completed use only for confirmed reservations; otherwise it releases the reservation.

Override island `PathStore.record_load` to call the legacy implementation and then confirm from `GIGAEVO_PROGRAM_ID` and `GIGAEVO_MUTATION_REGIME`. Confirmation is idempotent.

- [ ] **Step 5: Implement the execution wrapper**

Add `_program_env_updates` to `CachedCallProgramFunction` and merge its result into the existing subprocess environment. `IslandPathAwareCallProgramFunction` resolves the route from program metadata, extracts the exact path, reserves it, exposes the route to the subprocess, calls the parent computation, and finalizes in `finally`. A denied or unresolvable refinement choice returns `ProgramStageResult.failure` before expensive execution.

- [ ] **Step 6: Make outcome accounting non-duplicating**

Change `record_route_result` so it attaches rank/improvement evidence to an already finalized use and never increments completed usage a second time. Keep `processed_program_ids` idempotency. Hard timeouts confirmed by `record_load` consume reuse even if validator aux is absent; pre-load failures do not.

- [ ] **Step 7: Configure and verify GREEN**

Replace only the island pipeline's `CallProgramFunction` target with `IslandPathAwareCallProgramFunction` and pass the four configured limits. Run:

```bash
pytest -q tests/problems/test_vartodd_gf_islands_path_store.py tests/custom/test_vartodd_islands_context.py tests/custom/test_additional_stages.py
```

Expected: all tests pass and concurrent reservations obey both exact-path and family caps.

---

### Task 5: Prompt correction and end-to-end composition

**Files:**
- Modify: `problems/vartodd_evo_gf_islands/prompts/islands/mid_margin.txt`
- Modify: `problems/vartodd_evo_gf_islands/prompts/islands/near_end.txt`
- Modify: `problems/vartodd_evo_gf_islands/task_description.txt`
- Modify: `tests/integration/test_vartodd_gf_islands_flow.py`
- Modify: `tests/problems/test_vartodd_gf_islands_prompts.py`

**Interfaces:**
- Consumes the two-section mid cards and detailed near family cards.
- Prompts tell the LLM how to interpret `producer_start_rank`, root/family uses, and the global `INITIAL_RANK` margin span.

- [ ] **Step 1: Add failing prompt-consumer integration tests**

Build route contexts through `VartoddIslandsRouteContextProvider` and assert behavior-visible context:

```text
ab-initio context contains no "Selectable Shared Paths";
mid context contains both inventory sections and uppercase INITIAL_RANK formula;
mid context does not call producer_start_rank the global initial rank;
near context explains exact-path and family counters and contains one representative per tree.
```

- [ ] **Step 2: Run prompt integration tests and verify RED**

Run:

```bash
pytest -q tests/problems/test_vartodd_gf_islands_prompts.py tests/integration/test_vartodd_gf_islands_flow.py
```

Expected: mid-margin still recommends lowercase `initial_rank - final_rank`, and current cards have no two-level counters.

- [ ] **Step 3: Apply minimal prompt edits**

State that mid-margin uses `INITIAL_RANK - selected_path_rank`; `producer_start_rank`, `_i<rank>`, and `loaded_rank` are execution/restart evidence rather than the global rank. Explain the two mid inventories and near-end `path uses/2` plus `family uses/7` without adding unrelated optimizer or policy biases.

- [ ] **Step 4: Run focused and full verification**

Run:

```bash
pytest -q tests/test_tools/test_run_gf_islands.py \
  tests/integration/test_vartodd_gf_islands_flow.py \
  tests/problems/test_vartodd_gf_islands_path_store.py \
  tests/problems/test_vartodd_gf_islands_prompts.py \
  tests/custom/test_vartodd_islands_context.py \
  tests/evolution/test_vartodd_island_penalty_selector.py \
  tests/test_config.py
```

Expected: all selected tests pass.

- [ ] **Step 5: Perform launcher smoke composition**

Run the launcher builder in a temporary runtime root for GF16 with explicit non-default limits and penalties. Assert the generated overlay links only island assets and the resulting override list contains the six requested values. Do not start an evolution or mutate Redis.

---

## Self-Review

- Spec coverage: all three route inventories, global-rank correction, 6/8/7/2 limits, route provenance, promotion behavior, concurrency reservations, 12/16 penalties, launcher controls, and timeout neutrality are assigned to tasks.
- Placeholder scan: no implementation step delegates an undefined behavior.
- Type consistency: launcher keys match Hydra keys; `PathSelectionLimits` is consumed by inventory and reservation APIs; route names remain `ab_initio`, `mid_margin`, and `near_end`; all counters use completed plus in-flight values.
