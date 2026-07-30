# VarTODD Regime Prompt Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately runnable VarTODD GF island experiment with size-aware parent mixing, regime-specific mutation evidence, evidence-rich shared path cards, and a dedicated Path Store while preserving the legacy GF experiment.

**Architecture:** The generic evolution layer carries a route, destination island, context profile, and parent roles. A configurable multi-island selector bootstraps empty refinement islands from `ab_initio` and later mixes at most one donor. A VarTODD-specific route-context provider builds filtered parent aux and unified path cards from the new problem's dedicated store; the legacy problem receives no provider and keeps its current behavior.

**Tech Stack:** Python 3.12, asyncio, Pydantic, Hydra/OmegaConf, Redis-backed MAP-Elites, pytest, YAML, existing VarTODD Python/NumPy runtime.

## Global Constraints

- Work only on branch `public-vartodd`; do not modify or merge `main`.
- Preserve unrelated dirty-worktree changes and stage only task-owned hunks.
- Keep `problems/vartodd_evo_gf`, `config/pipeline/vartodd_pipeline.yaml`, its legacy experiment, `run_gf.py`, `data_gf<matrix>`, and the legacy Redis prefix behavior unchanged.
- New source problem: `problems/vartodd_evo_gf_islands`.
- New generated problem and Redis prefix: `vartodd_evo_gf_islands<matrix>`.
- New Path Store: `data_gf_islands<matrix>/path_backups`.
- Route probabilities: `ab_initio=0.40`, `mid_margin=0.35`, `near_end=0.25`.
- Initial island: `ab_initio`.
- Bootstrap source: `ab_initio`; bootstrap applies while target size is below `8`.
- Bootstrap mixing probability: `0.70`; steady one-parent mixing probability: `0.10`.
- Migration remains disabled.
- `ab_initio` prompts contain no saved-path block or loadable path handles.
- Both refinement routes receive the same unified selectable-path inventory.
- Path context defaults: `selectable_path_top_k=8`, `path_cards_max_chars=24000`, `parent_aux_max_chars=12000`.
- Default cross-population memory is disabled; donor-local evidence enters only through an actual donor parent.

---

## File Structure

### Generic route transport

- `gigaevo/evolution/strategies/base.py`: route, selection, and parent-role values.
- `gigaevo/evolution/strategies/models.py`: validated route configuration.
- `gigaevo/evolution/strategies/route_context.py`: protocol shared by strategy and mutation prompt builder.
- `gigaevo/evolution/strategies/multi_island.py`: initial placement, bootstrap selection, and steady mixing.
- `gigaevo/evolution/mutation/base.py`: route-aware mutation interface.
- `gigaevo/evolution/mutation/constants.py`: persisted role metadata key.
- `gigaevo/evolution/engine/mutant_task.py`: selection-to-generation role transport.
- `gigaevo/evolution/engine/mutation.py`: mutation call and metadata persistence.

### Prompt assembly

- `gigaevo/evolution/mutation/mutation_operator.py`: inject provider and pass route/roles.
- `gigaevo/llm/agents/factories.py`: construct the agent with the provider.
- `gigaevo/llm/agents/mutation.py`: render assignment, role-labeled parents, filtered aux, external cards, and binding guidance.
- `custom/vartodd_islands_context.py`: VarTODD route-context provider, island statistics collector, and path-card enrichment stage.

### Separate problem

- `problems/vartodd_evo_gf_islands/variant.py`: island variant naming and dedicated data root.
- `problems/vartodd_evo_gf_islands/path_store.py`: legacy-compatible Path Store subclass with cards and unified eligibility.
- `problems/vartodd_evo_gf_islands/task_description.txt`: island prompt source.
- `problems/vartodd_evo_gf_islands/prompts/`: island-specific insight/mutation/lineage prompts.

The launcher links unchanged `helper.py`, `node.py`, `mcts_dao.py`, `todd.py`,
`full_pso.py`, `validate.py`, and initial-program directories from
`problems/vartodd_evo_gf` into the generated overlay.

### Separate configuration and launcher

- `config/pipeline/vartodd_islands_pipeline.yaml`: island-local statistics and card enrichment.
- `config/algorithm/vartodd_diverse_gf_islands.yaml`: three islands, routes, provider, and mixing.
- `config/experiment/vartodd_evo_gf_islands_steady.yaml`: separate runnable experiment.
- `run_gf_islands.py`: matrix/rank launcher with isolated names and data.

---

### Task 1: Extend the Generic Route Contract

**Files:**
- Modify: `gigaevo/evolution/strategies/base.py`
- Modify: `gigaevo/evolution/strategies/models.py`
- Create: `gigaevo/evolution/strategies/route_context.py`
- Modify: `gigaevo/evolution/mutation/constants.py`
- Test: `tests/evolution/test_strategy_base.py`
- Test: `tests/evolution/test_multi_island_extended.py`

**Interfaces:**
- Produces: `ParentRole`, `MutationRoute.context_profile`, `MutationSelection.parent_roles`.
- Produces: `MutationRouteContextProvider` protocol used by Tasks 2 and 3.
- Preserves: default non-routed `EvolutionStrategy.select_for_mutation()`.

- [ ] **Step 1: Write failing contract tests**

Add tests equivalent to:

```python
from gigaevo.evolution.strategies.base import MutationRoute, MutationSelection


def test_route_carries_context_profile() -> None:
    route = MutationRoute(
        regime_id="near_end",
        island_id="near_end",
        guidance="Load a selectable path.",
        context_profile="path_refinement",
    )
    assert route.context_profile == "path_refinement"


def test_selection_roles_align_with_parents(program_factory) -> None:
    parents = [program_factory("a"), program_factory("b")]
    selection = MutationSelection(
        parents=parents,
        route=None,
        parent_roles=("target_island", "mixed_donor"),
    )
    assert len(selection.parent_roles) == len(selection.parents)
```

Add route-config assertions that `context_profile` rejects blank strings and
defaults to `"default"` for legacy route dictionaries.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
direnv exec . pytest \
  tests/evolution/test_strategy_base.py \
  tests/evolution/test_multi_island_extended.py -q
```

Expected: failures mention missing `context_profile` and `parent_roles`.

- [ ] **Step 3: Add the route values and provider protocol**

Use these public shapes:

```python
from typing import Literal, Protocol

ParentRole = Literal["target_island", "bootstrap_donor", "mixed_donor"]


@dataclass(frozen=True, slots=True)
class MutationRoute:
    regime_id: str
    island_id: str
    guidance: str
    context_profile: str = "default"


@dataclass(frozen=True, slots=True)
class MutationSelection:
    parents: list[Program]
    route: MutationRoute | None = None
    parent_roles: tuple[ParentRole, ...] = ()
```

Create the provider protocol:

```python
class MutationRouteContextProvider(Protocol):
    def route_is_available(self, route: MutationRoute) -> bool: ...

    def build_assignment(
        self,
        route: MutationRoute,
        parents: list[Program],
        parent_roles: tuple[ParentRole, ...],
    ) -> str: ...

    def filter_parent_context(
        self,
        route: MutationRoute,
        parent: Program,
        role: ParentRole,
        mutation_context: str,
    ) -> str: ...

    def build_external_context(self, route: MutationRoute) -> str: ...
```

Add `context_profile: str = "default"` to `MutationRouteConfig` with a
whitespace-stripping validator. Add
`MUTATION_PARENT_ROLES_METADATA_KEY = "mutation_parent_roles"` to constants.

- [ ] **Step 4: Run the focused tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit only the route-contract hunks**

```bash
git add -p gigaevo/evolution/strategies/base.py \
  gigaevo/evolution/strategies/models.py \
  gigaevo/evolution/mutation/constants.py
git add gigaevo/evolution/strategies/route_context.py \
  tests/evolution/test_strategy_base.py \
  tests/evolution/test_multi_island_extended.py
git commit -m "feat: extend mutation route context contract"
```

### Task 2: Implement Initial Placement and Size-Aware Parent Mixing

**Files:**
- Modify: `gigaevo/evolution/strategies/multi_island.py`
- Test: `tests/evolution/test_multi_island_extended.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `MutationRouteContextProvider.route_is_available()`.
- Produces: `MutationSelection` with aligned parent roles.
- Produces constructor options:
  `initial_island_id`, `bootstrap_source_island`, `bootstrap_until_size`,
  `bootstrap_mix_probability`, `steady_mix_probability`,
  `route_context_provider`.

- [ ] **Step 1: Add failing selection tests**

Cover these deterministic cases by monkeypatching `random.random`,
`random.choices`, and each island's `select_elites`:

```python
async def test_empty_target_uses_two_ab_initio_donors(strategy, ab_parents):
    selection = await strategy.select_for_mutation(2)
    assert [p.id for p in selection.parents] == [p.id for p in ab_parents]
    assert selection.parent_roles == ("bootstrap_donor", "bootstrap_donor")
    assert selection.route.island_id == "mid_margin"


async def test_small_target_forces_mix_when_only_one_local(
    strategy, local_parent, ab_parent
):
    selection = await strategy.select_for_mutation(2)
    assert {p.id for p in selection.parents} == {local_parent.id, ab_parent.id}
    assert selection.parent_roles == ("target_island", "bootstrap_donor")


async def test_established_target_replaces_at_most_one_parent(
    strategy, local_parent, second_local_parent, donor
):
    selection = await strategy.select_for_mutation(2)
    assert selection.parent_roles.count("mixed_donor") == 1
    assert selection.parent_roles.count("target_island") == 1
```

Also test:

- target sizes `0`, `1`, `7`, and `8`;
- `0.70` bootstrap and `0.10` steady probability boundaries;
- uniform donor-island choice before donor elite selection;
- route exclusion when `route_is_available()` returns `False`;
- no duplicate parent IDs;
- all roots route to `initial_island_id`;
- legacy `seed_island_map` still works when `initial_island_id` is absent;
- invalid island names and probabilities fail in construction.

- [ ] **Step 2: Run the focused selector tests**

```bash
direnv exec . pytest \
  tests/evolution/test_multi_island_extended.py \
  tests/test_config.py -q
```

Expected: new selection tests fail against local-only routing.

- [ ] **Step 3: Implement validated constructor options**

Add defaults that preserve legacy behavior:

```python
initial_island_id: str | None = None
bootstrap_source_island: str | None = None
bootstrap_until_size: int = 0
bootstrap_mix_probability: float = 0.0
steady_mix_probability: float = 0.0
route_context_provider: MutationRouteContextProvider | None = None
```

Validate island references, `bootstrap_until_size >= 0`, and both probabilities
inside `[0.0, 1.0]`. Reject simultaneous `initial_island_id` and a non-empty
`seed_island_map` to avoid ambiguous root placement.

In `add()`, route an `initial_program` root to `initial_island_id` before
falling back to the legacy map/router.

- [ ] **Step 4: Implement selection helpers**

Add helpers with these responsibilities:

```python
async def _island_size(self, island_id: str) -> int:
    return int(await self.islands[island_id].__len__())


async def _select_unique(
    self,
    island_id: str,
    count: int,
    *,
    exclude_ids: set[str],
) -> list[Program]:
    candidates = await self.islands[island_id].select_elites(count + len(exclude_ids))
    result: list[Program] = []
    for candidate in candidates:
        if candidate.id in exclude_ids or any(p.id == candidate.id for p in result):
            continue
        result.append(candidate)
        if len(result) == count:
            break
    return result
```

Implement route selection in this order:

1. build immutable `MutationRoute` objects;
2. remove routes unavailable through the provider;
3. require a non-empty target or bootstrap source;
4. sample the route using configured weights;
5. apply the `0`, `1..7`, or `8+` rule;
6. preserve parent order as local first, donor second;
7. return aligned roles and record the successful route counter.

If unique parents cannot fill `total`, return the available unique parents
without duplication.

- [ ] **Step 5: Run selector and legacy strategy tests**

```bash
direnv exec . pytest \
  tests/evolution/test_multi_island_extended.py \
  tests/evolution/test_island.py \
  tests/evolution/test_strategy_base.py \
  tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the selector**

```bash
git add -p gigaevo/evolution/strategies/multi_island.py
git add tests/evolution/test_multi_island_extended.py tests/test_config.py
git commit -m "feat: add regime island bootstrap mixing"
```

### Task 3: Carry Parent Roles into Route-Aware Prompt Assembly

**Files:**
- Modify: `gigaevo/evolution/mutation/base.py`
- Modify: `gigaevo/evolution/engine/mutant_task.py`
- Modify: `gigaevo/evolution/engine/mutation.py`
- Modify: `gigaevo/evolution/mutation/mutation_operator.py`
- Modify: `gigaevo/llm/agents/factories.py`
- Modify: `gigaevo/llm/agents/mutation.py`
- Test: `tests/evolution/test_evolution_engine.py`
- Test: `tests/evolution/test_mutant_task_two_sema.py`
- Test: `tests/evolution/test_generate_mutations_exceptions.py`
- Test: `tests/evolution/test_mutation_operator.py`
- Test: `tests/llm/test_mutation_agent.py`

**Interfaces:**
- Consumes: route and roles from `MutationSelection`.
- Consumes: optional `MutationRouteContextProvider`.
- Persists: `mutation_parent_roles` alongside `mutation_regime` and
  `target_island`.
- Preserves: legacy `mutate_single()` and current generic Live Path Store
  behavior when no provider is configured.

- [ ] **Step 1: Add failing transport and prompt tests**

Test that:

```python
assert captured["route"].regime_id == "mid_margin"
assert captured["parent_roles"] == ("target_island", "bootstrap_donor")
assert child.metadata["mutation_parent_roles"] == [
    "target_island",
    "bootstrap_donor",
]
```

With a fake provider, assert prompt order:

```python
assignment_at = prompt.index("## Mutation Assignment")
parent_at = prompt.index("=== Parent 1")
external_at = prompt.index("## Selectable Shared Paths")
guidance_at = prompt.index("## Required Island Regime")
assert assignment_at < parent_at < external_at < guidance_at
```

Assert each parent header contains its role and that the provider receives the
same route/role pairing. Assert the provider is not called for legacy
`route=None` mutations.

- [ ] **Step 2: Run the focused tests and confirm failure**

```bash
direnv exec . pytest \
  tests/evolution/test_evolution_engine.py \
  tests/evolution/test_mutant_task_two_sema.py \
  tests/evolution/test_generate_mutations_exceptions.py \
  tests/evolution/test_mutation_operator.py \
  tests/llm/test_mutation_agent.py -q
```

Expected: failures show that roles and provider are not accepted.

- [ ] **Step 3: Extend the mutation operator interface**

Use this backward-compatible signature:

```python
async def mutate_with_route(
    self,
    selected_parents: list[Program],
    route: MutationRoute | None,
    parent_roles: tuple[ParentRole, ...] = (),
) -> MutationSpec | None:
    return await self.mutate_single(selected_parents)
```

Pass `selection.parent_roles` through `run_one_mutant()` and
`generate_one_mutation()`. Persist roles as a JSON list before constructing the
child `Program`.

- [ ] **Step 4: Inject and use the context provider**

Add `route_context_provider: MutationRouteContextProvider | None = None` to
`LLMMutationOperator`, `create_mutation_agent()`, and `MutationAgent`.

Add `explicit_route` and `parent_roles` to `MutationState`. For a routed prompt
with a provider:

```python
assignment = provider.build_assignment(route, parents, parent_roles)
parent_blocks = self._build_parent_blocks(
    parents,
    route=route,
    parent_roles=parent_roles,
)
external = provider.build_external_context(route)
user_prompt = "\n\n".join(
    part
    for part in (
        assignment,
        self.user_prompt_template.format(
            count=len(parents),
            parent_blocks=parent_blocks,
        ),
        external,
        route.guidance.strip(),
    )
    if part.strip()
)
```

Inside `_build_parent_blocks`, call `filter_parent_context()` and render:

```text
=== Parent 1 [role=target_island] ===
```

Validate that non-empty role tuples have exactly one entry per parent.
Retain the existing internal regime sampler and global Live Path Store builder
only when no explicit provider route is present.

- [ ] **Step 5: Run transport, cancellation, and legacy tests**

Run the Step 2 command plus:

```bash
direnv exec . pytest \
  tests/evolution/test_engine_invariants.py \
  tests/evolution/test_engine_cancellation.py \
  tests/evolution/test_engine_ghost_persist.py \
  tests/evolution/test_coalesce_refresh.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the transport changes**

Stage only task-owned hunks in pre-existing dirty files:

```bash
git add -p gigaevo/evolution/mutation/base.py \
  gigaevo/evolution/engine/mutant_task.py \
  gigaevo/evolution/engine/mutation.py \
  gigaevo/evolution/mutation/mutation_operator.py \
  gigaevo/llm/agents/factories.py \
  gigaevo/llm/agents/mutation.py
git add tests/evolution/test_evolution_engine.py \
  tests/evolution/test_mutant_task_two_sema.py \
  tests/evolution/test_generate_mutations_exceptions.py \
  tests/evolution/test_mutation_operator.py \
  tests/llm/test_mutation_agent.py
git commit -m "feat: build mutation prompts from route context"
```

### Task 4: Create the Dedicated Island Problem and Evidence Cards

**Files:**
- Create: `problems/vartodd_evo_gf_islands/variant.py`
- Create: `problems/vartodd_evo_gf_islands/path_store.py`
- Test: `tests/problems/test_vartodd_gf_islands_path_store.py`
- Test: `tests/problems/test_vartodd_gf_islands_variant.py`

**Interfaces:**
- Produces: `PathStore.has_selectable_paths()`.
- Produces: `PathStore.render_selectable_path_cards(top_k, max_chars)`.
- Produces: `PathStore.update_evidence_card(name, producer)`.
- Preserves: legacy `PathStore.save/load/record_result` behavior through
  subclassing.

- [ ] **Step 1: Write failing variant and card tests**

Use a temporary repository root and matrix file. Assert:

```python
spec = resolve_variant(tmp_path / "vartodd_evo_gf_islands16")
assert spec.data_path == tmp_path / "data_gf_islands16" / "path_backups"
```

Build three fake path records and assert:

```python
cards = store.render_selectable_path_cards(top_k=8, max_chars=24000)
assert "## Selectable Shared Paths" in cards
assert "policy_bands:" in cards
assert "policy_profiles:" in cards
assert "nonimproved-child-name" not in cards
assert store.has_selectable_paths() is True
```

Test one representative per final rank, higher starting rank on a rank tie,
lower reuse as the second tie-breaker, missing-card derivation, atomic card
updates, and deterministic budget reduction.

- [ ] **Step 2: Run the new problem tests**

```bash
direnv exec . pytest \
  tests/problems/test_vartodd_gf_islands_variant.py \
  tests/problems/test_vartodd_gf_islands_path_store.py -q
```

Expected: import failures because the new problem does not exist.

- [ ] **Step 3: Implement the island variant**

Copy the legacy resolver structure but require:

```python
match = re.fullmatch(
    r"vartodd_evo_gf_islands(?P<degree>\d+)",
    variant_dir.name,
)
```

Return:

```python
VariantSpec(
    degree=degree,
    matrix_path=matrices[0],
    data_path=repository_root
    / f"data_gf_islands{degree}"
    / "path_backups",
    target_final_rank=target_final_rank,
)
```

- [ ] **Step 4: Implement the compatible Path Store subclass**

Dynamically load the legacy source module from
`$VARTODD_REPOSITORY_ROOT/problems/vartodd_evo_gf/path_store.py`, re-export
`DATA_PATH` and `X0_LENGTH`, and subclass its `PathStore`.

Use `evidence_card.json` inside each saved path directory:

```json
{
  "version": 1,
  "path_stats": "path_summary:\n  depth=15 init_rank=433 final_rank=395 total_reduction=38",
  "producer": {
    "metrics": {},
    "runtime": null,
    "total_evals": null,
    "best_seen_times": null,
    "last_improvement": null,
    "timeout_salvaged": null
  }
}
```

Override `save()` to call the base implementation and atomically write a base
card derived from `paths[0].format_path_stats()`. Preserve an existing
`producer` object when the same path is rewritten.

`update_evidence_card()` must write through a sibling temporary file followed
by `os.replace()`.

- [ ] **Step 5: Implement unified eligibility and rendering**

`selectable_records()` must:

1. start from `iter_path_records()`;
2. retain records accepted by inherited `_is_live_record()` with
   `max_nonimproved_reuse=10`;
3. reject `child_improved_loaded is False` and
   `is_stale_improved_child`;
4. keep one record per final rank using
   `(rank ascending, init_rank descending, used_count ascending, name)`;
5. return at most `top_k`.

Render every surviving record with:

```text
### <exact name>
selection:
  rank=<rank> depth=<depth> u/i=<used>/<improved> kind=<kind> band=<band>
path_summary:
  <trajectory>
policy_bands:
  <compact group lines>
policy_profiles:
  <matching profile and deduplicated score lines>
producer_search:
  runtime=651.0 total_evals=5535 best_seen_times=63 timeout_salvaged=0
```

Reduce cards in deterministic priority order until `len(rendered) <= max_chars`.
If a complete card cannot fit, remove that path from the rendered collection;
never render a bare path name.

- [ ] **Step 6: Run the path-store tests**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 7: Commit the new problem core**

```bash
git add problems/vartodd_evo_gf_islands/variant.py \
  problems/vartodd_evo_gf_islands/path_store.py \
  tests/problems/test_vartodd_gf_islands_variant.py \
  tests/problems/test_vartodd_gf_islands_path_store.py
git commit -m "feat: add isolated vartodd island path store"
```

### Task 5: Implement VarTODD Route Context and Island-Local Pipeline Evidence

**Files:**
- Create: `custom/vartodd_islands_context.py`
- Create: `config/pipeline/vartodd_islands_pipeline.yaml`
- Test: `tests/custom/test_vartodd_islands_context.py`
- Test: `tests/problems/test_vartodd_gf_islands_pipeline.py`

**Interfaces:**
- Implements: `MutationRouteContextProvider`.
- Produces stages: `IslandEvolutionaryStatisticsCollector`,
  `PathCardEnrichmentStage`.
- Consumes: new problem `PathStore`.

- [ ] **Step 1: Add failing context-provider tests**

With a fake imported Path Store, assert:

```python
assert provider.route_is_available(ab_route) is True
assert provider.build_external_context(ab_route) == ""
assert "loaded_path_name" not in provider.filter_parent_context(
    ab_route, parent, "target_island", parent_context
)
assert "this path name" not in provider.filter_parent_context(
    ab_route, parent, "target_island", parent_context
)
assert (
    provider.build_external_context(mid_route)
    == provider.build_external_context(near_route)
)
```

Assert mid aux retains branch threshold/improvement and reopened groups. Assert
near aux keeps only the last three group lines, their profile/score blocks,
TODD z/pool fields, and search-stat plateau fields. Assert all outputs respect
`parent_aux_max_chars=12000`.

- [ ] **Step 2: Add failing stage tests**

Create programs in all three islands and assert the collector's
`archive_valid_fitnesses` contains only the focal island's values. For an
initial root without route metadata, assert it uses `ab_initio`.

For enrichment input containing:

```text
total_evals: 5535
best_seen_times: 63
timeout_salvaged: 1
this path name: f395_i433_deadbeef_z502of1024
```

assert the exact card gains runtime, metrics, evaluation counts, and salvage
state.

- [ ] **Step 3: Run context and stage tests**

```bash
direnv exec . pytest \
  tests/custom/test_vartodd_islands_context.py \
  tests/problems/test_vartodd_gf_islands_pipeline.py -q
```

Expected: import failures for the new custom module and pipeline.

- [ ] **Step 4: Implement the provider**

Load `problem_dir/path_store.py` using the established `importlib` pattern.
Implement:

```python
def route_is_available(self, route: MutationRoute) -> bool:
    if route.context_profile == "ab_initio":
        return True
    return self._path_store().has_selectable_paths()
```

`build_assignment()` must render target regime, destination, and each
`parent_N_role`. `build_external_context()` returns cards only for
`path_refinement`.

Before appending a new aux excerpt, strip the generic
`## Execution Signal Digest` and `## Program Aux Excerpt` blocks from the
stored mutation context. Extract the replacement directly from
`parent.metadata["aux_info"]`, using explicit section parsing rather than
prefix truncation.

- [ ] **Step 5: Implement island-local statistics**

Subclass `EvolutionaryStatisticsCollector`. Determine the focal island in this
order:

1. `current_island`;
2. `target_island`;
3. `ab_initio` when `source == "initial_program"`.

Fetch with `exclude=EXCLUDE_STAGE_RESULTS` so island metadata remains
available, filter programs using the same resolution rule, then call the
inherited processing logic on that filtered collection.

- [ ] **Step 6: Implement card enrichment**

Define an input model with optional `metrics`, `non_metrics`, and `runtime`
boxes. Parse the exact path name from non-metrics aux. Extract
`total_evals`, `best_seen_times`, last-improvement checkpoint, and
`timeout_salvaged`; call `PathStore.update_evidence_card()`. Return a
`StringContainer` naming the enriched path, or an empty string if no path was
saved.

- [ ] **Step 7: Create the separate pipeline**

Copy the current `vartodd_pipeline.yaml` into
`vartodd_islands_pipeline.yaml`, then make only these structural changes:

- target `custom.vartodd_islands_context.IslandEvolutionaryStatisticsCollector`
  instead of the generic collector;
- add `PathCardEnrichmentStage`;
- wire `EnsureMetricsStage -> PathCardEnrichmentStage.metrics`;
- wire `EnsureNonMetricsStage -> PathCardEnrichmentStage.non_metrics`;
- wire `ComputeTimeStage -> PathCardEnrichmentStage.runtime`;
- make enrichment run after all three upstream stages;
- keep the existing insight, lineage, digest, and mutation-context flow.

Do not add a global `MemoryContextStage`; the experiment continues to use
`memory=none`.

- [ ] **Step 8: Run context, pipeline, and current pipeline tests**

```bash
direnv exec . pytest \
  tests/custom/test_vartodd_islands_context.py \
  tests/problems/test_vartodd_gf_islands_pipeline.py \
  tests/evolution/test_mutation_context.py \
  tests/stages/test_mutation_context.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit provider and pipeline**

```bash
git add custom/vartodd_islands_context.py \
  config/pipeline/vartodd_islands_pipeline.yaml \
  tests/custom/test_vartodd_islands_context.py \
  tests/problems/test_vartodd_gf_islands_pipeline.py
git commit -m "feat: add regime-specific vartodd evidence pipeline"
```

### Task 6: Create the Island-Specific Task and Mutation Prompts

**Files:**
- Create: `problems/vartodd_evo_gf_islands/task_description.txt`
- Create: `problems/vartodd_evo_gf_islands/prompts/insights/system.txt`
- Create: `problems/vartodd_evo_gf_islands/prompts/insights/user.txt`
- Create: `problems/vartodd_evo_gf_islands/prompts/mutation/system.txt`
- Create: `problems/vartodd_evo_gf_islands/prompts/mutation/user.txt`
- Create: `problems/vartodd_evo_gf_islands/prompts/lineage/system.txt`
- Create: `problems/vartodd_evo_gf_islands/prompts/lineage/user.txt`
- Test: `tests/problems/test_vartodd_gf_islands_prompts.py`

**Interfaces:**
- Consumes the assignment, role headers, route-filtered aux, and cards from
  Tasks 3 and 5.
- Preserves the current Action API and grouped `map_par` rules.

- [ ] **Step 1: Write failing prompt-contract tests**

Assert the new mutation prompt explains:

- target-island parent is primary when present;
- a donor contributes mechanisms without changing the destination regime;
- only `Selectable Shared Paths` names are loadable;
- historical parent handles are not selection authority;
- island-local percentile drives the archetype gate;
- raw aux evidence overrides Program Insights.

Reject mentions of TOHPE-prefix controls, bucket temperature, random fraction,
or named non-Pymoo optimizer libraries.

- [ ] **Step 2: Run prompt tests**

```bash
direnv exec . pytest tests/problems/test_vartodd_gf_islands_prompts.py -q
```

Expected: missing-file failures.

- [ ] **Step 3: Create the separate prompt tree**

Mechanically copy the current GF task description and prompt tree as the API
baseline. In the new mutation system prompt, replace the multi-parent rule with
this contract:

```text
Parent roles are assigned before this call. When a target_island parent is
present, use it as the primary parent unless direct invalidity makes that
impossible. bootstrap_donor and mixed_donor parents contribute reusable
mechanisms and evidence; they do not change target_regime or
destination_island. If every parent is a bootstrap donor, choose the best
implementation base and transform it to satisfy the required regime.
```

Add evidence definitions for `Mutation Assignment` and
`Selectable Shared Paths`. State that only the current selectable block grants
permission to load a handle.

- [ ] **Step 4: Replace the task description's Live Path Store section**

Describe the new unified card format and these factual distinctions:

- path identity/selection fields choose a candidate;
- path bands and profiles show the policy associated with the saved descent;
- producer search statistics show how that artifact was obtained;
- historical aux handles are not current availability;
- `mid_margin` and `near_end` use the same inventory but different branch
  contracts;
- `ab_initio` receives no inventory.

Keep the full current Action API, mapped-parameter group API, score semantics,
runtime semantics, optimizer-landscape explanation, and TODD-specific z
interpretation.

- [ ] **Step 5: Run prompt and API contract tests**

```bash
direnv exec . pytest \
  tests/problems/test_vartodd_gf_islands_prompts.py \
  tests/problems/test_vartodd_gf_parameter_groups.py \
  tests/problems/test_vartodd_gf_initial_programs.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit the separate prompts**

```bash
git add problems/vartodd_evo_gf_islands/task_description.txt \
  problems/vartodd_evo_gf_islands/prompts \
  tests/problems/test_vartodd_gf_islands_prompts.py
git commit -m "docs: add vartodd island mutation prompts"
```

### Task 7: Add the Separate Algorithm, Experiment, and Launcher

**Files:**
- Create: `config/algorithm/vartodd_diverse_gf_islands.yaml`
- Create: `config/experiment/vartodd_evo_gf_islands_steady.yaml`
- Create: `run_gf_islands.py`
- Modify: `config/algorithm/vartodd_diverse_gf16_tohpe_updated.yaml`
- Modify: `run_gf.py`
- Test: `tests/test_config.py`
- Test: `tests/test_tools/test_run_gf_islands.py`
- Test: `tests/problems/test_vartodd_gf_islands_shared_source.py`

**Interfaces:**
- Consumes all route/provider/selector options from Tasks 1–5.
- Produces the command:
  `python run_gf_islands.py experiment=vartodd_evo_gf_islands_steady matrix=16 lb=380 ub=421`.
- Restores legacy files to their pre-island behavior.

- [ ] **Step 1: Add failing composition and launcher tests**

Assert Hydra composition yields exactly three island IDs and routes, migration
disabled, all initial programs assigned through `initial_island_id`, and the
approved probabilities/mixing values.

Assert launcher overrides contain:

```python
assert "problem.name=vartodd_evo_gf_islands16" in overrides
assert "redis.prefix=vartodd_evo_gf_islands16" in overrides
assert any("data_gf_islands16" in str(path) for path in created_paths)
```

Assert the generated overlay links runtime files and initial programs from the
legacy source but links `variant.py`, `path_store.py`, task description, and
prompts from the new source.

Add a regression snapshot showing `run_gf.py` still produces
`vartodd_evo_gf16`, the legacy Redis prefix, and `data_gf16`.

- [ ] **Step 2: Run config/launcher tests**

```bash
direnv exec . pytest \
  tests/test_config.py \
  tests/test_tools/test_run_gf_islands.py \
  tests/problems/test_vartodd_gf_islands_shared_source.py -q
```

Expected: failures for missing new configs and launcher.

- [ ] **Step 3: Create the island algorithm**

Define three independent behavior spaces and three `IslandConfig`s with IDs
`ab_initio`, `mid_margin`, and `near_end`. Reuse the current GF selectors and
`lambda_=5.0`.

Configure one shared provider:

```yaml
vartodd_islands_route_context:
  _target_: custom.vartodd_islands_context.VartoddIslandsRouteContextProvider
  problem_dir: ${problem.dir}
  selectable_path_top_k: 8
  path_cards_max_chars: 24000
  parent_aux_max_chars: 12000
```

Pass the same referenced provider to `mutation_operator` and
`evolution_strategy`. Disable the legacy internal regime sampler with an empty
guidance list and probability `0.0`.

Configure:

```yaml
initial_island_id: ab_initio
bootstrap_source_island: ab_initio
bootstrap_until_size: 8
bootstrap_mix_probability: 0.70
steady_mix_probability: 0.10
enable_migration: false
```

Set route `context_profile` to `ab_initio` for the builder and
`path_refinement` for both refinement routes. Use the approved binding
guidance and route probabilities.

- [ ] **Step 4: Create the experiment**

Inherit `base`, then override:

- pipeline to `vartodd_islands_pipeline`;
- algorithm to `vartodd_diverse_gf_islands`;
- LLM to `openrouter_vartodd_evolution`;
- evolution to `steady_state`;
- memory to `none`.

Retain strict in-flight behavior and the current timeout controls without
changing the legacy experiment.

- [ ] **Step 5: Implement the separate launcher**

Reuse parsing and metrics helpers from `run_gf.py`, but use:

```python
SOURCE_PROBLEM_NAME = "vartodd_evo_gf_islands"
LEGACY_SOURCE_PROBLEM_NAME = "vartodd_evo_gf"
VARIANT_PREFIX = "vartodd_evo_gf_islands"
DATA_PREFIX = "data_gf_islands"
```

Build overlays under `.run_gf/islands_overlays` and caches under
`.run_gf/islands_cache`. Link island-specific assets from the new source and
unchanged runtime/initial-program assets from the legacy source. Create only
`data_gf_islands<matrix>/path_backups`.

The help text must show a fresh Redis DB for the first run and
`redis.resume=true` only for later runs with the same three-island topology.

- [ ] **Step 6: Restore legacy configuration surfaces**

Move the uncommitted three-island algorithm content into the new algorithm,
then restore `vartodd_diverse_gf16_tohpe_updated.yaml` to its preceding
single-island content. Remove the island-specific note previously added to
`run_gf.py` help. Do not restore or overwrite unrelated user edits.

- [ ] **Step 7: Run composition and launcher tests**

Run the Step 2 command. Expected: PASS.

Run dry composition:

```bash
MPLCONFIGDIR=/tmp OPENAI_API_KEY=smoke direnv exec . python run.py \
  experiment=vartodd_evo_gf_islands_steady \
  problem.name=vartodd_evo_gf_islands16 \
  problem.dir=/tmp/vartodd_evo_gf_islands16 \
  --cfg job
```

Expected: three route/island IDs, no migration, no legacy regime sampler.

- [ ] **Step 8: Commit the runnable experiment**

```bash
git add config/algorithm/vartodd_diverse_gf_islands.yaml \
  config/experiment/vartodd_evo_gf_islands_steady.yaml \
  run_gf_islands.py \
  tests/test_tools/test_run_gf_islands.py \
  tests/problems/test_vartodd_gf_islands_shared_source.py
git add -p config/algorithm/vartodd_diverse_gf16_tohpe_updated.yaml run_gf.py \
  tests/test_config.py
git commit -m "feat: add isolated vartodd regime island experiment"
```

### Task 8: End-to-End Verification and Operational Documentation

**Files:**
- Create: `tests/integration/test_vartodd_gf_islands_flow.py`
- Modify: `docs/superpowers/specs/2026-07-30-vartodd-regime-prompt-context-design.md`

**Interfaces:**
- Verifies the complete selection → prompt → child metadata → destination
  island flow.
- Documents the final launch and resume commands.

- [ ] **Step 1: Write an integration test with fakes**

Use in-memory/fake storage, fake islands, a temporary Path Store, and a capture
mutator. Verify:

1. initial programs enter `ab_initio`;
2. an available `mid_margin` route with an empty target uses two donors;
3. the prompt contains the assignment, donor roles, cards, and only one regime
   block;
4. the persisted child contains `mutation_regime=mid_margin`,
   `target_island=mid_margin`, and two donor roles;
5. strategy ingestion sends the child to `mid_margin`;
6. the next small-island mutation can select one local plus one `ab_initio`
   donor.

- [ ] **Step 2: Run the integration test**

```bash
direnv exec . pytest tests/integration/test_vartodd_gf_islands_flow.py -q
```

Expected: PASS.

- [ ] **Step 3: Run all focused regression suites**

```bash
direnv exec . pytest \
  tests/evolution/test_strategy_base.py \
  tests/evolution/test_multi_island_extended.py \
  tests/evolution/test_island.py \
  tests/evolution/test_evolution_engine.py \
  tests/evolution/test_mutant_task_two_sema.py \
  tests/evolution/test_generate_mutations_exceptions.py \
  tests/evolution/test_mutation_operator.py \
  tests/evolution/test_engine_invariants.py \
  tests/evolution/test_engine_cancellation.py \
  tests/evolution/test_engine_ghost_persist.py \
  tests/evolution/test_coalesce_refresh.py \
  tests/llm/test_mutation_agent.py \
  tests/custom/test_vartodd_islands_context.py \
  tests/problems/test_vartodd_gf_islands_variant.py \
  tests/problems/test_vartodd_gf_islands_path_store.py \
  tests/problems/test_vartodd_gf_islands_pipeline.py \
  tests/problems/test_vartodd_gf_islands_prompts.py \
  tests/problems/test_vartodd_gf_islands_shared_source.py \
  tests/test_tools/test_run_gf_islands.py \
  tests/integration/test_vartodd_gf_islands_flow.py \
  tests/test_config.py -q
```

Expected: PASS with only pre-existing intentional skips.

- [ ] **Step 4: Run static verification**

```bash
direnv exec . ruff check \
  gigaevo/evolution/strategies/base.py \
  gigaevo/evolution/strategies/models.py \
  gigaevo/evolution/strategies/route_context.py \
  gigaevo/evolution/strategies/multi_island.py \
  gigaevo/evolution/mutation/base.py \
  gigaevo/evolution/engine/mutant_task.py \
  gigaevo/evolution/engine/mutation.py \
  gigaevo/evolution/mutation/mutation_operator.py \
  gigaevo/llm/agents/factories.py \
  gigaevo/llm/agents/mutation.py \
  custom/vartodd_islands_context.py \
  problems/vartodd_evo_gf_islands \
  run_gf_islands.py \
  tests/custom/test_vartodd_islands_context.py \
  tests/problems/test_vartodd_gf_islands_path_store.py \
  tests/integration/test_vartodd_gf_islands_flow.py

direnv exec . python -m compileall \
  gigaevo/evolution \
  gigaevo/llm/agents \
  custom/vartodd_islands_context.py \
  problems/vartodd_evo_gf_islands \
  run_gf_islands.py

git diff --check
```

Expected: all commands succeed.

- [ ] **Step 5: Document exact commands**

Append:

```bash
python run_gf_islands.py \
  experiment=vartodd_evo_gf_islands_steady \
  matrix=16 lb=380 ub=421 \
  cache=false call_timeout=2800 \
  max_concurrent_dags=12 max_in_flight=12 \
  runner_config.prefetch_factor=1 \
  redis.db=6
```

For a later resume:

```bash
python run_gf_islands.py \
  experiment=vartodd_evo_gf_islands_steady \
  matrix=16 lb=380 ub=421 \
  cache=false call_timeout=2800 \
  max_concurrent_dags=12 max_in_flight=12 \
  runner_config.prefetch_factor=1 \
  redis.db=6 redis.resume=true
```

State that the new run writes paths only to
`data_gf_islands16/path_backups`.

- [ ] **Step 6: Commit verification and run instructions**

```bash
git add tests/integration/test_vartodd_gf_islands_flow.py
git add -p docs/superpowers/specs/2026-07-30-vartodd-regime-prompt-context-design.md
git commit -m "test: verify vartodd regime island flow"
```
