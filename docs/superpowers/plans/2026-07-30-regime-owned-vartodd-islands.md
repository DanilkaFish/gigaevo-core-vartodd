# Regime-Owned VarTODD Islands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three isolated VarTODD parent populations whose regime is sampled before parent selection, whose child returns to the producing island, and which share the existing matrix-specific Path Store.

**Architecture:** Add a generic `MutationSelection`/`MutationRoute` contract between evolution strategies and the engine. `MapElitesMultiIsland` samples one configured route, selects every parent from that route's island, and the engine carries the route through the LLM mutation call into persisted child metadata. Routed children and configured initial roots are inserted deterministically into their assigned islands; migration remains disabled while the Path Store remains global.

**Tech Stack:** Python 3.12, asyncio, Pydantic, Hydra/OmegaConf YAML configuration, Redis-backed program and archive storage, pytest.

## Global Constraints

- Work only on the `public-vartodd` branch.
- Do not merge or modify `main`.
- The shared Path Store and matrix-specific `DATA_PATH` behavior must remain unchanged.
- Routed islands share saved paths but never parents, children, or migrants.
- A route is sampled before its parents.
- A routed child is inserted into the route's island; child code and metrics never determine its island.
- Existing non-routed algorithms must retain their current behavior.
- An old single-island Redis archive is not automatically converted to the three-island topology.
- Use `apply_patch` for source and documentation edits.

---

### Task 1: Add the Generic Mutation-Selection Contract

**Files:**
- Modify: `gigaevo/evolution/strategies/base.py`
- Test: `tests/evolution/test_strategy_base.py`

**Interfaces:**
- Produces: `MutationRoute(regime_id: str, island_id: str, guidance: str)`
- Produces: `MutationSelection(parents: list[Program], route: MutationRoute | None = None)`
- Produces: `EvolutionStrategy.select_for_mutation(total: int) -> MutationSelection`
- Consumes: existing `EvolutionStrategy.select_elites(total)`

- [ ] **Step 1: Write tests for immutable route/selection values and the legacy default**

Add tests equivalent to:

```python
from gigaevo.evolution.strategies.base import (
    EvolutionStrategy,
    MutationRoute,
    MutationSelection,
)


async def test_default_select_for_mutation_wraps_legacy_elites():
    parents = [Program(code="def solve(): return 1")]
    strategy = StubStrategy(parents)

    selection = await strategy.select_for_mutation(total=1)

    assert selection == MutationSelection(parents=parents, route=None)
    assert strategy.selected_total == 1


def test_mutation_route_is_immutable():
    route = MutationRoute(
        regime_id="ab_initio",
        island_id="ab_initio",
        guidance="Build from path_name='init'.",
    )
    with pytest.raises(FrozenInstanceError):
        route.island_id = "near_end"
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
pytest tests/evolution/test_strategy_base.py -q
```

Expected: import or attribute failures for `MutationRoute`,
`MutationSelection`, and `select_for_mutation`.

- [ ] **Step 3: Add the immutable values and default strategy method**

Implement in `base.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MutationRoute:
    regime_id: str
    island_id: str
    guidance: str


@dataclass(frozen=True, slots=True)
class MutationSelection:
    parents: list[Program]
    route: MutationRoute | None = None


class EvolutionStrategy(ABC):
    async def select_for_mutation(self, total: int) -> MutationSelection:
        return MutationSelection(
            parents=await self.select_elites(total),
            route=None,
        )
```

Do not change the existing abstract `select_elites` contract.

- [ ] **Step 4: Run the strategy contract tests**

Run:

```bash
pytest tests/evolution/test_strategy_base.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the contract**

```bash
git add gigaevo/evolution/strategies/base.py tests/evolution/test_strategy_base.py
git commit -m "feat: add routed mutation selection contract"
```

---

### Task 2: Add Configurable Route Sampling to Multi-Island MAP-Elites

**Files:**
- Modify: `gigaevo/evolution/strategies/models.py`
- Modify: `gigaevo/evolution/strategies/multi_island.py`
- Test: `tests/evolution/test_multi_island_extended.py`

**Interfaces:**
- Consumes: `MutationRoute` and `MutationSelection` from Task 1
- Produces: `MutationRouteConfig`
- Produces: constructor arguments
  `mutation_routes: list[MutationRouteConfig] | None` and
  `seed_island_map: dict[str, str] | None` on `MapElitesMultiIsland`
- Produces: route-aware `MapElitesMultiIsland.select_for_mutation(total)`

- [ ] **Step 1: Write route-configuration validation tests**

Cover:

```python
def test_routes_require_unique_regime_ids():
    with pytest.raises(ValueError, match="duplicate regime_id"):
        _make_multi_island(
            mutation_routes=[
                _route("builder", "island_0", 1.0),
                _route("builder", "island_1", 1.0),
            ]
        )


def test_route_must_reference_existing_island():
    with pytest.raises(ValueError, match="unknown island"):
        _make_multi_island(
            mutation_routes=[_route("builder", "missing", 1.0)]
        )


def test_routed_topology_rejects_enabled_migration():
    with pytest.raises(ValueError, match="migration must be disabled"):
        _make_multi_island(
            mutation_routes=[_route("builder", "island_0", 1.0)],
            enable_migration=True,
        )
```

Also verify Pydantic rejects blank IDs/guidance and non-positive probability.

- [ ] **Step 2: Write route-first parent-selection tests**

Add deterministic tests by patching `random.choices`:

```python
async def test_select_for_mutation_uses_only_sampled_route_island(monkeypatch):
    multi, islands, _ = _make_multi_island(
        n=3,
        enable_migration=False,
        mutation_routes=[
            _route("ab_initio", "island_0", 0.4),
            _route("mid_margin", "island_1", 0.35),
            _route("near_end", "island_2", 0.25),
        ],
    )
    parent_a, parent_b = _prog("island_1"), _prog("island_1")
    islands["island_0"].__len__ = AsyncMock(return_value=2)
    islands["island_1"].__len__ = AsyncMock(return_value=2)
    islands["island_2"].__len__ = AsyncMock(return_value=2)
    islands["island_1"].select_elites = AsyncMock(
        return_value=[parent_a, parent_b]
    )
    monkeypatch.setattr(
        "gigaevo.evolution.strategies.multi_island.random.choices",
        lambda routes, weights, k: [routes[1]],
    )

    selection = await multi.select_for_mutation(total=2)

    assert selection.route.regime_id == "mid_margin"
    assert selection.route.island_id == "island_1"
    assert selection.parents == [parent_a, parent_b]
    islands["island_0"].select_elites.assert_not_called()
    islands["island_2"].select_elites.assert_not_called()
```

Add cases for:

- empty islands being excluded before sampling;
- all islands empty returning `MutationSelection([], None)`;
- one available parent being returned without borrowing a second parent;
- a parent whose `current_island` differs from the route causing a
  `RuntimeError`;
- configurations without routes delegating to the legacy selection behavior.

- [ ] **Step 3: Run the multi-island tests and verify failure**

Run:

```bash
pytest tests/evolution/test_multi_island_extended.py -q
```

Expected: failures for the missing config type, constructor fields, and routed
selection method.

- [ ] **Step 4: Add `MutationRouteConfig`**

Implement in `models.py`:

```python
class MutationRouteConfig(BaseModel):
    regime_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    island_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    probability: float = Field(gt=0.0)
    guidance: str = Field(min_length=1)

    @field_validator("guidance")
    @classmethod
    def guidance_must_not_be_whitespace(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("guidance must not be blank")
        return stripped
```

- [ ] **Step 5: Validate and store routes in `MapElitesMultiIsland`**

Extend the constructor:

```python
def __init__(
    self,
    island_configs: list[IslandConfig],
    program_storage: RedisProgramStorage,
    migration_interval: int = 50,
    enable_migration: bool = True,
    max_migrants_per_island: int = 5,
    island_selector: WeightedIslandSelector | None = None,
    mutant_router: RandomMutantRouter | None = None,
    mutation_routes: list[MutationRouteConfig] | None = None,
    seed_island_map: dict[str, str] | None = None,
):
```

Normalize Hydra dictionaries before storing them:

```python
self.mutation_routes = tuple(
    route
    if isinstance(route, MutationRouteConfig)
    else MutationRouteConfig.model_validate(route)
    for route in (mutation_routes or [])
)
self.seed_island_map = dict(seed_island_map or {})
```

Store routes as an immutable tuple and the seed map as a copied dictionary.
Reject:

- duplicate `regime_id`;
- references to missing islands;
- seed-map targets that are not configured islands;
- `enable_migration=True` when routes are configured.

- [ ] **Step 6: Implement weighted route-first selection**

Implement `select_for_mutation` with this order:

1. If no routes are configured, call `super().select_for_mutation(total)` by
   wrapping the existing `select_elites` result.
2. Read each routed island size concurrently.
3. Remove routes whose island is empty.
4. Sample one remaining route with
   `random.choices(available_routes, weights=available_weights, k=1)`.
5. Call only the selected island's `select_elites(total)`.
6. Verify every returned parent's `current_island`.
7. Increment and persist strategy generation exactly once when parents exist.
8. Return `MutationSelection(parents, MutationRoute(route.regime_id,
   route.island_id, route.guidance))`.

Factor the existing generation increment into one private helper so both
`select_elites` and `select_for_mutation` retain the same resume semantics.

- [ ] **Step 7: Run route-selection and existing multi-island tests**

Run:

```bash
pytest tests/evolution/test_multi_island_extended.py \
       tests/evolution/test_island.py \
       tests/evolution/test_resume.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit route-aware multi-island selection**

```bash
git add gigaevo/evolution/strategies/models.py \
        gigaevo/evolution/strategies/multi_island.py \
        tests/evolution/test_multi_island_extended.py
git commit -m "feat: sample mutation routes before island parents"
```

---

### Task 3: Carry Routes Through the Concurrent Mutation Engine

**Files:**
- Modify: `gigaevo/evolution/engine/core.py`
- Modify: `gigaevo/evolution/engine/mutant_task.py`
- Modify: `gigaevo/evolution/engine/mutation.py`
- Modify: `gigaevo/evolution/mutation/base.py`
- Modify: `gigaevo/evolution/mutation/constants.py`
- Test: `tests/evolution/test_evolution_engine.py`
- Test: `tests/evolution/test_mutant_task_two_sema.py`
- Test: `tests/evolution/test_generate_mutations_exceptions.py`

**Interfaces:**
- Consumes: `EvolutionStrategy.select_for_mutation`
- Produces: `MutationOperator.mutate_with_route(selected_parents, route)`
- Produces: optional `route: MutationRoute | None` on
  `generate_one_mutation`
- Produces: child metadata keys `mutation_regime` and `target_island`

- [ ] **Step 1: Write engine-selection tests**

Update the parent-selection tests to expect:

```python
selection = MutationSelection(
    parents=[parent_a, parent_b],
    route=MutationRoute(
        regime_id="mid_margin",
        island_id="mid_margin",
        guidance="Reopen with a medium margin.",
    ),
)
engine.strategy.select_for_mutation.return_value = selection

result = await engine._select_parents_for_mutation()

assert result == selection
engine.strategy.select_for_mutation.assert_awaited_once_with(total=2)
```

The method name may remain `_select_parents_for_mutation`, but its return type
becomes `MutationSelection`.

- [ ] **Step 2: Write concurrent-task propagation tests**

In `test_mutant_task_two_sema.py`, make the fake engine return a
`MutationSelection`. Assert the patched `generate_one_mutation` receives:

```python
assert captured["parents"] == [parent]
assert captured["route"].regime_id == "near_end"
assert captured["route"].island_id == "near_end"
```

Retain every existing producer semaphore, buffer semaphore, refresh-ticket,
cancellation, and in-flight assertion.

- [ ] **Step 3: Write mutation stamping tests**

Add focused tests:

```python
async def test_generate_one_mutation_stamps_route_metadata():
    route = MutationRoute("mid_margin", "mid_margin", "medium guidance")
    mutator.mutate_with_route.return_value = MutationSpec(
        code="def solve(): return 2",
        parents=[parent],
        name="test",
    )

    child_id = await generate_one_mutation(
        parents=[parent],
        route=route,
        mutator=mutator,
        storage=storage,
        state_manager=state_manager,
        iteration=4,
        task_id=0,
    )

    child = storage.add.await_args.args[0]
    assert child.metadata["mutation_regime"] == "mid_margin"
    assert child.metadata["target_island"] == "mid_margin"
```

Also test `route=None` leaves both keys absent.

- [ ] **Step 4: Run the focused tests and verify failure**

Run:

```bash
pytest tests/evolution/test_evolution_engine.py \
       tests/evolution/test_mutant_task_two_sema.py \
       tests/evolution/test_generate_mutations_exceptions.py -q
```

Expected: route-aware assertions fail before implementation.

- [ ] **Step 5: Add route metadata constants and the compatibility hook**

In `MutationSpec` add:

```python
META_MUTATION_REGIME: ClassVar[str] = MUTATION_REGIME_METADATA_KEY
META_TARGET_ISLAND: ClassVar[str] = TARGET_ISLAND_METADATA_KEY
```

Define the canonical strings in `mutation/constants.py`:

```python
MUTATION_REGIME_METADATA_KEY = "mutation_regime"
TARGET_ISLAND_METADATA_KEY = "target_island"
```

In `MutationOperator` add:

```python
async def mutate_with_route(
    self,
    selected_parents: list[Program],
    route: MutationRoute | None,
) -> MutationSpec | None:
    return await self.mutate_single(selected_parents)
```

The generic fallback intentionally ignores route guidance. This preserves
subclasses whose `mutate_single` accepts only the parent list.
`LLMMutationOperator` provides the route-aware override in Task 4.

- [ ] **Step 6: Propagate `MutationSelection` through the engine**

Change `EvolutionEngine._select_parents_for_mutation` to call
`strategy.select_for_mutation`.

In `run_one_mutant`:

```python
selection = await engine._select_parents_for_mutation()
parents = selection.parents
route = selection.route
```

Refresh only `parents`, then pass both `parents` and `route` to
`generate_one_mutation`. Do not place route state on the engine or operator:
multiple producer tasks run concurrently.

- [ ] **Step 7: Stamp the child before persistence**

Change `generate_one_mutation`:

```python
mutation_spec = await mutator.mutate_with_route(parents, route)
if mutation_spec is None:
    return None
if route is not None:
    mutation_spec.metadata[MUTATION_REGIME_METADATA_KEY] = route.regime_id
    mutation_spec.metadata[TARGET_ISLAND_METADATA_KEY] = route.island_id
```

Stamp before `Program.from_mutation_spec` so Redis always contains ownership
metadata, including while the child is queued or running.

- [ ] **Step 8: Update fake engines and mutation operators**

Update tests whose fake `_select_parents_for_mutation` returns a raw list to
return `MutationSelection(parents=[parent])`. Tests that mock the generic
operator call must configure `mutator.mutate_with_route`, while concrete
operator classes may preserve their existing `mutate_single`
implementations.

- [ ] **Step 9: Run engine and concurrency regression tests**

Run:

```bash
pytest tests/evolution/test_evolution_engine.py \
       tests/evolution/test_mutant_task_two_sema.py \
       tests/evolution/test_generate_mutations_exceptions.py \
       tests/evolution/test_engine_invariants.py \
       tests/evolution/test_engine_stress.py \
       tests/evolution/test_engine_cancellation.py -q
```

Expected: PASS with all semaphore and cancellation invariants unchanged.

- [ ] **Step 10: Commit route propagation**

```bash
git add gigaevo/evolution/engine/core.py \
        gigaevo/evolution/engine/mutant_task.py \
        gigaevo/evolution/engine/mutation.py \
        gigaevo/evolution/mutation/base.py \
        gigaevo/evolution/mutation/constants.py \
        tests/evolution
git commit -m "feat: propagate mutation routes through engine"
```

---

### Task 4: Make the LLM Mutation Prompt Use the Explicit Route

**Files:**
- Modify: `gigaevo/evolution/mutation/mutation_operator.py`
- Modify: `gigaevo/llm/agents/mutation.py`
- Test: `tests/evolution/test_mutation_operator.py`
- Test: `tests/llm/test_mutation_agent.py`

**Interfaces:**
- Consumes: `MutationOperator.mutate_with_route` and `MutationRoute`
- Produces: `LLMMutationOperator.mutate_with_route(selected_parents, route)`
- Produces:
  `MutationAgent.arun(input, mutation_mode, explicit_regime_guidance=None)`

- [ ] **Step 1: Write explicit-guidance precedence tests**

Add an agent test:

```python
def test_explicit_regime_guidance_skips_internal_sampling(agent, monkeypatch):
    monkeypatch.setattr(
        agent,
        "_sample_mutation_regime",
        lambda: (_ for _ in ()).throw(AssertionError("must not sample")),
    )
    state = {
        "input": [parent],
        "mutation_mode": "rewrite",
        "explicit_regime_guidance": "## Near-end route\nUse the near tail.",
    }

    result = agent.build_prompt(state)

    assert "## Near-end route" in result["user_prompt"]
    assert (
        result["selected_mutation_regime"]
        == "## Near-end route\nUse the near tail."
    )
```

Add a second test proving `explicit_regime_guidance=None` retains legacy
weighted internal sampling.

- [ ] **Step 2: Write operator route-forwarding tests**

Mock `operator.agent.arun` and assert:

```python
await operator.mutate_with_route([parent], route)

operator.agent.arun.assert_awaited_once_with(
    input=[parent],
    mutation_mode="rewrite",
    explicit_regime_guidance=route.guidance,
)
```

Also assert `mutate_single([parent])` still calls the agent without explicit
guidance.

- [ ] **Step 3: Run mutation-agent/operator tests and verify failure**

Run:

```bash
pytest tests/llm/test_mutation_agent.py \
       tests/evolution/test_mutation_operator.py -q
```

Expected: failures for the missing explicit-guidance state and override.

- [ ] **Step 4: Extend mutation state and agent entrypoint**

Add:

```python
class MutationState(TypedDict):
    explicit_regime_guidance: NotRequired[str | None]
```

Extend `arun`:

```python
async def arun(
    self,
    input: list[Program],
    mutation_mode: str,
    explicit_regime_guidance: str | None = None,
) -> dict:
```

Place the explicit value into `initial_state`.

- [ ] **Step 5: Give explicit guidance precedence in `build_user_prompt`**

Use:

```python
explicit = (
    state.get("explicit_regime_guidance")
    if state is not None
    else None
)
regime = explicit if explicit is not None else self._sample_mutation_regime()
```

Append exactly one regime block. Do not append both explicit and sampled
guidance.

- [ ] **Step 6: Override `LLMMutationOperator.mutate_with_route`**

Refactor the common mutation body into a private method accepting
`explicit_regime_guidance`. Keep the public methods:

```python
async def mutate_single(
    self,
    selected_parents: list[Program],
    memory_instructions: str | None = None,
) -> MutationSpec | None:
    return await self._mutate(
        selected_parents,
        explicit_regime_guidance=None,
    )


async def mutate_with_route(
    self,
    selected_parents: list[Program],
    route: MutationRoute | None,
) -> MutationSpec | None:
    return await self._mutate(
        selected_parents,
        explicit_regime_guidance=(
            route.guidance if route is not None else None
        ),
    )
```

The deprecated `memory_instructions` argument must not become the route
transport for this operator.

- [ ] **Step 7: Run mutation tests**

Run:

```bash
pytest tests/llm/test_mutation_agent.py \
       tests/evolution/test_mutation_operator.py \
       tests/memory/test_mutation_operator.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit explicit route prompting**

```bash
git add gigaevo/evolution/mutation/mutation_operator.py \
        gigaevo/llm/agents/mutation.py \
        tests/evolution/test_mutation_operator.py \
        tests/llm/test_mutation_agent.py
git commit -m "feat: drive mutation prompts from selected route"
```

---

### Task 5: Route Children and Initial Roots Deterministically

**Files:**
- Modify: `gigaevo/evolution/strategies/island.py`
- Modify: `gigaevo/evolution/strategies/multi_island.py`
- Test: `tests/evolution/test_multi_island_extended.py`
- Test: `tests/evolution/test_island.py`

**Interfaces:**
- Consumes: child metadata `target_island`
- Consumes: root metadata `source=initial_program` and `strategy_name`
- Produces: deterministic routed-child and root insertion

- [ ] **Step 1: Write routed-child insertion tests**

Cover:

```python
async def test_add_uses_child_target_island_without_router():
    multi, islands, _ = _make_multi_island(
        n=3,
        enable_migration=False,
        mutation_routes=[
            _route("ab_initio", "island_0", 0.40),
            _route("mid_margin", "island_1", 0.35),
            _route("near_end", "island_2", 0.25),
        ],
    )
    child = _prog()
    child.metadata["target_island"] = "island_2"

    assert await multi.add(child) is True

    islands["island_2"].add.assert_awaited_once_with(child)
    multi.mutant_router.route_mutant.assert_not_awaited()


async def test_unknown_child_target_is_rejected_without_random_fallback():
    child.metadata["target_island"] = "removed_island"
    assert await multi.add(child) is False
    multi.mutant_router.route_mutant.assert_not_awaited()
```

Add a defense test that a route-owned child with parent metadata from another
island never reaches insertion; the earlier parent validation should raise.

- [ ] **Step 2: Write deterministic root-assignment tests**

Test:

```python
async def test_initial_root_uses_seed_island_map():
    root = Program(code="def entrypoint(): return 1")
    root.metadata.update(
        source="initial_program",
        strategy_name="pso_tohpe_only",
    )
    multi.seed_island_map = {"pso_tohpe_only": "ab_initio"}

    assert await multi.add(root) is True
    islands["ab_initio"].add.assert_awaited_once_with(root)


async def test_unmapped_root_is_rejected_when_seed_map_is_configured():
    root.metadata.update(
        source="initial_program",
        strategy_name="unknown_seed",
    )
    assert await multi.add(root) is False
```

Legacy configurations without a seed map must retain mutant-router behavior.

- [ ] **Step 3: Run focused insertion tests and verify failure**

Run:

```bash
pytest tests/evolution/test_multi_island_extended.py \
       tests/evolution/test_island.py -q
```

Expected: routed-child and seed-map tests fail before implementation.

- [ ] **Step 4: Resolve target islands before random routing**

In `MapElitesMultiIsland.add`:

```python
metadata_target = program.metadata.get(TARGET_ISLAND_METADATA_KEY)
if isinstance(metadata_target, str):
    island_id = metadata_target
elif (
    self.seed_island_map
    and program.metadata.get("source") == "initial_program"
):
    strategy_name = str(program.metadata.get("strategy_name") or "")
    island_id = self.seed_island_map.get(strategy_name)
    if island_id is None:
        logger.error("Unmapped initial program {}", strategy_name)
        return False
```

Only call `mutant_router.route_mutant` when neither explicit target nor
configured seed assignment applies.

- [ ] **Step 5: Preserve island ownership metadata**

Keep `MapElitesIsland.add` as the only place that sets `home_island` and
`current_island`. Assert after insertion:

```python
assert child.metadata["home_island"] == child.metadata["target_island"]
assert child.metadata["current_island"] == child.metadata["target_island"]
```

Do not remove `target_island`; it is required for audit and resume.

- [ ] **Step 6: Run island and ingestion regression tests**

Run:

```bash
pytest tests/evolution/test_multi_island_extended.py \
       tests/evolution/test_island.py \
       tests/evolution/test_evolution_engine.py \
       tests/evolution/test_ingestor_releases_buffer.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit deterministic insertion**

```bash
git add gigaevo/evolution/strategies/island.py \
        gigaevo/evolution/strategies/multi_island.py \
        tests/evolution/test_multi_island_extended.py \
        tests/evolution/test_island.py
git commit -m "feat: preserve mutation island ownership"
```

---

### Task 6: Configure the Three VarTODD Regime Islands

**Files:**
- Modify: `config/algorithm/vartodd_diverse_gf16_tohpe_updated.yaml`
- Modify: `tests/test_config.py`
- Modify: `tests/problems/test_vartodd_gf_evolution_guidance.py`
- Modify: `tests/problems/test_vartodd_tohpe_updated_contract.py`

**Interfaces:**
- Consumes: `MutationRouteConfig` and `seed_island_map`
- Produces: islands `ab_initio`, `mid_margin`, and `near_end`
- Produces: sampling weights `0.40`, `0.35`, and `0.25`

- [ ] **Step 1: Replace two-regime assertions with route-owned-island assertions**

Add resolved-config tests:

```python
assert [i.island_id for i in cfg.islands] == [
    "ab_initio",
    "mid_margin",
    "near_end",
]
assert cfg.evolution_strategy.enable_migration is False

routes = cfg.evolution_strategy.mutation_routes
assert [(r.regime_id, r.island_id, r.probability) for r in routes] == [
    ("ab_initio", "ab_initio", 0.40),
    ("mid_margin", "mid_margin", 0.35),
    ("near_end", "near_end", 0.25),
]
assert list(cfg.mutation_operator.mutation_regime_guidance) == []
assert cfg.mutation_operator.mutation_regime_probability == 0.0
```

Assert the three instantiated behavior-space objects are distinct by identity
so dynamic bound changes in one island cannot affect another.

- [ ] **Step 2: Add prompt-contract assertions**

Assert route guidance contains:

- `ab_initio`: `path_name="init"`, TOHPE-primary/light-TODD role, reusable path;
- `mid_margin`: exact visible shared path, margin `30..100`, branch before the
  terminal funnel;
- `near_end`: exact near-tail path, margin `5..30`, plateau handling, terminal
  TODD z research;
- no route contains TOHPEprefix guidance;
- all routes refer to the same Live Path Store rather than island-local
  directories.

- [ ] **Step 3: Run config tests and verify failure**

Run:

```bash
pytest tests/test_config.py \
       tests/problems/test_vartodd_gf_evolution_guidance.py \
       tests/problems/test_vartodd_tohpe_updated_contract.py -q
```

Expected: failures because the configuration still has one island and two
internally sampled regimes.

- [ ] **Step 4: Define three independent island configs**

In `vartodd_diverse_gf16_tohpe_updated.yaml`, replace the inherited single
island list with three `IslandConfig` entries using stable IDs:

```yaml
ab_initio_behavior_space:
  _target_: gigaevo.config.helpers.build_behavior_space
  keys: [${primary_key}, runtime, loaded_rank, ${validity_key}]
  bounds:
    [${primary_bounds}, ${runtime_bounds}, ${loaded_rank_bounds}, ${validity_bounds}]
  resolutions: [25, 4, 10, 2]
  binning_types: [${binning_type}, linear, linear, linear]
  dynamic: true
  expansion_buffer_ratio: 0.1

mid_margin_behavior_space:
  _target_: gigaevo.config.helpers.build_behavior_space
  keys: [${primary_key}, runtime, loaded_rank, ${validity_key}]
  bounds:
    [${primary_bounds}, ${runtime_bounds}, ${loaded_rank_bounds}, ${validity_bounds}]
  resolutions: [25, 4, 10, 2]
  binning_types: [${binning_type}, linear, linear, linear]
  dynamic: true
  expansion_buffer_ratio: 0.1

near_end_behavior_space:
  _target_: gigaevo.config.helpers.build_behavior_space
  keys: [${primary_key}, runtime, loaded_rank, ${validity_key}]
  bounds:
    [${primary_bounds}, ${runtime_bounds}, ${loaded_rank_bounds}, ${validity_bounds}]
  resolutions: [25, 4, 10, 2]
  binning_types: [${binning_type}, linear, linear, linear]
  dynamic: true
  expansion_buffer_ratio: 0.1

islands:
  - _target_: gigaevo.evolution.strategies.map_elites.IslandConfig
    island_id: ab_initio
    max_size: 20
    behavior_space: ${ab_initio_behavior_space}
    archive_selector:
      _target_: custom.archive_selectors.RankAwareRetentionSelector
      fitness_keys: [${primary_key}]
      fitness_key_higher_is_better: [${higher_is_better}]
    elite_selector:
      _target_: gigaevo.evolution.strategies.map_elites.WeightedEliteSelector
      fitness_key: ${primary_key}
      fitness_key_higher_is_better: ${higher_is_better}
      lambda_: 5.0
      epsilon: 1e-8
    archive_remover:
      _target_: gigaevo.evolution.strategies.map_elites.FitnessArchiveRemover
      fitness_key: ${primary_key}
      fitness_key_higher_is_better: ${higher_is_better}
    migrant_selector:
      _target_: gigaevo.evolution.strategies.map_elites.TopFitnessMigrantSelector
      fitness_key: ${primary_key}
      fitness_key_higher_is_better: ${higher_is_better}

  - _target_: gigaevo.evolution.strategies.map_elites.IslandConfig
    island_id: mid_margin
    max_size: 20
    behavior_space: ${mid_margin_behavior_space}
    archive_selector:
      _target_: custom.archive_selectors.RankAwareRetentionSelector
      fitness_keys: [${primary_key}]
      fitness_key_higher_is_better: [${higher_is_better}]
    elite_selector:
      _target_: gigaevo.evolution.strategies.map_elites.WeightedEliteSelector
      fitness_key: ${primary_key}
      fitness_key_higher_is_better: ${higher_is_better}
      lambda_: 5.0
      epsilon: 1e-8
    archive_remover:
      _target_: gigaevo.evolution.strategies.map_elites.FitnessArchiveRemover
      fitness_key: ${primary_key}
      fitness_key_higher_is_better: ${higher_is_better}
    migrant_selector:
      _target_: gigaevo.evolution.strategies.map_elites.TopFitnessMigrantSelector
      fitness_key: ${primary_key}
      fitness_key_higher_is_better: ${higher_is_better}

  - _target_: gigaevo.evolution.strategies.map_elites.IslandConfig
    island_id: near_end
    max_size: 20
    behavior_space: ${near_end_behavior_space}
    archive_selector:
      _target_: custom.archive_selectors.RankAwareRetentionSelector
      fitness_keys: [${primary_key}]
      fitness_key_higher_is_better: [${higher_is_better}]
    elite_selector:
      _target_: gigaevo.evolution.strategies.map_elites.WeightedEliteSelector
      fitness_key: ${primary_key}
      fitness_key_higher_is_better: ${higher_is_better}
      lambda_: 5.0
      epsilon: 1e-8
    archive_remover:
      _target_: gigaevo.evolution.strategies.map_elites.FitnessArchiveRemover
      fitness_key: ${primary_key}
      fitness_key_higher_is_better: ${higher_is_better}
    migrant_selector:
      _target_: gigaevo.evolution.strategies.map_elites.TopFitnessMigrantSelector
      fitness_key: ${primary_key}
      fitness_key_higher_is_better: ${higher_is_better}
```

Keep current selector semantics in this change so the island-routing
experiment is not confounded with a simultaneous archive-selection rewrite.

- [ ] **Step 5: Add route configs and disable migration**

Configure:

```yaml
evolution_strategy:
  mutation_routes:
    - regime_id: ab_initio
      island_id: ab_initio
      probability: 0.40
      guidance: |
        ## Required Island Regime: Ab-initio builder
        Start from Evaluator(path_name="init").
        Build a complete reusable path. TOHPE is the primary source; TODD is
        disabled or light except for an evidence-based lower-rank schedule.
        Explore policy scores, rank bands, restarts, and optimizer behavior
        without inheriting a saved-path tail.
    - regime_id: mid_margin
      island_id: mid_margin
      probability: 0.35
      guidance: |
        ## Required Island Regime: Medium-margin path refinement
        Load an exact selectable path from the shared Live Path Store and
        reopen it with margin 30..100. Branch before the terminal funnel so
        the policy, source balance, scoring, and restart placement can change
        the inherited trajectory.
    - regime_id: near_end
      island_id: near_end
      probability: 0.25
      guidance: |
        ## Required Island Regime: Near-end refinement
        Load an exact selectable near-tail path and reopen it with margin
        5..30. Address the resulting parameter plateau through optimizer
        settings, initialization or restarts, or broad beam exploration.
        Where TOHPE no longer supplies positive actions, use terminal TODD
        z research to search for rare actions.
  enable_migration: false
```

Override the inherited legacy sampler so the agent cannot sample a second
contradictory regime:

```yaml
mutation_operator:
  mutation_regime_guidance: []
  mutation_regime_probability: 0.0
```

- [ ] **Step 6: Add deterministic seed assignments**

Configure all supported root stems:

```yaml
seed_island_map:
  pso_tohpe_only: ab_initio
  pso_chunked_light_todd: ab_initio
  de_light_terminal_todd: ab_initio
  lean_scout_restart: ab_initio
  tohpe_weights_budget_probe: ab_initio

  pso_three_band_restart: mid_margin
  pso_cma_grouped_tail: mid_margin
  pso_restart_terminal_todd: mid_margin
  lean_beam_de: mid_margin
  beam3_temp_probe: mid_margin

  pso_wide_terminal_beam: near_end
  pso_pattern_heavy_restart: near_end
  todd_hard_tail_budget_split: near_end
  full_pso_pattern: near_end
```

Pass `${seed_island_map}` into `MapElitesMultiIsland`.

- [ ] **Step 7: Keep the Path Store shared**

Retain:

```yaml
mutation_operator:
  live_path_store_root_dir: null
```

Do not add an island ID to `DATA_PATH`, `run_gf.py`, path names, or
`PathStore`.

- [ ] **Step 8: Run configuration and path tests**

Run:

```bash
pytest tests/test_config.py \
       tests/problems/test_vartodd_gf_evolution_guidance.py \
       tests/problems/test_vartodd_tohpe_updated_contract.py \
       tests/problems/test_vartodd_live_paths.py \
       tests/problems/test_vartodd_gf_shared_source.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit the three-island configuration**

```bash
git add config/algorithm/vartodd_diverse_gf16_tohpe_updated.yaml \
        tests/test_config.py \
        tests/problems/test_vartodd_gf_evolution_guidance.py \
        tests/problems/test_vartodd_tohpe_updated_contract.py
git commit -m "feat: specialize vartodd evolution by regime island"
```

---

### Task 7: Persist Route Counters and Verify Resume Ownership

**Files:**
- Modify: `gigaevo/evolution/strategies/multi_island.py`
- Modify: `gigaevo/evolution/strategies/base.py`
- Test: `tests/evolution/test_resume.py`
- Test: `tests/evolution/test_resume_e2e.py`
- Test: `tests/evolution/test_multi_island_extended.py`

**Interfaces:**
- Produces: strategy metrics `route_selections/<regime_id>`
- Produces: persisted run-state keys `strategy:route_selections:<regime_id>`
- Consumes: child `target_island` metadata after resume

- [ ] **Step 1: Write route-counter persistence tests**

Test that one selection of `mid_margin`:

```python
metrics = await strategy.get_metrics()
assert metrics.strategy_specific_metrics["route_selections/mid_margin"] == 1
storage.save_run_state.assert_any_await(
    "strategy:route_selections:mid_margin",
    1,
)
```

Test `restore_state()` loads the configured route counters and leaves missing
keys at zero.

- [ ] **Step 2: Write outstanding-child resume test**

Create a queued/done child with:

```python
child.metadata.update(
    mutation_regime="near_end",
    target_island="near_end",
)
```

After engine restoration and ingestion, assert only the `near_end` island
receives the child and its metadata remains unchanged.

- [ ] **Step 3: Run resume tests and verify failure**

Run:

```bash
pytest tests/evolution/test_resume.py \
       tests/evolution/test_resume_e2e.py \
       tests/evolution/test_multi_island_extended.py -q
```

Expected: route-counter assertions fail before implementation.

- [ ] **Step 4: Persist route-selection counters**

Initialize:

```python
self.route_selection_counts = {
    route.regime_id: 0 for route in self.mutation_routes
}
```

After successful routed parent selection, increment and save only that route:

```python
key = f"strategy:route_selections:{route.regime_id}"
self.route_selection_counts[route.regime_id] += 1
await self.program_storage.save_run_state(
    key,
    self.route_selection_counts[route.regime_id],
)
```

Load each configured key in `restore_state`. Add the counters to
`StrategyMetrics.strategy_specific_metrics`.

- [ ] **Step 5: Run resume and metrics tests**

Run:

```bash
pytest tests/evolution/test_resume.py \
       tests/evolution/test_resume_e2e.py \
       tests/evolution/test_multi_island_extended.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit route observability**

```bash
git add gigaevo/evolution/strategies/multi_island.py \
        tests/evolution/test_resume.py \
        tests/evolution/test_resume_e2e.py \
        tests/evolution/test_multi_island_extended.py
git commit -m "feat: persist mutation route statistics"
```

---

### Task 8: Run the Full Verification and Document Launch/Resume Rules

**Files:**
- Modify: `run_gf.py`
- Modify: `docs/superpowers/specs/2026-07-30-regime-owned-vartodd-islands-design.md`
- Test: `tests/problems/test_vartodd_gf_shared_source.py`

**Interfaces:**
- Consumes: completed three-island implementation
- Produces: launch help warning for fresh three-island namespaces

- [ ] **Step 1: Add a launcher-help contract test**

Assert `run_gf.py --help` or `_usage()` communicates:

- the three-island algorithm requires a fresh Redis namespace when switching
  from the old single-island topology;
- `redis.resume=true` is supported only after the namespace was created with
  the same three stable island IDs;
- `data_gf<matrix>` remains shared between islands.

- [ ] **Step 2: Update launcher usage text**

Add a concise note without changing launch behavior:

```text
Regime-island note: the ab_initio, mid_margin, and near_end archives share
data_gf<matrix> paths but require a fresh Redis namespace when first replacing
the legacy single-island archive. Later runs may use redis.resume=true with
the same topology.
```

- [ ] **Step 3: Run targeted VarTODD and engine suites**

Run:

```bash
pytest tests/evolution/test_strategy_base.py \
       tests/evolution/test_multi_island_extended.py \
       tests/evolution/test_island.py \
       tests/evolution/test_evolution_engine.py \
       tests/evolution/test_mutant_task_two_sema.py \
       tests/evolution/test_generate_mutations_exceptions.py \
       tests/evolution/test_mutation_operator.py \
       tests/evolution/test_resume.py \
       tests/evolution/test_resume_e2e.py \
       tests/llm/test_mutation_agent.py \
       tests/problems/test_vartodd_gf_evolution_guidance.py \
       tests/problems/test_vartodd_tohpe_updated_contract.py \
       tests/problems/test_vartodd_live_paths.py \
       tests/problems/test_vartodd_gf_shared_source.py \
       tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 4: Run format and static checks on modified Python**

Run the repository's configured checks:

```bash
ruff check gigaevo/evolution/strategies/base.py \
           gigaevo/evolution/strategies/models.py \
           gigaevo/evolution/strategies/multi_island.py \
           gigaevo/evolution/engine/core.py \
           gigaevo/evolution/engine/mutant_task.py \
           gigaevo/evolution/engine/mutation.py \
           gigaevo/evolution/mutation/base.py \
           gigaevo/evolution/mutation/mutation_operator.py \
           gigaevo/llm/agents/mutation.py
```

Expected: no errors.

- [ ] **Step 5: Perform a resolved-config smoke test**

Run a Hydra composition/config check without starting an evolution:

```bash
python run_gf.py --help
pytest tests/experiment/test_checks_resolved_config.py -q
```

Inspect the resolved algorithm in the config test and confirm:

- exactly three islands;
- route probabilities sum to `1.0`;
- migration is false;
- every current initial-program stem is mapped;
- `live_path_store_root_dir` is null;
- the mutation operator has no legacy random regime list.

- [ ] **Step 6: Self-review the implementation against isolation invariants**

Search:

```bash
rg -n "mutation_regime|target_island|select_for_mutation|mutation_routes|seed_island_map" \
  gigaevo config tests
```

Confirm there is no code path in which:

- guidance is sampled after parents for the three-island configuration;
- a routed child reaches `RandomMutantRouter`;
- a routed parent set contains multiple `current_island` values;
- migration is enabled;
- island ID changes `DATA_PATH`.

- [ ] **Step 7: Commit verification documentation**

```bash
git add run_gf.py \
        tests/problems/test_vartodd_gf_shared_source.py \
        docs/superpowers/specs/2026-07-30-regime-owned-vartodd-islands-design.md
git commit -m "docs: describe regime island launch contract"
```
