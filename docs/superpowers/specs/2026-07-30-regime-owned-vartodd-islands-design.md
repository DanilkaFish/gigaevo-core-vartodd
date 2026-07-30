# Regime-Owned VarTODD Islands Design

## Goal

Split the VarTODD GF evolution into independent, role-specific parent
populations while keeping one shared saved-path store:

1. `ab_initio`: construct complete paths from the matrix.
2. `mid_margin`: refine saved paths from a medium-distance branch point.
3. `near_end`: refine the inherited terminal tail.

The regime is sampled before parent selection. All parents come from the
regime's island, the regime's guidance is included in the mutation prompt, and
the resulting child returns to the same island.

## Chosen Approach

Introduce a first-class `MutationRoute` selected by the evolution strategy.
A route identifies:

- the regime;
- the source/destination island;
- the regime-specific prompt guidance.

The engine carries this route alongside the selected parents. It is not
re-sampled inside the LLM agent and is not inferred from the generated code or
the child's metrics.

This is preferred over:

- **Post-hoc routing:** inferring an island from child code can move a child
  away from the population that produced it and recreates regime mixing.
- **Soft cross-island parent bias:** giving the matching island only a higher
  probability still permits incompatible parent/prompt combinations and lets
  the strongest archetype leak into every population.

## Isolation Contract

- Parent selection for a route reads only the route's island archive.
- Every selected parent has `current_island` equal to the route island.
- The child is stamped with `mutation_regime` and `target_island` before it is
  persisted.
- Ingestion sends a stamped child only to `target_island`.
- Migration is disabled.
- Cross-island crossover is disabled.
- The global saved-path store remains shared. Any island may load a selectable
  saved path produced by another island.
- A program never changes island because of its observed policy, runtime,
  rank, or generated code.

## Core Interfaces

Add immutable strategy-layer values:

```python
@dataclass(frozen=True, slots=True)
class MutationRoute:
    regime_id: str
    island_id: str
    guidance: str


@dataclass(frozen=True, slots=True)
class MutationSelection:
    parents: list[Program]
    route: MutationRoute | None = None
```

`EvolutionStrategy.select_for_mutation(total)` returns a
`MutationSelection`. Its default implementation wraps the existing
`select_elites(total)` result with `route=None`, keeping non-island algorithms
unchanged.

`MapElitesMultiIsland.select_for_mutation(total)` samples one configured route
from the routes whose island has at least one elite, then calls only that
island's elite selector. Positive route weights are normalized at sampling
time.

The mutation stack receives the route explicitly. A default
`MutationOperator.mutate_with_route(selected_parents, route)` delegates to the
existing `mutate_single(selected_parents)` method; `LLMMutationOperator`
overrides it to pass the selected guidance into `MutationAgent`.

## Mutation Flow

```text
steady-state capacity slot
    -> strategy.select_for_mutation(num_parents)
    -> sample one available route
    -> select parents from that route's island only
    -> refresh those parents
    -> LLMMutationOperator receives route guidance
    -> MutationAgent builds one route-specific prompt
    -> MutationSpec is stamped with regime_id and target island
    -> child evaluates normally
    -> MapElitesMultiIsland.add reads target_island
    -> child competes only in the producing island
```

The legacy internal `MutationAgent` regime sampler remains available when no
explicit route is provided, so other algorithm configurations retain their
current behavior. The new VarTODD configuration does not configure that
legacy sampler.

## Route Configuration

The multi-island strategy accepts route configuration rather than hard-coding
the three VarTODD regimes:

```yaml
mutation_routes:
  - regime_id: ab_initio
    island_id: ab_initio
    probability: 0.40
    guidance: |
      Start from path_name="init" and build a complete reusable path.
  - regime_id: mid_margin
    island_id: mid_margin
    probability: 0.35
    guidance: |
      Load a selectable shared path and reopen it with margin 30..100.
  - regime_id: near_end
    island_id: near_end
    probability: 0.25
    guidance: |
      Load a selectable near-tail path and reopen it with margin 5..30.
```

Every route must have a unique `regime_id`, reference an existing island, use
a positive probability, and contain non-empty guidance. Adding another island
later requires another `IslandConfig`, route entry, and initial-seed
assignment; engine code remains unchanged.

## Regime Guidance

### `ab_initio`

- Start from `Evaluator(path_name="init")`, with any ordinary constructor
  arguments supplied by the program.
- Build a complete path and produce reusable saved-path artifacts.
- Use the current builder logic: TOHPE is the main source; TODD is disabled or
  light except for an evidence-based lower-rank schedule.
- Explore policy scores, policy bands, restarts, and optimizer behavior without
  inheriting a saved-path tail.

### `mid_margin`

- Load an exact selectable name from the shared Path Store.
- Reopen with a medium margin, initially `30..100`.
- Branch before the terminal funnel so the mutation can alter the inherited
  trajectory rather than replay only the last few actions.
- Optimize policy transitions, source balance, restart placement, and scoring
  across the reopened band.

### `near_end`

- Load an exact selectable near-tail path.
- Reopen close to the inherited tail, initially with margin `5..30`.
- Address the broad parameter plateaus caused by short inherited tails through
  optimizer settings, initialization/restarts, or sufficiently broad beam
  exploration.
- Use terminal TODD z research when action evidence shows that TOHPE no longer
  supplies positive actions.

These are route roles, not post-execution classification rules.

## Initial Population

Initial programs are assigned deterministically by their
`metadata.strategy_name`. The strategy receives a `seed_island_map`.

For `initial_programs=best`:

- `ab_initio`: `pso_tohpe_only`, `pso_chunked_light_todd`,
  `de_light_terminal_todd`
- `mid_margin`: `pso_three_band_restart`, `pso_cma_grouped_tail`,
  `pso_restart_terminal_todd`
- `near_end`: `pso_wide_terminal_beam`, `pso_pattern_heavy_restart`

For the default pool:

- `ab_initio`: `lean_scout_restart`, `tohpe_weights_budget_probe`
- `mid_margin`: `lean_beam_de`, `beam3_temp_probe`
- `near_end`: `todd_hard_tail_budget_split`, `full_pso_pattern`

When `seed_island_map` is configured, an unmapped root is rejected with a
clear error instead of being randomly inserted into an island.

## Shared Path Store

No path storage format or directory changes are required. The mutation agent
continues to load the live summary from the matrix-specific `DATA_PATH`.
Island metadata is not added to path names and does not restrict which island
may load a path.

This is intentional one-way artifact sharing: islands share search results
through saved paths, but not programs through parent selection or migration.

## Failure Handling

- Invalid route configuration fails during strategy construction.
- If a selected route's island becomes empty, selection retries from the other
  non-empty routes without borrowing parents.
- If every routed island is empty, mutation selection returns no parents and
  the steady-state producer backs off normally.
- A child with an unknown `target_island` is rejected rather than randomly
  routed.
- A routed mutation whose parents name another `current_island` fails before
  the LLM call.
- Existing algorithms without routes retain current selection and routing.

## Resume Behavior

Route identity and destination are persisted in child metadata. A resumed run
therefore continues to ingest outstanding children into their producing
islands.

The three-island topology uses stable IDs:

- `ab_initio`
- `mid_margin`
- `near_end`

An old single-island Redis archive is not automatically converted. The first
three-island run must use a fresh namespace/database, or an explicit archive
rebuild performed separately.

## Observability

Expose and log:

- route selections by `regime_id`;
- selected parent IDs and their common island;
- accepted/rejected children by route;
- archive size per island;
- child `mutation_regime`, `target_island`, `home_island`, and
  `current_island`.

These counters make it possible to compare regime productivity and retune
route probabilities without inspecting source code manually.

## Verification

Tests must prove:

- the route is sampled before parents;
- all parents come from exactly the selected island;
- the explicit route guidance replaces internal random regime sampling;
- child metadata carries the route and ingestion preserves it;
- no migration or random mutant routing occurs for routed children;
- initial roots enter their configured islands;
- all islands see the same live Path Store summary;
- legacy single-island and non-routed strategies still pass existing tests;
- resume preserves route ownership for in-flight children.
