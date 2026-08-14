# VarTODD Island Path-Family Selection Design

## Goal

Make saved-path evidence and reuse accounting match the three search regimes:

- ab-initio receives no saved paths;
- mid-margin receives separately bounded ab-initio roots and mid-margin
  refinement families;
- near-end receives the best representative of each eligible refinement tree
  and retires work at both exact-path and family levels.

The design also removes the rank-name ambiguity in the mid-margin prompt and
allows reuse limits and refinement penalties to be tuned from
`run_gf_islands.py`.

## Current Problems

The current shared inventory treats mid-margin and near-end as two filtered
views of nearly the same path list. Its counters belong to exact path names,
so an improving child receives a fresh allowance. Direct siblings are
canonicalized, but grandchildren are not managed as one refinement family.
The stored origin only distinguishes ab-initio roots from improved children;
it does not record which refinement route produced a child.

The compact mid-margin card labels the path-name `i<rank>` value as
`init_rank`. That value describes the producer's restart/start band, not the
matrix-wide `INITIAL_RANK`. This encourages mutations to calculate margins
from the wrong span.

## Terminology

- **Path**: one exact saved `path_name`.
- **Ab-initio root**: a path produced with `path_name="init"` and therefore no
  saved parent.
- **Mid family**: a mid-margin path and all improving mid-margin descendants
  reached from it.
- **Near family**: the path first selected by near-end and all improving
  near-end descendants reached from it.
- **Representative**: the lowest-rank selectable path in a family. Rank ties
  use deterministic existing evidence tie-breakers.
- **Use**: an evaluation that successfully loads the selected saved path.
  Once loading succeeds, a timeout or non-improving result still consumes the
  reservation. A syntax/API failure before loading does not.

Only improving children can become representatives. Non-improving child paths
remain evidence but are never selectable.

## Route Inventories

### Ab-initio

The ab-initio mutation context contains no selectable path inventory or path
handles. All initial programs belong to this island.

### Mid-margin

The mid-margin context contains two explicit sections.

#### Ab-initio roots

- Show the four best eligible ab-initio roots.
- Each exact root has a direct mid-margin reuse limit of 6.
- An improvement does not hide or replace the root before its allowance is
  exhausted.
- Print `uses/6` and the best descendant rank across every later generation
  rooted at that ab-initio path.
- Newly produced ab-initio roots may enter the top four by rank, but an
  improving refinement child is not reclassified as an ab-initio root.

#### Mid-margin families

- Show the four best eligible mid-margin family representatives.
- A family has a shared reuse limit of 8 across its representative and all
  improving descendants.
- When a child improves rank, it replaces the displayed representative while
  inheriting the family's accumulated usage.
- Paths produced by near-end refinement never enter the mid-margin inventory.

Mid-margin margin selection uses the matrix-wide rank span:

```text
global_span = INITIAL_RANK - selected_path_rank
margin = chosen_quantile * global_span
```

The prompt must explicitly refer to uppercase `INITIAL_RANK`. It must not use
the path-name `i<rank>` component, a compact-card producer/restart rank, or the
eventual `loaded_rank` as the global initial rank. Compact cards rename the
ambiguous `init_rank` label to `producer_start_rank` (or an equally explicit
name).

### Near-end

- Near-end may consider eligible best paths originating from ab-initio,
  mid-margin, or near-end refinement.
- The first near-end selection of a path establishes a near family rooted at
  that path.
- Display at most one path per family: its best improving descendant, or the
  root before any improvement.
- Each exact path may be reused at most 2 times in near-end.
- The whole near family may be reused at most 7 times across every descendant.
- An improving child becomes the representative with a fresh exact-path count
  of zero, while the family count remains unchanged.
- If the current representative exhausts its two exact-path uses without an
  improving child, the family becomes unavailable even if fewer than seven
  family uses were consumed. Do not fall back to a worse ancestor.
- At seven family uses, retire the entire family. A descendant must not re-enter
  later as a new root with reset accounting.

## Provenance and Family Accounting

Every saved path card or associated usage record needs explicit provenance:

```text
created_by_route
parent_path_name
mid_family_root
near_family_root
```

Family representatives and best descendant ranks can be derived from the
parent graph, but stable family identities and counters must be persisted so
promotion cannot reset a budget. Historical records without route provenance
must be handled conservatively and must not be guessed into the wrong route.

The usage ledger records, per route/family/path as appropriate:

```text
completed_uses
in_flight_uses
improved_count
nonimproved_count
best_child_rank
processed_program_ids
```

Reservations must be atomic. A path is reserved after the generated program's
exact path choice is known and before expensive evaluation begins. Selection
uses `completed_uses + in_flight_uses`, preventing concurrent programs from
oversubscribing a limit. Completion converts the reservation into a completed
use; failure before successful path loading releases it.

## Refinement Penalties

Failure to beat `loaded_path_rank` receives a larger route-specific selection
penalty:

- mid-margin default: 12 rank units;
- near-end default: 16 rank units.

The penalty applies only to a loaded-path child that does not improve its
loaded artifact rank. Soft-timeout salvage has no independent penalty.

Because validation does not know the mutation route, route-specific shaping is
performed by an island-aware selection component using `rank_improved`, rather
than inferred inside `validate.py`. The unshaped metrics remain available for
reporting.

## Launcher Configuration

`run_gf_islands.py` accepts and validates these managed arguments:

```text
mid_root_reuse_limit=6
mid_family_reuse_limit=8
near_family_reuse_limit=7
near_path_reuse_limit=2
mid_no_improvement_penalty=12
near_no_improvement_penalty=16
```

It forwards them as Hydra overrides to the dedicated island configuration.
They are included in `--help` examples and tests. Defaults preserve the values
above while allowing later runs to tune them without editing source files.

## Verification

Tests cover:

- ab-initio context contains no path inventory;
- mid-margin renders four ab-initio roots and four mid-family representatives;
- route provenance excludes near-end descendants from mid-margin;
- root improvements do not hide roots before `uses/6`;
- mid-family promotion preserves `uses/8`;
- near-end promotion preserves family usage and resets only exact-path usage;
- exact paths retire at 2 and near families retire at 7;
- exhausted near families cannot re-enter through descendants;
- non-improving children never appear in selectable inventories;
- concurrent reservations cannot exceed configured limits;
- mid-margin guidance uses uppercase global `INITIAL_RANK` and does not call
  the path-name `i<rank>` value `init_rank`;
- launcher defaults and explicit overrides reach the path provider and
  route-aware selection components;
- penalties apply only to non-improving loaded-path children and do not depend
  on `timeout_salvaged`.

