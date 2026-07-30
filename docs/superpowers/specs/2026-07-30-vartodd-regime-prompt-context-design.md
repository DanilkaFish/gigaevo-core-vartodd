# VarTODD Regime Prompt Context Design

## Goal

Create a separate VarTODD GF evolution problem whose mutation prompts,
parent populations, saved-path evidence, and auxiliary execution evidence are
specialized for three search regimes:

1. `ab_initio`: build complete paths from `path_name="init"`;
2. `mid_margin`: refine a selectable saved path from a medium-distance branch;
3. `near_end`: refine a selectable saved path near its terminal tail.

The legacy `vartodd_evo_gf` problem, pipeline, algorithm configuration,
experiment, launcher behavior, Redis namespace, execution cache, and
`data_gf<matrix>` Path Store remain available unchanged.

## Separate Experiment Namespace

The new implementation uses independent assets:

- problem source: `problems/vartodd_evo_gf_islands`;
- pipeline: `config/pipeline/vartodd_islands_pipeline.yaml`;
- algorithm: `config/algorithm/vartodd_diverse_gf_islands.yaml`;
- experiment: `config/experiment/vartodd_evo_gf_islands_steady.yaml`;
- launcher: `run_gf_islands.py`;
- generated problem name: `vartodd_evo_gf_islands<matrix>`;
- Redis prefix: `vartodd_evo_gf_islands<matrix>`;
- Path Store: `data_gf_islands<matrix>`;
- execution cache: a distinct island-problem cache under `.run_gf`.

The three regime islands share the new matrix-specific Path Store with one
another. They do not read or write the legacy `data_gf<matrix>` store.

Unchanged runtime modules and initial-program sources may be shared by link
from the legacy source problem. Island-specific prompts, task description,
variant/data-root logic, and Path Store behavior live in the new problem.

## Mutation Route

A mutation route is sampled before parent selection. It determines:

- the required mutation regime;
- the destination island;
- the route-specific prompt context profile;
- the binding regime guidance.

The child always enters the route's destination island. Parent origin,
generated code, runtime, path choice, and resulting rank never change that
destination.

The initial route probabilities are:

- `ab_initio`: `0.40`;
- `mid_margin`: `0.35`;
- `near_end`: `0.25`.

Unavailable routes are excluded and their probability is normalized across
the routes that can currently produce a valid mutation.

## Initial Population

Every initial program belongs to `ab_initio`, regardless of which initial
program pool the launcher selects. The strategy uses:

```yaml
initial_island_id: ab_initio
```

This replaces a filename-to-island map and automatically covers future initial
programs.

`mid_margin` and `near_end` begin empty and are populated by children produced
under their corresponding routes.

## Parent Selection and Controlled Mixing

For the configured two-parent evolution:

### Empty target island

If the sampled destination island has size `0`, both parents are selected from
`ab_initio`. The resulting child still enters the sampled destination island.

### Bootstrap population

If the destination island has size `1..7`, select:

- one local parent and one `ab_initio` parent with probability `0.70`;
- two local parents otherwise.

If the target contains only one unique program, the mixed selection is
mandatory so the program is not duplicated to manufacture a two-parent input.

### Established population

If the destination island has size `8` or greater:

- normally select both parents locally;
- with probability `0.10`, replace at most one parent with a parent from
  another non-empty island.

For steady mixing, choose uniformly among eligible donor islands before using
that island's configured elite selector. This prevents the largest island from
becoming the implicit donor for every mixed mutation.

The initial configuration is:

```yaml
bootstrap_source_island: ab_initio
bootstrap_until_size: 8
bootstrap_mix_probability: 0.70
steady_mix_probability: 0.10
```

Migration remains disabled. Cross-island transfer happens only through this
explicit parent-mixing rule and the shared island-problem Path Store.

Every parent block is labeled with one of:

- `target_island`;
- `bootstrap_donor`;
- `mixed_donor`.

When a target-island parent exists, it is the default primary parent. A donor
contributes reusable mechanisms and evidence without changing the required
regime. When the destination island is empty, an `ab_initio` parent may be the
primary implementation base but must be transformed to satisfy the sampled
refinement contract.

## Prompt Architecture

One shared system prompt and task/API description serve all three regimes.
Dynamic evidence is assembled by a route-aware context builder.

Each mutation prompt is ordered as follows:

1. common mutation system prompt and Action API/task mechanics;
2. compact mutation assignment containing the regime, destination island, and
   parent roles;
3. route-filtered parent blocks;
4. a route-specific saved-path block;
5. exactly one binding regime contract.

This avoids maintaining three copies of the Action API while making the
evidence presented to each regime materially different.

## Saved-Path Supply

### Ab-initio

The `ab_initio` prompt receives no Live Path Store block and no loadable saved
path handles. Historical `loaded_path_name` and `this path name` fields are
removed from its parent aux excerpts. Its binding contract requires
`Evaluator(path_name="init", ...)`.

### Refinement regimes

`mid_margin` and `near_end` receive the same unified set of selectable paths.
The set is the union of currently eligible near-tail and wide-margin records,
rendered under one `Selectable Shared Paths` heading rather than separate
groups.

Dead-end paths, stale replaced paths, and non-improving child paths are omitted
from the prompt. Only names in the current unified block are loadable. A path
name in parent code, lineage, or historical aux evidence is descriptive and
does not grant permission to load that path.

The two refinement islands differ through their margin and experiment
contracts, not through different path inventories.

If no selectable path exists, the two refinement routes are temporarily
unavailable. Evolution continues through `ab_initio` until a selectable path
is available.

## Evidence-Rich Path Cards

Every displayed selectable path has a compact evidence card. A card contains:

- exact path name;
- final rank, depth, use/improvement counts, search kind, restart band, and
  best-child summary;
- compact rank trajectory;
- policy bands with reduction, dimension, accepted source counts, retained
  source pools, and TODD-specific z research;
- the policy profiles used along the path;
- deduplicated exploration/final score weights and meaningful centers;
- action-pool, beam, sampling, bucket, cap, and actions-per-bucket settings
  associated with those profiles;
- producer search statistics when available: total evaluations,
  best-seen count, last improvement, restart/stage summary, runtime, and
  timeout-salvage status.

Card persistence has two phases:

1. the evaluator saves the base card with the path, policy, source, z, and
   optimizer-search evidence available inside `get_best()`;
2. a pipeline enrichment stage reads `this path name` after validation and
   timing, then attaches validated metrics, runtime, and pipeline-level
   timeout status to that exact path record.

For an older path without a persisted card, the provider derives path
statistics and policy profiles from the saved path and DAO objects.
Producer-only fields that cannot be recovered are explicitly marked
unavailable.

Cards use deterministic field selection and score-block deduplication rather
than arbitrary prefix truncation. When the aggregate prompt exceeds its
configured path-card budget, retain in this order:

1. path identity and selection state;
2. terminal bands and their policy profiles;
3. score blocks and TODD source/z evidence;
4. producer search summary;
5. less relevant early bands.

Every path that remains visible remains loadable and has a card. The provider
never presents a bare name after dropping its evidence.

Initial context limits are:

```yaml
selectable_path_top_k: 8
path_cards_max_chars: 24000
parent_aux_max_chars: 12000
```

The selectable set keeps at most one representative per final rank. A rank
tie prefers the path with the higher observed starting rank, then the less
reused path.

## Parent Evidence Profiles

All regimes receive parent code, validated metrics, errors, structured
insights, parent-role labels, relevant lineage, memory, and runtime relative to
the configured call budget.

### `ab_initio`

Include:

- the complete produced trajectory without its saved-path handle;
- all reached policy bands;
- every policy profile used by the best path;
- deduplicated score weights and meaningful centers;
- accepted and retained TOHPE/TODD actions per band;
- TODD research per band;
- optimizer stages, restarts, evaluations, last improvement, best-seen count,
  and rank quantiles.

The evidence explains complete-path construction and runtime use without
providing a saved path that could violate the regime.

### `mid_margin`

Include:

- historical loaded-path rank, actual branch threshold, and effective margin;
- whether the program beat the loaded path;
- the trajectory from the branch threshold to the final rank;
- policy bands and profiles reached in the reopened region;
- source/pool/z behavior in those bands;
- optimizer stages and restarts associated with the reopened search.

Historical path names are labeled non-selectable unless the same exact name
appears in the current path-card block.

### `near_end`

Include:

- historical loaded-path rank, branch threshold, effective margin, and
  improvement result;
- the terminal segment and last one to three relevant policy bands;
- active terminal score profiles;
- terminal TODD accepted actions, retained pool, `z_researched_todd`,
  `min_buckets`, `max_buckets`, `limit_bucket`, and actions per bucket;
- beam/search breadth;
- repeated-best count, last improvement, rank quantiles, restarts, and stage
  results that expose plateau behavior.

Early-path detail is omitted unless it is needed to explain the branch.

## Evolutionary Statistics and Memory

Mutation statistics are island-local:

- archive percentile;
- island worst/median/best;
- recent island trend;
- invalid streak;
- iterations since that island's last new best.

The mutation archetype gate compares a parent with its specialized population,
not with programs from another regime.

Intra-lineage memory remains attached to its parent. Island-local memory is the
default. A donor's own memory may enter with that donor parent, but a global
cross-island memory bank must not bypass the configured mixing probabilities.

Structured Program Insights remain hypotheses. The route-filtered raw aux
evidence stays in the mutation prompt so the mutator can reject an insight
that conflicts with direct execution evidence.

## Regime Contracts

### `ab_initio`

- Use `Evaluator(path_name="init", ...)`.
- Produce a complete reusable path.
- Use TOHPE as the primary builder source and TODD disabled or light until
  lower-rank evidence justifies a schedule.
- Explore policy scores, reached policy bands, restarts, and search behavior
  without inheriting a saved tail.

### `mid_margin`

- Load one exact name from `Selectable Shared Paths`.
- Use an evidence-based medium margin, initially `30..100`.
- Branch early enough to alter the inherited trajectory, policy scoring,
  source balance, and restart placement.

### `near_end`

- Load one exact name from `Selectable Shared Paths`.
- Use an evidence-based near-tail margin, initially `5..30`.
- Address short-tail plateaus through suitable initialization, optimizer
  behavior, restarts, or broad beam exploration.
- Use terminal TODD z research when the supplied action evidence shows rare or
  starved positive actions.

These are required roles for the produced child, not post-run classifiers.

## Components

### `RegimeParentSelector`

Owns empty-island bootstrap, size-dependent mixing, local selection, donor
selection, route-availability filtering, and parent-role labels. It depends on
`SelectablePathCardProvider.has_selectable_paths()` for routes that require a
saved path and returns the sampled destination unchanged.

### `RegimePromptContextBuilder`

Owns the mutation-assignment block, parent aux profile, path inclusion policy,
historical-handle labeling, prompt budget, and island-local statistics.

### `SelectablePathCardProvider`

Owns unified eligibility, evidence-card persistence/derivation, deterministic
compression, and rendering. It uses the dedicated island-problem Path Store.

### `PathCardEnrichmentStage`

Adds validated metrics, pipeline timing, and timeout state to the base card
after the program and validator have completed. It updates only the exact path
named by the program's aux report.

The generic mutation engine carries route and parent-role metadata but does not
contain VarTODD-specific evidence rules.

## Failure Handling

- An invalid regime/mixing configuration fails during construction.
- An unavailable refinement route is excluded before parent selection.
- An unreadable path is omitted; the LLM is never asked to invent a handle.
- If the destination and bootstrap-source islands are both empty, mutation
  production backs off without an LLM call.
- If an optional donor is unavailable, selection falls back to local parents.
- A child with an unknown destination island is rejected.
- Missing card fields are labeled unavailable rather than inferred.
- Legacy algorithms without route context retain their existing prompt and
  selection behavior.

## Verification

Tests must verify:

- legacy GF problem/config/launcher assets are unchanged by this feature;
- the new launcher uses a distinct problem name, Redis prefix, cache, and
  `data_gf_islands<matrix>` store;
- every initial program enters `ab_initio`;
- parent selection for target sizes `0`, `1..7`, and `8+`;
- bootstrap mixing at `0.70` and steady one-parent mixing at `0.10`;
- no mutation replaces both local parents after bootstrap;
- the child always enters the sampled destination island;
- no saved-path block or handles appear in an `ab_initio` prompt;
- both refinement regimes receive the same unified selectable inventory;
- dead, stale, and non-improving paths are absent;
- every displayed path includes policy/path evidence;
- persisted and derived path cards render compatible fields;
- aux evidence is filtered according to the three profiles;
- evolutionary statistics and memory obey island boundaries;
- refinement routes remain unavailable until a selectable path exists;
- prompt-budget reduction is deterministic and never leaves a bare path name.
