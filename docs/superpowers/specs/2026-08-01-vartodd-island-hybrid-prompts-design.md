# VarTODD Island Hybrid Prompts Design

## Goal

Strengthen the dedicated `vartodd_evo_gf_islands` prompt stack without
discarding either of its useful sources:

- preserve the statistically detailed insight, lineage, and mutation contracts
  restored from the successful `problems/vartodd_evo` setup;
- preserve the current island task description's matrix-generic API, grouped
  parameter model, unified selectable-path cards, and route semantics;
- give each island a detailed, binding search playbook instead of a short
  destination label.

This is a prompt-only change. Parent-selection pressure, route probabilities,
archive admission, fitness penalties, path-store selection, and evaluator code
remain unchanged.

## Prompt Ownership

The prompt stack uses three layers with non-overlapping responsibilities.

### Shared analytical contracts

The restored files under `prompts/insights` and `prompts/lineage` remain
unchanged. They retain the successful setup's evidence schema, population
statistics interpretation, diff quantification, and transferable-lesson rules.

The restored mutation system prompt also remains the base contract. It retains
the successful setup's archive-percentile gate, archetype selection, evidence
precedence, primary-parent selection, causal hypotheses, and output schema.

### Shared VarTODD mechanics

`task_description.txt` remains the source of truth for mechanics that apply to
all three regimes:

- the current GF-island helper API;
- policy mapping and stable parameter groups;
- action generation, retention, scoring, and selection;
- optimizer lifecycle, active-vector dimensions, and restarts;
- selectable shared-path card semantics;
- execution-evidence interpretation;
- the common cost model and evidence-to-mutation decision process.

It describes the three regimes only at a high level. Detailed regime strategy
does not belong here.

### Binding island playbooks

Each file in `prompts/islands` owns one route's requirements and search
strategy. The selected overlay is appended after assignment, parent evidence,
and external path cards, so it can refer directly to the evidence visible in
that mutation call.

Shared API definitions are not duplicated into every overlay. Each overlay may
name shared controls and evidence fields while relying on `task_description.txt`
for their definitions.

## Shared Mutation-Prompt Addendum

Add a compact island-specific section to the restored mutation system prompt.
It must explain:

- `Mutation Assignment` is binding for the generated child;
- a `target_island` parent is normally the primary implementation base;
- `bootstrap_donor` and `mixed_donor` parents contribute mechanisms and
  evidence but do not change the destination regime;
- when all parents are bootstrap donors, select the strongest implementation
  base and transform it to satisfy the assigned regime;
- evolutionary statistics are island-local;
- only exact names in the current `Selectable Shared Paths` block are loadable;
  historical handles in code, aux, lineage, or comments are evidence only.

This section augments rather than replaces the restored generic multi-parent
and evidence rules. If the two overlap, the island role assignment is the more
specific constraint.

## Task Description Changes

Retain the current island task description's structure and API. Add the useful
diagnostic content that the successful `vartodd_evo` task description contains,
adapted to the current implementation.

### Quality and cost priority

State the decision order:

1. valid execution;
2. lower reliable final rank;
3. productive runtime only when rank quality is not weakened.

Do not reproduce obsolete numeric fitness shaping or an absolute minimum
runtime. Runtime is interpreted through the work it purchased.

Describe approximate program cost as proportional to:

```text
optimizer evaluations × seeds/seed parallelism × reached path depth
× effective beamwidth × per-step action-generation cost
```

Use this as a reasoning model, not a measured runtime formula. Every runtime
recommendation must name the multiplier being increased, decreased, or moved.

### Sampling versus optimizer breadth

Explain two legitimate sampling strategies:

- more complete action sampling reduces per-policy stochastic variance and
  makes scoring among generated actions more representative, but increases
  each evaluation's cost and reduces the number of parameter vectors that fit
  in the deadline;
- lighter stochastic sampling makes each evaluation less complete but permits
  more optimizer evaluations, restarts, seeds, or independent trajectories,
  increasing the number of chances to discover an unusually productive path.

Neither strategy is universally preferred. The model must use actual source
acceptance, pool fill, rank quantiles, segment yield, and repeated-best counts
to choose between them.

The finite-y-space qualification is binding: a binary null space of dimension
`dim` has `2**dim - 1` nonzero vectors. High-dimensional early spaces cannot
normally be exhausted, whereas terminal dimensions `1..3` saturate after only
`1..7` distinct nonzero vectors. Once y is saturated, more y samples do not
increase recall; z coverage, retention, branch point, or trajectory breadth
becomes the relevant lever.

### Rank-band diagnosis

Describe the common observed bands:

- early: high dimension, strong TOHPE acceptance, full or nearly full pools,
  and large reductions; deep TODD normally wastes budget here;
- middle: declining dimension and acceptance, where a scheduled TODD
  transition may become productive;
- terminal: low dimension, rare positive actions, high z research,
  underfilled pools, and small reductions; z coverage and path traversal often
  matter more than fine score tuning.

Schedules must target rank regions actually reached in the evidence. A
threshold outside the reached region is inactive.

### Optimizer and restart diagnosis

Use `search_stat` segments to distinguish:

- improvement only at the beginning followed by silence: an exhausted basin;
- a restart with no improvement: a poor branch/restart placement or the same
  basin re-entered;
- improvements near the end of a segment: productive remaining search budget;
- `best_seen_times` close to `total_evals`: repeated saturation;
- flat rank quantiles: many vectors map to the same effective policy/path;
- broad rank quantiles: the parameters still influence behavior and optimizer
  tuning may pay.

The response must move budget away from dead segments, preserve productive
segments, and rebuild optimizer state after parameter-group or active-vector
changes.

### Shared actionability rules

Preserve and expand the current configured-versus-actual checks:

- increase final pool size only when the actual pool reaches its capacity;
- diagnose generation or retention when a pool is underfilled;
- infer source contribution from accepted `src=H/T` counts, not configured
  sizes alone;
- use wider beam only when several distinct positive actions survive;
- treat score signs according to the score formula and same-region evidence;
- for a failed path refiner, change a causal mechanism rather than replaying
  the same path, margin, and policy.

## Ab-Initio Overlay

The `ab_initio` overlay requires `Evaluator(path_name="init")` and forbids
loading or inferring a saved path.

Its objective is to construct a strong reusable full path while maximizing the
number of useful policy hypotheses tested. Guidance must:

- prefer TOHPE as the main early action source;
- keep early TODD disabled or light so deep z research does not consume the
  optimizer budget in easy bands;
- move saved cost into optimizer evaluations, suitable population size,
  independent seeds, and evidence-supported restarts;
- explicitly compare more-complete sampling with lighter stochastic sampling;
- use source acceptance, pool fill, rank quantiles, `last_improvement`, and
  `best_seen_times` to decide which sampling/budget strategy to test;
- introduce a scheduled heavier TODD policy only in lower bands where TOHPE
  starvation or hard-frontier evidence supports it;
- examine optimizer family, initialization, population/social or differential
  parameters, termination, restart placement, and staged parameter groups;
- require one causal budget-allocation hypothesis rather than unrelated knob
  changes.

The overlay must not imply that exhaustive y sampling is possible in large
high-dimensional early spaces.

## Mid-Margin Overlay

The `mid_margin` overlay requires an exact name from the current
`Selectable Shared Paths` block and a margin in `30..100`.

Its objective is to reopen enough inherited depth to create an alternative
descent. Guidance must:

- select a path using final rank, `u/i`, start/depth/band information, policy
  bands, and producer-search yield;
- choose the margin so the branch occurs before the inherited failure region;
- allow heavier TODD than ab-initio because less total depth is replayed;
- keep TODD scheduled to the hard reached bands rather than paying its cost
  across the inherited easy prefix;
- compare spending the shorter-depth budget on optimizer evaluations,
  sampling, z coverage, beamwidth, seeds, and restarts;
- apply the same segment-yield and plateau analysis as ab-initio;
- compare the final result directly with `loaded_path_rank`;
- prohibit replaying a failed path/margin/mechanism without a material change
  to path choice, branch point, optimizer flow, source balance, recall,
  scoring, beamwidth, or restart structure.

## Near-End Overlay

The `near_end` overlay requires an exact current selectable path and a margin
in `5..30`.

Its quality-first objective is to research the terminal positive actions and
plausible alternative trajectories as thoroughly as the deadline permits.
The inherited prefix makes the path shallow, so the saved depth budget may be
spent on terminal action and trajectory breadth.

Guidance must:

- treat flat rank quantiles, repeated best ranks, and replayed tails as evidence
  that many optimizer vectors map to the same effective behavior;
- move budget from fine optimizer tuning to direct action/path coverage when
  such a plateau is present;
- use heavy TODD and high or unlimited z coverage when terminal action
  starvation is evidenced;
- explain that `min_buckets` is guaranteed research, `max_buckets` is a soft
  reserve-driven target, and `limit_bucket=-1` removes the hard cap;
- keep source reserve/retention and `actions_per_bucket` large enough that
  discovered terminal alternatives survive and the intended z research is not
  rendered ineffective;
- avoid increasing y sampling after the low-dimensional y space is saturated;
- use high beamwidth only when several distinct positive actions survive, and
  use it to traverse competing terminal paths rather than merely rescore one
  action;
- allow wide-beam or near-exhaustive trajectory traversal as a deliberate
  alternative to optimizer search on a flat mapping;
- compare the final result directly with `loaded_path_rank`.

The overlay does not claim full z search is always affordable. It makes high or
full terminal coverage the preferred response when evidence shows action
starvation and the shallow path leaves sufficient budget.

## Files and Scope

Modify:

- `problems/vartodd_evo_gf_islands/task_description.txt`;
- `problems/vartodd_evo_gf_islands/prompts/mutation/system.txt`;
- `problems/vartodd_evo_gf_islands/prompts/islands/ab_initio.txt`;
- `problems/vartodd_evo_gf_islands/prompts/islands/mid_margin.txt`;
- `problems/vartodd_evo_gf_islands/prompts/islands/near_end.txt`;
- prompt-focused tests under `tests/problems` and, only if needed for composed
  prompt assertions, the existing island integration test.

Do not modify:

- restored `prompts/insights/*` or `prompts/lineage/*`;
- mutation user prompt;
- route probabilities or bootstrap mixing;
- `lambda_` or elite-selector implementation;
- validator penalties or archive admission;
- path-store filtering or evidence-card rendering;
- legacy `problems/vartodd_evo` and `problems/vartodd_evo_gf` prompts.

## Validation and Testing

Prompt tests must verify semantic contracts without snapshotting entire files.
They should assert:

- the mutation prompt retains restored statistical/archetype rules and adds
  parent-role, path-authority, and island-local-statistics rules;
- the task description retains grouped parameter APIs and unified path-card
  semantics;
- the task description contains cost multipliers, dual sampling strategies,
  finite-y-space reasoning, rank-band diagnosis, and optimizer-segment
  diagnosis;
- ab-initio requires `path_name="init"`, contains no loadable path inventory,
  prioritizes early TOHPE and optimizer/restart breadth, and covers both
  sampling strategies;
- mid-margin requires an exact selectable path, margin `30..100`, evidence-
  selected branching, scheduled heavier TODD, and `loaded_path_rank` comparison;
- near-end requires margin `5..30`, terminal plateau diagnosis, high/unlimited
  z coverage semantics, y-space saturation, evidence-gated high beamwidth, and
  `loaded_path_rank` comparison;
- retired or unavailable controls remain absent.

Run the focused prompt, provider, and routed-flow tests. No evaluator execution,
network service, or long VarTODD run is required for this prompt-only change.

## Success Criteria

- The restored statistically rich analytical prompts remain intact.
- Every mutation receives an unambiguous island assignment and path authority.
- Shared mechanics are defined once in the task description.
- Each island overlay provides a detailed, evidence-driven playbook with a
  distinct compute-allocation strategy.
- Recommendations distinguish action-generation completeness from optimizer
  breadth and correctly account for finite terminal y spaces.
- The near-end route explicitly supports high/full terminal z and trajectory
  coverage as an escape from optimizer plateaus.
- No prompt describes obsolete APIs, retired controls, or stale fitness rules.
