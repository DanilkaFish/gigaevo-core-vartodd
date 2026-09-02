# VarTODD Ultra-Expensive Initial Programs Design

## Goal

Add an eight-program initial pool for matrices where one complete policy
evaluation can take as long as ten minutes. The intended problem shape is an
ab-initio descent from roughly rank 5000 to rank 3800, with a maximum search
depth of at least 1000 so depth does not truncate a long successful descent.

Each initial program should normally finish below the 2--3 hour program cap.
The initial target is approximately 40--100 minutes per program, leaving room
for evolved children to use more of the cap.

## Why There Is No Numerical Optimizer

PSO, DE, CMA-ES, PatternSearch, and similar methods spend a substantial part
of a small budget merely initializing a population or local model. When only
a handful of complete descents are affordable, their convergence semantics
are weak and their hyperparameters obscure what was actually tested.

The new pool therefore uses small, explicit fixed-profile portfolios. A
program selects a profile, rebuilds the evaluator policy, and runs one seed.
The profile lists are experiments chosen by construction rather than points
proposed by an optimizer.

This restriction applies only to the new seed pool. Evolution may later add a
different search procedure when evidence supports it.

## Pool and Launcher Contract

Create:

`problems/vartodd_evo_gf/initial_programs_ultra_expensive/`

Register it under launcher value:

`initial_programs=ultra_expensive`

Both `run_gf.py` and `run_gf_islands.py` must accept that value. The island
launcher continues to link the selected initial pool from the shared
`vartodd_evo_gf` problem.

When `initial_programs=ultra_expensive` is selected and `call_timeout` is not
specified, the effective call timeout is 9000 seconds. An explicit
`call_timeout` continues to take precedence. All other initial pools keep the
existing 3800-second default. Generated metrics use the effective timeout as
the runtime upper bound and sentinel, as they do for other pools.

The overlay and execution-cache names continue to include the selected pool,
so this pool does not reuse another pool's overlay or cache namespace.

## Shared Program Contract

Every ultra-expensive program:

- is a standalone Python seed loaded by the existing directory loader;
- imports `INITIAL_RANK` and `TARGET_FINAL_RANK` from `helper`;
- starts with `Evaluator(path_name="init", ...)`;
- passes `fin_rank=TARGET_FINAL_RANK` explicitly, preventing an unnecessary
  descent toward `BaseEvaluator`'s unrelated default final rank;
- derives rank switches from fractions of
  `INITIAL_RANK - TARGET_FINAL_RANK`, not embedded GF-specific ranks;
- uses `max(1000, round((INITIAL_RANK - TARGET_FINAL_RANK) / 2))` as its
  maximum depth, so every program allows at least 1000 search steps while
  remaining matrix-relative for substantially larger rank spans;
- uses no optimizer library, `ElementwiseProblem`, `minimize`, optimizer
  class, or `optimize*` function;
- declares no `map_par` parameter slots and calls the evaluator with `[]`;
- runs one seed per expensive call rather than hiding multiple descents in one
  call;
- makes between four and eight expensive policy calls in its successful
  maximal path;
- calls `get_best()` exactly once and returns it from `entrypoint()`;
- leaves TOHPEprefix disabled by using only the current action API;
- ends with an `AUX_DESCRIPTION` explaining its role, cost, and intended
  evidence.

The programs expose `SEARCH_STRATEGY = "fixed_policy_portfolio"` or a more
specific fixed-strategy label and `TARGET_POLICY_EVALUATIONS` matching the
maximum planned number of expensive calls. Profile tables, seeds, rank
fractions, caps, pools, and beams are visible constants or straightforward
literal data.

Every `TohpeSearch` sampling budget uses numeric `one_hot`, `sparse`, and
`dense` values no larger than 10. The `one_hot="all"` form is not used in this
pool. This bound applies only to TOHPE: light or scheduled TODD may retain a
larger sampling budget when its profile intentionally spends work on
lower-region action discovery.

## Profile Execution

An evaluator may store a `profile_index` or a named profile before
construction. For each full-descent trial it:

1. selects the next fixed profile;
2. calls `reinit()` so `policy_mapping()` installs that profile;
3. calls the evaluator with `[]` and a one-element seed tuple;
4. lets the evaluator retain the best path across trials.

Changing a fixed profile must not change any mapped-parameter layout; the new
pool has no mapped parameters.

Restart programs may call `set_up_new_init()` between trials. They still use
fixed policies and thresholds rather than fitting an optimizer. If a restart
cannot be constructed, the program returns the best completed path instead of
failing.

## The Eight Programs

### 1. `tohpe_greedy_profiles.py`

Six ab-initio TOHPE-only profiles using `mode="best"`. Profiles vary low beam
width, retained pool size, TOHPE keep/reserve, sampling, and z choices. TODD
is explicitly disabled. This is the deterministic, inexpensive source
baseline.

### 2. `tohpe_softmax_profiles.py`

Six TOHPE-only profiles using softmax selection with several fixed
temperatures and beam widths. It explores whether stochastic trajectory
selection gives a better descent than the greedy baseline without adding TODD
cost.

### 3. `score_shape_profiles.py`

Eight profiles sharing one moderate TOHPE generation policy while varying
fixed exploration/finalization weights and selected finalization centers.
This isolates the most influential policy lever without creating a continuous
parameter space.

### 4. `late_todd_schedule_profiles.py`

Six rank-scheduled profiles. Early search is TOHPE-dominated with TODD
disabled or light; lower bands enable finite TODD. Profiles vary the switch
fraction and a modest terminal cap while keeping the schedule readable.

### 5. `todd_bucket_breadth_profiles.py`

Five profiles concentrating on lower-region TODD action starvation. They vary
`max_buckets`, `limit_bucket`, actions per bucket, and retention. Wider
`max_buckets` represents research of more distinct z buckets; limits remain
finite and explicit.

### 6. `wide_beam_profiles.py`

Four profiles using progressively wider beams and matching final pools. The
small profile count compensates for the higher per-evaluation tree cost. This
tests trajectory coverage independently of numerical optimization.

### 7. `fixed_restart_ladder.py`

One full fixed-policy descent followed by at most five fixed restart trials.
Restart thresholds form a visible ladder derived from the rank span. The tail
policy is fixed or selected from a short literal table; no parameters are
fitted.

### 8. `seed_trajectory_portfolio.py`

One balanced fixed policy evaluated sequentially over eight individual seeds.
This is the stochastic-trajectory control: unlike other programs, its policy
does not vary, so observed differences come from search randomness.

## Cost Model

The maximum intended expensive calls are:

| Program | Calls |
|---|---:|
| TOHPE greedy | 6 |
| TOHPE softmax | 6 |
| Score shapes | 8 |
| Late TODD schedules | 6 |
| TODD breadth | 5 |
| Wide beams | 4 |
| Restart ladder | 6 |
| Seed trajectories | 8 |

At ten minutes per full descent, the nominal maxima range from 40 to 80
minutes. Restart trials may be shorter because they begin from an inherited
partial path. More expensive beam or TODD profiles have fewer calls. The
9000-second timeout is a safety cap, not a target that every initial seed must
consume.

## Validation and Failure Behavior

- A completed earlier trial remains salvageable if a later trial reaches the
  soft deadline.
- Fixed profiles must contain valid action-pool/source-pool relationships and
  finite TODD caps.
- Disabled TODD uses the established zero-budget/zero-pool representation.
- Empty seed lists and optimizer fallbacks are not present.
- Profile selection is deterministic from source constants.

## GF64 Follow-up: Dimension-Preserving Alternatives

The first GF64 execution produced one rank-3847 path from the smallest greedy
TOHPE profile, while completed larger-pool profiles stopped around rank
4847--4891. Preserve `tohpe_greedy_profiles.py` exactly as the control.

For the other seven programs:

- reduce TOHPE sampling to roughly 2--6 one-hot, 0--3 sparse, and 0--1 dense
  candidates;
- make a light TODD fallback available from the initial rank with
  `min_buckets=10`, `max_buckets=200`, a finite limit between 200 and 1000,
  nonzero keep, and reserve 0 or 1;
- retain wider scheduled terminal TODD where a program already tests it;
- give four programs an explicitly positive final TOHPE weight;
- give three programs negative exploration and final reduction weights to
  test whether avoiding aggressive early reduction preserves dimension.

Only programs still incomplete in the GF64 CSV receive a further call-budget
reduction: wide beam 2 to 1, late TODD 3 to 2, and seed trajectory 4 to 2.

## Tests

Add a dedicated structural test module for the new pool. It verifies:

- exactly the eight expected `.py` files exist;
- all files parse and end with `AUX_DESCRIPTION`;
- every program starts ab initio and explicitly passes
  `fin_rank=TARGET_FINAL_RANK`;
- maximum depth is derived from the rank span and is at least 1000;
- every TOHPE `one_hot`, `sparse`, and `dense` sampling value is numeric and at
  most 10, without imposing the same cap on TODD sampling;
- `TARGET_POLICY_EVALUATIONS` matches the approved 6/6/8/6/5/4/6/8 budget;
- there are no optimizer imports/classes/functions, `ElementwiseProblem`,
  `minimize`, or `map_par` calls;
- evaluator calls use an empty parameter vector and one seed per call;
- TOHPEprefix and retired controls do not appear;
- all eight strategy roles are distinct and readable.

Extend launcher tests to verify:

- `ultra_expensive` is accepted by both launchers;
- each runtime overlay links the new directory;
- the pool-specific overlay and cache names are isolated;
- omitted `call_timeout` yields 9000 seconds and matching runtime metrics;
- explicit `call_timeout` overrides 9000;
- existing pools continue to default to 3800 seconds;
- help text documents the new pool and its timeout behavior.

## Out of Scope

This change does not alter evaluator search mechanics, score formulas, path
storage, island routing, mutation prompts, existing initial pools, or the
global experiment timeout configuration. It does not benchmark the eight
programs on a real ten-minute evaluation during unit tests.
