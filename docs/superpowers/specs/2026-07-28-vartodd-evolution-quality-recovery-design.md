# VarTODD Evolution Quality Recovery Design

## Objective

Recover the search quality demonstrated by the archived GF16 evolutions while
keeping the current shared `vartodd_evo_gf` action API and optimizer-group API.
The change addresses four diagnosed regressions:

1. mutation guidance biased evolution toward small finite TODD caps;
2. generated insights confused low-dimensional y saturation with z-space
   coverage;
3. the grouped TOHPE seed optimized interacting policy blocks in isolation and
   restarted between them;
4. the prompt made optimizer novelty too salient and encouraged multi-axis
   mutations without causal isolation.

The initial six-program pool will also be reduced from roughly 20--38 mapped
parameters per program to a budget-aware portfolio of 12--24 parameters.

## Evidence From Archived Runs

The complete eight-hour GF16 RDB, rather than its stale CSV snapshot, reached:

- rank 389 with a finite TODD cap around 2342;
- rank 387 after escalating the same broad lineage to full search.

Four separate rank-383 backups show the consistent progression:

1. use finite search to obtain a productive rank-389/rank-385 path;
2. reopen that path near the terminal band;
3. switch the terminal TODD policy to `limit_bucket=-1`;
4. control cost through `min_buckets`, the soft `max_buckets` reserve target,
   y sampling, retention, action-pool size, beamwidth, and optimizer budget.

The rank-383 programs used approximately 12, 16, 23, and 31 mapped parameters.
The compact 12--23-dimensional programs demonstrate that an initial program
does not need more than 24 parameters. The 31-dimensional program worked only
with a correspondingly large DE budget.

## Mutation-Regime Guidance

The algorithm prompt will describe a staged z-search policy rather than a
universal cap rule:

- Fresh and early/middle-rank search normally starts with a finite hard cap.
- If actual z research reaches `max_buckets`, first determine whether the soft
  reserve target stopped search before the hard cap.
- If actual research reaches a finite `limit_bucket` and improvements are
  still arriving or the terminal pool is generation-starved, raising the hard
  cap is a falsifiable next experiment.
- A productive capped path near the terminal frontier may be reopened with
  `limit_bucket=-1`. Full search is an escalation, not a default and not a
  forbidden cost.
- Cost under full hard search is controlled with rank schedules, `min_buckets`,
  `max_buckets`, sampling, retention, actions per bucket, beamwidth, pool size,
  branch point, and optimizer budget.

The guidance must keep the three z controls distinct:

- `min_buckets`: guaranteed research;
- `max_buckets`: soft reserve-target stopping control;
- `limit_bucket`: hard number of z candidates considered, with `-1` meaning
  the full current candidate space.

## y-Space and z-Space Interpretation

For a fixed researched z bucket, a binary null space of dimension `dim`
contains at most `2**dim - 1` nonzero y vectors. This bounds useful y sampling
and `actions_per_bucket` for that one bucket.

It does not bound:

- the number of z buckets that may need to be researched;
- the position of rare productive z buckets;
- the total candidate z space;
- whether a finite hard cap covers the useful terminal region.

Low `dim` may justify reducing redundant y work after the distinct y space is
covered. It must never, by itself, justify claiming that z coverage is
complete. High z research with low acceptance is ambiguous: it can indicate
over-provisioned soft research, or rare productive buckets that require a
larger/full hard search. Rank progress, binding min/max/limit controls, accepted
source counts, and runtime decide between those interpretations.

## Mutation Scope and Optimizer Reasoning

Each mutation will identify one primary causal hypothesis. Multiple concrete
values may change only when they implement one named mechanism, such as:

- reallocating budget from saturated y sampling to z research;
- replacing an exhausted optimizer segment while holding the policy fixed;
- aligning retention and final-pool capacity;
- switching a productive capped terminal path to a controlled full-search
  schedule.

The mutation prompt will no longer require a manufactured failure at both the
policy and optimizer levels. It will require:

1. a primary bottleneck supported by raw execution evidence;
2. the level being changed: policy, optimizer, or their interaction;
3. the important controls intentionally held fixed;
4. one predicted observation in the next execution output.

An optimizer-family change requires optimizer-specific evidence such as a flat
tail, collapsed decoded profiles, unsuitable population coverage, or a restart
that produced no new configurations. A policy-only hypothesis should normally
preserve the optimizer so the result remains attributable.

## Score Priority

Exploration and finalization scores are normally the most influential mapped
policy parameters:

- `ExplorationScore` determines which generated candidates survive source-pool
  filtering and enter the merged action pool.
- `FinalizationScore` determines which retained candidates remain competitive
  for action selection.
- A sign or scale error can systematically discard the rare high-reduction
  action that expensive TODD research produced.
- Score weights affect every reached rank step, while many sampling, pool, and
  bucket parameters bind only in particular bands.

Therefore initial programs and mutations should preserve meaningful score
weight freedom before exposing many low-level budget knobs. When candidates
are generated but useful source contributions disappear between `src` and
`pool`, or a sufficiently filled pool does not translate into descent, score
weights are the first policy parameters to inspect.

Scores are not a substitute for generation. They cannot select an action that
TOHPE, TOHPEprefix, or TODD never generated. When all relevant source counts
are near zero, generation and z/y coverage remain the primary bottleneck.
Score centers are lower-priority than weights and should normally be fixed
unless execution evidence shows that displacement matters.

## Parameter-Dimension Policy

The six initial programs form a portfolio:

- four compact programs with 12--18 mapped parameters;
- two medium programs with 19--24 mapped parameters;
- no initial program above 24 mapped parameters.

Parameters are retained only when they affect a rank region reached by the
program and expose a meaningful behavioral choice. The following become fixed
literals by default:

- score centers without evidence that displacement matters;
- saturated y-sampling budgets;
- settings for disabled or ineffective sources;
- duplicated controls that move the same cost/behavior together;
- schedule values for unreachable rank bands.

Within each program's dimension budget, score weights receive priority. A
typical compact program should devote most continuous dimensions to
exploration/finalization weights and expose only the few source, pool, bucket,
or selection controls that define its distinct search mechanism.

Evolution may create a program above 24 dimensions, but its code and
justification must match dimension to credible optimizer coverage. The prompt
will not impose a universal dimension cap on evolved programs.

## Initial-Program Portfolio

The six existing optimizer families remain distinct. Their policy
parameterizations will be simplified without making their mechanisms
identical.

- Four light/compact seeds use finite early and terminal caps and expose only
  their defining policy controls.
- One medium seed represents efficient moderate-cap terminal search, similar
  to the archived route that reached rank 389.
- One medium heavy-tail seed uses a rank schedule with finite early search and
  `limit_bucket=-1` only in the terminal band, representing the successful
  rank-389/rank-385 to rank-383 escalation.

Exactly two programs continue to demonstrate explicit parameter-group
optimization.

## Grouped TOHPE Seed

The TOHPE-focused seed will use approximately 17 total parameters:

- about 11 score weights with fixed, transparent centers;
- about 6 search-structure parameters.

It will:

1. select all parameter groups and run joint TPE optimization so scores and
   search structure are evaluated together;
2. retain the resulting full `x0`;
3. select only the score group and perform CMA refinement on the same evaluator
   and same path state;
4. avoid `set_up_new_init` between the group stages.

Any later branch/restart must occur only after the joint and score-refinement
stages and must carry the combined full parameter state.

## Tests

New source-level contracts will verify:

- mutation guidance contains no universal 10,000 cap;
- finite, raised, and full-search escalation are all documented;
- y-per-z saturation is explicitly separated from z coverage;
- the prompt requires one primary causal hypothesis and controlled variables;
- optimizer changes require optimizer evidence;
- score weights are documented as the highest-priority mapped policy
  parameters without claiming that scores can repair missing generation;
- the grouped TOHPE seed jointly optimizes before score-only refinement and
  does not branch between those stages;
- each initial program's mapped-parameter count falls in its assigned compact
  or medium band;
- the pool still contains six optimizer families and exactly two explicit
  grouped programs;
- at least one initial program contains finite early TODD and full terminal
  TODD.

Existing action-API and parameter-group tests remain applicable. Legacy prompt
tests for obsolete problem copies are not extended.
