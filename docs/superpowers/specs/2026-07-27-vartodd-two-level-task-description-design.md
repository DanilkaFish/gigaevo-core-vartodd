# VarTODD Two-Level Task Description Design

## Objective

Rewrite `problems/vartodd_evo_gf/task_description.txt` as one self-contained,
two-layer prompt. The prompt must make the LLM reason about both coupled
problems in every mutation:

1. designing an effective rank-dependent policy parameterization; and
2. designing an effective numerical optimization procedure for that
   parameter space.

The rewrite should be materially shorter and less repetitive while retaining
the mechanics, evidence rules, lifecycle constraints, and guardrails needed to
produce valid programs.

## Core Principle

Every mutation must diagnose both levels, but it does not have to modify both.
The mutation may change only the policy or only the optimizer when evidence
isolates the failure. Coordinated changes are appropriate when parameter-space
shape, policy cost, restart behavior, or optimizer dynamics interact.

The prompt must explicitly teach these failure modes:

- A capable optimizer cannot recover behavior excluded by the policy space.
- A rich policy space is ineffective when the optimizer cannot explore it.
- Weak, saturated, or quantized mappings can make many vectors produce the
  same policy.
- Expensive policies reduce the optimizer evaluations available under the
  call timeout.
- Conditional schedules and restarts can change the active parameter count.

## Layer A: Decision Framework

### 1. Objective and evaluation contract

State validity and reliable final rank as the first priorities. Runtime is a
secondary comparison signal and is not an improvement by itself. Distinguish
fresh builders from saved-path refiners and describe only fitness shaping that
matches the current validator.

Use matrix-independent language and the exported `INITIAL_RANK` and
`TARGET_FINAL_RANK` concepts. Do not describe the shared problem as fixed
GF32/rank 567.

### 2. The two coupled optimization problems

Define policy parameterization as the design of source mixture, y sampling,
z coverage, retention, scores, action selection, rank schedules, and the
choice of fixed versus optimizer-controlled values.

Define optimizer procedure as optimizer family, initialization, population,
bounds, parameter mappings, multi-seed objective, evaluation allocation,
stages, restarts, continuation, and reinitialization.

Give their interaction its own subsection. Explain that parameterization
defines the landscape while the optimizer determines how effectively that
landscape is explored.

### 3. Mandatory evidence-first diagnosis

Require each mutation justification to cover:

1. policy diagnosis;
2. optimizer diagnosis;
3. interaction diagnosis;
4. the selected intervention level; and
5. the expected observable change in the next execution report.

This requirement belongs near the beginning, before detailed API reference.

## Layer B: Technical Reference

### 4. Valid program and evaluator lifecycle

Consolidate imports, local `map_par` mappings, evaluator construction,
`policy_mapping`, `extract_active`, `run`, `set_up_new_init`, saved-path and
fresh starts, and the single final `get_best()` call. State that the optimizer
problem must be rebuilt when a restart or conditional mapping changes the
active parameter count.

### 5. Policy parameterization reference

Define action `(z, y)`, the three sources, sampling, source retention,
z-bucket search, merged pool, scores, selection, and rank schedules once.

Preserve and strengthen these rules:

- scores select generated candidates and do not create recall;
- accepted-source counts and retained-pool counts are different;
- low-dimensional y spaces saturate;
- `max_buckets` is a reserve-driven soft cap;
- `limit_bucket` is the hard cap;
- aggregate `z` evidence is not automatically TODD-only;
- beamwidth helps only when several useful actions exist.

Compress the current asymptotic discussion into a short list of runtime
multipliers rather than a long derivation.

### 6. Optimizer procedure reference

Give optimizer design comparable prominence to policy mechanics. Explain that
the objective is noisy, discrete, piecewise-constant, and expensive. Cover
mapping geometry, integer plateaus, bounds, population size, seed count,
evaluation allocation, staged search, continuation versus reinitialization,
and optimizer-family choice from evidence.

Do not prescribe one universal optimizer. Require dimensions to come from
`len(evaluator.extract_active())` for the current stage.

### 7. Reading evidence

Merge report definitions and diagnostic advice so each field is explained
once:

- Policy evidence: `path_policy_groups`, configured profiles, source
  acceptance, retained pools, dimensions, z research, and score tendencies.
- Optimizer evidence: rank quantiles, segment yield, `last_improvement`,
  `best_seen_times`, restart outcomes, and total evaluations.
- Joint evidence: runtime per useful improvement, policy cost versus optimizer
  trial count, and whether distinct vectors produce distinct descents.

Keep early/mid/terminal band interpretation inside this evidence section.

### 8. Saved paths and restart strategy

Keep all Live Path Store and restart rules in one section: selectable names,
`f<final>_i<loaded>_<hash>_lim<cap>`, loaded-path rank versus branch rank,
near-tail versus wide-margin roles, margins, reuse evidence, and the
difference between saved-path names and in-memory path indices.

### 9. Mutation checklist and guardrails

End with a short operational checklist. A mutation must identify the reached
rank region, diagnose both levels, cite raw evidence, choose a coherent
intervention, predict report-level effects, and preserve lifecycle/path rules.

Retain these guardrails:

- negative score weights are not automatically defects;
- configured capacity must be compared with actual utilization;
- aggregate z research must not be attributed to TODD without source evidence;
- cheap runtime alone is not success;
- saturated y budgets should not be raised;
- exhausted optimizer segments should not simply receive more evaluations;
- productive late segments may justify more budget;
- a multi-change improvement does not prove one changed mechanism caused it.

## Material to Remove or Correct

- Duplicate explanations of starts, margins, restarts, rank bands, pool
  underfill, and y-sampling saturation.
- Fixed GF32/rank-567 statements.
- Old `limit_<cap>` path-suffix wording.
- Fitness penalties and rewards that do not match current validator constants.
- Overly absolute claims such as “cheap runtime is harmful evidence.”
- Long constructor examples where a compact signature is sufficient.
- Regime-specific advice already supplied by algorithm YAML.
- Grammar errors and inconsistent TODD/ToDD terminology.

## Scope

This change rewrites only the shared `task_description.txt`. It does not alter
search implementation, validation, metrics, algorithm-regime probabilities,
mutation output schema, or prompt-loading code.

## Verification

Verification should check that:

- the prompt contains explicit policy, optimizer, and interaction diagnoses;
- all current public helper APIs referenced by the prompt exist;
- fixed GF32/rank-567 and old path-suffix language are absent;
- validator claims match `validate.py`;
- duplicated concepts occur in one authoritative section;
- existing shared-problem contract tests still pass.
