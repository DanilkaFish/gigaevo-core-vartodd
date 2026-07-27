# VarTODD Optimizer Freedom Prompt Design

## Goal

Give evolved programs freedom to choose among useful installed derivative-free
optimizers while making optimizer choice evidence-driven rather than
library-driven.

## Available libraries

The task description will explicitly allow the optimizer packages already
importable in the evolution runtime:

- NumPy;
- SciPy;
- pymoo;
- PySwarms;
- cma;
- Optuna;
- Nevergrad.

Evolved programs must not install packages or access the network at runtime.
No new project dependencies are part of this change.

## Landscape model

The optimizer guidance will explain that one objective call runs a costly
FastTODD policy evaluation. The raw objective is:

- discontinuous because small policy changes can select different actions and
  descent paths;
- piecewise constant because many continuous coordinates map to the same
  integer or categorical configuration;
- noisy or seed-sensitive when sampling and FastTODD seeds vary;
- conditionally dimensional when schedules or branches expose different
  active parameters;
- non-uniform in cost because policy settings change beam, sampling, bucket,
  retention, depth, and restart work.

Consequently, raw gradients are normally uninformative. Mapping design and
behavioral diversity must be diagnosed before spending more evaluations or
changing optimizer libraries.

## Optimizer families

The prompt will describe optimizer families with libraries as examples:

- population global search: DE and PSO through pymoo, SciPy, or PySwarms;
- evolution strategies: CMA-ES through cma, pymoo, or Nevergrad;
- model/adaptive search: Optuna TPE and Nevergrad portfolios;
- stochastic global search: SciPy differential evolution, dual annealing, and
  basin hopping where appropriate;
- low-discrepancy or randomized scouting: NumPy and SciPy QMC;
- local derivative-free refinement: Powell or Nelder-Mead only around a
  productive basin, because plateaus can stop them immediately.

The prompt will not prescribe one universally best optimizer or provide a
large API cookbook.

## Evidence-driven choice

Guidance will connect execution evidence to optimizer decisions:

- broad quantiles with rare strong outcomes favor global scouting followed by
  refinement;
- population collapse or repeated identical policies favors restarts,
  diversity, or different mappings;
- flat quantiles and fixed profiles indicate mapping/policy insensitivity,
  which another optimizer cannot repair;
- improvements near the end of a segment support continuation or more budget;
- long flat tails support a changed mechanism, initialization, bounds, or
  branch point rather than identical continuation.

## Budget and state safety

Programs must reason in actual objective calls:

```text
policy evaluations = population size * generations or iterations
FastTODD runs = policy evaluations * seeds per evaluation
```

Library iteration counts are not comparable without this conversion.
Parallel candidate evaluation is allowed only with isolated evaluator state
and explicit merging of discovered paths/statistics. Seed parallelism through
`run(..., max_workers=...)` remains distinct from candidate parallelism.

Warm starts are valid only when parameter meanings, bounds, and active
dimension remain compatible. After `set_up_new_init` or conditional mapping,
the optimizer problem must be rebuilt from the current active vector.

## Tests

Prompt contracts will verify:

- all seven installed optimizer libraries are named;
- no runtime package installation is allowed;
- discontinuity, integer plateaus, cost, and seed sensitivity are explained;
- at least the major optimizer families and evidence triggers are present;
- selection temperature and action-policy API guidance remain intact.
