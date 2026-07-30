# `run_gf.py` Initial-Program Pool Selection

## Goal

Let `run_gf.py` select either the existing initial-program pool or the curated
universal best pool, while keeping the current behavior as the default.
Reduce only the optimizer evaluation budgets in the best pool so its estimated
mean runtime is approximately 1000–1500 seconds.

## Launcher Interface

The launcher accepts:

```text
initial_programs=default
initial_programs=best
```

The argument defaults to `default`.

- `default` selects
  `problems/vartodd_evo_gf/initial_programs/`.
- `best` selects
  `problems/vartodd_evo_gf/initial_programs_best/`.
- Any other value raises a `ValueError` that lists the supported values.
- The launcher consumes this argument; it is not forwarded as a Hydra
  override.

The usage example includes `initial_programs=best`.

## Overlay and Cache Behavior

Downstream code continues to see the selected pool at
`<problem overlay>/initial_programs`. The generated overlay symlinks that
fixed name to the selected shared-source directory.

Default compatibility is preserved:

- Default overlay:
  `gf<M>_lb<LB>_ub<UB>/vartodd_evo_gf<M>`
- Default execution cache:
  `cache/gf<M>`

The best pool uses isolated locations:

- Best overlay:
  `gf<M>_lb<LB>_ub<UB>_seeds_best/vartodd_evo_gf<M>`
- Best execution cache:
  `cache/gf<M>/best`

This prevents a hidden overlay from retaining a symlink to another pool and
prevents cached executions of default seed files from being reused for the
best pool. `cache=false` still produces `initial_exec_cache_dir=null`.

Redis prefix, matrix-specific path storage, problem name, rank metrics, timeout
arguments, and runtime environment variables do not depend on the selected
pool.

## Best-Pool Evaluation Budgets

Only evaluation-count constants change. Policy mappings, score ranges, source
pools, rank schedules, optimizer parameters, seeds, restart placement, and
auxiliary descriptions remain unchanged.

| Program | Current budget | New budget |
|---|---:|---:|
| `de_light_terminal_todd.py` | `TOTAL_EVALS = 8000` | `TOTAL_EVALS = 3800` |
| `pso_restart_terminal_todd.py` | `TOTAL_EVALS = 3000` | `TOTAL_EVALS = 1500` |
| `pso_tohpe_only.py` | `TOTAL_EVALS = 5000` | `TOTAL_EVALS = 2500` |
| `pso_wide_terminal_beam.py` | `TOTAL_EVALS = 5000` | `TOTAL_EVALS = 2500` |
| `pso_chunked_light_todd.py` | `TOTAL_EVALS = 2000` | `TOTAL_EVALS = 1000` |
| `pso_cma_grouped_tail.py` | `144 + 512 + 64` | `72 + 256 + 32` |
| `pso_three_band_restart.py` | `1200 + 800` | `600 + 400` |
| `pso_pattern_heavy_restart.py` | `64 + 96` | unchanged |

The reductions are proportional to the GF16 source runtimes, which were
usually near 2600 seconds. They target an approximate pool mean near 1200
seconds; runtime is not guaranteed because policy evaluations have
rank-dependent cost.

The PatternSearch heavy-restart source already ran in about 659 seconds.
Cutting its evaluation count cannot move it toward the requested range, so it
remains the short structural-diversity seed.

## Implementation Boundaries

Modify:

- `run_gf.py`
- `tests/test_tools/test_run_gf.py`
- evaluation-count assignments in the seven affected files under
  `problems/vartodd_evo_gf/initial_programs_best/`
- `tests/problems/test_vartodd_gf_best_initial_programs.py`

Do not modify:

- active programs under `problems/vartodd_evo_gf/initial_programs/`
- Redis naming
- path storage
- metrics generation
- policy or optimizer logic in the best programs

## Verification

Launcher tests require:

- omitted selector preserves the default symlink, overlay, and cache paths;
- `initial_programs=best` selects the curated directory and isolated paths;
- invalid selectors fail clearly;
- `cache=false` remains effective with the best pool;
- the selector is absent from forwarded Hydra overrides;
- help output documents the new argument.

Best-pool tests require the exact approved evaluation constants and confirm
that all eight files and their existing structural contracts remain present.

Compile all eight best programs and run the focused launcher, best-pool, shared
source, and active-initial-program regression tests.
