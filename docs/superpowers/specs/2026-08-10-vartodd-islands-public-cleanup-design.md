# VarTODD Islands Public Cleanup Design

## Goal

Make the repository present one supported workflow: matrix-specific VarTODD
evolution with the three-island strategy launched by `run_gf_islands.py`.
Retain the shared GF implementation and the island-specific overrides as two
separate source directories, while removing every unrelated problem payload.

## Final Problem Layout

The `problems/` directory will contain exactly these two immediate
subdirectories:

- `problems/vartodd_evo_gf/` — shared evaluator, native bindings facade,
  metrics template, seed programs, and common launcher assets.
- `problems/vartodd_evo_gf_islands/` — island path-store behavior, routing
  prompts, task description, and island variant resolution.

Every other current immediate subdirectory of `problems/` will be deleted,
including the older `vartodd_evo` and `vartodd_gf32` payloads and all upstream
example problems. The two retained directories will not be merged: the islands
runtime overlay will continue to compose shared assets from
`vartodd_evo_gf/` with island-owned assets from `vartodd_evo_gf_islands/`.

The cleanup is limited to problem payload directories. Configuration and
framework modules outside `problems/` remain unless a direct launcher or test
reference must be adjusted to keep the supported islands workflow correct.

## Public Launcher And Internal Boundary

`run_gf_islands.py` becomes the only launcher documented as supported. It will
continue to expose `build_gf_islands_overrides()` for tests and programmatic
use, and its command-line `main()` will execute `run.py` with a generated,
matrix-specific problem overlay.

`run_gf.py` remains in the repository because it owns the shared launcher
mechanics used by the islands launcher:

- parsing launcher arguments;
- resolving a GF degree or exact `.npy` filename;
- deriving safe matrix and data identifiers;
- writing the matrix manifest and metrics overlay;
- linking shared assets and seed pools;
- constructing the VarTODD execution environment.

Its standalone legacy entry point remains functional for compatibility, but it
will not be presented as a normal workflow in the README. Shared helpers will
be cleaned up where necessary so `run_gf_islands.py` does not depend on stale
usage text or ambiguous argument handling.

## Islands Launcher Contract

The supported invocation is:

```text
python run_gf_islands.py matrix=<degree|filename.npy> lb=<rank> ub=<rank> [launcher options] [Hydra overrides]
```

The islands launcher will inject
`experiment=vartodd_evo_gf_islands_steady` when the caller omits an experiment.
If an explicit conflicting experiment is supplied, it will fail with a clear
error instead of silently launching a non-islands pipeline. User attempts to
override launcher-owned `problem.name`, `problem.dir`, `redis.prefix`, or
`initial_exec_cache_dir` will also fail clearly.

Required launcher arguments:

- `matrix`: a positive GF degree when exactly one matching
  `npy/gf2^<degree>_*.npy` file exists, or an exact filename under `npy/`.
- `lb`: lower fitness/rank bound and target final rank.
- `ub`: upper fitness/rank bound; it must be greater than `lb`.

Optional launcher arguments:

- `cache=true|false`: enable or disable the initial-program execution cache;
  default `true`.
- `call_timeout=<seconds>`: hard evaluator-call timeout; default `3800`.
- `soft_timeout_grace=<seconds>`: non-negative grace subtracted from the hard
  timeout to form the cooperative soft deadline.
- `initial_programs=default|best|expensive`: select the seed portfolio;
  default `default`.

All other arguments are forwarded unchanged to `run.py` as Hydra overrides.
The README will show concurrency, Redis DB/resume, logging, and model-config
overrides through this forwarding mechanism.

The generated runtime identity remains matrix-specific:

- problem and Redis prefix: `vartodd_evo_gf_islands<N>` for a numeric degree,
  or a sanitized filename-derived suffix for an exact matrix filename;
- data directory: `data_gf_islands<N>` or its filename-derived equivalent;
- overlays and caches beneath `.run_gf/`;
- path archives beneath the corresponding data directory.

## README Structure

`README.md` will be rewritten rather than incrementally edited. It will contain
only information needed to understand, install, configure, run, resume, and
inspect the islands version:

1. A concise description of LLM-guided VarTODD search and the three regimes:
   `ab_initio`, `mid_margin`, and `near_end`.
2. The two retained problem directories and the island-specific configuration,
   custom stages, matrix inputs, and curated `evolution_results/` layout.
3. Python, Redis, compiler/CMake, OpenRouter credential, and native VarTODD
   installation instructions.
4. A complete launcher API table matching the contract above.
5. Fresh-run, resume, alternate Redis DB, exact-filename matrix, seed-pool,
   timeout, concurrency, and smoke-run examples using only
   `run_gf_islands.py`.
6. A short explanation of the generated overlay, Redis namespace, saved Path
   Store, initial cache, soft-deadline salvage, and output locations.
7. Troubleshooting limited to the supported islands workflow.

The README will not advertise `vartodd_evo`, `vartodd_gf32`, batch presets, the
legacy single-island experiment, or any removed example problem.

## Error Handling

Launcher errors must be raised before creating or executing an invalid run
where practical. Messages will identify missing required arguments, malformed
integers or booleans, invalid bound ordering, unsupported seed pools,
ambiguous/missing matrix files, conflicting experiments, and reserved
overrides. `--help` and `-h` must remain side-effect free.

Existing overlays retain collision protection: a pre-existing asset or matrix
manifest with incompatible contents is an error rather than an overwrite.

## Tests And Verification

Implementation will update launcher tests before changing behavior. Coverage
will prove:

- the default islands experiment is injected;
- the correct islands problem name, Redis prefix, data directory, metrics, seed
  links, prompts, and cache paths are generated;
- conflicting experiments and reserved overrides are rejected;
- every documented launcher-only option is consumed by the launcher, while
  ordinary Hydra overrides remain in the forwarded argument list;
- exact filenames and unambiguous numeric degrees resolve correctly;
- help output and README API names stay aligned;
- Hydra composes `vartodd_evo_gf_islands_steady` with the islands pipeline and
  algorithm;
- the final immediate directory set under `problems/` is exactly
  `{vartodd_evo_gf, vartodd_evo_gf_islands}`;
- README scans contain no obsolete public problem or legacy launch commands.

Relevant unit, integration, configuration, and documentation checks will be
run in the project environment. A final `git diff --check` and complete status
review will ensure unrelated pre-existing worktree changes were not folded into
the cleanup.

## Non-Goals

- Merging the two retained problem source directories.
- Redesigning the three-island algorithm, prompts, archive policy, or native
  VarTODD search logic.
- Deleting historical configuration outside `problems/` merely because it is
  not documented.
- Changing the curated matrices or evolution-result artifacts.
- Committing unrelated existing worktree changes.
