# GigaEvo VarTODD Islands

This repository evolves Python search programs for VarTODD matrix
decomposition. An LLM mutates candidate programs, the native `pyvartodd`
extension executes them, and a three-island MAP-Elites strategy retains diverse
search logic while sharing useful saved paths.

`run_gf_islands.py` is the supported launcher. It selects a matrix, creates an
isolated runtime problem overlay, and starts the fixed
`vartodd_evo_gf_islands_steady` experiment.

## How The Three-Island Search Works

Each mutation is assigned to one of three regimes before its parents are
selected:

- `ab_initio` starts from the input matrix and develops a fresh decomposition.
  It receives no saved-path inventory.
- `mid_margin` loads a selectable saved path and branches with enough earlier
  context to search for a different descent.
- `near_end` loads a selectable saved path and concentrates on its difficult
  inherited tail.

Each regime has its own MAP-Elites archive. The two refinement islands read the
same evidence-rich Path Store, so an improvement found by one search can become
a starting point for later searches without collapsing the three parent
populations into one archive.

## Repository Layout

The `problems/` directory intentionally contains exactly two source packages:

- `problems/vartodd_evo_gf/` provides the shared evaluator, VarTODD helper API,
  validator, metrics template, and seed portfolios.
- `problems/vartodd_evo_gf_islands/` provides island routing, prompts, variant
  resolution, and the shared Path Store interface.

The launcher composes those packages into a generated matrix-specific overlay;
neither source directory needs to be copied or edited for a normal run.

Other important paths are:

- `run_gf_islands.py`: supported command-line launcher.
- `config/experiment/vartodd_evo_gf_islands_steady.yaml`: steady-state runtime
  preset.
- `config/algorithm/vartodd_diverse_gf_islands.yaml`: the three archives,
  routing probabilities, and bootstrap behavior.
- `config/pipeline/vartodd_islands_pipeline.yaml`: evaluation and island-context
  DAG.
- `custom/vartodd_islands_context.py`: route context, statistics, and path-card
  enrichment.
- `npy/`: input matrices.
- `evolution_results/<circuit>/`: curated final matrix and `evolution.csv` for
  each completed circuit.
- `scripts/install_pyvartodd.sh`: native extension installer.

## Requirements And Installation

You need:

- Python 3.11 or newer (Python 3.12 is the tested environment);
- Redis;
- CMake 3.20 or newer;
- a compiler with C++23 support;
- the native VarTODD build dependencies available to CMake;
- an OpenRouter-compatible API key.

Install the package and, when needed, its test dependencies:

```bash
python -m pip install -e .
python -m pip install -e ".[test]"
```

## Build `pyvartodd`

Build and install the native extension into `pyvartodd/Release/`:

```bash
scripts/install_pyvartodd.sh
```

By default, the installer clones the `no-data-scripts` branch of
`https://github.com/DanilkaFish/VarTodd.git`. It uses the active `python`; set
`PYTHON` to build for a particular environment:

```bash
PYTHON=/path/to/python scripts/install_pyvartodd.sh
```

Useful build overrides include:

```bash
VARTODD_SOURCE_DIR=/path/to/VarTodd scripts/install_pyvartodd.sh
VARTODD_REPO_URL=git@github.com:DanilkaFish/VarTodd.git scripts/install_pyvartodd.sh
VARTODD_BRANCH=no-data-scripts scripts/install_pyvartodd.sh
VARTODD_UPDATE=1 VARTODD_BUILD_JOBS=16 scripts/install_pyvartodd.sh
```

If the extension is installed elsewhere, expose that directory at runtime:

```bash
VARTODD_PYVARTODD_DIR=/path/to/pyvartodd/Release \
  python run_gf_islands.py matrix=16 lb=380 ub=421
```

## Configure OpenRouter

Create `.env` in the repository root or export the key in your shell:

```bash
export OPENAI_API_KEY="your-openrouter-api-key"
```

The launcher uses `config/llm/openrouter_vartodd_evolution.yaml`. Edit that
file to change model routing or supply normal Hydra overrides for settings the
config exposes.

## Start Redis

Start a local Redis server before launching evolution:

```bash
redis-server
```

The default connection is `localhost:6379`, database `0`. Connection settings
can be forwarded as Hydra overrides such as `redis.host=...`, `redis.port=...`,
or `redis.db=...`.

## Launcher API

```text
python run_gf_islands.py matrix=<degree|filename.npy> lb=<rank> ub=<rank> [options] [Hydra overrides]
```

| Argument | Required/default | Meaning |
| --- | --- | --- |
| `matrix` | required | Positive GF degree when exactly one `npy/gf2^<degree>_*.npy` matches, or an exact filename under `npy/`. |
| `lb` | required | Lower fitness bound and target final rank. |
| `ub` | required | Upper fitness bound and failure sentinel; must be greater than `lb`. |
| `cache` | `true` | Cache initial-program executions. Use `false` to evaluate every seed again. |
| `call_timeout` | `3800` | Hard timeout in seconds for one candidate evaluation. |
| `soft_timeout_grace` | `200` | Non-negative grace subtracted from the hard timeout to form the cooperative soft deadline. |
| `initial_programs` | `default` | Seed portfolio: `default`, `best`, or `expensive`. |

Every other argument is forwarded unchanged to `run.py` as a Hydra override.
The launcher supplies `experiment=vartodd_evo_gf_islands_steady` automatically.
An explicit different experiment is rejected, as are overrides of the
launcher-owned `problem.name`, `problem.dir`, `redis.prefix`, and
`initial_exec_cache_dir` values.

Run `python run_gf_islands.py --help` for the compact command reference.

## Run A Fresh Evolution

This GF(2^16) example uses the curated `best` seeds, disables seed-cache reuse,
and keeps six candidate evaluations in flight:

```bash
python run_gf_islands.py \
  matrix=16 lb=380 ub=421 \
  initial_programs=best cache=false \
  max_concurrent_dags=6 max_in_flight=6 \
  runner_config.prefetch_factor=1 \
  logging.level=INFO
```

The experiment already defaults to six concurrent DAGs, six in-flight
mutations, strict in-flight accounting, two parents, and prefetch factor 1, so
the minimal equivalent command is:

```bash
python run_gf_islands.py matrix=16 lb=380 ub=421
```

Use the heavier seed portfolio when the additional initial evaluation cost is
appropriate:

```bash
python run_gf_islands.py \
  matrix=16 lb=380 ub=421 initial_programs=expensive
```

Use custom hard and soft timeouts for unusually expensive matrices. Here the
cooperative deadline is approximately 12,000 seconds:

```bash
python run_gf_islands.py \
  matrix=32 lb=1150 ub=1250 \
  call_timeout=13200 soft_timeout_grace=1200
```

For a one-mutant smoke run:

```bash
python run_gf_islands.py \
  matrix=16 lb=380 ub=421 max_mutants=1 \
  max_concurrent_dags=1 max_in_flight=1 \
  runner_config.prefetch_factor=1 logging.level=INFO
```

Initial seeds are loaded before mutants, so a smoke run can still be expensive
when its seed cache is cold.

## Resume Or Isolate A Run

Resume with the same matrix, bounds, and seed portfolio that created the Redis
state:

```bash
python run_gf_islands.py \
  matrix=16 lb=380 ub=421 initial_programs=best redis.resume=true
```

Start an independent run in another Redis database:

```bash
python run_gf_islands.py \
  matrix=16 lb=380 ub=421 initial_programs=best redis.db=1
```

Without `redis.resume=true`, the launcher refuses to reuse a non-empty
namespace. To intentionally erase an entire Redis database before a fresh run,
use `redis-cli -n <db> FLUSHDB`; this removes every key in that database.

Ctrl-C requests an orderly stop of the evolution engine and DAG runner and
shuts down active executor workers.

## Matrix Selection Examples

A numeric degree is convenient when it has exactly one matching input:

```bash
python run_gf_islands.py matrix=16 lb=380 ub=421
```

When several matrices share a degree, or for a named circuit, pass the exact
filename under `npy/`:

```bash
python run_gf_islands.py "matrix=gf2^8_khor.npy" lb=120 ub=150
python run_gf_islands.py matrix=adder_8.qc.matrix.npy lb=110 ub=140
```

The launcher derives a safe matrix ID from exact filenames. A missing file or
an ambiguous numeric degree fails before evolution starts.

## Runtime Files And Curated Results

For each invocation, the launcher generates:

- a composed problem overlay below `.run_gf/islands_overlays/`;
- an initial-program cache below `.run_gf/islands_cache/<matrix-id>/`, separated
  by non-default seed portfolio;
- a matrix-specific Redis prefix such as `vartodd_evo_gf_islands16`;
- a shared saved-path store such as `data_gf_islands16/path_backups/`;
- Hydra logs and run artifacts below `outputs/<date>/<time>/`.

Only initial programs use the persistent execution cache. `cache=false`
disables reads and writes for that invocation; it does not delete an existing
cache.

Candidate execution receives a cooperative soft deadline at
`call_timeout - soft_timeout_grace`. A program that observes the deadline can
return its best completed decomposition; the hard timeout remains the final
limit for code that does not stop cleanly.

Curated results use one directory per circuit:

```text
evolution_results/<circuit>/
├── <circuit>_<final-rank>.npy
└── evolution.csv
```

The `.npy` file is the selected final decomposition and `evolution.csv` records
the corresponding best evolution trace.

## Troubleshooting

- **`expected one ... matrix ... found N`**: use the exact `.npy` filename
  instead of a numeric degree.
- **Redis namespace is not empty**: add `redis.resume=true`, choose another
  `redis.db`, or deliberately flush the database.
- **`pyvartodd` cannot be imported**: rebuild it with the same Python used to
  launch GigaEvo, or set `VARTODD_PYVARTODD_DIR`.
- **Seeds take a long time**: keep the default `cache=true` after the first
  successful evaluation, or select a lighter seed portfolio.
- **A launcher-owned override is rejected**: remove it. Matrix-specific
  problem paths, cache paths, and Redis prefixes are generated together to
  keep the overlay consistent.
