# GigaEvo VarTODD Fork

This repository is a public, VarTODD-focused fork of GigaEvo. It keeps the
GigaEvo execution engine and adds the minimal custom pipeline, configs, prompts,
and problem payloads needed to evolve VarTODD search programs.

The fork is not meant to expose every upstream GigaEvo example. The runnable
experiments here are:

- `vartodd_evo`: the current GF(2^7) through GF(2^16) VarTODD evolution target.
- `vartodd_gf32`: the GF(2^32) VarTODD target.

Each evolved program is Python code that builds a VarTODD/FastTODD search
strategy. The program chooses score shapes, source pools, TOHPE/TODD budgets,
restart schedules, saved-path loading, and refinement behavior. GigaEvo mutates
that program with an LLM, executes it through the native `pyvartodd` extension,
validates the produced path, and keeps diverse candidates with MAP-Elites.

## Repository Layout

- `problems/vartodd_evo/`: GF(2^7) through GF(2^16) problem code, prompts,
  metrics, and seed programs.
- `problems/vartodd_gf32/`: GF(2^32) problem code, prompts, metrics, and seed
  programs.
- `config/experiment/vartodd_evo_gf16_batch.yaml`: recommended GF(2^16) batch
  preset.
- `config/experiment/vartodd_gf32_batch.yaml`: recommended GF(2^32) batch
  preset.
- `config/pipeline/vartodd_pipeline.yaml`: VarTODD DAG stages and execution
  timeout policy.
- `config/algorithm/vartodd_diverse_gf16.yaml`: MAP-Elites and mutation-regime
  config for `vartodd_evo`.
- `config/algorithm/vartodd_diverse_gf32.yaml`: MAP-Elites and mutation-regime
  config for `vartodd_gf32`.
- `config/llm/openrouter_vartodd_evolution.yaml`: OpenRouter model routing for
  mutation, insights, and lineage.
- `custom/`: VarTODD-specific stages and the batch evolution engine.
- `npy/`: tracked matrix inputs loaded by the helper code when running from the
  repository root.
- `scripts/install_pyvartodd.sh`: builds and installs the native `pyvartodd`
  extension into `pyvartodd/Release/`.

The experiment name describes the preset, but the actual target circuit/matrix
is selected in each problem's `helper.py`. Check `DEFAULT_MATRIX_PATH` there
before starting a run or when retargeting the problem to another matrix.

## Requirements

- Python 3.11+ for GigaEvo. Python 3.12 is the tested environment for the
  current VarTODD runs.
- Redis.
- CMake 3.20+.
- A C++ compiler with C++23 support.
- An OpenRouter-compatible API key in `OPENAI_API_KEY`.
- VarTODD native build dependencies available to CMake.

Install the Python package:

```bash
python -m pip install -e .
```

For tests, install the test extra:

```bash
python -m pip install -e ".[test]"
```

## Build `pyvartodd`

The VarTODD Python extension is not committed. Build it from VarTODD with:

```bash
scripts/install_pyvartodd.sh
```

By default the script clones:

```text
https://github.com/DanilkaFish/VarTodd.git
```

from branch:

```text
no-data-scripts
```

and installs:

```text
pyvartodd/Release/pyvartodd*.so
pyvartodd/Release/libcnpy++.so
```

The installer uses the active `python` unless `PYTHON` is set, so it does not
force a Python version:

```bash
PYTHON=/home/danilkaf/pyenv/metaevolve312/bin/python scripts/install_pyvartodd.sh
```

Useful overrides:

```bash
VARTODD_SOURCE_DIR=/path/to/VarTodd scripts/install_pyvartodd.sh
VARTODD_REPO_URL=git@github.com:DanilkaFish/VarTodd.git scripts/install_pyvartodd.sh
VARTODD_BRANCH=no-data-scripts scripts/install_pyvartodd.sh
VARTODD_UPDATE=1 scripts/install_pyvartodd.sh
VARTODD_BUILD_JOBS=16 scripts/install_pyvartodd.sh
VARTODD_WITH_STUBS=ON scripts/install_pyvartodd.sh
```

If the extension is installed somewhere else, point the problem helpers to it:

```bash
VARTODD_PYVARTODD_DIR=/path/to/pyvartodd/Release python run.py ...
```

## Configure Credentials

Create `.env` or export the variable in your shell:

```bash
OPENAI_API_KEY=<your-openrouter-api-key>
```

The configured OpenRouter preset currently routes:

- mutation calls through DeepSeek V4 Flash and GPT-5 Mini;
- insights through GPT-5 Mini and DeepSeek V4 Flash;
- lineage through a colder GPT-5 Mini/DeepSeek V4 Flash mix.

Edit `config/llm/openrouter_vartodd_evolution.yaml` if you want different
models or probabilities.

## Run Evolution

Start Redis:

```bash
redis-server
```

Run the current GF(2^16) VarTODD experiment:

```bash
python run.py problem.name=vartodd_evo experiment=vartodd_evo_gf16_batch
```

Run the GF(2^32) experiment:

```bash
python run.py problem.name=vartodd_gf32 experiment=vartodd_gf32_batch
```

Use INFO-level console/file logs instead of the default DEBUG logging:

```bash
python run.py problem.name=vartodd_evo experiment=vartodd_evo_gf16_batch logging.level=INFO
```

The `vartodd_evo_gf16_batch` preset expands to the same core choices as:

```bash
python run.py problem.name=vartodd_evo \
  pipeline=vartodd_pipeline \
  algorithm=vartodd_diverse_gf16 \
  llm=openrouter_vartodd_evolution \
  evolution=batch \
  num_parents=2
```

For a short smoke run:

```bash
python run.py problem.name=vartodd_evo experiment=vartodd_evo_gf16_batch \
  max_generations=1 \
  max_mutations_per_generation=1 \
  max_concurrent_dags=1 \
  logging.level=INFO
```

Even a smoke run can take time if the initial-program execution cache is cold.

## Actual Batch Defaults

The VarTODD presets use `config/evolution/batch.yaml`.

Default batch settings:

- `num_parents=2`
- `max_elites_per_generation=10`
- `max_mutations_per_generation=16`
- `max_concurrent_dags=8`
- `prefetch_factor=0`
- `prefetch_extra=0`

The batch engine selects parent combinations, creates a generation of mutants,
waits for their DAG evaluations, ingests the results, refreshes the archive, and
then starts the next generation.

Operational overrides are normal Hydra overrides:

```bash
python run.py problem.name=vartodd_evo experiment=vartodd_evo_gf16_batch \
  max_concurrent_dags=4 \
  max_mutations_per_generation=8 \
  redis.db=1 \
  logging.level=INFO
```

## Execution Semantics

`config/pipeline/vartodd_pipeline.yaml` runs this high-level DAG:

1. Validate Python code syntax and basic safety.
2. Execute `entrypoint()` through `CachedCallProgramFunction`.
3. Validate the returned VarTODD payload with `validate.py`.
4. Split validator output into numeric metrics and non-metric aux text.
5. Add runtime and complexity metrics.
6. Ensure sentinel metrics exist for failed programs.
7. Build insights, lineage summaries, execution digests, evolutionary statistics,
   and mutation context.

Candidate execution for `vartodd_evo` uses:

- hard timeout: `3200` seconds;
- soft timeout grace: `200` seconds;
- effective soft deadline: about `3000` seconds after stage start.

The `vartodd_gf32_batch` experiment overrides that policy:

- hard timeout: `13200` seconds;
- soft timeout grace: `1200` seconds;
- effective soft deadline: about `12000` seconds after stage start.

The helper checks `GIGAEVO_EXEC_SOFT_DEADLINE_EPOCH`. If the program reaches that
deadline, it raises `GracefulEvaluationTimeout`. The executor then asks
`helper.get_active_evaluator_best()` for the best completed payload and returns
that salvaged result when available. The hard timeout still kills programs that
do not stop cleanly.

Only initial programs use the persistent execution cache by default:

```text
problems/<problem>/initial_programs/.exec_cache/
```

This cache is keyed by the program code and `helper.py`. It avoids re-running
expensive seeds across fresh Redis runs. Mutants are not cached by default.
Delete the cache directory to force seed re-evaluation.

## Metrics And Selection

Both VarTODD problems minimize `fitness`.

For `vartodd_evo`, fitness is:

```text
rank + mean(P)
```

where `P` is the found decomposition.

For `vartodd_gf32`, fitness also includes shaping around saved-path reuse and
effective TODD bucket limits. Rank differences are intended to dominate that
shaping.

MAP-Elites uses a behavior space over:

- primary fitness;
- runtime;
- `loaded_rank`;
- validity.

`loaded_rank` is a strategy descriptor. It separates ab-initio programs from
saved-path refiners by the rank they started from or branched from; it is not a
separate objective.

## Redis, Resume, And Fresh Runs

Redis stores programs, metrics, stage results, lineage, and engine state under
the problem key prefix.

If Redis already contains data and `redis.resume=false`, the run refuses to
start. Choose one of these:

```bash
# Continue the same run.
python run.py problem.name=vartodd_evo experiment=vartodd_evo_gf16_batch redis.resume=true

# Start isolated in another Redis DB.
python run.py problem.name=vartodd_evo experiment=vartodd_evo_gf16_batch redis.db=1

# Delete DB 0 and start over.
redis-cli -n 0 FLUSHDB
```

Pressing Ctrl-C asks the evolution engine and DAG runner to stop. Active
executor workers are killed so native VarTODD searches should not keep computing
after shutdown.

## Outputs

Hydra writes each run under:

```text
outputs/<date>/<time>/
```

Important local runtime artifacts:

- `outputs/`: Hydra logs and run outputs.
- `data/path_backups/`: live saved VarTODD paths used by mutation context.
- `problems/*/initial_programs/.exec_cache/`: initial-program execution cache.
- `pyvartodd/Release/`: native extension built by `scripts/install_pyvartodd.sh`.

These runtime directories are ignored by git. The root `npy/` matrix directory
is tracked and required for normal runs from the repository root.

## Troubleshooting

If `pyvartodd` cannot be imported, build it:

```bash
scripts/install_pyvartodd.sh
```

If `libcnpy++.so` is missing or cannot be loaded, rebuild and confirm it exists
next to the extension:

```text
pyvartodd/Release/libcnpy++.so
```

If Redis is not empty:

```bash
python run.py problem.name=vartodd_evo experiment=vartodd_evo_gf16_batch redis.resume=true
```

or use a different Redis DB:

```bash
python run.py problem.name=vartodd_evo experiment=vartodd_evo_gf16_batch redis.db=1
```

If initial programs keep reusing stale results, delete:

```text
problems/vartodd_evo/initial_programs/.exec_cache/
```

or disable the cache for a run:

```bash
python run.py problem.name=vartodd_evo experiment=vartodd_evo_gf16_batch initial_exec_cache_dir=null
```

If logs are too verbose:

```bash
python run.py problem.name=vartodd_evo experiment=vartodd_evo_gf16_batch logging.level=INFO
```

## Notes For Fork Maintenance

Keep VarTODD-specific changes scoped to:

- `problems/vartodd_evo/`
- `problems/vartodd_gf32/`
- `config/experiment/vartodd_*`
- `config/algorithm/vartodd_*`
- `config/pipeline/vartodd_pipeline.yaml`
- `config/llm/openrouter_vartodd_evolution.yaml`
- `custom/`
- `scripts/install_pyvartodd.sh`
- executor fixes needed for soft-deadline salvage and clean shutdown.

Avoid committing generated run outputs, local path backups, native build
artifacts, or old copied experiment folders.
