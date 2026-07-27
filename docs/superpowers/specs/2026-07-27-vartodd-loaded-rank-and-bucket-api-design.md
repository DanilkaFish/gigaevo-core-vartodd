# VarTODD Loaded-Rank Bounds and Bucket API Cleanup

## Goal

Make generated GF metrics describe the full valid loaded-rank interval and
stop exposing bucket exploration fields that are not currently implemented
correctly.

## Generated metrics

`run_gf.py` will resolve the selected matrix file and read its first dimension
as `INITIAL_RANK`. Generated `metrics.yaml` files will use:

- fitness bounds: `[lb, ub]`;
- loaded-rank bounds: `[lb, INITIAL_RANK]`;
- loaded-rank sentinel: `INITIAL_RANK`;
- runtime bounds and sentinel: unchanged.

The matrix remains the single source of truth for the initial rank. No
per-degree rank table or new launcher argument will be introduced.

## Bucket API cleanup

Within the shared `problems/vartodd_evo_gf` source:

- remove bucket `temperature` and `random_fraction` from the task-description
  API and explanatory text;
- stop reading those keys in helper and DAO mapping conversions;
- stop passing explicit zero values when constructing default
  `ZBucketSearch` objects.

`ActionSelection.temperature` is a separate, working selection control and
will remain documented and implemented.

## Tests

Regression tests will verify:

- GF14 and GF16 generated loaded-rank upper bounds match their matrix row
  counts;
- loaded-rank lower bounds remain equal to `lb`;
- fitness continues to use `[lb, ub]`;
- shared mutation-facing source no longer references bucket
  `random_fraction` or passes bucket `temperature`;
- selection-temperature references remain intact.
