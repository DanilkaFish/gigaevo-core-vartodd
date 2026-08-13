# TODD Beam-Level Soft-Timeout Design

## Goal

Allow a long `Todd.run()` descent to stop cooperatively at a completed beam
level so the evaluator can preserve the best path found during the current
seed and use the existing graceful-timeout salvage mechanism.

The native `policy_iteration()` call remains unchanged and non-interruptible.

## Current Behavior

`BaseEvaluator.run()` checks the executor soft deadline before starting a run
and between seeds. With the default single seed worker, one seed calls
`Todd.run()`, which may execute many beam levels before returning. The soft
deadline cannot be observed during that descent. If the descent remains inside
`Todd.run()` until the external hard timeout, the executor cannot call
`get_active_evaluator_best()` and the program returns no fitness.

`Todd.run()` has two nested loops:

1. the outer loop advances one beam level;
2. the inner loop calls `policy_iteration()` for each node in the current
   beam and collects its children.

The end of a completed beam level is a safe cooperative interruption point:
all native calls for that level have returned, the children can be merged, and
`best_node` can be updated without exposing a partially constructed node.

## Chosen Design

### Stop callback

Add an optional keyword-only `stop_requested` callback to `Todd.run()`.
The default is `None`, preserving existing callers and behavior.

The helper worker passes `_soft_deadline_reached` as this callback. `todd.py`
does not import `helper.py`, avoiding a circular dependency and keeping the
timeout source injectable in tests.

### Checkpoint placement

Do not check time inside `policy_iteration()` or native TODD bucket research.
For each outer beam iteration:

1. finish the inner `for node in nodes` loop;
2. merge and deduplicate all children from that level;
3. update `best_node`, counters, and the next beam;
4. call `stop_requested()`;
5. break the outer depth loop when it returns true.

The check occurs after the best-node update so the just-completed beam level is
included in the salvaged path.

If no children are produced, `Todd.run()` keeps its existing normal exit.
Exceptions raised by the callback are not hidden.

### Propagating graceful timeout

After the single-worker call to `Todd.run()` returns, `BaseEvaluator.run()`
checks `_soft_deadline_reached()` immediately. When true, it:

1. keeps the returned seed result;
2. records the returned node through the existing result-processing path;
3. raises `GracefulEvaluationTimeout`.

The executor then invokes the existing `get_active_evaluator_best()` hook, and
the resulting aux information contains `timeout_salvaged: 1`.

No new exception type, native result field, or evolved-policy API is required.

## Scope

Change only the shared implementation in:

- `problems/vartodd_evo_gf/todd.py`;
- `problems/vartodd_evo_gf/helper.py`;
- focused tests for the checkpoint and result preservation.

The generated GF variants and island problem continue to receive behavior from
the shared `vartodd_evo_gf` source. No changes are made to `pyvartodd`, policy
configuration, prompts, metrics, path storage, or initial programs.

## Compatibility and Limitations

- Runs without a soft deadline behave exactly as before.
- A single `policy_iteration()` remains non-interruptible. The soft deadline
  may therefore be exceeded by the duration of one complete beam level.
- The current default uses one seed worker, which is the path this change is
  designed to salvage. Parallel seed cancellation retains its existing
  behavior and is not expanded by this change.
- A deadline reached after a naturally completed seed is also treated as a
  graceful timeout. Its completed result is recorded before salvage.

## Verification

Add focused tests that establish:

1. `Todd.run()` does not invoke `stop_requested` until an inner node loop has
   completed.
2. When the callback becomes true, children from that completed level have
   already updated `best_node` and deeper levels are not evaluated.
3. With no callback, the previous depth behavior is unchanged.
4. The single-worker evaluator records the returned seed result and then raises
   `GracefulEvaluationTimeout` when the deadline is reached.
5. The existing salvage hook returns that recorded best path with
   `timeout_salvaged: 1`.
