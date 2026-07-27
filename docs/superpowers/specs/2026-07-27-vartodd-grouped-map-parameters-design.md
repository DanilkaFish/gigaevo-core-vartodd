# Grouped `map_par` Parameters

## Goal

Allow one `BaseEvaluator` policy mapping to divide its mapped parameters into
named groups. An optimization stage can select a subset of those groups,
optimize only their compact parameter vector, and leave every other mapped
parameter frozen at its current full `x0` value.

The interface must remain compatible with existing programs and with optimizers
that consume flat NumPy arrays, including pymoo, pyswarms, CMA-ES, and scipy.
It must also be explicit enough for LLM-generated programs to use without
accidentally changing parameter meanings.

## Stable Slots and Compact Active Vectors

Every call to `map_par` always consumes the next global `x0` slot, whether its
group is selected or disabled. This preserves the existing stable relationship
between policy declarations, saved path vectors, and `x0` indices.

The optimizer-facing active vector is compact. It contains only slots whose
groups are selected, in global policy-declaration order. The order in which
group names are passed to the selection method never changes vector ordering.

For example:

```text
slot:    0       1       2       3       4
group: scores  search  scores  budget  search
x0:      0.2     1.1    -0.4     3.0     0.7
```

Selecting `scores` produces `[0.2, -0.4]`. Inserting `[0.5, -0.8]`
updates slots 0 and 2 while slots 1, 3, and 4 retain their current values.

This is an active-coordinate projection, not a full vector with masked
coordinates. Optimizers therefore do not waste dimensions on frozen values.

## Public API

### Declaring a group

`map_par` gains a keyword-only `group` argument:

```python
def map_par(
    self,
    mapping: callable,
    *,
    group: str = "default",
    **mapping_kwargs,
):
    ...
```

Typical evaluator helpers expose the group explicitly:

```python
def float_range(
    self,
    low: float,
    high: float,
    *,
    group: str = "scores",
) -> float:
    return self.map_par(
        lambda x: low + (high - low) * sigmoid(x),
        group=group,
    )
```

Calls that omit `group` belong to the `"default"` group. All groups are
selected by default, so existing evaluators keep the same active vector and
behavior without modification.

Positional arguments after `mapping` are rejected. This makes an accidental
`map_par(mapping, "scores")` fail immediately instead of silently treating the
group name as an unused legacy argument.

### Selecting groups

```python
params = evaluator.select_parameter_groups("scores", "sampling")
params = evaluator.select_all_parameter_groups()
```

Both methods reinitialize the policy mapping and return a fresh compact active
vector, equivalent to calling `extract_active()` after selection. Using a
variadic string interface avoids treating a single group name as an iterable of
characters.

`select_parameter_groups` replaces the current selection; it is not additive.
At least one group must be selected. Group switching must happen before
constructing or restarting an optimizer because it may change `n_var` and the
meaning of the compact vector.

`run`, `insert`, `extract_active`, and `set_up_new_init(..., xopt=...)` continue
to operate on the current compact active layout. They do not accept an
independent group selection argument. Keeping selection separate from
evaluation prevents the optimizer dimension from changing inside an objective
call.

### Inspecting groups

```python
evaluator.parameter_group_names()                 # tuple[str, ...]
evaluator.parameter_group_size("scores")          # int
evaluator.parameter_group_sizes()                 # dict[str, int]
evaluator.active_parameter_group_names()          # tuple[str, ...]
evaluator.extract_parameter_group("scores")       # np.ndarray
evaluator.extract_active()                        # np.ndarray
```

Group-name and size results follow first declaration order. Extracted group
values follow global slot order and are copies, matching `extract_active`.
`extract_parameter_group` works for both selected and frozen groups.

The implementation may retain a private mapping from group names to stable
indices, but no public method exposes mutable internal lists.

## Lifecycle and Persistence

`reinit()` rebuilds the group registry deterministically while preserving the
requested selection. It still evaluates every mapping against the full `x0`.
Only selected slots are appended to `active_params`.

The complete fixed-length `x0` remains the saved-path representation. No path
format or migration is required. Loading a path restores all group values,
including values currently frozen by the selected layout.

`set_up_new_init` inserts an optional `xopt` only into the currently selected
slots. All frozen slots retain the values supplied by the loaded or current
`x0`.

Evaluator cloning must copy the selected-group configuration and rebuild or
copy group metadata without sharing mutable registries between clones.

## Validation and Errors

The API fails early with descriptive errors:

- `map_par` rejects group names that are not non-empty strings.
- `select_parameter_groups` rejects an empty selection.
- Duplicate selected names are rejected rather than silently deduplicated.
- Unknown names raise `ValueError` and list the known group names.
- If a selected group disappears or changes membership during a later
  `reinit`, the evaluator raises an error describing the unstable policy
  mapping instead of silently changing optimizer coordinates.
- Existing active-vector length checks remain in force.

Changing group selection invalidates any population, covariance state, bounds,
or `xopt` produced for the previous active layout. A plain NumPy array carries
no layout identity, so this cannot be detected reliably when two layouts happen
to have the same length. The Action API must state that the optimizer must be
rebuilt after every selection change.

## LLM-Facing Guidance

The shared GF task description's Action API will include a short staged example:

```python
evaluator = Evaluator(...)

# Stage 1: tune search generation and retention.
evaluator.select_parameter_groups("sampling", "pools")
xopt = optimize(evaluator, evaluations=...)

# Preserve the Stage 1 result in full x0, then tune policy scores.
evaluator.insert(xopt)
evaluator.select_parameter_groups("scores")
optimize(evaluator, evaluations=...)
```

The guidance will emphasize:

- use a small number of semantic, stable group names;
- group related parameters by optimization purpose, not by incidental code
  location;
- select groups before constructing the optimizer problem;
- rebuild optimizer state after changing groups;
- disabled groups remain effective but frozen at current `x0` values;
- omit grouping entirely when staged optimization is not useful.

## Tests

Focused tests will cover:

1. Existing ungrouped evaluators retain identical active indices and values.
2. Group declarations always consume stable full-vector slots.
3. A selected subset produces the expected compact vector.
4. Inserting and running update selected slots only.
5. Disabled slots remain frozen across `reinit` and repeated runs.
6. Selection argument order does not affect compact-vector ordering.
7. Selecting all groups restores the original active layout.
8. Group inspection reports correct names, sizes, active names, and values.
9. Invalid, empty, duplicate, and unknown names produce useful errors.
10. Saved-path loading and `set_up_new_init` preserve full `x0` semantics.
11. Evaluator clones preserve selection without sharing mutable metadata.
12. Unstable conditional group declarations are detected.
13. The shared task description documents the API and contains no obsolete
    alternative syntax.

## Scope

This change adds grouping to `BaseEvaluator`, tests its lifecycle, and documents
the API in the shared GF task description. It does not add optimizer-specific
adapters, tagged array subclasses, nested groups, group-level bounds, temporary
selection context managers, or automatic optimizer rebuilding.
