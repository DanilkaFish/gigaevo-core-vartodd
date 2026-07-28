# Vartodd Minimal API Clarification Design

## Goal

Clarify only the API contracts responsible for the current generated-program
failures, without restructuring or materially lengthening
`problems/vartodd_evo_gf/task_description.txt`.

## Changes

In the grouped-parameter section:

- require every `map_par` call to pass an explicit `group=` argument, including
  parameters intentionally assigned to `"default"`;
- state that `select_parameter_groups(name)` is valid only after at least one
  parameter has been declared with that exact group name; and
- state that group names, parameter declaration order, and the complete
  parameter layout must remain identical across `reinit()` and restarts.
  Conditional policy selection must not conditionally declare mapped
  parameters.

In the action-policy/rank-schedule section:

- state that policy constructors such as `ActionPool`, `ToddSearch`, and
  `PolicyScores` construct single configurations, while schedules are applied
  through the corresponding `set_*` method;
- state that a setter call must use either a single positional configuration
  or the keyword `ranks=..., values=...` schedule form, never both;
- include one compact valid setter schedule and two one-line invalid examples;
  and
- remove the duplicated `### Rank schedules` heading.

## Scope

No other guidance, examples, API behavior, source code, initial programs, or
tests will change. Existing unrelated worktree modifications to
`task_description.txt` will be preserved and excluded from the implementation
commit.

## Verification

Add focused source-level assertions to the existing evolution-guidance test:

- the explicit `group=` rule is present;
- the single-configuration versus setter-schedule distinction is present;
- both prohibited schedule forms are shown;
- the stable-layout rule is present; and
- the rank-schedule heading appears exactly once.

