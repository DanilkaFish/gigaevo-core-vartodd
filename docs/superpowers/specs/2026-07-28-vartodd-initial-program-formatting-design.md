# VarTODD Initial Program Formatting Design

## Goal

Make the six shared `vartodd_evo_gf` initial programs shorter and easier to
scan without changing their behavior, search hyperparameters, or optimization
structure.

## Scope

Format these files in place:

- `beam3_temp_probe.py`
- `full_pso_pyswarms.py`
- `lean_beam_de.py`
- `lean_scout_restart.py`
- `todd_hard_tail_budget_split.py`
- `tohpe_weights_budget_probe.py`

The programs remain standalone. No shared optimizer or policy utility module
will be introduced.

## Behavior That Must Remain Frozen

The refactor must preserve:

- every numerical and categorical search hyperparameter;
- optimizer families, bounds, seeds, budgets, population sizes, and stages;
- evaluator start mode, restarts, margins, objectives, and seed lists;
- rank schedules and finite/full TODD behavior;
- score weights, centers, mapped ranges, and parameter groups;
- sampling, pools, bucket controls, selection, and action-pool settings;
- the exact mapped parameter counts: `18, 24, 18, 18, 19, 17`;
- the two explicitly grouped programs and their group-selection order.

## Formatting Rules

- Remove docstrings and comments that merely restate the following code.
- Retain short comments only where they explain stage intent or a non-obvious
  constraint.
- Collapse unnecessarily multiline signatures, calls, lists, arithmetic
  expressions, and small constructors when they remain readable.
- Keep semantically distinct policy sections separated by one blank line.
- Prefer direct returns and concise local variables over one-use scaffolding.
- Keep important policy controls named when reuse or schedule comparison makes
  the name informative.
- Do not pursue minimum line count at the expense of transparent policy flow.

## Verification

Regression tests will snapshot the optimizer/search constants and continue to
check optimizer roles, mapped parameter counts, group flow, retention bounds,
and the finite-to-full TODD schedule. All six files must compile. A source-size
check will confirm that formatting reduces the total line count without
enforcing fragile per-file formatting.
