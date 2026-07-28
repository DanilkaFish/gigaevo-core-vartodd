# Vartodd Initial Score Centers Design

## Goal

Diversify score-center handling across the six shared
`problems/vartodd_evo_gf/initial_programs` seeds without changing their
optimizers, score weights, sampling budgets, search schedules, or other search
structure.

Exploration centers will be fixed at zero in every seed. Finalization centers
will use a small set of deliberately different strategies. Parameterization
will favor `yw` and `zw`; `red`, `bucket`, and `tohpe` centers will remain zero.

## Program distribution

The finalization center order is:

`red, dim, bucket, yw, zw, tohpe`

The six seeds will use these strategies:

| Program | Parameterized final centers |
| --- | --- |
| `lean_beam_de.py` | none |
| `beam3_temp_probe.py` | none |
| `lean_scout_restart.py` | `yw` |
| `tohpe_weights_budget_probe.py` | `zw` |
| `full_pso_pyswarms.py` | `dim`, `yw` |
| `todd_hard_tail_budget_split.py` | `yw`, `zw` |

Every fixed center is `0.0`. Each parameterized center maps one optimization
parameter into `[0.0, 1.0]` through the program's local `float_range` method.
Programs that already divide parameters into groups will assign center
parameters to the existing `scores` group.

This yields two zero-center baselines, two single-center variants, and two
two-center variants. It introduces center diversity while adding at most two
dimensions to any seed.

## Scope

Only score-center expressions in the six initial programs will change.
Exploration and finalization weights, score powers, optimizer settings,
optimization stages, parameter groups, sampling and pool sizes, action
selection, TODD/TOHPE configuration, rank schedules, and restart behavior will
remain unchanged.

## Verification

A regression test will inspect the shared initial-program sources and verify:

1. every exploration-center vector is all zero;
2. the six final-center strategies match the table above;
3. `red`, `bucket`, and `tohpe` centers remain zero;
4. grouped programs place parameterized centers in the `scores` group; and
5. existing initial-program contract tests continue to pass, except for any
   independently modified user-owned behavior already present in the dirty
   worktree.

