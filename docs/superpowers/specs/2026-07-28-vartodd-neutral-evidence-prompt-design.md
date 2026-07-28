# VarTODD Neutral Evidence Prompt Design

## Goal

Make the shared `vartodd_evo_gf` task description explain evidence and
optimizer mechanics without steering mutations toward predetermined rank-band
interpretations, optimizer families, or restart responses.

## Scope

Revise four areas in `problems/vartodd_evo_gf/task_description.txt`:

1. optimizer-family descriptions;
2. interpretation of optimizer evidence;
3. stages, restarts, and warm starts;
4. rank-band descriptions.

## Neutrality Rules

- Describe what a field, optimizer, stage, or API operation represents.
- Do not infer one failure mechanism from a single observed pattern.
- Do not prescribe a mutation response from flat/broad quantiles, late
  improvements, low `dim`, pool fill, or rank-band position alone.
- Do not label optimizer families as valid only for a particular role.
- Present early, middle, and terminal as relative positions in the reached
  trajectory, not fixed behavioral regimes.
- Present `set_up_new_init`, `xopt`, group selection, and rebuilding an
  optimizer problem as lifecycle mechanics rather than judgments about whether
  a stage is a genuine restart.

## Guidance That Remains

Retain:

- action-policy API and lifecycle validity requirements;
- factual meanings of `path_policy_groups`, `converged_policy_profiles`, and
  `search_stat`;
- score weights as the most influential mapped policy parameters;
- the distinction between per-bucket y diversity and z-bucket coverage;
- correctness and path-loading constraints;
- factual runtime multipliers and evaluation-count formulas;
- the requirement to ground claims in same-run evidence.

## Resulting Wording

The rank-band passage will say that early/middle/terminal identify relative
positions only. Their properties must be read from reported reductions,
dimensions, accepted source counts, retained pool composition, and researched
z counts.

The optimizer-evidence passage will state that quantile shape and improvement
timing describe observed outcomes but do not uniquely identify their causes.
Comparisons are strongest when path, policy space, optimizer, or budget is
held fixed.

The restart passage will define stage labels and the evaluator lifecycle.
After a branch or active-group change, optimizer dimensions come from the
current `extract_active()` vector. Reusing or omitting `xopt` is described by
its effect on coordinates, without recommending either choice.

Optimizer-family descriptions will list relevant mechanics without declaring
one family a default, poor scout, or valid only near a productive basin.

## Verification

Text regression tests will reject the removed prescriptive phrases and require
the neutral distinctions. Existing task-description, optimizer-family, action
API, and initial-program tests must continue to pass.
