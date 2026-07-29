# GF Evolution Policy and Path Guidance Design

## Goal

Revise `problems/vartodd_evo_gf/task_description.txt` so mutations primarily
improve policy mapping, saved-path selection, and restart placement. Numerical
optimization remains supporting machinery.

## Required Guidance

1. Rank schedules must follow observed policy-band transitions. Do not use
   `TARGET_FINAL_RANK + 55` as an unexplained universal switch point.
2. In the lower-rank region, TODD may become the only source of positive
   actions. Low `max_buckets` or `limit_bucket` values can prevent a rare
   existing action from being discovered. Raising TODD `max_buckets` searches
   more distinct z buckets, increasing both the probability of finding an
   action and the diversity of actions available to scoring and beam
   selection. `limit_bucket` must be large enough not to truncate that search.
3. Saved paths should branch before the observed failing rank band. Close
   reopening is appropriate only when the intended experiment genuinely
   targets the inherited tail.

## Algorithm Guidance

Update `config/algorithm/vartodd_diverse_gf16_tohpe_updated.yaml` to:

- sample wide-margin exploration with probability `0.6` and saved-path
  exploitation with probability `0.4`;
- guide wide-margin branches toward margins `70..140` and near-tail branches
  toward margins `15..40`;
- explain that lower-region TODD action starvation can justify raising
  `max_buckets` and `limit_bucket`, including an unrestricted terminal hard cap;
- connect higher `max_buckets` with both action discovery and action diversity.

## Scope

This change edits the shared task description, the updated TOHPE algorithm
guidance, and their prompt-contract tests. It does not modify initial programs,
runtime code, path storage, or metrics.

## Verification

- Contract tests require the lower-region TODD action-diversity guidance.
- Contract tests require the updated wide-margin/exploitation probabilities
  and margin ranges.
- Existing action API, parameter-group, saved-path, runtime, and source-specific
  z-statistics contracts remain present.
- The task description contains no named optimizer-family catalog.
