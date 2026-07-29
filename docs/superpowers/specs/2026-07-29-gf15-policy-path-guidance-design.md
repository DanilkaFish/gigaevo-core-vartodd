# GF Evolution Policy and Path Guidance Design

## Goal

Revise `problems/vartodd_evo_gf/task_description.txt` so mutations primarily
improve policy mapping, saved-path selection, and restart placement. Numerical
optimization remains supporting machinery.

## Required Guidance

1. Rank schedules must follow observed policy-band transitions. Do not use
   `TARGET_FINAL_RANK + 55` as an unexplained universal switch point.
2. Keep inexpensive standalone TOHPE behavior through rank bands where it
   continues to generate and retain useful actions. Activate heavy TODD only
   in the evidenced lower-rank region.
3. In the lower-rank region, TODD may become the only source of positive
   actions. Low `max_buckets` or `limit_bucket` values can prevent a rare
   existing action from being discovered.
4. Raising TODD `max_buckets` searches more distinct z buckets. This increases
   both the probability of finding an action and the diversity of actions
   available to scoring and beam selection. `limit_bucket` must be large
   enough not to truncate that search.
5. TOHPEprefix traversal is separate. Do not automatically give it the same
   breadth as terminal TODD.
6. Saved paths should branch before the observed failing rank band. Close
   reopening is appropriate only when the intended experiment genuinely
   targets the inherited tail.
7. Encourage diversity in score shapes and selection behavior when the archive
   has converged to one policy family. Avoid presenting one beamwidth,
   temperature, cap, or schedule as universally correct.

## Scope

This change edits only the shared task description and its prompt-contract
tests. It does not modify initial programs, runtime code, path storage, metrics,
or algorithm configuration.

## Verification

- Contract tests require the lower-region TODD action-diversity guidance.
- Existing action API, parameter-group, saved-path, runtime, and source-specific
  z-statistics contracts remain present.
- The task description contains no named optimizer-family catalog.
