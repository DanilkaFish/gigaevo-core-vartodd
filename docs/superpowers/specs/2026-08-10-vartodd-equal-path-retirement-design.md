# VarTODD Equal Path Retirement Design

## Goal

Remove permanent frontier-path pinning from both `mid_margin` and `near_end` refinement. Treat every selectable path with the same route-local non-improvement limit of four valid attempts.

## Selection behavior

`PathStore._selection_inventory()` will continue to compute the current frontier rank because rank is still needed for ordering and the near-end reproduced-frontier filter. It will no longer choose or return a pinned record.

After sibling canonicalization and stale-path filtering, every record will pass through the same failure check:

- read `nonimproved_count` for the requested route;
- keep the path while the count is below four;
- remove the path once the count reaches four.

The counter remains independent for `mid_margin` and `near_end`. Rank does not change the failure allowance.

Near-end sampling will no longer exclude a path merely because its rank is more than four ranks above the frontier. An above-frontier path will still be excluded after its near-end trials have already reproduced or beaten the frontier, because it has fulfilled its role as an alternative descent.

Eligible paths keep their existing deterministic order: rank, non-improvement count, use count, descending initial rank, and name. Thus better ranks are displayed first, but receive no extra capacity or permanent prompt placement.

## Prompt rendering and public API

`render_selectable_path_cards()` will remove the separate `Pinned Best Long-Descent Path` section. All cards, including frontier-rank cards, come from the normal bounded selection and count toward `top_k` and `max_per_rank`.

`has_selectable_paths()` will depend only on the eligible-record list. `_selection_inventory()` will return the frontier rank and eligible records without a pinned value; its tuple shape may be retained temporarily as `(frontier_rank, None, eligible)` to minimize unrelated caller changes.

The public `selectable_records()` defaults will use the same non-improvement limit of four as prompt rendering, eliminating the current inconsistent frontier limit.

## Compatibility and scope

This change preserves current sibling canonicalization, route-specific usage accounting, path cards, rank ordering, prompt `top_k` values, and the near-end reproduced-frontier filter. It removes the near-end `frontier_rank + 4` rank-window filter. It does not alter route probabilities, archive selection, saved-path creation, or existing usage counters.

Previously accumulated failures take effect immediately. A path already at four or more non-improvements for a route will no longer be selectable for that route after the process reloads this code.

## Tests

Regression tests will demonstrate that:

1. a frontier path with three failures remains eligible;
2. a frontier path with four failures is retired;
3. a worse-rank path follows the identical four-failure boundary;
4. no pinned prompt section is rendered;
5. frontier paths count toward `top_k` and `max_per_rank`;
6. `has_selectable_paths()` becomes false when every candidate reaches the common limit;
7. a near-end path more than four ranks above the frontier remains eligible while below the common failure limit;
8. an above-frontier near-end path is still retired after reproducing or beating the frontier;
9. existing sibling canonicalization tests remain green.

## Alternatives considered

- A seven-failure frontier allowance without pinning was rejected because best paths would still receive privileged treatment.
- A capped pinned section was rejected because it would remain outside normal `top_k` selection and keep biasing the LLM toward one path.
- Keeping the near-end `frontier_rank + 4` window was rejected because it unnecessarily prevents the route from revisiting alternative higher-rank descents.
- Randomized path ordering is outside this change; deterministic ordering is retained while eligibility and retirement become uniform.
