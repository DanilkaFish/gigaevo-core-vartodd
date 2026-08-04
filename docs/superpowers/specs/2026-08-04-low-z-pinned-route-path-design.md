# Independent Low-Z Route Path Design

## Objective

Ensure both `mid_margin` and `near_end` mutation routes always receive the
most under-researched promising path: the valid path with the lowest known z
research among the seven best distinct final ranks.

This path is a second structural pin. It is independent of the existing
long-descent pin and of the ordinary sampled route inventory. No prompt file
changes are required.

## Candidate Population

Use the same common validity rules as the current route inventory:

- include ab-initio root paths;
- include children only when they improved their loaded parent;
- exclude non-improving children;
- exclude stale improved children;
- require a parseable final rank.

Sort the distinct final-rank values ascending and keep the seven lowest
values. Every valid path whose final rank belongs to this set is a low-z pin
candidate.

Candidates without a known z-research value are excluded from low-z pin
selection. Missing evidence must not be interpreted as zero research. The
selector uses `max_z_researched`, with the legacy
`max_todd_z_researched` field accepted as a compatibility fallback.

## Selection

Choose the low-z pin using this deterministic ordering:

1. lower z-research value;
2. lower route-local `used_count`;
3. lower final rank;
4. higher initial/restart rank;
5. path name.

Selection is calculated independently for `mid_margin` and `near_end`, so a
z-value tie may choose different paths when their route-local usage differs.

The low-z pin does not obey the ordinary frontier/non-frontier
non-improvement exhaustion limits. It remains visible because the purpose of
the role is to prevent under-researched promising paths from disappearing.
Actual mutations using it continue to update the existing route-local usage,
improvement, and non-improvement counters.

## Inventory Semantics

Extend the route selection snapshot to contain:

- `frontier_rank`;
- the existing best-rank, highest-initial-rank long-descent pin;
- the independent low-z pin;
- the ordinary eligible records.

Neither pinned record consumes `top_k` or `max_per_rank`. Both are excluded
from the ordinary eligible list so they cannot reappear as sampled cards.

If one record wins both pinned roles, render it once and attach both role
labels. Do not choose a second-best low-z path merely to force another card;
the literal lowest-z result must be preserved.

`has_selectable_paths` returns true when either pin exists or the ordinary
inventory is non-empty. `selectable_records` continues to expose only bounded
ordinary records, preserving its existing public behavior.

## Rendering

Pinned content remains before `## Selectable Shared Paths`.

When the low-z pin is distinct, render a separate section titled:

```text
## Pinned Lowest-Z Top-Rank Path
```

The section declares:

- `selection_role=lowest_z_top_7_distinct_ranks`;
- `independent_pin=1`;
- the selected path's z-research value;
- the route and frontier rank.

When both roles select the same record, keep the existing long-descent
section and add both role labels to that section. Render only one path card.

The existing route-specific card formats remain unchanged:

- `near_end` receives the detailed evidence card with route statistics,
  policy bands, policy profiles, and producer search statistics;
- `mid_margin` receives the compact path record.

Pinned sections are guaranteed inventory and therefore are not charged
against the ordinary card count or character budget.

## Error and Compatibility Behavior

Malformed ranks are handled by the existing common-record filter. Malformed
or absent z values make a path ineligible only for the low-z role; the path
may still qualify for the long-descent pin or ordinary inventory.

If fewer than seven distinct ranks exist, use all available distinct ranks.
If no candidate has known z research, omit the low-z section and retain the
current inventory behavior.

The inventory is derived from one `iter_path_records()` snapshot per render,
preventing the two pins and ordinary records from observing different stores.

## Verification

Tests must demonstrate that:

1. The selector considers seven distinct ranks rather than seven paths.
2. The minimum known z value wins across every path in those ranks.
3. An eighth distinct rank is excluded even when it has lower z research.
4. Missing z evidence does not beat known evidence.
5. The low-z pin ignores exhaustion and ordinary near-end frontier-window
   filtering.
6. The pin is excluded from `selectable_records` and does not consume
   `top_k` or `max_per_rank`.
7. Distinct pins render as two sections before the ordinary inventory.
8. A shared winner renders once with both role labels.
9. Route-local usage breaks z ties independently for `mid_margin` and
   `near_end`.
10. Existing route inventory, rendering, and result-accounting tests remain
    green.
