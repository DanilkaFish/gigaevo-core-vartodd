# VarTODD Near-End Rank Window Design

## Goal

Limit the near-end Selectable Shared Paths inventory to paths relevant to the
current frontier. Paths more than 100 ranks above the best selectable rank are
too early for the near-end refinement regime.

## Behavior

The near-end inventory first computes `frontier_rank` from all otherwise
eligible saved paths. It then exposes only family representatives satisfying:

```text
frontier_rank <= path_rank <= frontier_rank + 100
```

The upper boundary is inclusive. A representative at `frontier_rank + 100`
remains visible; one at `frontier_rank + 101` does not.

Filtering happens after family representatives and their reuse eligibility are
determined. It does not delete paths, alter provenance or usage counters, or
change the mid-margin inventory. A path hidden from near-end can remain
available to mid-margin.

## Evidence

The near-end inventory header includes the effective inclusive range:

```text
rank_window=<frontier_rank>..<frontier_rank + 100>
```

The near-end overlay explains that paths above this window belong to an earlier
refinement region and are intentionally omitted from this regime's live cards.

## Verification

Tests verify that:

- the frontier and `frontier_rank + 100` are included;
- `frontier_rank + 101` is excluded;
- the rendered header reports the window;
- mid-margin continues to expose an otherwise eligible higher-rank path.
