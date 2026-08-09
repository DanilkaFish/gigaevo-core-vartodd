# GF Cost Diagnostics Manuscript Typography Design

## Goal

Regenerate the six GF10--GF15 cost-diagnostic panels with text that remains
readable when each panel is embedded at `0.32\textwidth` in the manuscript.

## Scope

- Modify `info/tools/plot_gf_cost_diagnostics.py` as the single plotting
  source.
- Preserve the plotted points, rolling statistics, best-so-far curve, colors,
  axis ranges, and output formats.
- Reconstruct the missing run manifest from the backups and API-cost CSVs
  already under `info`, validating the run mapping against the expected final
  best T-counts 187, 215, 243, 293, 319, and 363 for GF10 through GF15.
- Do not modify unrelated evolution code or the manuscript TeX.

## Presentation

Author each plot near its final panel size instead of relying on a large
Matplotlib canvas that LaTeX heavily reduces. Use a 3.6 by 2.7 inch canvas and
explicit serif typography:

- axis labels: 14 pt, bold;
- tick labels: 12 pt;
- legend: 11 pt.

Omit the internal `GF(2^n)` axes title. The existing LaTeX subcaption already
identifies each field size, so removing the duplicate title also leaves more
room for the enlarged text. Keep the current legend entries and use tight
layout so labels are not clipped.

## Outputs

Write new, non-destructive manuscript variants beneath
`info/outputs/gf10_gf15_cost_diagnostics_manuscript/`. For each exponent 10
through 15, create:

- `gfN_cost_diagnostics.pdf` for manuscript inclusion;
- `gfN_cost_diagnostics.png` at 240 DPI for inspection.

Retain aligned-points and run-summary CSV outputs in the same directory so the
figures remain auditable and reproducible.

## Failure Behavior

Retain the script's explicit errors for missing manifests, backups, API CSVs,
program records, and required Redis/Valkey server binaries. Regeneration must
stop rather than silently omit a panel or substitute a different run.

## Verification

- A focused figure test will assert the canvas size, absence of an axes title,
  and exact axis-label, tick-label, and legend font sizes.
- Run the plotting script against all six backups and confirm that it emits
  six PDFs, six PNGs, `aligned_points.csv`, and `run_summary.csv`.
- Confirm the regenerated run summary reports the expected best T-count for
  every exponent.
- Inspect a rasterized six-panel manuscript preview to ensure the enlarged
  text is readable and no labels or legends are clipped.
