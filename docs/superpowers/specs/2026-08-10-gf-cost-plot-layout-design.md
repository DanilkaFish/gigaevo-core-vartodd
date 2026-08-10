# GF Cost Plot Layout Design

## Goal

Produce title-free GF10–GF15 manuscript panels whose typography is readable without overwhelming the data region, and correct GF12's cumulative API-cost trajectory.

## Typography and layout

Each individual panel remains 3.6 by 2.7 inches. Axis labels use 11-point normal-weight serif text, tick labels use 9 points, and legend text uses 8 points. Axis labels are rendered on one line as `Cumulative LLM API cost (USD)` and `Program final T count`.

The four legend entries remain unchanged, but the legend uses two columns at the upper center. This reduces its height from four rows to two while keeping it inside the figure. Per-graph titles remain omitted. Existing line, marker, grid, axis-limit, PNG, and vector-PDF behavior remains unchanged.

## GF12 cost source

The GF12 run spans 2026-08-02 21:27 UTC through 2026-08-03 06:55 UTC. Its current `asus_gf6.csv` mapping begins only at 04:57 UTC, omitting roughly 7.5 hours and producing a truncated total of $0.501 from 166 calls.

GF12 will instead use `this_backup/kostrome2.csv`. Within the GF12 evolution window, this export contains 402 calls totaling approximately $1.287 and covers 21:32–06:53 UTC. The manifest change corrects the source rather than rescaling the plotted result.

## Outputs

Regenerate all GF10–GF15 PNG and PDF files in `info/outputs/gf10_gf15_cost_diagnostics_manuscript/`, together with `aligned_points.csv` and `run_summary.csv`. Existing files are intentional generated outputs and may be replaced by their corrected variants.

## Verification

Tests will inspect the Matplotlib figure object to verify title omission, exact typography, normal label weight, one-line labels, a two-column legend, and text containment inside the canvas. A manifest/data regression test will verify GF12 resolves to `kostrome2.csv`, has full-window call coverage, and accumulates more than $1.2.

After generation, inspect GF12 and the six-panel preview visually to ensure labels fit, the legend does not dominate the plot, and GF12 extends to approximately $1.29.
