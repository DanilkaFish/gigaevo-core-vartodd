# GF(2^32) Evolution Appendix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current interpretive GF(2^32) evolutionary trace with a three-part, evidence-only appendix containing the initial tuner, descriptive run statistics, and the final tuner.

**Architecture:** Keep the appendix in `GF32_EXAMPLE/LATEX.TEX` and preserve its section label and both listing labels. Replace the middle phase narrative with one route-outcome table, and put concise parameter evidence directly in comments in the final listing.

**Tech Stack:** LaTeX, `listings`, CSV-derived execution statistics, local TeX compiler when available.

## Global Constraints

- The appendix has three substantive subsections only: initial tuner, recorded evolution statistics, and final tuner.
- Report measured evidence without judging the evolutionary process or claiming that bundled changes are controlled ablations.
- Use the terminal finalization profile from `metadata_aux_info`; do not use sign prevalence among top programs as evidence of benefit.
- Preserve `app:gf32_evolution_trace`, `lst:gf32_best_initial`, and `lst:gf32_best_final_program`.
- Do not alter unrelated manuscript files or existing user changes.

---

### Task 1: Rewrite and verify the GF(2^32) appendix

**Files:**
- Modify: `GF32_EXAMPLE/LATEX.TEX`

**Interfaces:**
- Consumes: `GF32_EXAMPLE/vartodd_gf32.csv`, `GF32_EXAMPLE/GF32_EVIDENCE.md`, and the approved design specification.
- Produces: a standalone appendix fragment retaining the existing section and listing references.

- [ ] **Step 1: Record the current appendix boundaries and available compiler**

Run:

```bash
wc -l GF32_EXAMPLE/LATEX.TEX
cp GF32_EXAMPLE/LATEX.TEX /tmp/gf32-evolution-appendix.before.tex
command -v latexmk
command -v pdflatex
```

Expected: `LATEX.TEX` is 428 lines, the baseline copy exists under `/tmp`, and
at least one compiler path may be empty. The baseline is required because
`GF32_EXAMPLE` is currently untracked and ordinary `git diff` cannot display
its changes.

- [ ] **Step 2: Replace the prose and statistical table**

Use `apply_patch` to make these exact structural changes:

1. Reduce the section introduction to the snapshot size, valid-result count,
   initial/final T-counts, and the meaning of ancestry generation.
2. Keep the initial listing and reduce its introduction to the two executed
   stages. Remove the commented-out diagnostic paragraph.
3. Replace the entire `Evolution under changing search bottlenecks`
   subsection with `Recorded evolution statistics` and one table containing:
   ab-initio `118/360`, mid-margin `118/360`, near-end `124/360`; their valid
   rates; median/IQR final counts; counts reaching `T<=1179`; strict cached
   improvement rates; and best observed counts.
4. Follow the table only with the observed incumbent sequence
   `1205 -> 1201 -> 1195 -> 1193 -> 1183 -> 1181 -> 1179 -> 1177` and its
   ancestry generations `1,2,4,5,7,9,10,11`.
5. Reduce the final-tuner introduction to its executable structure and remove
   the interpretive paragraph after the listing.

- [ ] **Step 3: Add concise evidence comments to the final listing**

Use comments adjacent to the relevant constants/settings to record:

- score comparison: same-loaded-path advantages of `+red=1.44`,
  `+tohpe=0.74`, `-yw=0.92`, and `-bucket=0.67` ranks; dimension and z-weight
  differences are only `0.13`, so their signs remain flexible;
- threshold 1210: mean path dimension changes from `20.8` to `8.0` and then
  `2.7`, with only `3--7` nonzero y vectors at dimension 2--3;
- beam: configured 12--16, realized 14, stated as an observed setting rather
  than an optimum;
- margins: 40 reopens near 1224, proportional margin 49 reopens near 1234,
  and 1177 appears at evaluation 1659/1758 from the latter branch;
- guard: 213 to 1758 completed policy evaluations in 6224.423 versus
  6222.993 seconds, with the returned path using 2126/5000 buckets;
- DE, population, seed, and restart values are settings of this tuner only.

- [ ] **Step 4: Check content and formatting invariants**

Run:

```bash
rg -n "Early phase|Middle phase|Late phase|Population-level accumulation|The final program illustrates" GF32_EXAMPLE/LATEX.TEX
rg -n "118/360|28/86|29/95|1\.44|0\.74|0\.92|0\.67|1659|1758|2126" GF32_EXAMPLE/LATEX.TEX
rg -n "label=\{lst:gf32_best_initial\}|label=\{lst:gf32_best_final_program\}|label\{app:gf32_evolution_trace\}" GF32_EXAMPLE/LATEX.TEX
git diff --no-index --check /tmp/gf32-evolution-appendix.before.tex GF32_EXAMPLE/LATEX.TEX
```

Expected: the stale-prose search returns no matches; every evidence value and
all three labels are present; the no-index check reports no whitespace errors.
Its status is allowed to be 1 because the two files intentionally differ.

- [ ] **Step 5: Compile the containing manuscript when a compiler is available**

The appendix is a fragment, so compile its containing file from
`GF32_EXAMPLE`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error 'main_line_numbered_left(5).tex'
```

If `latexmk` is unavailable but `pdflatex` exists, run two passes:

```bash
pdflatex -interaction=nonstopmode -halt-on-error 'main_line_numbered_left(5).tex'
pdflatex -interaction=nonstopmode -halt-on-error 'main_line_numbered_left(5).tex'
```

Expected: exit status 0 and no undefined references to the retained appendix
or listing labels. If neither compiler exists, report that limitation and
retain the successful static checks.

- [ ] **Step 6: Review the final diff**

Run:

```bash
diff -u /tmp/gf32-evolution-appendix.before.tex GF32_EXAMPLE/LATEX.TEX
```

Expected: only the requested appendix prose, table, and listing comments have
changed; no unrelated files appear in the diff.
