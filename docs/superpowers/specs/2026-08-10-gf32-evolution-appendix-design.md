# GF(2^32) Evolution Appendix Rewrite

## Purpose

Rewrite `GF32_EXAMPLE/LATEX.TEX` as a short, evidence-only appendix. The text
will report what was executed and measured, without evaluating the quality of
the evolutionary process or assigning isolated causal effects to bundled
mutations.

## Structure

The appendix will contain exactly three substantive subsections.

### 1. Initial tuner

Introduce the best generation-1 tuner in one short paragraph and retain its
condensed listing. The paragraph will state only its measured result
(`T=1205`) and its two executable stages: an 80-evaluation DE scout followed
by a 120-evaluation PatternSearch tail. Listing comments will identify the
roles of the scout, path reopening, candidate sources, pool sizes, beam width,
and evaluation budgets. The commented-out diagnostic paragraph after the
listing will be removed.

### 2. Recorded evolution statistics

Replace the current phase narrative and phase table with one descriptive
table organized by the mutation route recorded in the CSV. The table will use
all 360 non-root programs and report:

| Route | Mutations | Valid programs | Median final T-count [IQR] | Valid programs reaching T<=1179 | Strict cached-path improvements | Best observed T-count |
|---|---:|---:|---:|---:|---:|---:|
| Ab-initio | 118/360 (32.8%) | 97/118 (82.2%) | 1209 [1203, 1215] | 0/97 | not applicable | 1193 |
| Mid-margin | 118/360 (32.8%) | 86/118 (72.9%) | 1195 [1185, 1201] | 10/86 (11.6%) | 28/86 (32.6%) | 1177 |
| Near-end | 124/360 (34.4%) | 95/124 (76.6%) | 1187 [1181, 1201] | 20/95 (21.1%) | 29/95 (30.5%) | 1177 |

`Strict cached-path improvement` means that the returned path ended below the
loaded path's final T-count. It is not meaningful for ab-initio programs,
whose baseline is the input matrix. A short table note will state that the
routes operated on different starting paths and generations and that many
programs changed several mechanisms together; the numbers are descriptive,
not isolated ablations.

After the table, include only a compact incumbent sequence:
`1205 -> 1201 -> 1195 -> 1193 -> 1183 -> 1181 -> 1179 -> 1177`, with the
corresponding ancestry generations `1, 2, 4, 5, 7, 9, 10, 11`. State that
generation is lineage depth rather than wall-clock order. Do not retain the
five interpretive phase paragraphs.

### 3. Final tuner

Introduce the first `T=1177` tuner in one short factual paragraph and retain
only its condensed listing; remove the interpretive paragraph after it.
Listing comments will attach evidence directly to the relevant choices:

- Among the 30 valid programs ending at `T<=1179`, the realized finalization
  weights were positive for reduction in 28/30 and TOHPE lookahead in 27/30,
  negative for y-weight in 25/30, and positive for z-weight in 22/30. Thus the
  listing will describe a negative y-weight bias, not a negative z-weight
  bias; z-weight remains sign-flexible. These frequencies are inherited
  associations rather than ablations.
- The terminal TOHPE threshold is 1210. On the returned `T=1177` path, the
  mean null-space dimension changed from 20.8 above 1210 to 8.0 and then 2.7
  below it. At dimension 2--3 there are only 3--7 nonzero y vectors, supporting
  the small terminal sparse/dense sampling budgets.
- The configured beam range is 12--16 and the realized beam is 14. The
  high-retention parent realized 43 terminal actions, while the final program
  realized 30.3 terminal actions; the comment will not claim that 12--16 is
  an optimal beam width.
- Margin 40 reopened the loaded `T=1183` trajectory near `T=1224`; the
  proportional reopen margin evaluates to 49, and the `T=1177` result was
  recorded after reopening near `T=1234`, at policy evaluation 1659 of 1758.
- Introducing the 5000-bucket hard guard changed the high-retention parent's
  213 completed policy evaluations to 1758 in essentially the same runtime
  (6224.423 versus 6222.993 seconds). The returned `T=1177` path used 2126
  buckets, below the guard. This is the closest comparison in the snapshot to
  a one-mechanism change.
- DE, its population settings, one evaluation seed, and restart counts will
  be identified as the settings of this program, not as globally preferred
  choices.

## Evidence and reproducibility

All counts come from `GF32_EXAMPLE/vartodd_gf32.csv`. Per-program path,
policy, score, and optimizer statistics come from each row's
`metadata_aux_info`. The detailed provenance remains in
`GF32_EXAMPLE/GF32_EVIDENCE.md`; program identifiers stay out of the
manuscript.

The rewrite will preserve the existing labels and listing style, then compile
the containing LaTeX manuscript if the local toolchain permits. Verification
will include a scan for stale phase prose and a review of all numerical claims
against the CSV/evidence ledger.
