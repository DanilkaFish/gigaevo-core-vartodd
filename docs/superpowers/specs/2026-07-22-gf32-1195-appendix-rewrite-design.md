# GF(2^32) 1195-T Evolution Appendix Rewrite Design

## Goal

Rewrite `appendix/gf32_evolution_appendix_section.tex` around the supplied
`appendix/gf32_1195/dump.rdb` run, replacing the obsolete 1205-T programs and
lineage while preserving the appendix's organization, prose style, annotated
pseudocode, and explanation order.

## Source of truth

Program records and execution evidence come from the isolated Redis instance
loaded from `appendix/gf32_1195/dump.rdb`. True T-counts come from each valid
program's `path_summary.final_rank`, not from shaped `fitness`.

The dump contains 171 program records spanning generations 1--10, including
160 valid evaluations. The best initial program is `26250e5d` at T-count 1235.
The best evolved program is `4fb701ef` at T-count 1195, first reached in
generation 6.

## Editorial structure

Keep the current section title and labels, followed by:

1. run summary;
2. best initial program and condensed listing;
3. best evolved program and generation-annotated condensed listing;
4. comparison of the two listings; and
5. the non-monotonic-evolution conclusion.

Retain both listing labels and the `vtpycompact` API-level pseudocode style.
Comments identify program IDs, generations, rationale, feature provenance,
and observed behavior. A short LLM mutation rationale remains clearly labeled
as rationale; surrounding prose distinguishes proposed mechanisms from facts
demonstrated by execution evidence.

## Initial program content

Listing `lst:gf32_best_initial` represents generation-1 program `26250e5d`:

- PSO population 16 and 144 optimizer evaluations;
- one global learnable score profile with power 1;
- beam-2 softmax selection at temperature 0.20 and final pool size 12;
- exhaustive one-hot plus learnable dense sampling;
- TOHPE keep 12, reserve 0, and two z choices; and
- effectively disabled TODD because its source keep is zero.

The listing caption and prose report the true final T-count 1235.

## Evolved program content

Listing `lst:gf32_best_final_program` represents generation-6 program
`4fb701ef`:

- reopen saved path `i1701_m1244_f1197_fc0b0d69` with margin 50, producing
  the observed branch start at rank 1248;
- PSO population 10, 80 evaluations, seed 29 during tuning, and final
  reevaluation seeds 29, 87, and 43;
- a global quadratic score profile with learnable weights and zero centers;
- beam-2 softmax at 0.20 and final pool size 16;
- one-hot cap 8, sparse budget 4, sparse weight 3, and learnable dense budget
  3--7;
- TOHPE keep/reserve 4/2 with two z choices;
- TODD keep/reserve 6/2, four actions per bucket, learnable minimum z research,
  and a rank schedule using finite cap 30937 through rank 1225 and full search
  below it.

Generation annotations follow the actual winner ancestry: G1 established the
PSO/beam-2/TOHPE baseline and produced the 1235 path; G2 introduced saved-path
refinement and effective TODD retention; G4 reached 1197 with full tail TODD;
G5 produced the exact `fc0b0d69` path and introduced cost-aware cap scheduling;
G6 reopened that path and combined the final settings.

## Evidence-aware commentary

Comments will report these measured facts:

- the G6 path descends from 1248 to 1195;
- finite-cap early groups stop at 43 researched buckets with the selected
  parameters, while the terminal group averages 255871 researched buckets
  and only 0.04 accepted actions per z bucket;
- terminal dimensions are 1--4, so y budgets often saturate there and the
  changed y sampling mainly affects the earlier mid-band;
- the smaller PSO budget follows parent saturation evidence, but runtime did
  not fall: G6 took 12134 seconds because unrestricted terminal TODD remained
  dominant; and
- the mutation changed several mechanisms at once, so the run demonstrates
  the combined program's success rather than isolated causality for power 2,
  sampling, reserve, or optimizer-budget changes.

## Verification

After editing:

- search for stale 1205/G12/DE/restart identifiers and values;
- compare every program ID, path name, generation, T-count, and policy value
  against dump-derived evidence;
- check listing environments, braces, labels, and TeX syntax structurally;
- compile the containing TeX document if a standalone build target exists;
  otherwise report that the appendix is a section fragment and run the
  strongest available static checks; and
- inspect the final diff to confirm style/order preservation and no unrelated
  changes.
