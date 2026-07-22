# GF(2^32) 1195-T LLM Commentary Design

## Goal

Revise the comments around the generation-6 program in
`appendix/gf32_evolution_appendix_section.tex` so they explain how the LLM
turned the supplied evolutionary evidence into the mutation that reached
T-count 1195. Preserve the appendix's existing organization, condensed
pseudocode, comment style, and explanation order.

The revision paraphrases the relevant decision process. It does not reproduce
the prompt, private reasoning trace, or completion verbatim.

## Source of truth

The relevant OpenRouter generation is
`gen-1784459807-5caO3v2YHwmbdadrLyWG`. Its request began at
2026-07-19 11:16:47.867 and completed at 11:18:21.179; program `4fb701ef`
was created at 11:18:21.894. The immediately following generation produced
the separate margin-200 program and must not be used for this commentary.

The attached system prompt explains how the model was instructed to read
program metrics, execution evidence, optimizer statistics, and the live path
store. The attached reasoning, the stored structured mutation output, the
winner's code, and its evaluation record agree on the selected path and all
material parameter changes.

## Narrative approach

Distribute the LLM's evidence-to-decision chain across the comments beside the
mechanism that it motivated. This is preferable to one long rationale block,
which separates reasoning from code, or repeated `LLM evidence` labels, which
would make the listing noisy.

Each revised comment follows the same compact sequence:

1. evidence available to the LLM;
2. inference made from that evidence;
3. corresponding program change; and
4. observed contribution to the successful combined run where relevant.

Use phrases such as `the LLM selected`, `the model inferred`, and `this
motivated` to make the LLM's role explicit. Retain the distinction between a
reasoned hypothesis and an observed run, but focus on the reasoning that led
to the lower rank.

## Decision chain to explain

### Path selection and branch point

The live path store showed two full-search paths at rank 1197. The LLM chose
`i1701_m1244_f1197_fc0b0d69` because it had only one recorded reuse and no
improvement, whereas the alternative 1197 path had already been reused three
times without improvement. Since the selected path was already full-searched,
the model did not treat a larger z cap as the main lever. It chose an earlier
branch plus a changed policy mechanism.

The model reasoned that `margin=50` gives a branch threshold of
`1197 + 50 = 1247`, reopening at or near the top of the saved prefix. The
execution record reports `init_rank_thr=1247` and an actual loaded rank of
1248. This gave the new policy substantially more branch room than a direct
return to the degenerate tail.

### Optimizer budget

Direct parent `04b87669` last improved at evaluation 14 of 188 and rediscovered
its best rank 18 times without beating the loaded rank 1203. The LLM treated
this as optimizer saturation and reduced `PSO_POP` from 12 to 10 and the
nominal evaluation budget from 120 to 80. The comments describe this as a
reduction in optimizer evaluation work. They do not make or discuss an
end-to-end wall-time claim.

### Scoring and y sampling

Parent `a592774c` supplied the explicit `pow=1` and `sparse=0` insights, while
the parent evidence also showed wasteful high dense sampling and exhaustive
one-hot enumeration. The LLM combined these signals as follows:

- raise both score powers from 1 to 2 to increase action-ranking contrast;
- cap one-hot sampling at 8, which still covers the at-most-seven nonzero
  vectors when the terminal dimension is at most 3;
- enable four sparse samples with maximum weight 3 for earlier bands where
  the dimension leaves room for candidates that one-hot sampling misses; and
- keep dense sampling in the bounded range 3--7 rather than the second
  parent's much larger range.

The comments should make clear that terminal y sampling is already close to
saturation and that the sparse/dense changes mainly alter earlier mid-band
coverage.

### TODD schedule and reserve

The selected 1197 path had required a full terminal search. The LLM therefore
preserved the direct parent's two-region structure: a finite cap of 30937 in
the early band and full TODD below rank 1225. Its changed mechanism was not
more z coverage, but different scoring and y coverage plus a lower TODD
reserve. The low terminal acceptance evidence motivated reducing
`TODD_RESERVE` from 4 to 2 so fewer actions were forced through an underfilled
source pool.

### Observed successful descent

The closing optimization comment connects the model's decisions to the
observed descent: the reopened path began at rank 1248, reached 1201 at
evaluation 2, 1199 at evaluation 18, and 1195 at evaluation 38; 68 evaluations
completed before timeout salvage. This validates the combined guided-
innovation mutation. It does not claim that any single changed parameter was
individually causal.

## Scope of the edit

Only explanatory prose and listing comments related to the best evolved
program are revised. Program constants, pseudocode operations, captions,
labels, section structure, the initial-program listing, and the underlying
1195-T factual trace remain unchanged.

Remove the short quasi-quotation currently labeled `LLM mutation rationale`
and replace it with a connected paraphrase. Remove the existing statement
that contrasts the smaller optimizer budget with unchanged total runtime, in
accordance with the requested focus on successful reasoning. Do not introduce
the incorrect margin-200 mutation or its OpenRouter generation.

## Verification

After editing:

- compare every paraphrased decision with the attached reasoning and stored
  structured mutation output;
- confirm the listing still uses path `fc0b0d69`, margin 50, PSO 10/80,
  score power 2, one-hot 8, sparse 4/weight 3, TODD reserve 2, and the
  30937/full rank schedule;
- search for stale margin-200 text, direct prompt quotations, and the removed
  wall-time contrast;
- check listing environments, braces, labels, and TeX syntax structurally;
  compile the containing document if a build target is available; and
- inspect the final diff to ensure that no unrelated files or pseudocode were
  changed.
