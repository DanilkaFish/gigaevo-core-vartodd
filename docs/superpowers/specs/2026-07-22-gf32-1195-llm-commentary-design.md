# GF(2^32) 1195-T LLM Commentary Design

## Goal

Lightly polish the comments around the generation-6 program in
`appendix/gf32_evolution_appendix_section.tex`. Make them easier to read and
align their explanations with the program insights and evidence used to form
the successful mutation. Preserve the appendix's organization, condensed
pseudocode, comment style, and explanation order.

The appendix remains an explanation of the evolved program, not an analysis
of the model's reasoning. It does not reproduce or cite the prompt, reasoning
trace, completion, or raw insight labels.

## Source of truth

The relevant OpenRouter generation is
`gen-1784459807-5caO3v2YHwmbdadrLyWG`. Its request began at
2026-07-19 11:16:47.867 and completed at 11:18:21.179; program `4fb701ef`
was created at 11:18:21.894. The immediately following generation produced
the separate margin-200 program and must not be used for this commentary.

The attached system prompt, attached reasoning, stored structured mutation
output, winner's code, and evaluation record are verification sources only.
They agree on the selected path and material parameter changes. The appendix
will summarize the resulting technical explanations in ordinary reader-facing
language.

## Editorial approach

Keep the current comment blocks and their order. Replace dense metadata and
quasi-quotation with short explanations placed beside the setting they
describe. Each block should answer at most three questions:

1. what changed or was retained;
2. which observed search problem made that choice relevant; and
3. what the winning run subsequently did, when useful.

Do not use phrases such as `the LLM selected` or `the model inferred` to
narrate the reasoning process. Do not include raw insight identifiers such as
`excessive_eval_budget_given_no_improvement`. The relationship to the LLM is
shown by making the comments follow the same evidence-based explanations, not
by narrating its internal process.

## Comment revisions

### Program summary

Replace the quoted mutation rationale and the raw list of diagnosed signals
with a compact overview: the mutation reopened a lightly reused rank-1197 path
near its top, retained capped-early/full-tail TODD, changed scoring and
mid-band sampling, and reduced the PSO evaluation budget.

### Saved path and branch point

Explain that `i1701_m1244_f1197_fc0b0d69` was one of the best available paths
and had been reused only once, while the alternative rank-1197 path had already
failed three reuses. `margin=50` produced `init_rank_thr=1247` and an observed
loaded rank of 1248, providing a broad branch above the difficult tail.

### Optimizer budget

The direct parent last improved at evaluation 14 of 188 and encountered its
best rank 18 times without beating the loaded path. Connect that saturation to
the smaller `PSO_POP=10` and `N_EVAL=80`. Describe this only as reduced
optimizer evaluation work; do not discuss or claim end-to-end wall time.

### Scoring

Explain `pow=2` as a change intended to separate candidate scores more clearly
when few useful terminal actions survive. Keep the existing caution that the
successful run changed several mechanisms together and does not isolate the
effect of score power alone.

### Tree-search width

Keep beam-2 softmax at temperature 0.20. Explain it as the retained compromise
between greedy width 1 and a wider, more expensive tree. The terminal action
pool was underfilled and acceptance was very low, so the evidence did not
support widening beyond 2.

### Sampling

Explain that one-hot 8 still covers all nonzero y vectors when terminal
dimension is at most 3, while sparse weight-3 and dense 3--7 sampling add
coverage mainly in earlier bands where the dimension is larger. Avoid claiming
that these larger budgets create additional terminal candidates once the
small nullspace is saturated.

### Candidate sources and TODD schedule

Keep the existing explanation that both TOHPE and TODD contributed accepted
actions. Clarify that the selected rank-1197 path had already depended on full
terminal search, so the mutation retained a finite early cap and full TODD
below rank 1225. Connect reserve 4-to-2 to the very low terminal acceptance and
underfilled pool: the smaller reserve lets TODD stop earlier instead of
continuing z research merely to fill a larger retained quota.

### Observed descent

Retain the compact measured sequence: the reopened path began at rank 1248,
reached 1201 at evaluation 2, 1199 at evaluation 18, and 1195 at evaluation
38; 68 evaluations completed before timeout salvage. Present this as the
result of the combined program rather than proof that one setting was solely
responsible.

## Scope of the edit

Only listing comments related to the best evolved program are revised. Program
constants, pseudocode operations, surrounding prose, captions, labels, section
structure, the initial-program listing, and the underlying 1195-T factual trace
remain unchanged.

Remove the quasi-quotation, raw insight-name list, and the statement contrasting
the smaller optimizer budget with unchanged total runtime. Do not add a prompt
analysis, reasoning transcript, OpenRouter identifier, or the incorrect
margin-200 mutation to the appendix.

## Verification

After editing:

- compare every summarized explanation with the attached reasoning, stored
  insights, program comments, and structured mutation output;
- confirm the listing still uses path `fc0b0d69`, margin 50, PSO 10/80,
  beamwidth 2, score power 2, one-hot 8, sparse 4/weight 3, TODD reserve 2,
  and the 30937/full rank schedule;
- search for stale margin-200 text, raw insight identifiers, direct prompt or
  reasoning quotations, and the removed wall-time contrast;
- check listing environments, braces, labels, and TeX syntax structurally;
  compile the containing document if a build target is available; and
- inspect the final diff to ensure that no unrelated files or pseudocode were
  changed.
