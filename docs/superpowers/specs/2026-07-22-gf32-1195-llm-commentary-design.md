# GF(2^32) 1195-T Appendix Commentary Design

## Goal

Revise `appendix/gf32_evolution_appendix_section.tex` so its terminology
matches the main manuscript and its evolved-program listing is easy to read.
Keep only short comments that connect an observed search behavior to a program
choice and, where the stored run supports it, to the resulting improvement.

## Source of truth

Use the terminology and conceptual hierarchy in
`appendix/main_submission_prxquantum_section2B_ftqc_revised(1).tex` and
`appendix/vartodd_appendix_prxquantum.tex`. Use the winner's stored program,
trajectory, lineage, and evaluation record only to check factual claims.

The appendix must explain the evolved tuner, not reproduce or analyze the
LLM's prompt, completion, or hidden reasoning. Its annotations may summarize
the useful observations reflected in those sources without presenting them as
quotations or individually proven causes.

## Terminology

Use the manuscript's reader-facing terms:

- current \(T\)-count or parity-matrix column count, rather than rank;
- saved trajectory and restart point, rather than saved path and branch point;
- numerical policy search or particle-swarm optimization, rather than refiner;
- candidate actions, TOHPE-style source, and TODD/FastTODD source;
- admissible-vector sampling and admissible space;
- source retention, final action pool, score power, and beam width;
- high-\(T\)-count and terminal regimes, rather than early, tail, or mid-band.

The actual `PATH_NAME` identifier remains because it is part of the condensed
API. Rename the display-only schedule constants to `HIGH_TCOUNT` and
`TERMINAL_TCOUNT` and update their use in `set_todd_search`.

## Comment structure

Use sparse causal annotations rather than preserving a comment for every
component. Keep at most five compact blocks, in the program's existing order:

1. **Saved trajectory and restart point.** The selected trajectory had reached
   \(T\)-count 1197 and was reopened at \(T\)-count 1248, leaving room for a
   different descent.
2. **Numerical policy search and seeds.** The parent had stopped improving, so
   the PSO population and evaluation budget were reduced. Tuning uses one seed;
   the selected policy is then rerun with three seeds to reduce dependence on
   that tuning seed and retain the best resulting trajectory.
3. **Scoring and admissible-vector sampling.** Quadratic score power increases
   candidate-score separation. Capped one-hot sampling plus sparse and dense
   samples adds coverage mainly when the admissible space is larger.
4. **Candidate sources and \(T\)-count schedule.** Retain both sources, use a
   finite TODD/FastTODD search in the high-\(T\)-count regime, and reserve the
   unrestricted search for the terminal regime.
5. **Observed outcome.** The combined program improved the saved trajectory
   from 1197 to 1195. Do not attribute the improvement to one setting alone.

Short descriptive inline comments may identify a source or an unchanged
setting, but they must not restore the removed diagnostic narrative.

## `FINAL_SEEDS` evidence

`FINAL_SEEDS = [29, 87, 43]` reruns the selected policy under three stochastic
seeds after the one-seed numerical policy search. The evaluator retains the
best trajectory across these runs, so this is a best-of-three final search as
well as a check against dependence on one tuning seed; it is not an independent
algebraic correctness validation.

The complete evolved program improved the saved trajectory from \(T\)-count
1197 to 1195. The stored diagnostics do not identify the final per-seed results
or isolate the contribution of reseeding from the other simultaneous changes.
The appendix may therefore say the final rerun served its intended purpose,
but it must not claim that `FINAL_SEEDS` alone caused or independently proved
the 1195 result.

## Material to remove

Remove detailed generation chronology, reuse counts, raw path diagnostics,
H/T ratios, `apz`, z-saturation descriptions, timeout-salvage details, and
internal labels such as tail or mid-band. Remove the incorrect suggestion that
one-hot 8 exhausts every admissible space of dimension four. Do not include
prompt excerpts, model narration, insight identifiers, or precise LLM
reasoning.

## Scope and verification

Revise terminology in the whole section, while preserving the two programs,
their parameter values, captions, labels, and explanatory order. Verify that:

- all reader-facing terms match the manuscript;
- the evolved listing still uses trajectory `fc0b0d69`, margin 50, PSO 10/80,
  beam width 2, score power 2, one-hot 8, sparse 4/weight 3, retention reserve
  2, and the 30937/unrestricted TODD schedule;
- no removed diagnostic language remains;
- listing environments, braces, and cross-references are structurally valid;
- the containing TeX document compiles if a local TeX engine is available;
  and
- the final diff contains no unrelated edits.
