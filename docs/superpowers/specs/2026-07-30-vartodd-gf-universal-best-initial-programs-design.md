# Universal Best VarTODD GF Initial Programs

## Goal

Create `problems/vartodd_evo_gf/initial_programs_best/` containing eight
high-quality, logically diverse ab-initio programs selected from
`vartodd_gf16.csv`.

The programs adopt the source programs' policy and optimization archetypes but
must work with any matrix and target rank configured through `run_gf.py`.
The directory is a curated library. This change does not replace the active
`initial_programs/` pool and does not change `run_gf.py`.

## Selection

Archive state is not a selection criterion. A valid discarded program is
eligible when its observed result and logic justify inclusion.

The selected source programs are:

| File | Source program | GF16 rank | Preserved archetype |
|---|---|---:|---|
| `de_light_terminal_todd.py` | `a899151c-4745-4b00-8be9-e0ca0618391a` | 387 | Single DE search with light scheduled terminal TODD |
| `pso_restart_terminal_todd.py` | `e8f52447-6132-4ecc-91f3-20630378c82a` | 387 | Multi-restart PSO with moderate terminal TODD |
| `pso_tohpe_only.py` | `67b2f195-25c2-4f82-b50a-ab70864fac20` | 387 | Multi-restart PSO with TODD disabled |
| `pso_wide_terminal_beam.py` | `e350705c-c6f2-479f-b153-922fad712cf8` | 387 | Rank-scheduled beam widening from 5 to 8 |
| `pso_chunked_light_todd.py` | `4e37489e-8572-46ac-8303-e426d6b9ce8e` | 393 | Beam-3 chunked PSO with light terminal TODD |
| `pso_cma_grouped_tail.py` | `f7aff6f1-3e02-4bf4-8f6c-f2aaea42b200` | 397 | Deterministic beam 4 and group-wise PSO/CMA-ES/PSO |
| `pso_three_band_restart.py` | `86ccecc3-e4f1-4fcc-b0ac-9cd4488069d0` | 399 | Beam-2 three-band policy with a mid-run restart |
| `pso_pattern_heavy_restart.py` | `0d61fdac-8502-4e32-a416-4180b6a44691` | 399 | Grouped PSO/PatternSearch with heavy TODD after restart |

This set covers beamwidths 2, 3, 4, 5, and 8; disabled, light, moderate, and
heavy-after-restart TODD; one-, two-, and three-band schedules; and several
distinct optimization and restart procedures.

## Universal Rank Model

Every program imports `INITIAL_RANK` and `TARGET_FINAL_RANK` from `helper`.
It defines:

```python
RANK_SPAN = max(1, INITIAL_RANK - TARGET_FINAL_RANK)


def rank_from_target(fraction: float) -> int:
    offset = max(1, round(RANK_SPAN * fraction))
    return min(INITIAL_RANK, TARGET_FINAL_RANK + offset)
```

Rank-band fractions preserve the corresponding source program's approximate
position within the GF16 rank span. Programs with one terminal transition use
fractions near `0.08`, `0.10`, or `0.12`. The three-band program preserves an
early transition near `0.66` and a terminal transition near `0.13`.

Restart margins use the same principle:

```python
def rank_margin(fraction: float) -> int:
    return max(1, round(RANK_SPAN * fraction))
```

Optimizer evaluation counts, population sizes, score ranges, source pools,
beam behavior, and TODD breadth remain explicit tuning constants. They are
part of each archetype and are not scaled speculatively by matrix degree.
`max_depth` is derived from the search span so it does not assume GF16's
initial rank.

## Adaptation Rules

- Every evaluator starts from `path_name="init"` and never loads a saved path.
- Fixed source ranks such as `567`, `394`, `402`, and `504` are replaced by
  the universal rank model.
- Each `map_par` call has an explicit parameter group.
- Parameter-group optimization remains only in the source archetypes that used
  it.
- The programs use only optimizer implementations from `pymoo`.
- TOHPEprefix is absent.
- Source logic may be reformatted for clarity, but its score shape, source
  regime, beam strategy, optimizer sequence, and restart structure remain
  recognizable.
- No program claims that its GF16 result will reproduce on another matrix.

## Auxiliary Description

Each file ends with a module-level `AUX_DESCRIPTION` string. It records:

- the full GF16 source program ID;
- observed source rank, fitness, runtime, and timeout-salvage status;
- the program's role in the diverse pool;
- the preserved policy and optimization structure;
- the rank-relative changes made for universality;
- an evidence-based limitation or tuning note.

The description is passive metadata and does not affect `entrypoint()`.

## Verification

Add focused tests for `initial_programs_best/` that require:

- exactly the eight expected Python files;
- syntactically valid modules with an `entrypoint`;
- `path_name="init"` and no saved-path names;
- imports of `INITIAL_RANK` and `TARGET_FINAL_RANK`;
- explicit `group=` on every `map_par` call;
- no TOHPEprefix or non-`pymoo` optimizer library;
- an end-of-file `AUX_DESCRIPTION` containing source provenance;
- source-specific structural markers for each archetype.

Run the focused tests and compile all eight files. Existing dirty worktree
changes outside this directory and its focused test remain untouched.
