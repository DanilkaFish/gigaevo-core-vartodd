# Expensive-TODD Initial Program Portfolio Design

## Objective

Add an eight-program `initial_programs_expensive` pool for large matrices where
one TODD-backed policy evaluation is costly. The pool must be a useful starting
population without changing mutation, insight, lineage, task-description, or
island prompts.

The launcher-facing selector is `initial_programs=expensive`. It must work in
both `run_gf.py` and `run_gf_islands.py` through the existing shared initial
program pool mapping.

## Evidence from the Current Run

The current `vartodd_gf.csv` contains six initial programs for a matrix with
initial rank 1011. Every initial program ran for approximately 6,000 seconds
and completed only 350--438 reported evaluations before timeout salvage. The
best initial results came from staged programs:

- heavy TODD after a path restart reached rank 573;
- PSO followed by restarted PatternSearch reached rank 575;
- broad joint or repeated refinement searches reached ranks 583--623.

This favors short, staged, low-dimensional searches over large populations
that optimize every policy field at once.

## Portfolio

The pool contains eight standalone Python programs. Six use one search seed
throughout. Two use one seed for the initial descent and switch to two seeds
only after a successful `set_up_new_init`.

| Role | Target reported evaluations | Search shape |
|---|---:|---|
| TOHPE stochastic scout | 100 | One seed; randomized or low-discrepancy sampling over cheap action generation and scores |
| TOHPE score refinement | 120 | One seed; cheap scout followed by restarted score-only local refinement |
| Low-beam restart search | 140 | One seed; low sampling cost and two short descent stages |
| Grouped DE search | 160 | One seed; optimize action generation and scoring in separate stages |
| Restarted terminal TODD | 180 | One seed; TOHPE-first descent, then substantial finite TODD in the tail |
| High-z narrow tail | 200 | One seed; high bucket coverage after restart with constrained retention and beam width |
| Multi-seed tail validation | 180 | 60 one-seed scout evaluations, then 60 two-seed tail objective calls |
| Multi-seed exhaustive tail | 220 | 80 one-seed scout evaluations, then 70 two-seed tail objective calls |

The values are scheduling targets. Programs must not add a hard guard around
`BaseEvaluator.total_eval`. Pymoo generation boundaries and normal framework
overhead may cause small differences between the declared target and the final
reported value. The existing soft timeout remains the compute safety bound.

## Policy Design

All eight programs use general matrix and target constants from `helper.py`.
Rank switches and restart margins derive from `INITIAL_RANK -
TARGET_FINAL_RANK`; no seed contains GF16- or matrix-specific rank constants.

Initial descent policies emphasize inexpensive action generation:

- TOHPE is the primary source;
- TODD is disabled or tightly bounded;
- beam width and retained pools remain small;
- stochastic selection and independent search seeds provide trajectory
  diversity.

Tail policies are installed only after a successful path restart. The two
TODD-focused single-seed programs and two multi-seed programs cover distinct
tail tradeoffs:

- moderate finite z coverage with tunable scoring;
- high z coverage with narrow source retention;
- robustness validation across two seeds;
- wider or more exhaustive action generation across two seeds.

Bucket ranges scale from the current matrix state or the available bucket
space rather than relying on fixed ranges chosen for GF16. High coverage is a
tradeoff: fewer policies are evaluated, but each policy supplies tree search
with more rare TODD actions.

## Optimization Design

Each program is self-contained so mutation agents can inspect and modify the
entire strategy. No shared portfolio engine is introduced.

The programs deliberately mix randomized sampling, PSO, DE, and
PatternSearch. Population sizes remain small and are compatible with the
target evaluation schedule. Parameter groups separate scores, cheap action
generation, and tail action generation. Restarted stages activate only the
groups relevant to that stage, avoiding broad optimization on a plateaued
integer-rank objective.

The two multi-seed programs use an evaluator stage flag or equivalent explicit
seed selection. Their initial stage calls `run` with one seed. Only after
`set_up_new_init` succeeds does their objective call `run` with two seeds.

## Launcher Integration

Extend `INITIAL_PROGRAM_POOLS` with:

```python
"expensive": "initial_programs_expensive"
```

Launcher validation and help text must advertise `default|best|expensive`.
Overlay and cache naming already include a non-default pool selector, so the
new pool remains isolated from default and best-seed runs without another
namespace mechanism.

## Failure Handling

If a requested restart cannot extract a branch, the program returns the best
completed path from its earlier stage. Optimizer failure to produce `result.X`
falls back to the evaluator's current active parameters, following existing
seed conventions. Soft-timeout salvage remains owned by the execution
framework.

No program may catch or suppress timeout exceptions, enforce a hard
`total_eval` ceiling, or change shared prompt behavior.

## Verification

Tests must establish that:

1. `initial_programs=expensive` links the new directory in regular and island
   overlays and uses an isolated cache suffix.
2. The directory contains exactly eight executable seed files.
3. Every seed uses universal rank constants and declares its intended
   evaluation schedule within 100--220 reported evaluations.
4. Exactly two seeds use multiple search seeds, and both activate the second
   seed only after `set_up_new_init`.
5. No seed implements a hard check against `total_eval`.
6. Existing default and best pool selection remains unchanged.

Targeted launcher and initial-program contract tests must pass. Import and
compile checks must cover all eight new programs without running the expensive
search itself.
