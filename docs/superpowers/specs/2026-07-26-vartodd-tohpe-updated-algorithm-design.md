# Updated GF16 TOHPE Search-Guidance Design

## Goal

Create a GF16 algorithm variant for `vartodd_evo_tohpe_updated` that teaches
the mutation agent to use the new TOHPEprefix source and to avoid unnecessarily
large standalone-TOHPE sample budgets.

## Configuration

Add `config/algorithm/vartodd_diverse_gf16_tohpe_updated.yaml`. It inherits the
existing `vartodd_diverse_gf16` configuration and replaces only
`mutation_operator.mutation_regime_guidance`.

The archive behavior space, rank-aware retention, parent selector, selector
pressure, source-path integration, and regime probabilities remain unchanged.
The steady experiment selects the new algorithm variant.

## Mutation Guidance

Both mutation regimes use concise source guidance:

- Keep standalone TOHPE sampling modest by default.
- Prefer one-hot and low-weight sparse samples to large dense budgets.
- Increase standalone TOHPE sampling only when execution evidence shows that
  it continues to produce useful accepted actions.
- Use TOHPEprefix as the intermediate search source: it searches a reduced
  space and is generally cheaper but less expressive than full TODD.
- Reserve full TODD for difficult regions where the reduced sources stop
  finding useful actions.
- Preserve the distinction between configured and effective unique samples;
  do not increase budgets after the coefficient space is saturated.

The existing saved-path rules remain in place. The guidance does not impose a
hard sample cap or numerical cost comparison.

## Validation

Add a Hydra composition contract covering:

- the new algorithm variant composes;
- it retains `WeightedEliteSelector(lambda_=5.0)`;
- its mutation guidance contains the standalone-TOHPE and TOHPEprefix policy;
- the steady experiment resolves to the new algorithm while preserving six
  concurrent DAGs, six prefetched tasks, and two parents.

No runtime validation, metric, fitness-shaping, or source implementation
changes are included.
