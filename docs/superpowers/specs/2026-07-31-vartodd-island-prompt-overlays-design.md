# VarTODD Island Prompt Overlays Design

## Goal

Make `problems/vartodd_evo_gf_islands` the source of truth for each island's
mutation guidance. Keep the algorithm YAML responsible for routing and
probabilities, not prompt prose.

This change applies only to the dedicated island pipeline. The legacy
`vartodd_evo_gf` problem and its single-island configuration remain unchanged.

## Prompt Ownership

The common prompt remains in the island problem:

- `task_description.txt` defines shared VarTODD mechanics and API.
- `prompts/mutation/system.txt` and `prompts/mutation/user.txt` define the
  common mutation contract.
- `prompts/insights/*` and `prompts/lineage/*` remain common analytical
  prompts.

The problem gains one mutation overlay per route:

- `prompts/islands/ab_initio.txt`
- `prompts/islands/mid_margin.txt`
- `prompts/islands/near_end.txt`

Each overlay contains only the binding requirements and useful guidance
specific to that island. Shared API, score semantics, runtime interpretation,
and general evidence rules must not be copied into all three files.

## Prompt Composition

For a routed mutation, the final user prompt is assembled in this order:

1. mutation assignment and parent roles;
2. common mutation user instructions and filtered parent evidence;
3. route-specific external context, including shared path cards for refinement
   routes and no path cards for ab-initio;
4. the problem-owned overlay matching `route.regime_id`.

The route-context provider resolves overlays by the safe convention
`prompts/islands/<regime_id>.txt`. It reads the selected overlay from the
configured problem directory. Runtime GF overlays already link the complete
`prompts` directory, so matrix-specific runs receive these files without
copying prompt text.

The overlay should be read when a prompt is built rather than permanently
cached. Editing a problem-owned overlay therefore affects subsequent mutations
after the normal process/config reload behavior without requiring duplicated
YAML changes.

## Configuration

`config/algorithm/vartodd_diverse_gf_islands.yaml` retains:

- route and destination island IDs;
- route sampling probabilities;
- context profiles;
- parent mixing and bootstrap settings.

The multiline `guidance` bodies are removed from YAML. Route configuration
allows guidance to be omitted when a route-context provider supplies a
problem-owned overlay. Existing routes that use inline guidance and do not use
this provider remain supported.

The generic mutation path resolves guidance as follows:

1. ask the route-context provider for a route overlay when the provider
   implements that capability;
2. otherwise use the route's existing inline guidance;
3. fail clearly if neither source supplies nonblank guidance.

It is an error for both sources to define competing binding guidance for the
same route. This prevents silent concatenation and ambiguous instructions.

## Island Overlay Content

The initial overlays preserve the current three regime contracts while moving
them out of YAML:

- `ab_initio`: require `Evaluator(path_name="init")`, construct a reusable
  path, use standalone TOHPE as the main early source, and keep TODD disabled
  or light until lower-rank evidence supports a schedule.
- `mid_margin`: require an exact selectable shared path and margin `30..100`;
  branch before the inherited tail so policy, source balance, restart
  placement, or trajectory can change materially.
- `near_end`: require an exact selectable shared path and margin `5..30`;
  acknowledge small-margin plateau stagnation and the possible need for
  plateau-resilient optimization, restarts, or very broad beam search; explain
  terminal TODD action starvation and z-bucket breadth without declaring one
  fixed parameter choice universally correct.

These files are mutation overlays only. Insight extraction and lineage
analysis continue to use their common prompts and the child's recorded island
and execution evidence, avoiding three duplicated analytical prompt families.

## Validation and Errors

For a provider-owned overlay:

- the regime ID must already satisfy the route identifier validation;
- the expected file must exist;
- the file must be regular/readable and nonblank;
- filesystem or validation errors identify the regime and expected path;
- unknown regimes do not silently fall back to another island's overlay.

Inline-guidance routes preserve their current validation and behavior.

## Verification

Tests will verify:

- all three problem-owned overlay files exist and are nonblank;
- each overlay contains its current regime-specific contract;
- the island YAML contains route structure but no embedded regime prose;
- prompt assembly appends the selected overlay after assignment, parents, and
  external path context;
- ab-initio still receives no shared path cards;
- a missing or blank overlay raises a clear error;
- inline-guidance routes without an overlay provider remain compatible;
- competing inline and provider guidance is rejected;
- legacy single-island VarTODD configuration remains unchanged.

## Out of Scope

This change does not alter route probabilities, parent selection, archive
admission, bootstrap behavior, path sampling, insight semantics, lineage
semantics, initial programs, or the legacy GF pipeline.
