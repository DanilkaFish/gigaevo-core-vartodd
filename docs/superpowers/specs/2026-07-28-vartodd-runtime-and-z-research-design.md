# VarTODD Runtime and Observed Z-Research Design

## Goal

Make evolution favor productive use of the available runtime without treating
soft-timeout salvage as either a reward or a validation failure. Strongly
discourage loaded-path children that fail to improve, while retaining a narrow
exception for children that test a genuinely cheaper observed TODD search.

Keep mutation guidance neutral about optimizer and policy mechanisms. The two
explicit priorities are:

1. use the runtime budget productively while avoiding routine dependence on
   timeout salvage;
2. balance the recall benefit of broader TODD research against its runtime
   cost using observed evidence.

## Metrics and Fitness Shaping

- Set the non-primary `runtime` metric to `higher_is_better: true`.
- Keep `is_valid` as the validity gate.
- Do not add any fitness penalty or reward for `timeout_salvaged=1`.
- A loaded-path child that does not beat the applicable loaded path rank gets a
  `+7` fitness penalty by default.
- If that non-improving child reached a strictly smaller maximum observed TODD
  z-research value than its loaded parent, replace the `+7` penalty with `+1`.
- Configured `limit_bucket` does not qualify a child for this exception.
- If either observed value is unavailable, apply the default `+7` penalty.
- Remove the obsolete low-cap reward and associated prompt claim.

Fresh `path_name="init"` programs retain the existing rank-improvement
semantics; the no-improvement fitness penalty applies to loaded-path searches.

## Path Names

New saved paths use:

```text
f<final>_i<loaded>_<hash>_lim<cap>_z<max_observed_todd_z>
```

`lim` is the maximum active configured TODD `limit_bucket` in the policy that
produced the saved best path. `lim-1` means an active TODD band was
unrestricted.

`z` is the maximum TODD z-research actually observed along the saved best path.
It excludes TOHPEprefix research. The value is computed only from the DAOs on
the selected best path, never from an ancestor DAO outside that path. If the
value cannot be determined, use `zunknown`.

Path parsing remains backward compatible with legacy names that end after
`_lim<cap>`. A legacy parent cannot establish the reduced-observed-research
exception and therefore receives the default non-improvement penalty.

## Prompt Guidance

Update the task description and mutation-regime text so that:

- remaining runtime is a resource for additional justified evaluations,
  restarts, groups, paths, or optimizer stages;
- repeated `timeout_salvaged=1` indicates imperfect budget fitting but is not
  itself good or bad and has no direct fitness effect;
- larger TODD caps can expose actions inaccessible to smaller caps but make
  evaluations more expensive;
- smaller caps can permit more parameter evaluations but can reduce recall;
- configured caps and observed z-research are distinct evidence;
- no fixed optimizer, cap direction, schedule, or source mixture is prescribed;
- claims about saturation or wasted research must be supported by observed
  generation, retention, improvement, and runtime evidence.

Remove the false statement that low finite limits receive a fitness reward.

## Tests

Add focused regression tests for:

1. runtime being marked higher-is-better;
2. a non-improving loaded child receiving `+7`;
3. a non-improving child receiving only `+1` when its observed TODD z maximum
   is strictly smaller than the parent path's `_z` value;
4. equal, larger, missing, and legacy parent z values receiving `+7`;
5. timeout salvage not changing fitness;
6. new path names containing both `_lim` and `_z`;
7. observed z being derived from TODD on the selected best path only;
8. legacy path names remaining loadable and parseable;
9. prompts distinguishing cap from observed research without low-cap reward
   language.
