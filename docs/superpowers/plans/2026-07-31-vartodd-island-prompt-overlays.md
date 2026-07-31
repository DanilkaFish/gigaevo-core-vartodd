# VarTODD Island Prompt Overlays Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move each VarTODD island mutation regime prompt from algorithm YAML into a problem-owned overlay file selected by `route.regime_id`.

**Architecture:** Extend routed mutation guidance so a route-context provider may supply the binding guidance while existing inline-guidance routes remain compatible. The VarTODD island provider reads `prompts/islands/<regime_id>.txt` on each prompt build, and the island YAML retains only routing structure and probabilities.

**Tech Stack:** Python 3.12, Pydantic, Hydra/OmegaConf, pytest, plain-text prompt assets.

## Global Constraints

- Change only the dedicated `vartodd_evo_gf_islands` prompt behavior; preserve the legacy `vartodd_evo_gf` pipeline.
- Keep insight and lineage prompts common; add overlays only for mutation.
- Ab-initio must receive no shared path cards.
- Provider-owned overlay files are resolved as `prompts/islands/<regime_id>.txt`.
- Missing, unreadable, blank, or unsafe overlay names must fail clearly.
- Exactly one binding guidance source is allowed: provider overlay or inline route guidance.
- Read provider overlays at prompt-build time rather than caching their contents.
- Do not change route probabilities, archive behavior, parent selection, path sampling, or initial programs.

---

### Task 1: Support Provider-Owned Route Guidance

**Files:**
- Modify: `gigaevo/evolution/strategies/base.py`
- Modify: `gigaevo/evolution/strategies/models.py`
- Modify: `gigaevo/evolution/strategies/route_context.py`
- Modify: `gigaevo/llm/agents/mutation.py`
- Test: `tests/evolution/test_multi_island_extended.py`
- Test: `tests/llm/test_mutation_agent.py`

**Interfaces:**
- Consumes: existing `MutationRoute`, `MutationRouteConfig`, and `MutationRouteContextProvider`.
- Produces: `MutationRoute.guidance: str | None`, `MutationRouteConfig.guidance: str | None`, and optional provider method `build_route_guidance(route: MutationRoute) -> str | None`.
- Produces: `MutationAgent._resolve_route_guidance(route: MutationRoute) -> str`, enforcing exactly one nonblank guidance source.

- [ ] **Step 1: Write failing route-model tests**

In `tests/evolution/test_multi_island_extended.py`, add a test proving that a
route intended for provider-owned guidance may omit `guidance`, while the
existing whitespace rejection remains:

```python
def test_route_config_allows_provider_owned_guidance():
    config = MutationRouteConfig(
        regime_id="builder",
        island_id="island_0",
        probability=1.0,
    )

    assert config.guidance is None
    assert config.context_profile == "default"
```

Keep `"guidance", "   "` in
`test_route_config_rejects_invalid_values`; an explicitly supplied blank string
is invalid even though omission is valid.

- [ ] **Step 2: Write failing prompt-resolution tests**

In `tests/llm/test_mutation_agent.py`, update
`test_route_provider_orders_and_labels_prompt_context` so the mock provider
explicitly returns no overlay:

```python
provider.build_route_guidance.return_value = None
```

Then add these tests beside the other route-provider tests:

```python
def test_route_provider_supplies_problem_owned_guidance():
    provider = MagicMock()
    provider.build_assignment.return_value = "## Mutation Assignment\nnear_end"
    provider.filter_parent_context.side_effect = (
        lambda route, parent, role, context: context
    )
    provider.build_external_context.return_value = (
        "## Selectable Shared Paths\npath cards"
    )
    provider.build_route_guidance.return_value = (
        "## Required Island Regime: Near-end refinement\nuse margin 5..30"
    )
    agent = _make_agent(route_context_provider=provider)
    parent = _make_program()
    route = MutationRoute(
        regime_id="near_end",
        island_id="near_end",
        guidance=None,
        context_profile="path_refinement",
    )
    state = _make_state(
        parents=[parent],
        explicit_route=route,
        parent_roles=("target_island",),
        explicit_regime_guidance=None,
    )

    prompt = agent.build_prompt(state)["user_prompt"]

    assert "use margin 5..30" in prompt
    assert prompt.index("Selectable Shared Paths") < prompt.index(
        "Required Island Regime"
    )
```

```python
def test_inline_route_guidance_remains_supported_without_provider():
    agent = _make_agent(route_context_provider=None)
    parent = _make_program()
    route = MutationRoute(
        regime_id="builder",
        island_id="builder",
        guidance="Build from scratch.",
    )
    state = _make_state(
        parents=[parent],
        explicit_route=route,
        explicit_regime_guidance=route.guidance,
    )

    assert "Build from scratch." in agent.build_prompt(state)["user_prompt"]
```

```python
def test_route_rejects_competing_inline_and_provider_guidance():
    provider = MagicMock()
    provider.build_assignment.return_value = "assignment"
    provider.filter_parent_context.side_effect = (
        lambda route, parent, role, context: context
    )
    provider.build_external_context.return_value = ""
    provider.build_route_guidance.return_value = "provider guidance"
    agent = _make_agent(route_context_provider=provider)
    route = MutationRoute(
        regime_id="near_end",
        island_id="near_end",
        guidance="inline guidance",
    )
    state = _make_state(
        parents=[_make_program()],
        explicit_route=route,
        parent_roles=("target_island",),
        explicit_regime_guidance=route.guidance,
    )

    with pytest.raises(ValueError, match="both inline and provider"):
        agent.build_prompt(state)
```

```python
def test_route_requires_inline_or_provider_guidance():
    provider = MagicMock()
    provider.build_assignment.return_value = "assignment"
    provider.filter_parent_context.side_effect = (
        lambda route, parent, role, context: context
    )
    provider.build_external_context.return_value = ""
    provider.build_route_guidance.return_value = None
    agent = _make_agent(route_context_provider=provider)
    route = MutationRoute(
        regime_id="near_end",
        island_id="near_end",
        guidance=None,
    )
    state = _make_state(
        parents=[_make_program()],
        explicit_route=route,
        parent_roles=("target_island",),
        explicit_regime_guidance=None,
    )

    with pytest.raises(ValueError, match="no binding guidance"):
        agent.build_prompt(state)
```

Add `import pytest` to this test module if it is not already imported.

- [ ] **Step 3: Run the focused tests and confirm failure**

Run:

```bash
direnv exec . pytest \
  tests/evolution/test_multi_island_extended.py::TestMultiIslandConstruction::test_route_config_allows_provider_owned_guidance \
  tests/llm/test_mutation_agent.py::TestBuildPrompt::test_route_provider_supplies_problem_owned_guidance \
  tests/llm/test_mutation_agent.py::TestBuildPrompt::test_inline_route_guidance_remains_supported_without_provider \
  tests/llm/test_mutation_agent.py::TestBuildPrompt::test_route_rejects_competing_inline_and_provider_guidance \
  tests/llm/test_mutation_agent.py::TestBuildPrompt::test_route_requires_inline_or_provider_guidance -q
```

Expected: FAIL because `guidance` is required and the provider protocol/agent
does not yet resolve provider-owned guidance.

- [ ] **Step 4: Make inline guidance optional in route models**

In `gigaevo/evolution/strategies/base.py`, change the route field to:

```python
guidance: str | None = None
```

Preserve the existing field order so any positional callers remain compatible:

```python
@dataclass(frozen=True, slots=True)
class MutationRoute:
    regime_id: str
    island_id: str
    guidance: str | None = None
    context_profile: str = "default"
```

In `gigaevo/evolution/strategies/models.py`, replace the required field with:

```python
guidance: str | None = None
```

Split validation so `context_profile` remains required and nonblank, while
guidance is stripped only when supplied:

```python
@field_validator("context_profile")
@classmethod
def context_profile_must_not_be_whitespace(cls, value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("value must not be blank")
    return stripped

@field_validator("guidance")
@classmethod
def guidance_must_not_be_whitespace(
    cls, value: str | None
) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise ValueError("value must not be blank")
    return stripped
```

- [ ] **Step 5: Add the provider guidance interface**

In `gigaevo/evolution/strategies/route_context.py`, document the optional
capability on the protocol:

```python
def build_route_guidance(self, route: MutationRoute) -> str | None:
    """Return problem-owned binding guidance, or None for inline guidance."""
    ...
```

The mutation agent will still use `getattr`, so older runtime providers without
this method remain compatible despite the stronger static protocol.

- [ ] **Step 6: Implement exact-one-source guidance resolution**

In `gigaevo/llm/agents/mutation.py`, add:

```python
def _resolve_route_guidance(self, route: MutationRoute) -> str:
    inline = (route.guidance or "").strip()
    provider_text = ""
    builder = getattr(
        self.route_context_provider,
        "build_route_guidance",
        None,
    )
    if callable(builder):
        value = builder(route)
        if value is not None and not isinstance(value, str):
            raise TypeError(
                "route-context provider guidance must be str or None"
            )
        provider_text = (value or "").strip()

    if inline and provider_text:
        raise ValueError(
            f"route {route.regime_id!r} defines both inline and provider "
            "binding guidance"
        )
    guidance = provider_text or inline
    if not guidance:
        raise ValueError(
            f"route {route.regime_id!r} has no binding guidance"
        )
    return guidance
```

In the explicit-route branch of `build_user_prompt`, replace
`explicit_route.guidance.strip()` with:

```python
guidance = self._resolve_route_guidance(explicit_route)
```

Also handle an explicit route without a context provider before the legacy
sampled-regime branch:

```python
if explicit_route is not None:
    guidance = self._resolve_route_guidance(explicit_route)
    if state is not None:
        state["selected_mutation_regime"] = explicit_route.regime_id
    return f"{user_prompt}\n\n{guidance}"
```

Place this after the provider-aware explicit-route block. This preserves inline
route guidance without inventing assignment/path context when no provider
exists.

- [ ] **Step 7: Run the route guidance tests**

Run:

```bash
direnv exec . pytest \
  tests/evolution/test_multi_island_extended.py \
  tests/llm/test_mutation_agent.py \
  tests/evolution/test_mutation_operator.py -q
```

Expected: PASS. Existing inline route tests must remain green.

- [ ] **Step 8: Commit Task 1**

```bash
git add \
  gigaevo/evolution/strategies/base.py \
  gigaevo/evolution/strategies/models.py \
  gigaevo/evolution/strategies/route_context.py \
  gigaevo/llm/agents/mutation.py \
  tests/evolution/test_multi_island_extended.py \
  tests/llm/test_mutation_agent.py
git commit -m "feat: support provider-owned mutation guidance"
```

---

### Task 2: Move VarTODD Regime Contracts into Problem-Owned Overlays

**Files:**
- Modify: `custom/vartodd_islands_context.py`
- Create: `problems/vartodd_evo_gf_islands/prompts/islands/ab_initio.txt`
- Create: `problems/vartodd_evo_gf_islands/prompts/islands/mid_margin.txt`
- Create: `problems/vartodd_evo_gf_islands/prompts/islands/near_end.txt`
- Modify: `config/algorithm/vartodd_diverse_gf_islands.yaml`
- Test: `tests/custom/test_vartodd_islands_context.py`
- Test: `tests/problems/test_vartodd_gf_islands_prompts.py`
- Test: `tests/test_config.py`
- Test: `tests/test_tools/test_run_gf_islands.py`
- Test: `tests/integration/test_vartodd_gf_islands_flow.py`

**Interfaces:**
- Consumes: `MutationRoute.regime_id` and the provider's configured
  `problem_dir`.
- Produces:
  `VartoddIslandsRouteContextProvider.build_route_guidance(route: MutationRoute) -> str`.
- File contract: `<problem_dir>/prompts/islands/<regime_id>.txt`.
- Produces: one nonblank binding prompt overlay for each configured
  `regime_id`.
- Preserves: route probabilities `0.40`, `0.35`, `0.25`; context profiles
  `ab_initio`, `path_refinement`, `path_refinement`.

- [ ] **Step 1: Write failing provider overlay tests**

In `tests/custom/test_vartodd_islands_context.py`, add:

```python
import pytest
```

Then add a helper and tests:

```python
def _write_overlay(problem_dir, regime_id: str, text: str) -> None:
    directory = problem_dir / "prompts" / "islands"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{regime_id}.txt").write_text(text, encoding="utf-8")
```

```python
def test_provider_reads_route_overlay_on_each_prompt_build(tmp_path) -> None:
    _write_overlay(tmp_path, "near_end", "first near-end guidance")
    provider = VartoddIslandsRouteContextProvider(problem_dir=tmp_path)
    route = MutationRoute(
        regime_id="near_end",
        island_id="near_end",
        guidance=None,
    )

    assert provider.build_route_guidance(route) == "first near-end guidance"

    _write_overlay(tmp_path, "near_end", "updated near-end guidance")
    assert provider.build_route_guidance(route) == "updated near-end guidance"
```

```python
@pytest.mark.parametrize(
    "regime_id,text,error,match",
    [
        ("missing", None, FileNotFoundError, "missing"),
        ("blank", "   \n", ValueError, "blank"),
        ("../escape", "unused", ValueError, "unsafe"),
    ],
)
def test_provider_rejects_invalid_route_overlays(
    tmp_path,
    regime_id,
    text,
    error,
    match,
) -> None:
    if text is not None and regime_id != "../escape":
        _write_overlay(tmp_path, regime_id, text)
    provider = VartoddIslandsRouteContextProvider(problem_dir=tmp_path)
    route = MutationRoute(
        regime_id=regime_id,
        island_id="test",
        guidance=None,
    )

    with pytest.raises(error, match=match):
        provider.build_route_guidance(route)
```

- [ ] **Step 2: Run the provider tests and confirm failure**

Run:

```bash
direnv exec . pytest \
  tests/custom/test_vartodd_islands_context.py::test_provider_reads_route_overlay_on_each_prompt_build \
  tests/custom/test_vartodd_islands_context.py::test_provider_rejects_invalid_route_overlays -q
```

Expected: FAIL because `build_route_guidance` does not exist.

- [ ] **Step 3: Implement safe, uncached overlay loading**

In `custom/vartodd_islands_context.py`, add:

```python
_REGIME_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
```

Add this method to `VartoddIslandsRouteContextProvider`:

```python
def build_route_guidance(self, route: MutationRoute) -> str:
    regime_id = route.regime_id
    if _REGIME_ID_RE.fullmatch(regime_id) is None:
        raise ValueError(
            f"unsafe mutation regime id for prompt overlay: {regime_id!r}"
        )
    path = (
        self.problem_dir
        / "prompts"
        / "islands"
        / f"{regime_id}.txt"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"missing prompt overlay for route {regime_id!r}: {path}"
        )
    try:
        guidance = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise OSError(
            f"cannot read prompt overlay for route {regime_id!r}: {path}"
        ) from exc
    if not guidance:
        raise ValueError(
            f"blank prompt overlay for route {regime_id!r}: {path}"
        )
    return guidance
```

Do not add an overlay-content cache.

- [ ] **Step 4: Run the provider overlay tests**

Run:

```bash
direnv exec . pytest \
  tests/custom/test_vartodd_islands_context.py::test_provider_reads_route_overlay_on_each_prompt_build \
  tests/custom/test_vartodd_islands_context.py::test_provider_rejects_invalid_route_overlays -q
```

Expected: PASS.

- [ ] **Step 5: Write failing prompt-ownership tests**

In `tests/problems/test_vartodd_gf_islands_prompts.py`, add:

```python
def test_each_island_owns_a_nonblank_mutation_overlay() -> None:
    overlays = PROBLEM_DIR / "prompts" / "islands"
    expected = {
        "ab_initio.txt": ('path_name="init"', "standalone TOHPE"),
        "mid_margin.txt": ("Selectable Shared Paths", "margin 30..100"),
        "near_end.txt": ("Selectable Shared Paths", "margin 5..30"),
    }

    assert {path.name for path in overlays.glob("*.txt")} == set(expected)
    for filename, required in expected.items():
        text = (overlays / filename).read_text(encoding="utf-8").strip()
        assert text
        for phrase in required:
            assert phrase in text
```

Add neutral mechanism assertions for the near-end overlay:

```python
def test_near_end_overlay_explains_plateaus_and_todd_breadth() -> None:
    text = (
        PROBLEM_DIR / "prompts" / "islands" / "near_end.txt"
    ).read_text(encoding="utf-8")

    assert "plateau" in text.lower()
    assert "max_buckets" in text
    assert "limit_bucket" in text
    assert "action starvation" in text.lower()
    assert "universally" not in text.lower()
```

- [ ] **Step 6: Update the configuration contract test before YAML**

In `tests/test_config.py`, replace the block that concatenates
`route.guidance` with:

```python
assert all("guidance" not in route for route in routes)
```

Keep the route tuple assertion unchanged so IDs, islands, probabilities, and
context profiles remain protected.

- [ ] **Step 7: Protect runtime-overlay propagation**

In
`test_launcher_creates_isolated_overlay_store_and_redis_prefix` in
`tests/test_tools/test_run_gf_islands.py`, add:

```python
assert (overlay / "prompts").is_symlink()
assert (
    overlay / "prompts" / "islands" / "ab_initio.txt"
).read_text(encoding="utf-8").strip()
assert (
    overlay / "prompts" / "islands" / "mid_margin.txt"
).read_text(encoding="utf-8").strip()
assert (
    overlay / "prompts" / "islands" / "near_end.txt"
).read_text(encoding="utf-8").strip()
```

- [ ] **Step 8: Make the integration flow consume provider guidance**

In `tests/integration/test_vartodd_gf_islands_flow.py`, change
`_CaptureRouteMutator.mutate_with_route` so its final prompt block uses:

```python
self.provider.build_route_guidance(route)
```

instead of `route.guidance`. In the integration test's
`MutationRouteConfig`, remove the inline `guidance=...` argument:

```python
MutationRouteConfig(
    regime_id="mid_margin",
    island_id="mid_margin",
    probability=1.0,
    context_profile="path_refinement",
)
```

This ensures the end-to-end test exercises the problem-owned overlay rather
than retaining a private inline prompt.

- [ ] **Step 9: Run the new ownership tests and confirm failure**

Run:

```bash
direnv exec . pytest \
  tests/problems/test_vartodd_gf_islands_prompts.py \
  tests/test_config.py::test_vartodd_gf_islands_steady_experiment_contract \
  tests/test_tools/test_run_gf_islands.py::test_launcher_creates_isolated_overlay_store_and_redis_prefix \
  tests/integration/test_vartodd_gf_islands_flow.py -q
```

Expected: FAIL because the overlay files do not exist and YAML still owns
inline guidance.

- [ ] **Step 10: Create the ab-initio overlay**

Create `problems/vartodd_evo_gf_islands/prompts/islands/ab_initio.txt`:

```text
## Required Island Regime: Ab-initio search

Start from `Evaluator(path_name="init")` and construct a complete reusable
path. Use standalone TOHPE as the main early action source. Keep TODD disabled
or light until lower-rank execution evidence supports a scheduled transition.
Do not load or infer a saved path.
```

- [ ] **Step 11: Create the medium-margin overlay**

Create `problems/vartodd_evo_gf_islands/prompts/islands/mid_margin.txt`:

```text
## Required Island Regime: Medium-margin path refinement

Load an exact name from Selectable Shared Paths with margin 30..100. Branch
before the inherited tail so the policy, source balance, restart placement, or
trajectory can change materially. Judge success against `loaded_path_rank`,
not only the global initial rank.
```

- [ ] **Step 12: Create the near-end overlay**

Create `problems/vartodd_evo_gf_islands/prompts/islands/near_end.txt`:

```text
## Required Island Regime: Near-end refinement

Load an exact current Selectable Shared Paths name with margin 5..30. A small
margin can produce broad plateaus because many parameter vectors replay the
same inherited tail. Test a plateau-resilient optimizer initialization or
social parameters, a deliberate restart, or a beam wide enough to research
most plausible trajectories.

In the lower-rank region TODD may be the only source of positive actions. When
action starvation is evidenced, larger `max_buckets` researches more distinct
z buckets and can expose more diverse actions. Set `limit_bucket` high enough
not to truncate the intended search while balancing the resulting runtime.
```

- [ ] **Step 13: Remove inline prompt prose from YAML**

In `config/algorithm/vartodd_diverse_gf_islands.yaml`, keep each route's
`regime_id`, `island_id`, `probability`, and `context_profile`, and delete all
three `guidance: |` blocks. The route section becomes:

```yaml
  mutation_routes:
    - regime_id: ab_initio
      island_id: ab_initio
      probability: 0.40
      context_profile: ab_initio
    - regime_id: mid_margin
      island_id: mid_margin
      probability: 0.35
      context_profile: path_refinement
    - regime_id: near_end
      island_id: near_end
      probability: 0.25
      context_profile: path_refinement
```

- [ ] **Step 14: Run the island prompt/config/launcher tests**

Run:

```bash
direnv exec . pytest \
  tests/problems/test_vartodd_gf_islands_prompts.py \
  tests/test_config.py::test_vartodd_gf_islands_steady_experiment_contract \
  tests/test_tools/test_run_gf_islands.py \
  tests/integration/test_vartodd_gf_islands_flow.py -q
```

Expected: PASS.

- [ ] **Step 15: Run the complete routed-island regression set**

Run:

```bash
direnv exec . pytest \
  tests/evolution/test_multi_island_extended.py \
  tests/evolution/test_mutation_operator.py \
  tests/llm/test_mutation_agent.py \
  tests/custom/test_vartodd_islands_context.py \
  tests/problems/test_vartodd_gf_islands_prompts.py \
  tests/problems/test_vartodd_gf_islands_shared_source.py \
  tests/integration/test_vartodd_gf_islands_flow.py \
  tests/test_tools/test_run_gf_islands.py \
  tests/test_config.py -q
```

Expected: PASS with no prompt-order, Hydra-composition, legacy-single-island,
or overlay-linking regressions.

- [ ] **Step 16: Check prompt ownership and formatting**

Run:

```bash
rg -n "Required Island Regime|margin 30\\.\\.100|margin 5\\.\\.30" \
  config/algorithm/vartodd_diverse_gf_islands.yaml \
  problems/vartodd_evo_gf_islands/prompts/islands
git diff --check
```

Expected: regime prose appears only in the three problem-owned overlay files;
`git diff --check` emits no errors.

- [ ] **Step 17: Commit Task 2**

```bash
git add \
  custom/vartodd_islands_context.py \
  config/algorithm/vartodd_diverse_gf_islands.yaml \
  problems/vartodd_evo_gf_islands/prompts/islands \
  tests/custom/test_vartodd_islands_context.py \
  tests/problems/test_vartodd_gf_islands_prompts.py \
  tests/test_config.py \
  tests/test_tools/test_run_gf_islands.py \
  tests/integration/test_vartodd_gf_islands_flow.py
git commit -m "refactor: move vartodd island guidance into prompts"
```

---

## Final Verification

- [ ] Run the complete focused suite from Task 2 Step 15 once more after all
  commits.
- [ ] Run `git status --short` and confirm no implementation files remain
  uncommitted; preserve all unrelated pre-existing untracked files.
- [ ] Inspect one assembled prompt per route in tests or a local prompt dump and
  confirm the order is assignment → parents → path cards when applicable →
  selected island overlay.
