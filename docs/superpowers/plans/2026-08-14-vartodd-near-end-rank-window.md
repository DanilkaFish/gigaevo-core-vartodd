# VarTODD Near-End Rank Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose only saved-path family representatives within 100 ranks of the current near-end frontier while leaving storage and mid-margin selection unchanged.

**Architecture:** Add one named near-end window constant to the dedicated island `PathStore`. Compute the frontier from the complete eligible inventory, build family representatives with the existing reuse rules, then filter representatives against the inclusive frontier-relative upper bound before sorting and rendering. Report the range in the live-card header and explain it in near-end guidance.

**Tech Stack:** Python 3.12, pytest, filesystem-backed VarTODD Path Store, text prompt overlays.

## Global Constraints

- The inclusive near-end range is `frontier_rank <= path_rank <= frontier_rank + 100`.
- Compute `frontier_rank` before applying the range filter.
- Do not delete hidden paths or alter provenance, reservations, or usage counters.
- Do not apply this filter to the mid-margin inventory.
- Work only in the dedicated `vartodd_evo_gf_islands` problem and its tests.

---

### Task 1: Filter and report the near-end rank window

**Files:**
- Modify: `problems/vartodd_evo_gf_islands/path_store.py`
- Modify: `problems/vartodd_evo_gf_islands/prompts/islands/near_end.txt`
- Modify: `problems/vartodd_evo_gf_islands/task_description.txt`
- Test: `tests/problems/test_vartodd_gf_islands_path_store.py`
- Test: `tests/problems/test_vartodd_gf_islands_prompts.py`

**Interfaces:**
- Produces: module constant `NEAR_END_RANK_WINDOW = 100`.
- Preserves: `PathStore._near_end_inventory(*, limits) -> tuple[int | None, list[dict[str, Any]]]`.
- Preserves: `PathStore._mid_margin_inventory(...)` without rank-window filtering.
- Extends: near-end `## Selectable Shared Paths` header with `rank_window=<low>..<high>`.

- [ ] **Step 1: Write failing inventory boundary tests**

Add a test with otherwise eligible independent path families at ranks 380,
480, and 481:

```python
def test_near_end_inventory_uses_inclusive_frontier_plus_100_window(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_path_store(monkeypatch, tmp_path)
    store = module.PathStore(root_dir=str(tmp_path / "cards"))
    records = [
        _family_record(
            "frontier", 380, init_rank=567,
            created_by_route="ab_initio", route_id="near_end",
        ),
        _family_record(
            "boundary", 480, init_rank=567,
            created_by_route="ab_initio", route_id="near_end",
        ),
        _family_record(
            "outside", 481, init_rank=567,
            created_by_route="ab_initio", route_id="near_end",
        ),
    ]
    monkeypatch.setattr(store, "iter_path_records", lambda: records)

    frontier, eligible = store._near_end_inventory(
        limits=module.PathSelectionLimits()
    )

    assert frontier == 380
    assert [record["name"] for record in eligible] == ["frontier", "boundary"]
```

Add a second assertion using the same records and `_mid_margin_inventory` that
`outside` remains among the ab-initio roots when `root_top_k` is large enough.

- [ ] **Step 2: Write failing rendered-header and prompt tests**

Render near-end cards from a frontier rank of 380 and assert:

```python
assert "rank_window=380..480" in cards
assert "### outside" not in cards
```

In `test_vartodd_gf_islands_prompts.py`, assert the near-end overlay contains
`frontier_rank + 100` and states that the boundary is inclusive.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
nix-shell --run '../gigaevo-core-internal/.venv/bin/pytest tests/problems/test_vartodd_gf_islands_path_store.py tests/problems/test_vartodd_gf_islands_prompts.py -q' -p python312Packages.pytest-asyncio python312Packages.pytest-timeout
```

Expected: the rank-481 path is still returned, the header lacks
`rank_window=380..480`, and the overlay lacks the new boundary explanation.

- [ ] **Step 4: Implement the minimal filter and header**

Define beside the existing route constants:

```python
NEAR_END_RANK_WINDOW = 100
```

After `_near_end_inventory` has constructed reuse-eligible family
representatives, filter them without modifying their records or ledger state:

```python
max_near_rank = frontier_rank + NEAR_END_RANK_WINDOW
representatives = [
    record
    for record in representatives
    if int(record["rank"]) <= max_near_rank
]
```

Extend only the near-end rendered header:

```python
f"route={route_id} frontier_rank={frontier_value} "
f"rank_window={frontier_rank}..{frontier_rank + NEAR_END_RANK_WINDOW}\n\n"
```

Keep the empty-inventory header safe by emitting `rank_window=none` when
`frontier_rank is None`.

- [ ] **Step 5: Explain the filtering contract**

Add this concise rule to `near_end.txt` and the Selectable Shared Paths section
of `task_description.txt`:

```text
The live near-end inventory is restricted to the inclusive range from
frontier_rank through frontier_rank + 100. Higher-rank paths remain stored and
may still be used by mid-margin, but they are omitted here because they belong
to an earlier refinement region.
```

- [ ] **Step 6: Run focused and island regression tests**

Run:

```bash
nix-shell --run '../gigaevo-core-internal/.venv/bin/pytest tests/problems/test_vartodd_gf_islands_path_store.py tests/problems/test_vartodd_gf_islands_prompts.py tests/custom/test_vartodd_islands_context.py tests/integration/test_vartodd_gf_islands_flow.py -q' -p python312Packages.pytest-asyncio python312Packages.pytest-timeout
```

Expected: all selected tests pass.

- [ ] **Step 7: Verify the diff**

Run:

```bash
git diff --check
```

Expected: exit status 0 and no whitespace errors.
