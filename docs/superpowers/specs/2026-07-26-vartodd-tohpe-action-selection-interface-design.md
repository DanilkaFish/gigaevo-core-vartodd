# Updated VarTODD ActionSelection Interface Design

## Goal

Make `problems/vartodd_evo_tohpe_updated/helper.py` expose the same public
`ActionSelection` interface as `problems/vartodd_evo/helper.py`.

## Interface

The helper imports pyvartodd's class, retains it as `_PyActionSelection`, and
exposes:

```python
ActionSelection(
    beamwidth: int = 1,
    mode: str = "best",
    temperature: float = 0.0,
)
```

The factory constructs `_PyActionSelection` with:

- `count=int(beamwidth)`;
- the supplied `mode`;
- `temperature=float(temperature)`.

Unknown keyword arguments, including the underlying binding's legacy
`count`, raise `TypeError`. This intentionally matches `vartodd_evo` rather
than providing a second compatibility interface.

## Conversion

`_to_action_selection` recognizes `_PyActionSelection` instances. Mapping
input reads `beamwidth`, `mode`, and `temperature`, then calls the public
factory. Rank-scheduled action selection continues to use the existing
conversion path unchanged.

No changes are made to `node.py`, pyvartodd, initial programs, action pools,
or rank-schedule semantics.

## Validation

Behavioral tests verify:

- `ActionSelection(beamwidth=3, mode="softmax", temperature=0.2)` stores
  `count == 3` and preserves the other values;
- mapping conversion maps `beamwidth` to the internal `count`;
- `ActionSelection(count=3)` raises `TypeError`;
- the existing updated-problem contract tests continue to pass.
