from __future__ import annotations

from custom.additional_stages import _frontier_lever


def test_high_z_low_acceptance_frontier_requests_broader_z_research() -> None:
    hint = _frontier_lever(
        "  g4 401->393 red=8 dim:1..2 z:12000 [hard_hiZ,lo_acc]",
        "hard_hiZ,lo_acc",
    )

    assert "positive actions are rare" in hint
    assert "broader z-bucket research" in hint
    assert "action discovery and candidate diversity" in hint
    assert "NOT more max_buckets" not in hint
