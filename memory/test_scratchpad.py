from __future__ import annotations

from memory.scratchpad import Scratchpad
from memory.short_term import ShortTermBuffer, Turn


def test_scratchpad_tracks_goal_and_state():
    pad = Scratchpad()
    pad.set_goal("Review open rate exceptions before approving shipment 512")
    pad.set_sub_goal("Fetch rate exceptions for customer 12")
    pad.update_state("customer_id", 12)
    pad.update_state("exceptions_reviewed", 0)

    snap = pad.snapshot()
    assert snap["goal"].startswith("Review open rate exceptions")
    assert snap["working_state"]["customer_id"] == 12


def test_scratchpad_survives_short_term_buffer_pruning():
    """The whole point of splitting these two: pruning the transcript
    must never destroy what the agent is actively doing."""

    pad = Scratchpad()
    pad.set_goal("Review open rate exceptions before approving shipment 512")
    pad.update_state("customer_id", 12)

    buffer = ShortTermBuffer(max_turns=2)
    for i in range(10):
        buffer.add(Turn(role="tool", content=f"tool result {i}"))

    # The buffer has been pruned down to 2 turns, but the scratchpad
    # object was never touched by that process.
    assert len(buffer) == 2
    assert pad.goal == "Review open rate exceptions before approving shipment 512"
    assert pad.working_state["customer_id"] == 12


def test_reset_clears_scratchpad():
    pad = Scratchpad()
    pad.set_goal("Some task")
    pad.update_state("k", "v")
    pad.reset()

    assert pad.goal == ""
    assert pad.working_state == {}
