from __future__ import annotations

from memory.short_term import ShortTermBuffer, Turn


def _turn(i: int) -> Turn:
    return Turn(role="employee", content=f"message {i}")


def test_buffer_holds_up_to_max_turns():
    buffer = ShortTermBuffer(max_turns=3)
    for i in range(3):
        evicted = buffer.add(_turn(i))
        assert evicted is None

    assert len(buffer) == 3


def test_buffer_evicts_oldest_when_full():
    buffer = ShortTermBuffer(max_turns=3)
    for i in range(3):
        buffer.add(_turn(i))

    evicted = buffer.add(_turn(3))

    assert evicted is not None
    assert evicted.content == "message 0"
    assert len(buffer) == 3
    assert [t.content for t in buffer.all()] == [
        "message 1", "message 2", "message 3",
    ]


def test_buffer_clear():
    buffer = ShortTermBuffer(max_turns=2)
    buffer.add(_turn(0))
    buffer.clear()
    assert len(buffer) == 0


def test_max_turns_must_be_positive():
    try:
        ShortTermBuffer(max_turns=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
