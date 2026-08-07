"""Short-term memory: a rolling buffer of recent conversation turns.

This is intentionally the *only* thing that gets pruned during a
conversation. It holds raw turns (customer messages, tool calls, tool
results), not the agent's working plan -- that lives in Scratchpad
instead, so pruning the transcript never destroys what the agent is
actively doing (see scratchpad.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Turn:
    """One raw item in the short-term buffer."""

    role: str  # "customer" | "employee" | "tool" | "agent"
    content: Any
    customer_id: int | None = None
    session_id: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ShortTermBuffer:
    """Fixed-size rolling window over the current conversation.

    Unlike a plain ``collections.deque(maxlen=...)``, this keeps track of
    *which* turn got evicted when the buffer overflows, because the
    promote-or-drop router (router.py) needs that exact item to decide
    whether it should be forgotten or promoted to episodic memory.
    """

    def __init__(self, max_turns: int = 12):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    def add(self, turn: Turn) -> Turn | None:
        """Add a turn to the buffer.

        Returns the evicted Turn if adding this one pushed the buffer
        over capacity, otherwise None. The caller (normally the agent
        loop) is responsible for handing an evicted turn to the
        PromoteDropRouter -- this class does not decide what happens to
        aged-out items, it only tracks the window.
        """

        self._turns.append(turn)

        if len(self._turns) > self.max_turns:
            return self._turns.pop(0)

        return None

    def all(self) -> list[Turn]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
