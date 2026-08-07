"""Scratchpad: the agent's current plan and working state.

This is deliberately a separate object from ShortTermBuffer. The buffer
holds raw conversation turns and gets pruned constantly; the scratchpad
holds *what the agent is trying to do right now* and must survive that
pruning untouched.

Example (Swiftrail): a finance manager asks the agent to review every
open rate exception for a customer before approving a new shipment.
That goal has to survive even if the tool-call chatter used to gather
each rate exception overflows the short-term buffer along the way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Scratchpad:
    """Working memory for the agent's current task, immune to pruning."""

    goal: str = ""
    sub_goal: str = ""
    working_state: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def set_goal(self, goal: str) -> None:
        self.goal = goal
        self._touch()

    def set_sub_goal(self, sub_goal: str) -> None:
        self.sub_goal = sub_goal
        self._touch()

    def update_state(self, key: str, value: Any) -> None:
        """Record one piece of working state, e.g. an id already fetched
        this task so the agent doesn't re-fetch it after a prune."""

        self.working_state[key] = value
        self._touch()

    def snapshot(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "sub_goal": self.sub_goal,
            "working_state": dict(self.working_state),
            "updated_at": self.updated_at,
        }

    def reset(self) -> None:
        """Only called when a task genuinely completes -- never called
        as a side effect of short-term buffer pruning."""

        self.goal = ""
        self.sub_goal = ""
        self.working_state.clear()
        self._touch()

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()
