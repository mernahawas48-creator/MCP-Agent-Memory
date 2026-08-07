from typing import Any

from .base import ContextStrategy


class RecursiveSummarization(ContextStrategy):
    """Replace older messages with a compact summary."""

    def __init__(self, keep_last_messages: int = 5):
        self.keep_last_messages = keep_last_messages

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(messages) <= self.keep_last_messages:
            return messages

        old_messages = messages[:-self.keep_last_messages]
        recent_messages = messages[-self.keep_last_messages:]

        summary_parts = []

        for message in old_messages:
            role = message.get("role", "unknown")
            content = str(message.get("content", ""))
            summary_parts.append(f"{role}: {content}")

        summary = " | ".join(summary_parts)

        summary_message = {
            "role": "system",
            "content": f"Summary of older context: {summary}",
        }

        return [summary_message] + recent_messages
