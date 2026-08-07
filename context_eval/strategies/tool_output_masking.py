from typing import Any

from .base import ContextStrategy


class ToolOutputMasking(ContextStrategy):
    """Mask old tool outputs while keeping the conversation messages."""

    def __init__(self, keep_last_tool_outputs: int = 2):
        self.keep_last_tool_outputs = keep_last_tool_outputs

    def apply(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tool_indexes = [
            i for i, message in enumerate(messages)
            if message.get("role") == "tool"
        ]

        indexes_to_mask = tool_indexes[:-self.keep_last_tool_outputs]

        result = [message.copy() for message in messages]

        for index in indexes_to_mask:
            result[index]["content"] = "[TOOL OUTPUT MASKED]"

        return result
