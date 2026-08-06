"""Local schema and capability smoke checks.

Run the full protocol path with ``python ../agent/demo.py``. This file checks
that registered tool schemas reject unknown top-level arguments before a
database is contacted.
"""

from __future__ import annotations

import asyncio
import json

from app_instance import app

# Import modules for registration side effects.
import tools.auth  # noqa: F401
import tools.credit_hold  # noqa: F401
import tools.portfolio  # noqa: F401
import tools.rate_exception  # noqa: F401
import tools.read_tools  # noqa: F401
import tools.resources_prompts  # noqa: F401


async def main() -> None:
    tools = await app.list_tools()
    assert tools, "No tools were registered."

    failures: list[str] = []

    for tool in tools:
        schema = tool.inputSchema
        if schema.get("additionalProperties") is not False:
            failures.append(tool.name)
            print(f"\n[schema FAILED] {tool.name}")
            print(json.dumps(schema, indent=2, default=str))
        else:
            print(f"[schema OK] {tool.name}")

    assert not failures, (
        "These tools do not publish additionalProperties=false: "
        + ", ".join(failures)
    )

    options = app._mcp_server.create_initialization_options()
    assert options.capabilities.tools is not None
    assert options.capabilities.tools.listChanged is True
    print("[capability OK] tools.listChanged=true")


if __name__ == "__main__":
    asyncio.run(main())