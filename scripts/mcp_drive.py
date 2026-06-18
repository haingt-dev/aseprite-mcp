"""Tiny stdio MCP client driver for testing aseprite-mcp.

Usage:
  mcp_drive.py --list                 # print tool names + schemas (compact)
  mcp_drive.py calls.json             # run a sequence of tool calls
  echo '[{"tool":"create_canvas","args":{...}}]' | mcp_drive.py -

calls.json = [{"tool": "<name>", "args": {...}}, ...]
Prints one JSON line per call: {"tool":..., "ok":..., "result":...}
"""
import asyncio
import json
import sys
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main() -> None:
    # Self-sufficient: default ASEPRITE_PATH so the driver works without the
    # caller exporting it (mirrors the godot mcp_drive.py GODOT_PATH default).
    env = dict(os.environ)
    env.setdefault(
        "ASEPRITE_PATH",
        "/home/haint/.steam/steam/steamapps/common/Aseprite/aseprite",
    )
    server = StdioServerParameters(
        command=os.path.join(REPO, ".venv", "bin", "python"),
        args=["-m", "aseprite_mcp"],
        cwd=REPO,
        env=env,
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if sys.argv[1] == "--list":
                tools = await session.list_tools()
                for t in tools.tools:
                    params = list((t.inputSchema or {}).get("properties", {}).keys())
                    print(f"{t.name}({', '.join(params)})")
                return
            src = sys.stdin.read() if sys.argv[1] == "-" else open(sys.argv[1]).read()
            calls = json.loads(src)
            for call in calls:
                try:
                    res = await session.call_tool(call["tool"], call.get("args", {}))
                    text = "\n".join(
                        c.text for c in res.content if getattr(c, "text", None)
                    )
                    print(json.dumps({"tool": call["tool"], "ok": not res.isError, "result": text}))
                except Exception as e:  # noqa: BLE001
                    print(json.dumps({"tool": call["tool"], "ok": False, "result": f"driver error: {e}"}))


asyncio.run(main())
