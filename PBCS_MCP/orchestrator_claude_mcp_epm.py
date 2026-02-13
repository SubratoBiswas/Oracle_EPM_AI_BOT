from __future__ import annotations

import asyncio
import json
import os
import shlex
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class Config:
    anthropic_api_key: str
    anthropic_model: str
    mcp_command: str
    mcp_args: List[str]


def load_config() -> Config:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY")

    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest").strip()

    cmd = os.getenv("MCP_COMMAND", "py").strip()
    args_str = os.getenv("MCP_ARGS", "").strip()
    if not args_str:
        raise RuntimeError("Missing MCP_ARGS (e.g. '-3 C:\\path\\to\\epm_mcp_readonly_dualbase.py')")

    args = shlex.split(args_str, posix=(os.name != "nt"))
    return Config(
        anthropic_api_key=api_key,
        anthropic_model=model,
        mcp_command=cmd,
        mcp_args=args,
    )


SYSTEM_PROMPT = """You are an Oracle EPM Cloud assistant.
You may call MCP tools to retrieve read-only operational information (snapshots/files, backups, maintenance time, status).
STRICT SAFETY:
- Do NOT run jobs.
- Do NOT upload/download files.
- Do NOT create/update/delete anything.
If asked to do risky actions, refuse and suggest safe read-only checks.

When you use tool results, cite the exact items you observed (file names, timestamps, counts).
"""


# ----------------------------
# Helpers
# ----------------------------

def extract_text(blocks_or_str: Any) -> str:
    if isinstance(blocks_or_str, str):
        return blocks_or_str
    if not isinstance(blocks_or_str, list):
        return str(blocks_or_str)

    parts: List[str] = []
    for b in blocks_or_str:
        btype = getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)
        if btype == "text":
            parts.append(getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else ""))
    return "\n".join([p for p in parts if p]).strip()


def blocks_to_dicts(blocks: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for b in blocks:
        t = getattr(b, "type", None)
        if t == "text":
            out.append({"type": "text", "text": b.text})
        elif t == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        else:
            try:
                out.append(b.model_dump())  # type: ignore[attr-defined]
            except Exception:
                out.append({"type": str(t), "raw": str(b)})
    return out


def mcp_result_to_text(res: Any) -> str:
    content = getattr(res, "content", None)
    if content is None:
        return json.dumps({"result": str(res)}, ensure_ascii=False)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: List[str] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                texts.append(str(b.get("text", "")))
            elif hasattr(b, "type") and getattr(b, "type") == "text":
                texts.append(str(getattr(b, "text", "")))
        joined = "\n".join([t for t in texts if t]).strip()
        if joined:
            return joined
        try:
            return json.dumps(content, ensure_ascii=False)
        except Exception:
            return str(content)

    return str(content)


# ----------------------------
# Main loop
# ----------------------------

async def main():
    cfg = load_config()
    anthropic = Anthropic(api_key=cfg.anthropic_api_key)

    server_params = StdioServerParameters(
        command=cfg.mcp_command,
        args=cfg.mcp_args,
        env=dict(os.environ),  # pass EPM_* vars through to your MCP server
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover MCP tools
            tools_resp = await session.list_tools()
            mcp_tools = getattr(tools_resp, "tools", []) or []

            # Convert MCP tool schema → Anthropic tool schema
            anth_tools: List[Dict[str, Any]] = []
            for t in mcp_tools:
                input_schema = (
                    getattr(t, "inputSchema", None)
                    or getattr(t, "input_schema", None)
                    or {"type": "object", "properties": {}}
                )
                anth_tools.append({
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": input_schema,
                })

            print("\nConnected to MCP tools:")
            for tt in anth_tools:
                print(" -", tt["name"])
            print()

            messages: List[Dict[str, Any]] = []

            while True:
                user_q = input("You> ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ("exit", "quit"):
                    break

                messages.append({"role": "user", "content": user_q})

                # First Claude call
                resp = anthropic.messages.create(
                    model=cfg.anthropic_model,
                    max_tokens=1400,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=anth_tools,
                )

                # Tool-use loop
                while True:
                    # Add assistant content to conversation
                    assistant_blocks = blocks_to_dicts(list(resp.content))
                    messages.append({"role": "assistant", "content": assistant_blocks})

                    tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
                    if not tool_uses:
                        print(f"\nClaude> {extract_text(assistant_blocks)}\n")
                        break

                    # Execute tools and send tool_result blocks back to Claude
                    tool_results: List[Dict[str, Any]] = []
                    for tu in tool_uses:
                        try:
                            tool_res = await session.call_tool(tu.name, tu.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tu.id,
                                "content": mcp_result_to_text(tool_res),
                            })
                        except Exception as e:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tu.id,
                                "content": f"Tool error: {e}",
                                "is_error": True,
                            })

                    messages.append({"role": "user", "content": tool_results})

                    # Continue Claude with tool results
                    resp = anthropic.messages.create(
                        model=cfg.anthropic_model,
                        max_tokens=1400,
                        system=SYSTEM_PROMPT,
                        messages=messages,
                        tools=anth_tools,
                    )


if __name__ == "__main__":
    asyncio.run(main())
