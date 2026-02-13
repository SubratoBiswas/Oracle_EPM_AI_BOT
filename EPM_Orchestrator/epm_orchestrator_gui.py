from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import customtkinter as ctk
from PIL import Image
from anthropic import Anthropic

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# =========================
# Defaults (YOUR MCP PATH)
# =========================
DEFAULT_MCP_SERVER_PATH = r"C:\Users\subra\PBCS_MCP\epm_mcp_readonly_dualbase.py"
DEFAULT_MCP_ARGS = f'-3 "{DEFAULT_MCP_SERVER_PATH}"'
DEFAULT_MCP_COMMAND = "py"  # IMPORTANT on Windows if you use -3
DEFAULT_MODEL = "claude-3-5-sonnet-latest"

LOGO_FILENAME = "trinamix_logo.png"  # put this next to the .py (and bundle into EXE)

SYSTEM_PROMPT = """You are an Oracle EPM Cloud assistant.
You may call MCP tools to retrieve READ-ONLY operational info (repository files/snapshots, backups, maintenance time, status).
STRICT SAFETY:
- Do NOT run jobs.
- Do NOT upload/download files.
- Do NOT create/update/delete anything.
If the user requests risky actions, refuse and suggest safe read-only alternatives.

When you use tool results, cite exact items observed (file names, timestamps, counts).
"""


# ----------------------------
# UI defaults
# ----------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def resource_path(relative_path: str) -> str:
    """
    PyInstaller-safe path resolver:
    - Dev mode: uses file directory
    - Onefile EXE: uses sys._MEIPASS
    """
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


# ----------------------------
# Helpers
# ----------------------------
def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: List[str] = []
    for b in content:
        btype = getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)
        if btype == "text":
            parts.append(getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else ""))
    return "\n".join([p for p in parts if p]).strip()


def _blocks_to_dicts(blocks: List[Any]) -> List[Dict[str, Any]]:
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


def _mcp_result_to_text(res: Any) -> str:
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
# MCP bridge (Reliable stdio)
# ----------------------------
@dataclass
class MCPConfig:
    command: str
    args: List[str]


class MCPThreadBridge:
    """
    Keeps an MCP stdio session alive in its own asyncio loop/thread.
    Reliable implementation using AsyncExitStack.
    """
    def __init__(self, cfg: MCPConfig):
        self.cfg = cfg
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)

        self._exit_stack: Optional[AsyncExitStack] = None
        self.session: Optional[ClientSession] = None
        self._tools_cache: Optional[List[Dict[str, Any]]] = None

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _connect(self):
        self._exit_stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self.cfg.command,
            args=self.cfg.args,
            env=dict(os.environ),  # pass EPM_* vars to the MCP server
        )
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    async def _close(self):
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self.session = None
        self._tools_cache = None

    def start(self, timeout_s: float = 25.0):
        if not self.thread.is_alive():
            self.thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._connect(), self.loop)
        fut.result(timeout=timeout_s)

    def is_connected(self) -> bool:
        return self.session is not None

    def list_tools(self, timeout_s: float = 20.0) -> List[Dict[str, Any]]:
        if not self.session:
            raise RuntimeError("MCP not connected")

        fut = asyncio.run_coroutine_threadsafe(self.session.list_tools(), self.loop)
        resp = fut.result(timeout=timeout_s)
        tools = getattr(resp, "tools", []) or []

        schemas: List[Dict[str, Any]] = []
        for t in tools:
            input_schema = (
                getattr(t, "inputSchema", None)
                or getattr(t, "input_schema", None)
                or {"type": "object", "properties": {}}
            )
            schemas.append({
                "name": t.name,
                "description": t.description or "",
                "input_schema": input_schema,
            })

        self._tools_cache = schemas
        return schemas

    def call_tool(self, name: str, args: Dict[str, Any], timeout_s: float = 120.0) -> Any:
        if not self.session:
            raise RuntimeError("MCP not connected")

        fut = asyncio.run_coroutine_threadsafe(self.session.call_tool(name, args), self.loop)
        return fut.result(timeout=timeout_s)

    def close(self):
        if not self.thread.is_alive():
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(self._close(), self.loop)
            fut.result(timeout=10.0)
        except Exception:
            pass
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass


# ----------------------------
# Claude tool loop runner
# ----------------------------
class ClaudeToolRunner:
    def __init__(self, api_key: str, model: str):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def run(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], mcp: MCPThreadBridge) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1400,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools if tools else None,
        )

        while True:
            assistant_blocks = _blocks_to_dicts(list(resp.content))
            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                return _extract_text(assistant_blocks) or "(no text returned)"

            tool_results: List[Dict[str, Any]] = []
            for tu in tool_uses:
                try:
                    tool_res = mcp.call_tool(tu.name, tu.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": _mcp_result_to_text(tool_res),
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": f"Tool error: {repr(e)}",
                        "is_error": True,
                    })

            messages.append({"role": "user", "content": tool_results})

            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1400,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools if tools else None,
            )


# ----------------------------
# GUI App
# ----------------------------
class EPMOrchestratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EPM Read-Only Orchestrator (Claude + MCP)")
        self.geometry("1200x820")
        self.minsize(1050, 720)

        self.messages: List[Dict[str, Any]] = []
        self._pending_logs: List[Tuple[str, str]] = []  # buffer logs before chat_box exists

        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.mcp_command_default = os.getenv("MCP_COMMAND", DEFAULT_MCP_COMMAND).strip() or DEFAULT_MCP_COMMAND

        env_mcp_args = os.getenv("MCP_ARGS", "").strip()
        self.mcp_args_default = env_mcp_args if env_mcp_args else DEFAULT_MCP_ARGS

        self.mcp: Optional[MCPThreadBridge] = None
        self.tools_schema: List[Dict[str, Any]] = []

        self._build_ui()

    # ----------------------------
    # Logging (safe before chat_box exists)
    # ----------------------------
    def log(self, who: str, text: str):
        if not hasattr(self, "chat_box") or self.chat_box is None:
            self._pending_logs.append((who, text))
            return

        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"\n{who}:\n{text}\n")
        self.chat_box.insert("end", "-" * 70 + "\n")
        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")

    def _flush_pending_logs(self):
        for who, msg in self._pending_logs:
            self.log(who, msg)
        self._pending_logs.clear()

    # ----------------------------
    # UI
    # ----------------------------
    def _build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        root = ctk.CTkFrame(self, corner_radius=0)
        root.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(root)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 8))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        # Left: title/subtitle/tagline
        text_block = ctk.CTkFrame(header, fg_color="transparent")
        text_block.grid(row=0, column=0, sticky="w", padx=12, pady=12)

        title = ctk.CTkLabel(
            text_block,
            text="EPM Read-Only Orchestrator",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = ctk.CTkLabel(
            text_block,
            text="Python Front End → Claude API → MCP Server (stdio) → EPM REST APIs (read-only)",
            text_color="gray70",
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(2, 0))

        tagline = ctk.CTkLabel(
            text_block,
            text="Powered by Trinamix",
            text_color="gray60",
            font=ctk.CTkFont(size=12, slant="italic"),
        )
        tagline.grid(row=2, column=0, sticky="w", pady=(6, 0))

        # Right: logo (top-right)
        self.logo_label = None
        self.logo_img = None
        try:
            logo_path = resource_path(LOGO_FILENAME)
            img = Image.open(logo_path)
            self.logo_img = ctk.CTkImage(light_image=img, dark_image=img, size=(160, 52))
            self.logo_label = ctk.CTkLabel(header, image=self.logo_img, text="")
            self.logo_label.grid(row=0, column=1, sticky="ne", padx=(0, 14), pady=12)
        except Exception:
            # Logo optional; app still runs
            self.logo_label = None

        # Left panel (config + buttons)
        left = ctk.CTkFrame(root, width=340)
        left.grid(row=1, column=0, sticky="ns", padx=(10, 8), pady=10)
        left.grid_rowconfigure(20, weight=1)
        left.grid_columnconfigure(0, weight=1)

        cfg_label = ctk.CTkLabel(left, text="Config", font=ctk.CTkFont(size=16, weight="bold"))
        cfg_label.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        self.api_entry = ctk.CTkEntry(left, placeholder_text="Anthropic API key (sk-ant-...)", show="*")
        self.api_entry.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.model_entry = ctk.CTkEntry(left)
        self.model_entry.insert(0, self.anthropic_model)
        self.model_entry.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.mcp_command_entry = ctk.CTkEntry(left)
        self.mcp_command_entry.insert(0, self.mcp_command_default)
        self.mcp_command_entry.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.mcp_args_entry = ctk.CTkEntry(
            left,
            placeholder_text=r'MCP args, e.g. -3 "C:\path\epm_mcp_readonly_dualbase.py"',
        )
        self.mcp_args_entry.insert(0, self.mcp_args_default)
        self.mcp_args_entry.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.connect_btn = ctk.CTkButton(left, text="Connect MCP", command=self.connect_mcp)
        self.connect_btn.grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.status = ctk.CTkLabel(left, text="MCP: Not connected", text_color="gray70")
        self.status.grid(row=6, column=0, sticky="w", padx=12, pady=(0, 12))

        ops_label = ctk.CTkLabel(left, text="Read-Only Operations", font=ctk.CTkFont(size=16, weight="bold"))
        ops_label.grid(row=7, column=0, sticky="w", padx=12, pady=(8, 6))

        # 5 buttons
        self.op_buttons: List[Tuple[str, str]] = [
            ("Health Snapshot", "Use EPM read-only tools to provide: daily maintenance time, latest backup, and newest repository snapshot/file."),
            ("List Repository Files", "Use EPM read-only tools to list repository files/snapshots newest-first, then summarize the top 10 and total count."),
            ("Backup Check (7 days)", "Use EPM read-only tools to list backups and summarize count per day for the last 7 days."),
            ("Maintenance Window", "Use EPM read-only tools to retrieve daily maintenance start time and summarize it clearly."),
            ("Status Check", "Ask me for operation type and job id if missing, then use EPM read-only tools to check status and summarize."),
        ]

        for i, (label, prompt) in enumerate(self.op_buttons):
            btn = ctk.CTkButton(left, text=label, height=42, command=lambda p=prompt: self.run_operation(p))
            btn.grid(row=8 + i, column=0, sticky="ew", padx=12, pady=6)

        # Right panel (chat)
        right = ctk.CTkFrame(root)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 10), pady=10)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        chat_header = ctk.CTkFrame(right, fg_color="transparent")
        chat_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        chat_header.grid_columnconfigure(0, weight=1)

        chat_title = ctk.CTkLabel(chat_header, text="Chat", font=ctk.CTkFont(size=16, weight="bold"))
        chat_title.grid(row=0, column=0, sticky="w")

        self.chat_status = ctk.CTkLabel(chat_header, text="Ready", text_color="gray70")
        self.chat_status.grid(row=0, column=1, sticky="e")

        self.chat_box = ctk.CTkTextbox(right, wrap="word")
        self.chat_box.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 10))
        self.chat_box.configure(state="disabled")

        # now it's safe to flush logs
        self._flush_pending_logs()

        input_row = ctk.CTkFrame(right)
        input_row.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
        input_row.grid_columnconfigure(0, weight=1)

        self.input_entry = ctk.CTkEntry(input_row, placeholder_text="Ask a question…")
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(8, 6), pady=8)
        self.input_entry.bind("<Return>", lambda e: self.send_chat())

        send_btn = ctk.CTkButton(input_row, text="Send", width=100, command=self.send_chat)
        send_btn.grid(row=0, column=1, padx=(0, 8), pady=8)

    def set_busy(self, busy: bool, msg: str):
        self.chat_status.configure(text=msg)
        self.input_entry.configure(state="disabled" if busy else "normal")

    # ----------------------------
    # MCP connect
    # ----------------------------
    def connect_mcp(self):
        cmd = self.mcp_command_entry.get().strip() or DEFAULT_MCP_COMMAND
        args_str = self.mcp_args_entry.get().strip()

        if not args_str:
            self.status.configure(text="MCP: missing args/path", text_color="tomato")
            self.log("SYSTEM", 'Please provide MCP args. Example:\n-3 "C:\\path\\to\\epm_mcp_readonly_dualbase.py"')
            return

        self.status.configure(text="MCP: Connecting…", text_color="gray70")

        def _do():
            try:
                args = shlex.split(args_str, posix=(os.name != "nt"))
                self.mcp = MCPThreadBridge(MCPConfig(command=cmd, args=args))
                self.mcp.start()
                self.tools_schema = self.mcp.list_tools()

                tool_names = ", ".join([t["name"] for t in self.tools_schema])
                self.after(0, lambda: self.status.configure(
                    text=f"MCP: Connected ({len(self.tools_schema)} tools)", text_color="lightgreen"
                ))
                self.after(0, lambda: self.log("SYSTEM", f"MCP connected.\nTools: {tool_names}"))
            except Exception as e:
                self.after(0, lambda: self.status.configure(text="MCP: Connect failed (see chat)", text_color="tomato"))
                self.after(0, lambda: self.log("SYSTEM", f"MCP connect failed:\n{repr(e)}"))

        threading.Thread(target=_do, daemon=True).start()

    # ----------------------------
    # Running operations and chat
    # ----------------------------
    def _ensure_ready(self) -> Optional[str]:
        api_key = self.api_entry.get().strip()
        if not api_key:
            return "Missing Anthropic API key."
        if not (self.mcp and self.mcp.is_connected()):
            return "MCP not connected. Click 'Connect MCP' first."
        if not self.tools_schema:
            return "No MCP tools found."
        return None

    def run_operation(self, prompt: str):
        err = self._ensure_ready()
        if err:
            self.log("SYSTEM", err)
            return

        self.messages.append({"role": "user", "content": prompt})
        self.log("YOU", prompt)
        self.set_busy(True, "Running…")

        def _do():
            try:
                api_key = self.api_entry.get().strip()
                model = self.model_entry.get().strip() or DEFAULT_MODEL
                runner = ClaudeToolRunner(api_key, model)
                answer = runner.run(self.messages, self.tools_schema, self.mcp)  # type: ignore[arg-type]
                self.after(0, lambda: self.log("CLAUDE", answer))
            except Exception as e:
                self.after(0, lambda: self.log("SYSTEM", f"Error:\n{repr(e)}"))
            finally:
                self.after(0, lambda: self.set_busy(False, "Ready"))

        threading.Thread(target=_do, daemon=True).start()

    def send_chat(self):
        err = self._ensure_ready()
        if err:
            self.log("SYSTEM", err)
            return

        text = self.input_entry.get().strip()
        if not text:
            return
        self.input_entry.delete(0, "end")

        self.messages.append({"role": "user", "content": text})
        self.log("YOU", text)
        self.set_busy(True, "Thinking…")

        def _do():
            try:
                api_key = self.api_entry.get().strip()
                model = self.model_entry.get().strip() or DEFAULT_MODEL
                runner = ClaudeToolRunner(api_key, model)
                answer = runner.run(self.messages, self.tools_schema, self.mcp)  # type: ignore[arg-type]
                self.after(0, lambda: self.log("CLAUDE", answer))
            except Exception as e:
                self.after(0, lambda: self.log("SYSTEM", f"Error:\n{repr(e)}"))
            finally:
                self.after(0, lambda: self.set_busy(False, "Ready"))

        threading.Thread(target=_do, daemon=True).start()


if __name__ == "__main__":
    app = EPMOrchestratorApp()
    app.mainloop()
