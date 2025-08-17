#!/usr/bin/env python3
"""
OpenAI-backed agent implementation for the Sunstone `think-agents` CLI.

- Connects to a local MCP server over Streamable HTTP
- Runs an agent with streaming to surface tool args/results and (when available) reasoning summaries
- Emits JSON events compatible with the CLI (`start`, `tool_start`, `tool_end`, `thinking`, `finish`, `error`)
- Raises max_turns (configurable via env)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse, urlunparse

from agents import Agent, Runner, SQLiteSession
from agents.items import (
    MessageOutputItem,
    ReasoningItem,
    ToolCallItem,
    ToolCallOutputItem,
)
from agents.mcp.server import MCPServerStreamableHttp
from agents.model_settings import ModelSettings
from agents.run import RunConfig

# Optional: used only for raw text deltas if available
try:
    from openai.types.responses import ResponseTextDeltaEvent  # type: ignore
except Exception:  # pragma: no cover
    ResponseTextDeltaEvent = object  # type: ignore

from think.utils import agent_instructions

from .agents import JSONEventCallback, ThinkingEvent
from .models import GPT_5

DEFAULT_MODEL = os.getenv("OPENAI_AGENT_MODEL", GPT_5)
DEFAULT_MAX_TOKENS = int(os.getenv("OPENAI_AGENT_MAX_TOKENS", "8192"))
DEFAULT_MAX_TURNS = int(os.getenv("OPENAI_AGENT_MAX_TURNS", "32"))

LOG = logging.getLogger("think.openai")


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, stream=sys.stdout)
    return LOG


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_streamable_http_uri(http_uri: str) -> str:
    """
    Ensure the Streamable HTTP MCP URL points at the MCP endpoint.

    If no path or '/', append '/mcp'.
    If already '/mcp' (with or without trailing '/'), keep unchanged.
    Otherwise, leave as-is (user may have a reverse-proxy path).
    """
    try:
        parsed = urlparse(http_uri.strip())
        path = parsed.path or ""
        if path in ("", "/"):
            path = "/mcp"
        elif path.rstrip("/") == "/mcp":
            path = "/mcp"
        new_parsed = parsed._replace(path=path)
        return urlunparse(new_parsed)
    except Exception:
        return http_uri


def _json_maybe_loads(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def _extract_tool_name(raw_call: Any) -> str:
    for attr in ("name", "tool", "tool_name"):
        if hasattr(raw_call, attr):
            val = getattr(raw_call, attr)
            if isinstance(val, str) and val:
                return val
    return type(raw_call).__name__


def _extract_tool_call_id(raw_call: Any) -> Optional[str]:
    for attr in ("id", "call_id", "tool_call_id"):
        if hasattr(raw_call, attr):
            val = getattr(raw_call, attr)
            if isinstance(val, str) and val:
                return val
    return None


def _extract_tool_args(raw_call: Any) -> Any:
    """
    Function tools usually expose `.arguments` (often JSON string).
    Some tools expose `.input_json` / `.input`.
    """
    if hasattr(raw_call, "arguments"):
        return _json_maybe_loads(getattr(raw_call, "arguments"))
    for attr in ("input_json", "input", "arguments_json"):
        if hasattr(raw_call, attr):
            return getattr(raw_call, attr)
    if isinstance(raw_call, dict):
        if "arguments" in raw_call:
            return _json_maybe_loads(raw_call["arguments"])
        if "input" in raw_call:
            return raw_call["input"]
    return None


async def run_agent(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """
    Run a single prompt through the OpenAI Agents SDK using streaming.
    Emits JSON events and returns the final text output.
    """
    LOG.info("Running agent with model %s", model)
    cb = JSONEventCallback(on_event)
    cb.emit(
        {
            "event": "start",
            "prompt": prompt,
            "persona": persona,
            "model": model,
            "backend": "openai",
            "ts": _now_ms(),
        }
    )

    # Model settings: keep to widely-supported fields
    model_settings = ModelSettings(
        max_tokens=max_tokens,
        # If you later want to add temperature/top_p etc., do it here.
        # Avoid unsupported reasoning params to prevent 400s.
    )

    # Read MCP server URL from journal
    journal_path = os.getenv("JOURNAL_PATH")
    if not journal_path:
        raise RuntimeError("JOURNAL_PATH not set")
    uri_file = Path(journal_path) / "agents" / "mcp.uri"
    if not uri_file.exists():
        raise RuntimeError(f"MCP server URI file not found: {uri_file}")
    http_uri_raw = uri_file.read_text().strip()
    if not http_uri_raw:
        raise RuntimeError("MCP server URI file is empty")
    http_uri = _normalize_streamable_http_uri(http_uri_raw)

    mcp_server = MCPServerStreamableHttp(
        params={"url": http_uri},
        cache_tools_list=True,
    )

    # Use your repo's persona utility
    system_instruction, extra_context, _ = agent_instructions(persona)

    # Keep a map of in-flight tools so we can pair outputs with args
    pending_tools: Dict[str, Dict[str, Any]] = {}

    # Accumulate streamed text chunks as a fallback (if final_output missing)
    streamed_text: list[str] = []

    session = SQLiteSession("sunstone_oneshot")

    try:
        async with mcp_server:
            agent = Agent(
                name="SunstoneCLI",
                instructions=f"{system_instruction}\n\n{extra_context}".strip(),
                model=model,
                model_settings=model_settings,
                mcp_servers=[mcp_server],
            )

            result = Runner.run_streamed(
                agent,
                input=prompt,
                session=session,
                run_config=RunConfig(tracing_disabled=True),  # per docs
                max_turns=DEFAULT_MAX_TURNS,
            )

            async for ev in result.stream_events():
                # 1) Raw deltas (Responses API events)
                if ev.type == "raw_response_event":
                    data = getattr(ev, "data", None)
                    # If we have text deltas, capture them (optional)
                    if isinstance(data, ResponseTextDeltaEvent) and isinstance(
                        getattr(data, "delta", None), str
                    ):
                        streamed_text.append(data.delta)
                    continue

                # 2) Agent updates (handoffs)
                if ev.type == "agent_updated_stream_event":
                    new_agent = getattr(ev, "new_agent", None)
                    cb.emit(
                        {
                            "event": "agent_updated",
                            "agent": getattr(new_agent, "name", None),
                            "ts": _now_ms(),
                        }
                    )
                    continue

                # 3) Run items (messages, tools, reasoning, etc.)
                if ev.type == "run_item_stream_event":
                    name = ev.name
                    item = ev.item

                    # Tool call started — capture args
                    if name == "tool_called" and isinstance(item, ToolCallItem):
                        raw = item.raw_item
                        tool_name = _extract_tool_name(raw)
                        call_id = _extract_tool_call_id(raw) or tool_name
                        args = _extract_tool_args(raw)
                        pending_tools[call_id] = {"tool": tool_name, "args": args}
                        cb.emit(
                            {
                                "event": "tool_start",
                                "tool": tool_name,
                                "args": args,
                                "call_id": call_id,
                                "ts": _now_ms(),
                            }
                        )

                    # Tool output finished — mirror end, include original args if we have them
                    elif name == "tool_output" and isinstance(item, ToolCallOutputItem):
                        raw = item.raw_item
                        call_id = (
                            getattr(raw, "tool_call_id", None)
                            or getattr(raw, "call_id", None)
                            or getattr(raw, "id", None)
                        )
                        meta = pending_tools.pop(call_id, {}) if call_id else {}
                        cb.emit(
                            {
                                "event": "tool_end",
                                "tool": meta.get("tool", "tool"),
                                "args": meta.get("args"),
                                "result": item.output,
                                "call_id": call_id,
                                "ts": _now_ms(),
                            }
                        )

                    # Reasoning / "thinking" item created — no special params required
                    elif name == "reasoning_item_created" and isinstance(
                        item, ReasoningItem
                    ):
                        summary_text: Optional[str] = None
                        raw = item.raw_item

                        # Try raw.summary (list of text parts)
                        if hasattr(raw, "summary") and getattr(raw, "summary"):
                            try:
                                parts = getattr(raw, "summary")
                                texts = [
                                    getattr(p, "text", "")
                                    for p in parts
                                    if getattr(p, "text", None)
                                ]
                                if texts:
                                    summary_text = "".join(texts)
                            except Exception:
                                pass

                        # Fallback: raw.content (list of text parts)
                        if (
                            not summary_text
                            and hasattr(raw, "content")
                            and getattr(raw, "content")
                        ):
                            try:
                                parts = getattr(raw, "content")
                                texts = [
                                    getattr(p, "text", "")
                                    for p in parts
                                    if getattr(p, "text", None)
                                ]
                                if texts:
                                    summary_text = "".join(texts)
                            except Exception:
                                pass

                        if summary_text:
                            thinking_event: ThinkingEvent = {
                                "event": "thinking",
                                "summary": summary_text,
                                "model": model,
                                "ts": _now_ms(),
                            }
                            cb.emit(thinking_event)

                    # Completed assistant message (final text will be read from result)
                    elif name == "message_output_created" and isinstance(
                        item, MessageOutputItem
                    ):
                        pass  # no-op

            # Done streaming — prefer result.final_output, else join deltas
            final_text = getattr(result, "final_output", None)
            if not isinstance(final_text, str) or not final_text:
                final_text = "".join(streamed_text)

            cb.emit({"event": "finish", "result": final_text, "ts": _now_ms()})
            return final_text

    except Exception as exc:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        cb.emit({"event": "error", "error": str(exc), "trace": trace, "ts": _now_ms()})
        raise
    finally:
        # IMPORTANT: Don't explicitly close the SQLiteSession while streaming;
        # the SDK continues to read from it in background tasks and closing can race.
        pass


async def run_prompt(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Alias for run_agent for CLI parity."""
    return await run_agent(
        prompt,
        model=model,
        max_tokens=max_tokens,
        on_event=on_event,
        persona=persona,
    )


__all__ = [
    "run_agent",
    "run_prompt",
    "setup_logging",
    "DEFAULT_MODEL",
    "DEFAULT_MAX_TOKENS",
]
