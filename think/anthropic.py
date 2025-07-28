#!/usr/bin/env python3
"""think.anthropic

CLI utility launching an Anthropic Claude agent able to search ponder summaries,
occurrences and read full markdown files from the journal.

Usage:
    think-claude [TASK_FILE] [--model MODEL] [--max-tokens N] [-v] [-o OUT]

When TASK_FILE is omitted, an interactive chat prompt is started.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam, ToolUseBlock

from .agents import (
    BaseAgentSession,
    JournalEventWriter,
    JSONEventCallback,
    JSONEventWriter,
)
from .models import CLAUDE_OPUS_4
from .utils import agent_instructions, create_mcp_client, setup_cli


def setup_logging(verbose: bool) -> logging.Logger:
    """Return app logger configured for ``verbose``."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        os.environ["ANTHROPIC_LOG"] = "debug"
    else:
        logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


class ToolExecutor:
    """Handle MCP tool execution and result formatting for Anthropic."""

    def __init__(self, mcp_client: Any, callback: JSONEventCallback) -> None:
        self.mcp = mcp_client
        self.callback = callback

    async def execute_tool(self, tool_use: ToolUseBlock) -> dict:
        """Execute ``tool_use`` and return a Claude ``tool_result`` block."""
        self.callback.emit(
            {
                "event": "tool_start",
                "tool": tool_use.name,
                "args": tool_use.input,
            }
        )

        try:
            result = await self.mcp.session.call_tool(
                name=tool_use.name,
                arguments=tool_use.input,
            )
            self.callback.emit(
                {
                    "event": "tool_end",
                    "tool": tool_use.name,
                    "result": result,
                }
            )
            content = result if isinstance(result, str) else json.dumps(result)
        except Exception as exc:  # pragma: no cover - unexpected
            self.callback.emit(
                {
                    "event": "tool_end",
                    "tool": tool_use.name,
                    "result": {"error": str(exc)},
                }
            )
            content = f"Error: {exc}"

        return {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": content,
        }


class AgentSession(BaseAgentSession):
    """Context manager running Claude with MCP tools."""

    def __init__(
        self,
        model: str = CLAUDE_OPUS_4,
        *,
        max_tokens: int = 4096,
        on_event: Optional[Callable[[dict], None]] = None,
        persona: str = "default",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._callback = JSONEventCallback(on_event)
        self._mcp = None
        self.client: AsyncAnthropic | None = None
        self.messages: list[MessageParam] = []
        self.system_instruction = ""
        self._history: list[dict[str, str]] = []
        self.persona = persona
        self.tool_executor: ToolExecutor | None = None

    async def __aenter__(self) -> "AgentSession":
        self._mcp = create_mcp_client("fastmcp")
        await self._mcp.__aenter__()

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = AsyncAnthropic(api_key=api_key)

        self.system_instruction, first_user, _ = agent_instructions(self.persona)

        self.tool_executor = ToolExecutor(self._mcp, self._callback)

        if first_user:
            self.messages.append({"role": "user", "content": first_user})
            self._history.append({"role": "user", "content": first_user})

        return self

    @property
    def history(self) -> list[dict[str, str]]:
        """Return the accumulated chat history."""
        return list(self._history)

    def add_history(self, role: str, text: str) -> None:
        """Queue a history message for the next run."""
        self.messages.append({"role": role, "content": text})
        self._history.append({"role": role, "content": text})

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._mcp:
            await self._mcp.__aexit__(exc_type, exc, tb)

    def _get_mcp_tools(self) -> list[ToolParam]:
        if not self._mcp or not hasattr(self._mcp.session, "list_tools"):
            return []
        tools = []
        for tool in self._mcp.session.list_tools():
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema
                    or {"type": "object", "properties": {}, "required": []},
                }
            )
        return tools

    async def run(self, prompt: str) -> str:
        """Run ``prompt`` through Claude and return the result."""
        if self.client is None or self._mcp is None:
            raise RuntimeError("AgentSession not initialized")

        self.messages.append({"role": "user", "content": prompt})
        self._history.append({"role": "user", "content": prompt})

        self._callback.emit(
            {
                "event": "start",
                "prompt": prompt,
                "persona": self.persona,
                "model": self.model,
            }
        )

        tools = self._get_mcp_tools()

        while True:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_instruction,
                messages=self.messages,
                tools=tools if tools else None,
            )

            tool_uses = []
            final_text = ""
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    final_text += block.text
                elif getattr(block, "type", None) == "tool_use":
                    tool_uses.append(block)

            self.messages.append({"role": "assistant", "content": response.content})

            if not tool_uses:
                self._callback.emit({"event": "finish", "result": final_text})
                self._history.append({"role": "assistant", "content": final_text})
                return final_text

            results = []
            for tool_use in tool_uses:
                assert self.tool_executor
                result = await self.tool_executor.execute_tool(tool_use)
                results.append(result)

            self.messages.append({"role": "user", "content": results})


async def run_prompt(
    prompt: str,
    *,
    model: str = CLAUDE_OPUS_4,
    max_tokens: int = 4096,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Convenience helper to run ``prompt`` with a temporary session."""
    async with AgentSession(
        model, max_tokens=max_tokens, on_event=on_event, persona=persona
    ) as ag:
        return await ag.run(prompt)


async def main_async() -> None:
    """Main async entry point."""
    parser = argparse.ArgumentParser(description="Sunstone Claude Agent CLI")
    parser.add_argument(
        "task_file",
        nargs="?",
        help="Path to .txt file with request, '-' for stdin or omit for interactive",
    )
    parser.add_argument("--model", default=CLAUDE_OPUS_4, help="Claude model to use")
    parser.add_argument(
        "--max-tokens", type=int, default=4096, help="Maximum tokens for final response"
    )
    parser.add_argument(
        "-q", "--query", help="Direct prompt text for single query mode"
    )
    parser.add_argument(
        "-o", "--out", help="File path to write final result or error to"
    )
    parser.add_argument(
        "-p", "--persona", default="default", help="Persona instructions to load"
    )

    args = setup_cli(parser)
    out_path = args.out

    app_logger = setup_logging(args.verbose)
    event_writer = JSONEventWriter(out_path)
    journal_writer = JournalEventWriter()

    def emit_event(data: dict) -> None:
        event_writer.emit(data)
        journal_writer.emit(data)

    if args.query:
        user_prompt = args.query
    elif args.task_file is None:
        user_prompt = None
    elif args.task_file == "-":
        user_prompt = sys.stdin.read()
    else:
        if not os.path.isfile(args.task_file):
            parser.error(f"Task file not found: {args.task_file}")
        app_logger.info("Loading task file %s", args.task_file)
        user_prompt = Path(args.task_file).read_text(encoding="utf-8")

    try:
        async with AgentSession(
            model=args.model,
            max_tokens=args.max_tokens,
            on_event=emit_event,
            persona=args.persona,
        ) as agent_session:
            if user_prompt is None:
                app_logger.info("Starting interactive mode with model %s", args.model)
                try:
                    while True:
                        try:
                            loop = asyncio.get_event_loop()
                            prompt = await loop.run_in_executor(
                                None, lambda: input("chat> ")
                            )
                            if not prompt:
                                continue
                            await agent_session.run(prompt)
                        except EOFError:
                            break
                except KeyboardInterrupt:
                    pass
            else:
                app_logger.debug("Task contents: %s", user_prompt)
                app_logger.info("Running agent with model %s", args.model)
                await agent_session.run(user_prompt)
    except Exception as exc:
        emit_event({"event": "error", "error": str(exc)})
        raise
    finally:
        event_writer.close()
        journal_writer.close()


def main() -> None:
    """Entry point that runs the async main."""
    try:
        asyncio.run(main_async())
    except Exception as e:
        logging.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
