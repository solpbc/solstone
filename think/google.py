#!/usr/bin/env python3
"""think.google

CLI utility launching a Gemini agent able to search ponder summaries,
occurrences and read full markdown files from the journal.

Usage:
    think-google [TASK_FILE] [--model MODEL] [--max-tokens N] [-v] [-o OUT]

When ``TASK_FILE`` is omitted, an interactive ``chat>`` prompt is started.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from fastmcp import Client
from google import genai
from google.genai import types

from .agents import (
    BaseAgentSession,
    JournalEventWriter,
    JSONEventCallback,
    JSONEventWriter,
)
from .models import GEMINI_FLASH
from .utils import agent_instructions, create_mcp_client, setup_cli


def setup_logging(verbose: bool) -> logging.Logger:
    """Return app logger configured for ``verbose``."""

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    return logging.getLogger(__name__)


class ToolLoggingHooks:
    """Wrap ``session.call_tool`` to emit events."""

    def __init__(self, writer: JSONEventCallback) -> None:
        self.writer = writer

    def attach(self, session: Any) -> None:
        original = session.call_tool

        async def wrapped(name: str, arguments: dict | None = None, **kwargs) -> Any:
            self.writer.emit({"event": "tool_start", "tool": name, "args": arguments})
            result = await original(name=name, arguments=arguments, **kwargs)
            self.writer.emit({"event": "tool_end", "tool": name, "result": result})
            return result

        session.call_tool = wrapped  # type: ignore[assignment]


class AgentSession(BaseAgentSession):
    """Context manager running Gemini with MCP tools."""

    def __init__(
        self,
        model: str = GEMINI_FLASH,
        *,
        max_tokens: int = 8192,
        on_event: Optional[Callable[[dict], None]] = None,
        persona: str = "default",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._callback = JSONEventCallback(on_event)
        self._mcp: Client | None = None
        self.client: genai.Client | None = None
        self.chat: genai.chats.Chat | None = None
        self.system_instruction = ""
        self._history: list[dict[str, str]] = []
        self.persona = persona

    async def __aenter__(self) -> "AgentSession":
        self._mcp = create_mcp_client("fastmcp")
        await self._mcp.__aenter__()

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=api_key)

        self.system_instruction, first_user, _ = agent_instructions(self.persona)

        ToolLoggingHooks(self._callback).attach(self._mcp.session)
        self.chat = self.client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction
            ),
            history=[types.Content(role="user", parts=[types.Part(text=first_user)])],
        )
        return self

    @property
    def history(self) -> list[dict[str, str]]:
        """Return the accumulated chat history as ``role``/``content`` dicts."""

        return list(self._history)

    def add_history(self, role: str, text: str) -> None:
        """Record a message to the chat history."""
        if self.chat is not None:
            self.chat.record_history(
                types.Content(role=role, parts=[types.Part(text=text)])
            )
        self._history.append({"role": role, "content": text})

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._mcp:
            await self._mcp.__aexit__(exc_type, exc, tb)

    async def run(self, prompt: str) -> str:
        """Run ``prompt`` through Gemini and return the result."""

        if self._mcp is None or self.client is None or self.chat is None:
            raise RuntimeError("AgentSession not initialized")
        self._callback.emit({"event": "start", "prompt": prompt})
        session = self._mcp.session
        cfg = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            tools=[session],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
        )
        response = await asyncio.to_thread(self.chat.send_message, prompt, config=cfg)
        text = response.text
        self._callback.emit({"event": "finish", "result": text})
        self._history.append({"role": "user", "content": prompt})
        self._history.append({"role": "assistant", "content": text})
        return text


async def run_prompt(
    prompt: str,
    *,
    model: str = GEMINI_FLASH,
    max_tokens: int = 8192,
    on_event: Optional[Callable[[dict], None]] = None,
    persona: str = "default",
) -> str:
    """Convenience helper to run ``prompt`` with a temporary :class:`AgentSession`."""

    async with AgentSession(
        model, max_tokens=max_tokens, on_event=on_event, persona=persona
    ) as ag:
        return await ag.run(prompt)


async def main_async():
    """Main async entry point."""

    parser = argparse.ArgumentParser(description="Sunstone Gemini Agent CLI")
    parser.add_argument(
        "task_file",
        nargs="?",
        help="Path to .txt file with the request, '-' for stdin or omit for interactive",
    )
    parser.add_argument("--model", default=GEMINI_FLASH, help="Gemini model to use")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for the final response",
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Direct prompt text for single query mode",
    )
    parser.add_argument(
        "-o",
        "--out",
        help="File path to write the final result or error to",
    )
    parser.add_argument(
        "-p",
        "--persona",
        default="default",
        help="Persona instructions to load",
    )

    args = setup_cli(parser)
    out_path = args.out

    app_logger = setup_logging(args.verbose)
    event_writer = JSONEventWriter(out_path)
    journal_writer = JournalEventWriter()

    def emit_event(data: dict) -> None:
        event_writer.emit(data)
        journal_writer.emit(data)

    # Get task/prompt
    if args.query:
        user_prompt = args.query
    elif args.task_file is None:
        user_prompt = None  # Interactive mode
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
