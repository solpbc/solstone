#!/usr/bin/env python3
"""think.openai

CLI utility launching an OpenAI agent able to search ponder summaries,
occurrences and read full markdown files from the journal.

Usage:
    think-agent [TASK_FILE] [--model MODEL] [--max-tokens N] [-v] [-o OUT]

When ``TASK_FILE`` is omitted, an interactive ``chat>`` prompt is started.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional

from agents import (
    Agent,
    AgentHooks,
    ModelSettings,
    RunConfig,
    Runner,
    SQLiteSession,
    enable_verbose_stdout_logging,
    set_default_openai_key,
)
from agents.mcp import MCPServerStdio

from think.utils import agent_instructions, create_mcp_client, setup_cli

from .agents import (
    BaseAgentSession,
    JournalEventWriter,
    JSONEventCallback,
    JSONEventWriter,
)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration for agent and tools."""

    if verbose:
        enable_verbose_stdout_logging()

        openai_agents_logger = logging.getLogger("openai.agents")
        openai_agents_logger.setLevel(logging.DEBUG)
        if not openai_agents_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            openai_agents_logger.addHandler(handler)

        tracing_logger = logging.getLogger("openai.agents.tracing")
        tracing_logger.setLevel(logging.DEBUG)
        if not tracing_logger.handlers:
            tracing_logger.addHandler(handler)

    app_logger = logging.getLogger(__name__)
    if verbose:
        app_logger.setLevel(logging.DEBUG)
    else:
        app_logger.setLevel(logging.INFO)
    return app_logger


class ToolLoggingHooks(AgentHooks):
    """Hooks forwarding agent events to :class:`JSONEventCallback`."""

    def __init__(self, writer: JSONEventCallback) -> None:
        self.writer = writer

    def on_tool_call_start(self, context, tool_name: str, arguments: dict) -> None:
        self.writer.emit({"event": "tool_start", "tool": tool_name, "args": arguments})

    def on_tool_call_end(self, context, tool_name: str, result) -> None:

        self.writer.emit({"event": "tool_end", "tool": tool_name, "result": result})

    def on_agent_start(self, context, agent_name: str) -> None:
        self.writer.emit({"event": "agent_start", "agent": agent_name})

    def on_agent_end(self, context, agent_name: str, result) -> None:
        self.writer.emit({"event": "agent_end", "agent": agent_name})


class AgentSession(BaseAgentSession):
    """Context manager wrapping the Sunstone agent with optional session reuse."""

    def __init__(
        self,
        model: str = "gpt-4.1",
        *,
        max_tokens: int = 4096,
        on_event: Optional[Callable[[dict], None]] = None,
        persona: str = "default",
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._callback = JSONEventCallback(on_event)
        self.session = SQLiteSession("sunstone_cli_session")
        self.agent: Agent | None = None
        self._mcp: MCPServerStdio | None = None
        self.run_config = RunConfig()
        self._history: list[dict[str, str]] = []
        self._pending_history: list[dict[str, str]] = []
        self.persona = persona

    async def __aenter__(self) -> "AgentSession":
        self._mcp = create_mcp_client()
        await self._mcp.__aenter__()

        system_instruction, extra_context, _ = agent_instructions(self.persona)
        self.agent = Agent(
            name="SunstoneCLI",
            instructions=f"{system_instruction}\n\n{extra_context}",
            model=self.model,
            model_settings=ModelSettings(max_tokens=self.max_tokens, temperature=0.2),
            mcp_servers=[self._mcp],
            hooks=ToolLoggingHooks(self._callback),
        )
        return self

    @property
    def history(self) -> list[dict[str, str]]:
        """Return chat history added and generated during this session."""

        return list(self._history)

    def add_history(self, role: str, text: str) -> None:
        """Queue a history message for the next run."""

        self._pending_history.append({"role": role, "content": text})
        self._history.append({"role": role, "content": text})

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._mcp:
            await self._mcp.__aexit__(exc_type, exc, tb)

    async def run(self, prompt: str) -> str:
        """Run ``prompt`` through the agent and return the result."""

        if self.agent is None or self._mcp is None:
            raise RuntimeError("AgentSession not initialized")

        if self._pending_history:
            await self.session.add_items(self._pending_history)
            self._pending_history.clear()

        self._history.append({"role": "user", "content": prompt})

        self._callback.emit({"event": "start", "prompt": prompt})
        result = await Runner.run(
            self.agent, prompt, session=self.session, run_config=self.run_config
        )
        self._callback.emit({"event": "finish", "result": result.final_output})
        self._history.append({"role": "assistant", "content": result.final_output})
        return result.final_output


async def run_prompt(
    prompt: str,
    *,
    model: str = "gpt-4.1",
    max_tokens: int = 4096,
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
    parser = argparse.ArgumentParser(description="Sunstone Agent CLI")
    parser.add_argument(
        "task_file",
        nargs="?",
        help="Path to .txt file with the request, '-' for stdin or omit for interactive",
    )
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model to use")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
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

    # Set OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        set_default_openai_key(api_key)

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


def main():
    """Entry point that runs the async main."""
    try:
        asyncio.run(main_async())
    except Exception as e:
        logging.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
