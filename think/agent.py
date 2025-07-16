#!/usr/bin/env python3
"""think.agent

CLI utility launching an OpenAI agent able to search ponder summaries,
occurrences and read full markdown files from the journal.

Usage:
    python -m think.agent path/to/task.txt [--model MODEL] [--max-tokens N] [-v]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from agents import Agent, ModelSettings, RunConfig, Runner, set_default_openai_key

try:  # Optional import to avoid heavy dependencies during tests
    from think.mcp_server import read_markdown, search_occurrence, search_ponder
except Exception:  # pragma: no cover - replaced in tests

    def search_ponder(*_args, **_kwargs):
        raise NotImplementedError

    def search_occurrence(*_args, **_kwargs):
        raise NotImplementedError

    def read_markdown(*_args, **_kwargs):
        raise NotImplementedError


from think.utils import setup_cli

# Aliases for backwards compatibility with tests
tool_search_ponder = search_ponder
tool_search_occurrences = search_occurrence
tool_read_markdown = read_markdown


def build_agent(model: str, max_tokens: int) -> tuple[Agent, RunConfig]:
    """Return configured OpenAI agent and run configuration."""

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        set_default_openai_key(api_key)

    run_config = RunConfig(
        model=model,
        model_settings=ModelSettings(max_tokens=max_tokens, temperature=0.2),
    )

    try:
        from agents.mcp import MCPServerStdio
    except Exception:  # pragma: no cover - missing in test stubs
        MCPServerStdio = None

    mcp_servers = []
    if MCPServerStdio is not None:
        mcp_servers.append(
            MCPServerStdio(command="python", args=["think/mcp_server.py", "--stdio"])
        )

    agent = Agent(
        name="SunstoneCLI",
        instructions=(
            "You are the Sunstone journal assistant. Use the provided tools to "
            "search ponder summaries, search occurrences, and read markdown files."
        ),
        mcp_servers=mcp_servers,
    )
    return agent, run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Sunstone Agent CLI")
    parser.add_argument("task_file", help="Path to .txt file with the request, or '-' for stdin")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for the final response",
    )

    args = setup_cli(parser)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("openai.agents").setLevel(logging.DEBUG)

    if args.task_file == "-":
        user_prompt = sys.stdin.read()
    else:
        if not os.path.isfile(args.task_file):
            parser.error(f"Task file not found: {args.task_file}")

        logging.info("Loading task file %s", args.task_file)
        user_prompt = Path(args.task_file).read_text(encoding="utf-8")

    logging.debug("Task contents: %s", user_prompt)

    logging.info("Building agent with model %s", args.model)
    agent, run_config = build_agent(args.model, args.max_tokens)

    logging.info("Running agent")
    result = Runner.run_sync(agent, user_prompt, run_config=run_config)
    print(result.final_output)


if __name__ == "__main__":
    main()
