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
from pathlib import Path

from agents import (
    Agent,
    ModelSettings,
    RunConfig,
    Runner,
    function_tool,
    set_default_openai_key,
)

from think.indexer import search_occurrences, search_ponders
from think.utils import setup_cli


@function_tool(name="search_ponder", description="Full-text search over ponder summaries")
def tool_search_ponder(query: str) -> str:
    """Search ponder sentences using the journal index."""

    journal = os.getenv("JOURNAL_PATH", "")
    _total, results = search_ponders(journal, query, 5)
    return "\n".join(r["text"] for r in results)


@function_tool(name="search_occurrences", description="Search structured occurrences by keyword")
def tool_search_occurrences(query: str) -> str:
    """Search occurrences using the journal index."""

    journal = os.getenv("JOURNAL_PATH", "")
    results = search_occurrences(journal, query, 5)
    lines = []
    for r in results:
        meta = r.get("metadata", {})
        lines.append(f"{meta.get('day')} {meta.get('type')}: {r['text']}")
    return "\n".join(lines)


@function_tool(
    name="read_markdown", description="Read an entire Markdown file for a given date and filename"
)
def tool_read_markdown(date: str, filename: str) -> str:
    """Return markdown contents from ``journal/YYYYMMDD/filename.md``."""

    md_path = Path("journal") / date / f"{filename}.md"
    if not md_path.is_file():
        raise FileNotFoundError(f"Markdown not found: {md_path}")
    return md_path.read_text(encoding="utf-8")


def build_agent(model: str, max_tokens: int) -> tuple[Agent, RunConfig]:
    """Return configured OpenAI agent and run configuration."""

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        set_default_openai_key(api_key)

    run_config = RunConfig(
        model=model,
        model_settings=ModelSettings(max_tokens=max_tokens, temperature=0.2),
    )

    agent = Agent(
        name="SunstoneCLI",
        instructions=(
            "You are the Sunstone journal assistant. Use the provided tools "
            "to search ponder summaries, search occurrences, and read markdown files. "
            "When answering, always mention which tool was used."
        ),
        tools=[tool_search_ponder, tool_search_occurrences, tool_read_markdown],
    )
    return agent, run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Sunstone Agent CLI")
    parser.add_argument("task_file", help="Path to .txt file with the request")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for the final response",
    )

    args = setup_cli(parser)

    if not os.path.isfile(args.task_file):
        parser.error(f"Task file not found: {args.task_file}")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("openai.agents").setLevel(logging.DEBUG)

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
