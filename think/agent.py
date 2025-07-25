#!/usr/bin/env python3
"""think.agent

CLI utility launching an OpenAI agent able to search ponder summaries,
occurrences and read full markdown files from the journal.

Usage:
    think-agent [TASK_FILE] [--model MODEL] [--max-tokens N] [-v] [-o OUT]

When ``TASK_FILE`` is omitted, an interactive ``chat>`` prompt is started.
"""

import argparse
import asyncio
from datetime import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import zoneinfo

from agents import Agent, ModelSettings, RunConfig, Runner, set_default_openai_key, SQLiteSession
from agents.mcp import MCPServerStdio

from think.utils import get_topics, setup_cli

AGENT_PATH = Path(__file__).with_name("agent.txt")


def write_output(path: Optional[str], text: str) -> None:
    """Write *text* to *path* if provided."""
    if not path:
        return
    try:
        Path(path).write_text(text, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - display only
        logging.error("Failed to write output to %s: %s", path, exc)


def agent_instructions() -> tuple[str, str]:
    """Return system instruction and initial user message."""

    system_instruction = AGENT_PATH.read_text(encoding="utf-8")

    extra_parts: list[str] = []
    journal = os.getenv("JOURNAL_PATH")
    if journal:
        ent_path = Path(journal) / "entities.md"
        if ent_path.is_file():
            entities = ent_path.read_text(encoding="utf-8").strip()
            if entities:
                extra_parts.append("## Well-Known Entities\n" + entities)

    topics = get_topics()
    if topics:
        lines = ["## Topics", "These are the topics available for use in tool and resource requests:"]
        for name, info in sorted(topics.items()):
            desc = str(info.get("contains", ""))
            lines.append(f"* Topic: `{name}`: {desc}")
        extra_parts.append("\n".join(lines))

    now = datetime.now()

    # Try to get local timezone
    try:
        local_tz = zoneinfo.ZoneInfo(str(now.astimezone().tzinfo))
        now_local = now.astimezone(local_tz)
        time_str = now_local.strftime("%A, %B %d, %Y at %I:%M %p %Z")
    except:
        # Fallback without timezone
        time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
    
    extra_parts.append(f"## Current Date and Time\n{time_str}")

    extra_context = "\n\n".join(extra_parts).strip()
    return system_instruction, extra_context


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

    args = setup_cli(parser)
    out_path = args.out

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("openai.agents").setLevel(logging.DEBUG)

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
        logging.info("Loading task file %s", args.task_file)
        user_prompt = Path(args.task_file).read_text(encoding="utf-8")

    # Configure MCP server
    journal_path = os.getenv("JOURNAL_PATH", "journal")
    mcp_server = MCPServerStdio(
        params={
            "command": sys.executable,
            "args": ["-m", "think.mcp_server"],
            "env": {
                "JOURNAL_PATH": journal_path,
                "PYTHONPATH": os.pathsep.join([os.getcwd()] + sys.path),
            },
        },
        name="Sunstone MCP Server",
    )

    # Configure run settings
    run_config = RunConfig()

    # Connect to MCP server and run
    try:
        async with mcp_server:
            system_instruction, extra_context = agent_instructions()
            # Create agent with connected server and model configuration
            agent = Agent(
                name="SunstoneCLI",
                instructions=f"{system_instruction}\n\n{extra_context}",
                model=args.model,
                model_settings=ModelSettings(max_tokens=args.max_tokens, temperature=0.2),
                mcp_servers=[mcp_server],
            )

            if user_prompt is None:
                # Interactive mode with session management
                logging.info("Starting interactive mode with model %s", args.model)
                session = SQLiteSession("sunstone_cli_session")
                
                try:
                    while True:
                        try:
                            # Use asyncio-friendly input
                            loop = asyncio.get_event_loop()
                            prompt = await loop.run_in_executor(
                                None, lambda: input("chat> ")
                            )

                            if not prompt:
                                continue

                            result = await Runner.run(
                                agent, prompt, session=session, run_config=run_config
                            )
                            print(result.final_output)
                            write_output(out_path, result.final_output)

                        except EOFError:
                            break
                except KeyboardInterrupt:
                    pass
            else:
                # Single prompt mode
                logging.debug("Task contents: %s", user_prompt)
                logging.info("Running agent with model %s", args.model)

                result = await Runner.run(
                    agent, user_prompt, run_config=run_config
                )
                print(result.final_output)
                write_output(out_path, result.final_output)
    except Exception as exc:
        write_output(out_path, str(exc))
        raise


def main():
    """Entry point that runs the async main."""
    try:
        asyncio.run(main_async())
    except Exception as e:
        logging.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
