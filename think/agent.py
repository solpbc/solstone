#!/usr/bin/env python3
"""think.agent

CLI utility launching an OpenAI agent able to search ponder summaries,
occurrences and read full markdown files from the journal.

Usage:
    think-agent [TASK_FILE] [--model MODEL] [--max-tokens N] [-v]

When ``TASK_FILE`` is omitted, an interactive ``chat>`` prompt is started.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from agents import Agent, ModelSettings, RunConfig, Runner, set_default_openai_key
from agents.mcp import MCPServerStdio

from think.utils import agent_instructions, setup_cli


async def main_async():
    """Main async entry point."""
    parser = argparse.ArgumentParser(description="Sunstone Agent CLI")
    parser.add_argument(
        "task_file",
        nargs="?",
        help="Path to .txt file with the request, '-' for stdin or omit for interactive",
    )
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

    # Set OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        set_default_openai_key(api_key)

    # Get task/prompt
    if args.task_file is None:
        user_prompts = None  # Interactive mode
    elif args.task_file == "-":
        user_prompts = [sys.stdin.read()]
    else:
        if not os.path.isfile(args.task_file):
            parser.error(f"Task file not found: {args.task_file}")
        logging.info("Loading task file %s", args.task_file)
        user_prompts = [Path(args.task_file).read_text(encoding="utf-8")]

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
    run_config = RunConfig(
        model=args.model,
        model_settings=ModelSettings(max_tokens=args.max_tokens, temperature=0.2),
    )

    # Connect to MCP server and run
    async with mcp_server:
        # Create agent with connected server
        agent = Agent(
            name="SunstoneCLI",
            instructions=agent_instructions(),
            mcp_servers=[mcp_server],
        )

        if user_prompts is None:
            # Interactive mode
            logging.info("Starting interactive mode with model %s", args.model)
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

                        result = await Runner.run(agent, prompt, run_config=run_config)
                        print(result.final_output)

                    except EOFError:
                        break
            except KeyboardInterrupt:
                pass
        else:
            # Single prompt mode
            user_prompt = user_prompts[0]
            logging.debug("Task contents: %s", user_prompt)
            logging.info("Running agent with model %s", args.model)

            result = await Runner.run(agent, user_prompt, run_config=run_config)
            print(result.final_output)


def main():
    """Entry point that runs the async main."""
    try:
        asyncio.run(main_async())
    except Exception as e:
        logging.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
