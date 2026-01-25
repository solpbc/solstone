# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from .utils import load_prompt, setup_cli


async def _get_mcp_tools() -> str:
    """Return formatted MCP tools information for the prompt."""

    try:
        from think.mcp import mcp

        tools = await mcp.get_tools()
        if not tools:
            return ""

        lines = [
            "",
            "## Available Tools",
            "",
            "The following tools are available for use in your plans:",
            "",
        ]

        for name in sorted(tools.keys()):
            tool = tools[name]
            description = tool.description or "No description available"
            lines.append(f"**{name}**: {description}")

        return "\n".join(lines)
    except Exception as exc:
        logging.debug("Failed to fetch MCP tools: %s", exc)
        return ""


def _load_prompt() -> str:
    """Return system instruction text for planning."""
    prompt_content = load_prompt("planner", base_dir=Path(__file__).parent)
    base_prompt = prompt_content.text

    # Try to add MCP tools information
    try:
        tools_info = asyncio.run(_get_mcp_tools())
        if tools_info:
            return base_prompt + "\n" + tools_info
    except Exception as exc:
        logging.debug("Failed to load MCP tools for prompt: %s", exc)

    return base_prompt


def generate_plan(request: str) -> str:
    """Return a detailed agent plan for ``request`` using configured provider."""
    from think.models import generate

    return generate(
        contents=request,
        context="planner.generate",
        temperature=0.3,
        max_output_tokens=4096,
        thinking_budget=4096,
        system_instruction=_load_prompt(),
    )


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an agent plan using configured provider"
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Path to .txt file with the request or '-' for stdin",
    )
    parser.add_argument("-q", "--query", help="Request text directly")
    return parser


def main() -> None:
    parser = parse_args()
    args = setup_cli(parser)
    if args.query:
        request = args.query
    elif args.task is None:
        parser.error("request not provided")
    elif args.task == "-":
        request = sys.stdin.read()
    else:
        if not os.path.isfile(args.task):
            parser.error(f"File not found: {args.task}")
        request = Path(args.task).read_text(encoding="utf-8")

    plan = generate_plan(request)
    print(plan)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
