import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .models import GEMINI_PRO
from .utils import create_mcp_client, setup_cli

PROMPT_PATH = Path(__file__).with_name("planner.txt")


async def _get_mcp_tools() -> str:
    """Return formatted MCP tools information for the prompt."""
    try:
        mcp = create_mcp_client()
        if not hasattr(mcp, "list_tools"):
            return ""

        tool_list = await mcp.list_tools()
        if not tool_list:
            return ""

        lines = [
            "",
            "## Available Tools",
            "",
            "The following tools are available for use in your plans:",
            "",
        ]

        for tool in tool_list:
            name = tool.name
            description = tool.description or "No description available"
            lines.append(f"**{name}**: {description}")

        return "\n".join(lines)

    except Exception as exc:
        logging.debug("Failed to fetch MCP tools: %s", exc)
        return ""


def _load_prompt() -> str:
    """Return system instruction text for planning."""
    base_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()

    # Try to add MCP tools information
    try:
        tools_info = asyncio.run(_get_mcp_tools())
        if tools_info:
            return base_prompt + "\n" + tools_info
    except Exception as exc:
        logging.debug("Failed to load MCP tools for prompt: %s", exc)

    return base_prompt


def generate_plan(
    request: str, *, api_key: Optional[str] = None, model: str = GEMINI_PRO
) -> str:
    """Return a detailed agent plan for ``request`` using Gemini."""
    if api_key is None:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[request],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_budget=4096),
            system_instruction=_load_prompt(),
        ),
    )
    return response.text


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an agent plan with Gemini")
    parser.add_argument(
        "task",
        nargs="?",
        help="Path to .txt file with the request or '-' for stdin",
    )
    parser.add_argument("-q", "--query", help="Request text directly")
    parser.add_argument("--model", default=GEMINI_PRO, help="Gemini model to use")
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

    plan = generate_plan(request, model=args.model)
    print(plan)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
