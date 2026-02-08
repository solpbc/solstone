# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import argparse
import os
import sys
from pathlib import Path

from .prompts import load_prompt
from .utils import setup_cli


def _load_prompt() -> str:
    """Return system instruction text for planning."""
    prompt_content = load_prompt("planner", base_dir=Path(__file__).parent)
    return prompt_content.text


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
