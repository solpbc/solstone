import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .models import GEMINI_PRO
from .utils import setup_cli

PROMPT_PATH = Path(__file__).with_name("planner.txt")


def _load_prompt() -> str:
    """Return system instruction text for planning."""
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


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
