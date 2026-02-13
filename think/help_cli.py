# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI command for interactive help with sol commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from think.utils import setup_cli


def main() -> None:
    """Entry point for ``sol help``."""
    parser = argparse.ArgumentParser(
        prog="sol help",
        description="Get help with sol commands",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question about sol commands",
    )

    args = setup_cli(parser)

    if not args.question:
        # Imported here to avoid circular import (sol.py imports think.help_cli).
        from sol import print_help

        print_help()
        return

    question = " ".join(args.question).strip()
    config = {"name": "help", "prompt": question}
    config_json = json.dumps(config)

    try:
        result = subprocess.run(
            ["sol", "agents"],
            input=config_json + "\n",
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("Error: help request timed out after 120 seconds.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: failed to run help agent: {exc}", file=sys.stderr)
        sys.exit(1)

    finish_result: str | None = None
    errors: list[str] = []

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("event")
        if event_type == "error":
            errors.append(str(event.get("error", "Unknown error")))
        elif event_type == "finish":
            # Keep the last finish result seen in the stream.
            result_value = event.get("result")
            finish_result = "" if result_value is None else str(result_value)

    for message in errors:
        print(f"Error: {message}", file=sys.stderr)

    if finish_result is not None and finish_result.strip():
        print(finish_result)
        return

    if finish_result is not None:
        print("Error: help agent returned an empty result.", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0 and result.stderr.strip():
        print(f"Error: {result.stderr.strip()}", file=sys.stderr)
    else:
        print("Error: no help response received.", file=sys.stderr)
    sys.exit(1)
