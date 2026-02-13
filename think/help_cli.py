# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI command for interactive help with sol commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from think.utils import setup_cli


def _read_stdin() -> str:
    """Read a question from stdin. Shows a prompt if running in a terminal."""
    if sys.stdin.isatty():
        print(
            "Enter your question (Ctrl+D to submit):",
            file=sys.stderr,
        )
    try:
        return sys.stdin.read().strip()
    except KeyboardInterrupt:
        return ""


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
        question = _read_stdin()
        if not question:
            # Imported here to avoid circular import (sol.py imports think.help_cli).
            from sol import print_help

            print_help()
            return
    else:
        question = " ".join(args.question).strip()

    config = {"name": "help", "prompt": question}
    config_json = json.dumps(config)

    print("Thinking...", end="", file=sys.stderr, flush=True)

    try:
        proc = subprocess.Popen(
            ["sol", "agents"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        print(f"\rError: failed to run help agent: {exc}", file=sys.stderr)
        sys.exit(1)

    assert proc.stdin is not None  # for type checker
    proc.stdin.write(config_json + "\n")
    proc.stdin.close()

    finish_result: str | None = None
    errors: list[str] = []

    assert proc.stdout is not None  # for type checker
    for line in proc.stdout:
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
            result_value = event.get("result")
            finish_result = "" if result_value is None else str(result_value)

    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("\rError: help request timed out after 120 seconds.", file=sys.stderr)
        sys.exit(1)

    # Clear the "Thinking..." indicator.
    print("\r            \r", end="", file=sys.stderr, flush=True)

    for message in errors:
        print(f"Error: {message}", file=sys.stderr)

    if finish_result is not None and finish_result.strip():
        print(finish_result)
        return

    if finish_result is not None:
        print("Error: help agent returned an empty result.", file=sys.stderr)
        sys.exit(1)

    stderr_output = proc.stderr.read() if proc.stderr else ""
    if proc.returncode != 0 and stderr_output.strip():
        print(f"Error: {stderr_output.strip()}", file=sys.stderr)
    else:
        print("Error: no help response received.", file=sys.stderr)
    sys.exit(1)
