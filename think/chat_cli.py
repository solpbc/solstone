# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI command for chatting with the journal agent."""

from __future__ import annotations

import argparse
import sys
import threading

from think.callosum import CallosumConnection
from think.cortex_client import cortex_request, read_agent_events
from think.utils import setup_cli


def main() -> None:
    """Entry point for ``sol chat``."""
    parser = argparse.ArgumentParser(
        prog="sol chat",
        description="Chat with your journal",
    )
    parser.add_argument("message", nargs="*", help="Chat message")
    parser.add_argument("--facet", help="Facet context")
    parser.add_argument("--provider", help="AI provider override")
    parser.add_argument(
        "--talent", default="unified", help="Talent agent name (default: unified)"
    )
    args = setup_cli(parser)

    from think.awareness import ensure_sol_directory

    ensure_sol_directory()

    if not args.message:
        parser.print_help()
        return

    message = " ".join(args.message).strip()

    config: dict[str, str] = {}
    if args.facet:
        config["facet"] = args.facet

    agent_id = cortex_request(
        prompt=message,
        name=args.talent,
        provider=args.provider,
        config=config if config else None,
    )
    if agent_id is None:
        print(
            "Error: failed to connect to cortex (is the stack running?)",
            file=sys.stderr,
        )
        sys.exit(1)

    result: dict[str, str] = {}
    done = threading.Event()
    listener = CallosumConnection()

    def on_event(msg: dict) -> None:
        if msg.get("tract") != "cortex":
            return
        if msg.get("agent_id") != agent_id:
            return

        event_type = msg.get("event")
        if event_type == "start":
            if args.verbose:
                print(
                    f"Agent started (model={msg.get('model')}, provider={msg.get('provider')})",
                    file=sys.stderr,
                )
        elif event_type == "thinking":
            if args.verbose:
                print(
                    f"Thinking: {msg.get('summary', '')[:200]}",
                    file=sys.stderr,
                )
        elif event_type == "tool_start":
            if args.verbose:
                print(f"Tool: {msg.get('tool', 'unknown')}", file=sys.stderr)
        elif event_type == "tool_end":
            if args.verbose:
                print(f"Tool done: {msg.get('tool', '')}", file=sys.stderr)
        elif event_type == "finish":
            result["text"] = msg.get("result", "")
            done.set()
        elif event_type == "error":
            result["error"] = msg.get("error", "Unknown error")
            done.set()

    listener.start(callback=on_event)

    if not args.verbose:
        print("Thinking...", end="", file=sys.stderr, flush=True)

    try:
        done.wait(timeout=600)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        listener.stop()
        sys.exit(1)

    listener.stop()

    if not args.verbose:
        print("\r            \r", end="", file=sys.stderr, flush=True)

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if "text" in result and result["text"].strip():
        print(result["text"])
        return

    if "text" in result:
        print("Error: agent returned an empty result.", file=sys.stderr)
        sys.exit(1)

    try:
        events = read_agent_events(agent_id)
        for event in reversed(events):
            event_type = event.get("event")
            if event_type == "finish":
                text = event.get("result", "")
                if str(text).strip():
                    print(text)
                    return
                print("Error: agent returned an empty result.", file=sys.stderr)
                sys.exit(1)
            if event_type == "error":
                print(
                    f"Error: {event.get('error', 'Unknown error')}",
                    file=sys.stderr,
                )
                sys.exit(1)
    except FileNotFoundError:
        pass

    print("Error: request timed out.", file=sys.stderr)
    sys.exit(1)
