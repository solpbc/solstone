#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Muse CLI - Run AI agents directly or through Cortex.

Usage:
    muse "prompt"                       # Run through Cortex (default)
    muse -p default "prompt"            # Use specific persona
    muse --direct -p default "prompt"   # Run directly, bypass Cortex
    echo "prompt" | muse                # Read prompt from stdin
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from typing import Any, Dict, Optional, TextIO

from think.callosum import CallosumConnection
from think.utils import setup_cli

# --- Event Formatting ---


def format_event(event: Dict[str, Any], verbose: bool = False) -> Optional[str]:
    """Format an event for display. Returns None if event should be skipped."""
    event_type = event.get("event")

    if event_type == "thinking":
        summary = event.get("summary", "")
        if not verbose and len(summary) > 200:
            summary = summary[:200] + "..."
        return f"💭 {summary}"

    elif event_type == "tool_start":
        tool = event.get("tool", "unknown")
        args = event.get("args", {})
        if verbose:
            args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
        else:
            # Compact args: truncate long values
            args_parts = []
            for k, v in args.items():
                v_str = repr(v)
                if len(v_str) > 30:
                    v_str = v_str[:27] + "..."
                args_parts.append(f"{k}={v_str}")
            args_str = ", ".join(args_parts)
        return f"🔧 {tool}({args_str})"

    elif event_type == "tool_end":
        tool = event.get("tool", "unknown")
        if verbose:
            result = event.get("result", "")
            if isinstance(result, str) and len(result) > 100:
                result = result[:100] + "..."
            return f"✓ {tool} → {result}"
        return f"✓ {tool}"

    elif event_type == "error":
        error = event.get("error", "Unknown error")
        return f"❌ {error}"

    elif event_type == "start":
        # Skip start events in formatted output
        return None

    elif event_type == "finish":
        # Finish events handled separately (result goes to stdout)
        return None

    return None


def display_event(
    event: Dict[str, Any],
    verbose: bool = False,
    json_output: bool = False,
    file: TextIO = sys.stderr,
) -> None:
    """Display an event to the given file handle."""
    if json_output:
        # JSON mode: all events to stdout
        print(json.dumps(event), file=sys.stdout, flush=True)
    else:
        formatted = format_event(event, verbose)
        if formatted:
            print(formatted, file=file, flush=True)


# --- Config Building ---


def build_agent_config(
    prompt: str,
    persona: str = "default",
    provider: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build merged agent configuration from persona and overrides.

    Mirrors the config building logic in cortex.py for consistency.
    """
    from muse.mcp import get_tools
    from muse.models import resolve_model_for_provider, resolve_provider
    from think.utils import get_agent

    # Load persona configuration
    config = get_agent(persona)

    # Apply any additional overrides
    config.update({k: v for k, v in overrides.items() if v is not None})

    # Set prompt
    config["prompt"] = prompt
    config["persona"] = persona

    # Resolve provider and model from context
    # Context format: agent.{app}.{name} where app="system" for system agents
    if ":" in persona:
        app, name = persona.split(":", 1)
    else:
        app, name = "system", persona
    agent_context = f"agent.{app}.{name}"

    # Resolve default provider and model from context
    default_provider, model = resolve_provider(agent_context)

    # Provider can be overridden by parameter or persona config
    final_provider = provider or config.get("provider") or default_provider

    # If provider was overridden, re-resolve model for that provider
    if final_provider != default_provider:
        model = resolve_model_for_provider(agent_context, final_provider)

    config["provider"] = final_provider
    config["model"] = model

    # Expand tools if it's a string (tool pack name)
    tools_config = config.get("tools")
    if isinstance(tools_config, str):
        pack_names = [p.strip() for p in tools_config.split(",") if p.strip()]
        if not pack_names:
            pack_names = ["default"]

        expanded: list[str] = []
        for pack in pack_names:
            try:
                for tool in get_tools(pack):
                    if tool not in expanded:
                        expanded.append(tool)
            except KeyError:
                # Invalid pack, use default
                for tool in get_tools("default"):
                    if tool not in expanded:
                        expanded.append(tool)

        config["tools"] = expanded

    return config


# --- Cortex Mode ---


def run_via_cortex(
    prompt: str,
    persona: str = "default",
    provider: Optional[str] = None,
    timeout: float = 300.0,
    verbose: bool = False,
    json_output: bool = False,
) -> str:
    """Run agent through Cortex service via Callosum.

    Returns the result text, or raises an exception on error/timeout.
    """
    from muse.cortex_client import cortex_request

    # Submit request to Cortex
    config = {"provider": provider} if provider else {}
    agent_id = cortex_request(prompt, persona, provider=provider, config=config)

    # Track state
    result_text: Optional[str] = None
    error_text: Optional[str] = None
    finished = threading.Event()

    def on_message(message: Dict[str, Any]) -> None:
        nonlocal result_text, error_text

        # Filter for cortex tract and our agent_id
        if message.get("tract") != "cortex":
            return
        if message.get("agent_id") != agent_id:
            return

        event_type = message.get("event")

        # Display the event
        display_event(message, verbose, json_output)

        # Check for terminal events
        if event_type == "finish":
            result_text = message.get("result", "")
            finished.set()
        elif event_type == "error":
            error_text = message.get("error", "Unknown error")
            finished.set()

    # Connect to Callosum and listen for events
    conn = CallosumConnection()
    conn.start(callback=on_message)

    try:
        # Wait for completion with timeout
        if not finished.wait(timeout=timeout):
            raise TimeoutError(f"Agent {agent_id} timed out after {timeout}s")

        if error_text:
            raise RuntimeError(error_text)

        return result_text or ""
    finally:
        conn.stop()


# --- Direct Mode ---


def run_direct(
    prompt: str,
    persona: str = "default",
    provider: Optional[str] = None,
    verbose: bool = False,
    json_output: bool = False,
) -> str:
    """Run agent directly, bypassing Cortex.

    Returns the result text.
    """
    # Build config
    config = build_agent_config(prompt, persona, provider)

    # Determine provider
    provider_name = config.get("provider", "openai")

    # Create event callback
    result_holder: Dict[str, Any] = {"result": None, "error": None}

    def on_event(event: Dict[str, Any]) -> None:
        display_event(event, verbose, json_output)
        if event.get("event") == "finish":
            result_holder["result"] = event.get("result", "")
        elif event.get("event") == "error":
            result_holder["error"] = event.get("error", "Unknown error")

    # Route to appropriate provider using registry
    from muse.providers import PROVIDER_REGISTRY, get_provider_module

    if provider_name in PROVIDER_REGISTRY:
        provider_mod = get_provider_module(provider_name)
    else:
        valid = ", ".join(sorted(PROVIDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown provider: {provider_name!r}. Valid providers: {valid}"
        )

    # Run the agent
    asyncio.run(provider_mod.run_agent(config=config, on_event=on_event))

    if result_holder["error"]:
        raise RuntimeError(result_holder["error"])

    return result_holder["result"] or ""


# --- Main ---


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run AI agents directly or through Cortex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  muse "What time is it?"              Run with default persona through Cortex
  muse -p joke_bot "tell me a joke"    Run with specific persona
  muse --direct "prompt"               Run directly, bypass Cortex
  echo "prompt" | muse                 Read prompt from stdin
  muse --json "prompt" | jq .event     JSON output for scripting
""",
    )

    parser.add_argument("prompt", nargs="?", help="The prompt to send to the agent")
    parser.add_argument(
        "-p", "--persona", default="default", help="Agent persona (default: default)"
    )
    parser.add_argument(
        "-b", "--provider", help="Override provider (openai, anthropic, google)"
    )
    parser.add_argument(
        "--direct", action="store_true", help="Run directly, bypass Cortex"
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSONL events")
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds (Cortex mode, default: 300)",
    )

    # setup_cli adds -v/--verbose
    args = setup_cli(parser)

    # Get prompt from argument or stdin
    prompt = args.prompt
    if not prompt:
        if sys.stdin.isatty():
            parser.error(
                "No prompt provided. Use positional argument or pipe via stdin."
            )
        prompt = sys.stdin.read().strip()
        if not prompt:
            parser.error("Empty prompt from stdin")

    try:
        if args.direct:
            result = run_direct(
                prompt,
                persona=args.persona,
                provider=args.provider,
                verbose=args.verbose,
                json_output=args.json,
            )
        else:
            result = run_via_cortex(
                prompt,
                persona=args.persona,
                provider=args.provider,
                timeout=args.timeout,
                verbose=args.verbose,
                json_output=args.json,
            )

        # Print result to stdout (unless in JSON mode where it's already printed)
        if not args.json and result:
            print(result)

    except TimeoutError as e:
        print(f"❌ Timeout: {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
