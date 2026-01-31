#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Restart utility for Convey web service."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

from think.callosum import CallosumConnection
from think.utils import read_service_port, setup_cli


def _format_log(timestamp: float, stream: str, line: str) -> str:
    """Format a log line for display."""
    time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
    stream_prefix = "ERR" if stream == "stderr" else "OUT"
    return f"[{time_str}] [{stream_prefix}] {line}"


def wait_for_convey_restart(
    timeout: float = 30.0,
    verbose: bool = False,
) -> tuple[bool, list[tuple[float, str, str]]]:
    """Restart convey service and wait for it to be ready.

    Connects to CALLOSUM, sends restart request to supervisor, and monitors
    events until service successfully restarts or fails.

    Args:
        timeout: Maximum seconds to wait for restart (default: 30.0)
        verbose: If True, print progress messages and stream logs live

    Returns:
        (success, logs): True if service started successfully within timeout,
                        list of (timestamp, stream, line) tuples collected during restart

    Failure conditions:
        - Service restarts twice (crash detected)
        - Timeout expires before successful start
        - Connection to CALLOSUM fails
    """
    state = {
        "start_count": 0,
        "ref": None,
        "logs": [],  # List of (timestamp, stream, line)
        "started": False,
        "failed": False,
        "lock": threading.Lock(),
    }

    def handle_event(message: dict) -> None:
        """Process incoming CALLOSUM messages."""
        tract = message.get("tract")
        event = message.get("event")
        service = message.get("service")

        with state["lock"]:
            # Track supervisor events for convey service
            if tract == "supervisor" and service == "convey":
                if event == "restarting":
                    if verbose:
                        print("Restarting convey service...", file=sys.stderr)

                elif event == "stopped":
                    if verbose:
                        exit_code = message.get("exit_code", "?")
                        print(
                            f"Convey stopped (exit code: {exit_code})", file=sys.stderr
                        )

                elif event == "started":
                    state["start_count"] += 1
                    ref = message.get("ref")
                    pid = message.get("pid")

                    if state["start_count"] == 1:
                        # First start - this is expected
                        state["ref"] = ref
                        state["started"] = True
                        if verbose:
                            print(
                                f"Convey started (pid: {pid}, ref: {ref})",
                                file=sys.stderr,
                            )
                    elif state["start_count"] >= 2:
                        # Second start - service crashed and restarted
                        state["failed"] = True
                        # Always show crash errors
                        print(
                            f"ERROR: Convey crashed and restarted (attempt {state['start_count']})",
                            file=sys.stderr,
                        )

            # Collect and optionally stream logs from convey process
            elif tract == "logs" and event == "line":
                name = message.get("name", "")
                # Match both "convey" service and any convey-related processes
                if "convey" in name.lower():
                    timestamp = message.get("ts", time.time() * 1000) / 1000
                    stream = message.get("stream", "stdout")
                    line = message.get("line", "")
                    state["logs"].append((timestamp, stream, line))
                    # Stream live in verbose mode
                    if verbose:
                        print(_format_log(timestamp, stream, line), file=sys.stderr)

    # Connect to CALLOSUM
    if verbose:
        print("Connecting to Callosum...", file=sys.stderr)
    callosum = CallosumConnection()
    try:
        callosum.start(callback=handle_event)
    except Exception as e:
        print(f"ERROR: Failed to connect to CALLOSUM: {e}", file=sys.stderr)
        return False, []

    # Give connection a moment to establish
    time.sleep(0.2)

    # Send restart request
    if verbose:
        print("Sending restart request to supervisor...", file=sys.stderr)
    if not callosum.emit("supervisor", "restart", service="convey"):
        print("ERROR: Failed to send restart request", file=sys.stderr)
        callosum.stop()
        return False, []

    # Wait for restart with timeout
    if verbose:
        print(f"Waiting for restart (timeout: {timeout}s)...", file=sys.stderr)
    start_time = time.time()
    while time.time() - start_time < timeout:
        with state["lock"]:
            if state["failed"]:
                # Service crashed during restart
                callosum.stop()
                elapsed = time.time() - start_time
                if verbose:
                    print(f"Failed after {elapsed:.1f}s", file=sys.stderr)
                return False, state["logs"]

            if state["started"]:
                # Service restarted successfully
                callosum.stop()
                if verbose:
                    print("Waiting for Flask to bind port...", file=sys.stderr)
                # Small additional delay for Flask to fully bind port
                time.sleep(1.0)
                elapsed = time.time() - start_time
                if verbose:
                    print(f"Restarted in {elapsed:.1f}s", file=sys.stderr)
                return True, state["logs"]

        # Check again in 100ms
        time.sleep(0.1)

    # Timeout expired
    print(f"ERROR: Timeout waiting for convey to restart ({timeout}s)", file=sys.stderr)
    callosum.stop()
    return False, state["logs"]


def main() -> None:
    """CLI entry point for convey restart utility."""
    parser = argparse.ArgumentParser(
        description="Restart the Convey web service via supervisor"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Maximum seconds to wait for restart (default: 30.0)",
    )

    args = setup_cli(parser)

    journal_path = os.environ.get("JOURNAL_PATH", "(not set)")
    print(f"Journal: {journal_path}")
    print(f"Timeout: {args.timeout}s")

    success, logs = wait_for_convey_restart(timeout=args.timeout, verbose=args.verbose)

    if not success:
        print("\nERROR: Convey service failed to restart", file=sys.stderr)
        # Show collected logs on failure (already streamed if verbose)
        if logs and not args.verbose:
            print("\nCollected logs:", file=sys.stderr)
            print("-" * 60, file=sys.stderr)
            for timestamp, stream, line in logs:
                print(_format_log(timestamp, stream, line), file=sys.stderr)
            print("-" * 60, file=sys.stderr)
        sys.exit(1)

    # Success - print the URL with discovered port
    port = read_service_port("convey")
    if port:
        print(f"Convey running at http://localhost:{port}/")
    else:
        print("Convey restarted successfully (port unknown)")


if __name__ == "__main__":
    main()
