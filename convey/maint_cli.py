# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI for managing maintenance tasks.

Usage:
    sol maint                    # Run pending tasks
    sol maint --list             # Show status of all tasks
    sol maint <task>             # Show task details and log output
    sol maint --force <task>     # Re-run a specific task
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from think.utils import get_journal, setup_cli

from .maint import (
    get_state_file,
    get_task_by_name,
    get_task_status,
    list_tasks,
    run_pending_tasks,
    run_task,
)


def show_task_details(journal: Path, task_name: str) -> None:
    """Show details and log output for a maintenance task."""
    task = get_task_by_name(task_name)
    if not task:
        print(f"Task not found: {task_name}", file=sys.stderr)
        print("Use 'sol maint --list' to see available tasks.", file=sys.stderr)
        sys.exit(1)

    status, exit_code, ran_ts = get_task_status(journal, task.app, task.name)
    state_file = get_state_file(journal, task.app, task.name)

    duration_ms = None
    log_lines: list[str] = []
    errors: list[str] = []
    if status != "pending" and state_file.exists():
        with open(state_file, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("event")
                if event_type == "line":
                    text = event.get("line")
                    if isinstance(text, str):
                        log_lines.append(text)
                elif event_type == "exit":
                    if isinstance(event.get("duration_ms"), int):
                        duration_ms = event["duration_ms"]
                    if event.get("error"):
                        errors.append(str(event["error"]))

    print(task.qualified_name)
    if task.description:
        print(task.description)

    if status == "pending":
        print("Status: pending")
    elif status == "success":
        print("Status: success (exit 0)")
    elif exit_code is None:
        print("Status: failed")
    else:
        print(f"Status: failed (exit {exit_code})")

    if ran_ts is not None:
        ts_str = datetime.fromtimestamp(ran_ts / 1000).strftime("%Y-%m-%d %H:%M")
        if duration_ms is not None:
            print(f"Ran: {ts_str} ({duration_ms}ms)")
        else:
            print(f"Ran: {ts_str}")

    if state_file.exists():
        print(f"Log: {state_file}")

    print()

    if status == "pending":
        print("Task has not been run yet.")
        return

    for line in log_lines:
        print(line)

    for error in errors:
        print(f"Error: {error}")


def main() -> None:
    """CLI entry point for sol maint command."""
    parser = argparse.ArgumentParser(
        description="Run maintenance tasks for apps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    sol maint              Run all pending maintenance tasks
    sol maint --list       Show status of all tasks
    sol maint chat:fix_x   Show task details and log output
    sol maint -f fix_x     Re-run a specific task
""",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Task to show details for (or to re-run with --force)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all tasks with their status",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-run a specific task (requires task name)",
    )

    args = setup_cli(parser)
    journal = Path(get_journal())

    # List mode
    if args.list:
        tasks = list_tasks(journal)
        if not tasks:
            print("No maintenance tasks found.")
            return

        # Group by status
        pending = [t for t in tasks if t["status"] == "pending"]
        success = [t for t in tasks if t["status"] == "success"]
        failed = [t for t in tasks if t["status"] == "failed"]

        def print_task(t: dict) -> None:
            desc = f" - {t['description']}" if t["description"] else ""
            exit_info = ""
            if t["exit_code"] is not None and t["exit_code"] != 0:
                exit_info = f" (exit {t['exit_code']})"
            print(f"  {t['qualified_name']}{desc}{exit_info}")
            if t.get("ran_ts"):
                ts_str = datetime.fromtimestamp(t["ran_ts"] / 1000).strftime(
                    "%Y-%m-%d %H:%M"
                )
                print(f"    ran {ts_str}  log: {t['state_file']}")

        if pending:
            print(f"Pending ({len(pending)}):")
            for t in pending:
                print_task(t)

        if failed:
            print(f"Failed ({len(failed)}):")
            for t in failed:
                print_task(t)

        if success:
            print(f"Completed ({len(success)}):")
            for t in success:
                print_task(t)

        return

    # Force re-run a specific task
    if args.force:
        if not args.task:
            print("--force requires a task name.", file=sys.stderr)
            print("Usage: sol maint --force <task>", file=sys.stderr)
            sys.exit(1)
        task = get_task_by_name(args.task)
        if not task:
            print(f"Task not found: {args.task}", file=sys.stderr)
            print("Use 'sol maint --list' to see available tasks.", file=sys.stderr)
            sys.exit(1)
        success, exit_code = run_task(journal, task)
        sys.exit(0 if success else exit_code)

    # Show task details
    if args.task:
        show_task_details(journal, args.task)
        return

    # Run pending tasks
    ran, succeeded = run_pending_tasks(journal)
    if ran == 0:
        print("No pending maintenance tasks.")
    else:
        print(f"Completed {succeeded}/{ran} task(s)")
        sys.exit(0 if succeeded == ran else 1)


if __name__ == "__main__":
    main()
