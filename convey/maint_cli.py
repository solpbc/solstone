# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI for managing maintenance tasks.

Usage:
    sol maint              # Run all pending tasks
    sol maint --list       # Show status of all tasks
    sol maint chat:fix_x   # Run specific task (even if done)
    sol maint --force      # Re-run all tasks ignoring status
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from think.utils import get_journal, setup_cli

from .maint import (
    discover_tasks,
    get_task_by_name,
    list_tasks,
    run_pending_tasks,
    run_task,
)


def main() -> None:
    """CLI entry point for sol maint command."""
    parser = argparse.ArgumentParser(
        description="Run maintenance tasks for apps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    sol maint              Run all pending maintenance tasks
    sol maint --list       Show status of all tasks
    sol maint chat:fix_x   Run specific task (even if completed)
    sol maint --force      Re-run all tasks ignoring completion status
""",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Specific task to run (app:task or just task if unique)",
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
        help="Re-run tasks even if already completed",
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

    # Run specific task
    if args.task:
        task = get_task_by_name(args.task)
        if not task:
            print(f"Task not found: {args.task}", file=sys.stderr)
            print("Use 'sol maint --list' to see available tasks.", file=sys.stderr)
            sys.exit(1)

        success, exit_code = run_task(journal, task)
        sys.exit(0 if success else exit_code)

    # Run pending tasks (or all with --force)
    if args.force:
        # Run all tasks regardless of status
        tasks = discover_tasks()
        if not tasks:
            print("No maintenance tasks found.")
            return

        ran = 0
        succeeded = 0
        for task in tasks:
            ran += 1
            success, _ = run_task(journal, task)
            if success:
                succeeded += 1

        print(f"Completed {succeeded}/{ran} task(s)")
        sys.exit(0 if succeeded == ran else 1)
    else:
        # Run only pending tasks
        ran, succeeded = run_pending_tasks(journal)
        if ran == 0:
            print("No pending maintenance tasks.")
        else:
            print(f"Completed {succeeded}/{ran} task(s)")
            sys.exit(0 if succeeded == ran else 1)


if __name__ == "__main__":
    main()
