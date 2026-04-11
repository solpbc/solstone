# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unified CLI for solstone - AI-driven desktop journaling toolkit.

Usage:
    sol                     Show status and available commands
    sol <command> [args]    Run a subcommand
    sol <module> [args]     Run by module path (e.g., sol think.importers.cli)

Examples:
    sol import data.json    Import data into journal
    sol dream 20250101      Run daily processing for a day
    sol think.agents -h     Show help for specific module
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import setproctitle

# =============================================================================
# Command Registry
# =============================================================================
# Maps short command names to module paths.
# All modules must have a main() function as entry point.
#
# To add a new command:
#   1. Add entry here: "name": "package.module"
#   2. Ensure module has main() function
#
# Aliases for compound commands can be added to ALIASES dict below.
# =============================================================================

COMMANDS: dict[str, str] = {
    # think package - daily processing and analysis
    "import": "think.importers.cli",
    "dream": "think.dream",
    "planner": "think.planner",
    "indexer": "think.indexer",
    "supervisor": "think.supervisor",
    "schedule": "think.scheduler",
    "detect-created": "think.detect_created",
    "top": "think.top",
    "health": "think.health_cli",
    "callosum": "think.callosum",
    "notify": "think.notify_cli",
    "password": "think.password_cli",
    "streams": "think.streams",
    "segment": "think.segment",
    "journal-stats": "think.journal_stats",
    "config": "think.config_cli",
    "formatter": "think.formatters",
    # observe package - multimodal capture
    "transcribe": "observe.transcribe",
    "describe": "observe.describe",
    "sense": "observe.sense",
    "sync": "observe.sync",
    "transfer": "observe.transfer",
    "observer": "observe.observer_cli",
    # AI agents (talent package)
    "agents": "think.agents",
    "cortex": "think.cortex",
    "talent": "think.talent_cli",
    "call": "think.call",
    "engage": "think.engage",
    "help": "think.help_cli",
    "chat": "think.chat_cli",
    "heartbeat": "think.heartbeat",
    # convey package - web UI
    "convey": "convey.cli",
    "restart-convey": "convey.restart",
    "screenshot": "convey.screenshot",
    "maint": "convey.maint_cli",
    "service": "think.service",
}

# =============================================================================
# Aliases for Compound Commands
# =============================================================================
# Maps alias names to (module, default_args) tuples.
# These provide shortcuts for common operations with preset arguments.
#
# Example: "reindex": ("think.indexer", ["--rescan"])
#   Running "sol reindex" is equivalent to "sol indexer --rescan"
# =============================================================================

ALIASES: dict[str, tuple[str, list[str]]] = {
    "start": ("think.supervisor", []),
    "up": ("think.service", ["up"]),
    "down": ("think.service", ["down"]),
}

# Command groupings for help display
GROUPS: dict[str, list[str]] = {
    "Think (daily processing)": [
        "import",
        "dream",
        "planner",
        "indexer",
        "supervisor",
        "schedule",
        "top",
        "health",
        "callosum",
        "notify",
        "heartbeat",
    ],
    "Service": ["service"],
    "Observe (capture)": [
        "transcribe",
        "describe",
        "sense",
        "sync",
        "transfer",
        "observer",
    ],
    "Talent (AI agents)": [
        "agents",
        "cortex",
        "talent",
        "engage",
    ],
    "Convey (web UI)": [
        "convey",
        "restart-convey",
        "screenshot",
        "maint",
    ],
    "Specialized tools": [
        "config",
        "password",
        "streams",
        "segment",
        "journal-stats",
        "formatter",
        "detect-created",
    ],
    "Help": ["help", "chat"],
}


def get_status() -> dict[str, Any]:
    """Return current journal status information."""
    from think.utils import get_journal_info

    path, source = get_journal_info()

    return {
        "journal_path": path,
        "journal_source": source,
        "journal_exists": os.path.isdir(path),
    }


def print_status() -> None:
    """Print current journal status."""
    status = get_status()

    print(f"Journal: {status['journal_path']}")
    if status["journal_exists"]:
        # Count day directories
        journal = status["journal_path"]
        days = [
            d
            for d in os.listdir(journal)
            if os.path.isdir(os.path.join(journal, d)) and d.isdigit() and len(d) == 8
        ]
        print(f"Days: {len(days)}")
    print()


def print_help() -> None:
    """Print help with status and available commands."""
    print("sol - solstone unified CLI\n")
    print_status()

    print("Usage: sol <command> [args...]\n")

    # Print grouped commands
    for group_name, commands in GROUPS.items():
        print(f"{group_name}:")
        for cmd in commands:
            if cmd in COMMANDS:
                module = COMMANDS[cmd]
                print(f"  {cmd:16} {module}")
        print()

    # Print call sub-apps
    try:
        from think.call import call_app

        print("Apps (sol call <app>):")
        for group in call_app.registered_groups:
            name = group.name or ""
            help_text = group.typer_instance.info.help
            if not isinstance(help_text, str):
                help_text = ""
            print(f"  call {name:16} {help_text}")
        print()
    except Exception:
        pass

    # Print aliases if any
    if ALIASES:
        print("Aliases:")
        for alias, (module, args) in ALIASES.items():
            args_str = " ".join(args) if args else ""
            print(f"  {alias:16} → {module} {args_str}")
        print()

    print("Direct module syntax: sol <module.path> [args]")
    print("Example: sol think.importers.cli --help")


def resolve_command(name: str) -> tuple[str, list[str]]:
    """Resolve command name to module path and any preset args.

    Args:
        name: Command name, alias, or module path

    Returns:
        Tuple of (module_path, preset_args)

    Raises:
        ValueError: If command not found
    """
    # Check aliases first (they override commands)
    if name in ALIASES:
        module, preset_args = ALIASES[name]
        return module, preset_args

    # Check command registry
    if name in COMMANDS:
        return COMMANDS[name], []

    # Check if it looks like a module path (contains ".")
    if "." in name:
        return name, []

    # Not found
    available = sorted(set(COMMANDS.keys()) | set(ALIASES.keys()))
    raise ValueError(
        f"Unknown command: {name}\nAvailable commands: {', '.join(available[:10])}..."
    )


def run_command(module_path: str) -> int:
    """Import and run a module's main() function.

    Args:
        module_path: Dotted module path (e.g., "think.importers.cli")

    Returns:
        Exit code (0 for success)
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        print(f"Error: Could not import module '{module_path}': {e}", file=sys.stderr)
        return 1

    if not hasattr(module, "main"):
        print(f"Error: Module '{module_path}' has no main() function", file=sys.stderr)
        return 1

    # Call main - it may call sys.exit() internally
    try:
        module.main()
        return 0
    except SystemExit as e:
        # Preserve exit code from subcommand
        # SystemExit can have int code, string message, or None
        if isinstance(e.code, int):
            return e.code
        elif isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 1
        else:
            return 0 if not e.code else 1


def main() -> None:
    """Main entry point for sol CLI."""
    # No arguments - show status and help
    if len(sys.argv) < 2:
        print_help()
        return

    cmd = sys.argv[1]

    # Help flags
    if cmd in ("--help", "-h"):
        print_help()
        return
    if cmd == "help" and len(sys.argv) <= 2:
        print_help()
        return

    # Version flag
    if cmd in ("--version", "-V"):
        print("sol (solstone) 0.1.0")
        return

    # Path flag
    if cmd == "--path":
        from think.utils import get_journal_info

        path, _source = get_journal_info()
        print(path)
        return

    # Root command — print repo root for scripting: SOL=$(sol root)
    if cmd == "root":
        from think.utils import get_project_root

        print(get_project_root())
        return

    # Resolve command to module path
    try:
        module_path, preset_args = resolve_command(cmd)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Set process title for ps/top visibility
    setproctitle.setproctitle(f"sol:{cmd}")

    # Adjust sys.argv for the subcommand
    # Original: ["sol", "import", "--day", "20250101"]
    # Becomes:  ["sol import", "--day", "20250101"]
    # This makes argparse show "usage: sol import ..." in help
    remaining_args = sys.argv[2:]
    sys.argv = [f"sol {cmd}"] + preset_args + remaining_args

    # Run the command
    exit_code = run_command(module_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
