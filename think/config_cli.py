# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI for inspecting journal configuration.

Shows the resolved journal configuration as JSON, or prints JOURNAL_PATH with
its source for shell integration.

Usage:
    sol config          Show full config JSON
    sol config env      Show JOURNAL_PATH and source
"""

from __future__ import annotations

import argparse
import json

from think.utils import get_config, get_journal_info, setup_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Show journal configuration")
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.add_parser("env", help="Show journal path and source")

    # Capture journal info BEFORE setup_cli() loads .env
    journal_info = get_journal_info()

    args = setup_cli(parser)

    if args.subcommand == "env":
        path, source = journal_info
        print(f"JOURNAL_PATH={path} (from {source})")
    else:
        config = get_config()
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
