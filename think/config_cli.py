# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI for inspecting journal configuration.

Shows the resolved journal configuration as JSON, or prints JOURNAL_PATH with
its source for shell integration.

Usage:
    sol config                      Show full config JSON
    sol config env                  Show JOURNAL_PATH and source
    sol config facet rename OLD NEW Rename a facet
"""

from __future__ import annotations

import argparse
import json
import sys

from think.utils import get_config, get_journal_info, setup_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Show journal configuration")
    subparsers = parser.add_subparsers(dest="subcommand")
    subparsers.add_parser("env", help="Show journal path and source")

    # facet subcommand with its own sub-subcommands
    facet_parser = subparsers.add_parser("facet", help="Facet management")
    facet_sub = facet_parser.add_subparsers(dest="facet_action")
    rename_parser = facet_sub.add_parser("rename", help="Rename a facet")
    rename_parser.add_argument("old_name", help="Current facet name")
    rename_parser.add_argument("new_name", help="New facet name")

    # Capture journal info BEFORE setup_cli() loads .env
    journal_info = get_journal_info()

    args = setup_cli(parser)

    if args.subcommand == "env":
        path, source = journal_info
        print(f"JOURNAL_PATH={path} (from {source})")
    elif args.subcommand == "facet":
        if args.facet_action == "rename":
            from think.facets import rename_facet

            try:
                rename_facet(args.old_name, args.new_name)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
        else:
            facet_parser.print_help()
    else:
        config = get_config()
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
