# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI for journal source management.

Provides commands for creating, listing, revoking, and checking status
of journal source registrations for remote journal data import.

Usage:
    sol import journal-source create <name>
    sol import journal-source list
    sol import journal-source revoke <name>
    sol import journal-source status [name]
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from importlib import import_module
from pathlib import Path

from think.utils import get_journal, now_ms, setup_cli


def _fmt_time(ts: int | None) -> str:
    if ts is None:
        return "never"
    dt = datetime.datetime.fromtimestamp(ts / 1000)
    return dt.strftime("%Y-%m-%d %H:%M")


def _journal_sources():
    return import_module("apps.import.journal_sources")


def cmd_create(args: argparse.Namespace) -> int:
    from apps.utils import log_app_action

    journal_sources = _journal_sources()
    name = args.name

    if not journal_sources.is_valid_journal_source_name(name):
        print(f"Error: invalid journal source name '{name}'", file=sys.stderr)
        return 1

    if journal_sources.find_journal_source_by_name(name):
        print(f"Error: journal source '{name}' already exists", file=sys.stderr)
        return 1

    key = journal_sources.generate_key()
    source_data = {
        "key": key,
        "name": name,
        "created_at": now_ms(),
        "enabled": True,
        "revoked": False,
        "revoked_at": None,
        "stats": {
            "segments_received": 0,
            "entities_received": 0,
            "facets_received": 0,
            "imports_received": 0,
            "config_received": 0,
        },
    }

    if not journal_sources.save_journal_source(source_data):
        print("Error: failed to save journal source", file=sys.stderr)
        return 1

    journal_sources.create_state_directory(Path(get_journal()), key[:8])
    log_app_action(
        app="import",
        facet=None,
        action="journal_source_create",
        params={"name": name, "key_prefix": key[:8]},
    )

    if args.json_output:
        print(json.dumps({"name": name, "key": key, "prefix": key[:8]}))
        return 0

    print("Journal source created:")
    print(f"  Name:       {name}")
    print(f"  Prefix:     {key[:8]}")
    print(f"  api key:     {key}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    journal_sources = _journal_sources()
    sources = journal_sources.list_journal_sources()

    if args.json_output:
        print(
            json.dumps(
                [
                    {
                        "name": source.get("name", ""),
                        "prefix": source.get("key", "")[:8],
                        "status": "revoked" if source.get("revoked") else "active",
                        "created_at": source.get("created_at"),
                    }
                    for source in sources
                ]
            )
        )
        return 0

    if not sources:
        print("No journal sources registered.")
        return 0

    print(f"{'Name':<24} {'Prefix':<10} {'Status':<10} {'Created':<18}")
    print("-" * 66)
    for source in sources:
        print(
            f"{source.get('name', ''):<24} "
            f"{source.get('key', '')[:8]:<10} "
            f"{('revoked' if source.get('revoked') else 'active'):<10} "
            f"{_fmt_time(source.get('created_at')):<18}"
        )
    return 0


def _status_single(name: str, *, json_output: bool = False) -> int:
    journal_sources = _journal_sources()
    source = journal_sources.find_journal_source_by_name(name)
    if not source:
        print(f"Error: journal source '{name}' not found", file=sys.stderr)
        return 1

    key = source.get("key", "")
    prefix = key[:8]
    status = "revoked" if source.get("revoked") else "active"
    state_dir = str(Path(get_journal()) / "imports" / prefix)
    stats = source.get("stats", {})

    if json_output:
        print(
            json.dumps(
                {
                    "name": source.get("name", ""),
                    "prefix": prefix,
                    "status": status,
                    "created_at": source.get("created_at"),
                    "revoked": source.get("revoked", False),
                    "revoked_at": source.get("revoked_at"),
                    "state_dir": state_dir,
                    "stats": stats,
                }
            )
        )
        return 0

    print(f"Journal source: {source.get('name', '')}")
    print(f"  Prefix:     {prefix}")
    print(f"  Status:     {status}")
    print(f"  Created:    {_fmt_time(source.get('created_at'))}")
    if source.get("revoked"):
        print(f"  Revoked at: {_fmt_time(source.get('revoked_at'))}")
    print(f"  State dir:  {state_dir}")
    print("  Stats:")
    print(f"    segments:   {stats.get('segments_received', 0)}")
    print(f"    entities:   {stats.get('entities_received', 0)}")
    print(f"    facets:     {stats.get('facets_received', 0)}")
    print(f"    imports:    {stats.get('imports_received', 0)}")
    print(f"    config:     {stats.get('config_received', 0)}")
    return 0


def _status_all(*, json_output: bool = False) -> int:
    journal_sources = _journal_sources()
    sources = journal_sources.list_journal_sources()

    if json_output:
        print(
            json.dumps(
                [
                    {
                        "name": source.get("name", ""),
                        "prefix": source.get("key", "")[:8],
                        "status": "revoked" if source.get("revoked") else "active",
                        "created_at": source.get("created_at"),
                        "stats": source.get("stats", {}),
                        "state_dir": str(
                            Path(get_journal()) / "imports" / source.get("key", "")[:8]
                        ),
                    }
                    for source in sources
                ]
            )
        )
        return 0

    if not sources:
        print("No journal sources registered.")
        return 0

    print(
        f"{'Name':<20} {'Status':<10} {'Created':<18} "
        f"{'Seg':>5} {'Ent':>5} {'Fac':>5} {'Imp':>5} {'Cfg':>5}"
    )
    print("-" * 82)
    for source in sources:
        stats = source.get("stats", {})
        print(
            f"{source.get('name', ''):<20} "
            f"{('revoked' if source.get('revoked') else 'active'):<10} "
            f"{_fmt_time(source.get('created_at')):<18} "
            f"{stats.get('segments_received', 0):>5} "
            f"{stats.get('entities_received', 0):>5} "
            f"{stats.get('facets_received', 0):>5} "
            f"{stats.get('imports_received', 0):>5} "
            f"{stats.get('config_received', 0):>5}"
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if args.name:
        return _status_single(args.name, json_output=args.json_output)
    return _status_all(json_output=args.json_output)


def cmd_revoke(args: argparse.Namespace) -> int:
    from apps.utils import log_app_action

    journal_sources = _journal_sources()
    source = journal_sources.find_journal_source_by_name(args.name)
    if not source:
        print(f"Error: journal source '{args.name}' not found", file=sys.stderr)
        return 1
    if source.get("revoked"):
        print(f"Journal source '{args.name}' is already revoked.", file=sys.stderr)
        return 1

    name = source.get("name", "")
    prefix = source.get("key", "")[:8]
    source["revoked"] = True
    source["revoked_at"] = now_ms()

    if not journal_sources.save_journal_source(source):
        print("Error: failed to save journal source", file=sys.stderr)
        return 1

    log_app_action(
        app="import",
        facet=None,
        action="journal_source_revoke",
        params={"name": name, "key_prefix": prefix},
    )

    if args.json_output:
        print(json.dumps({"name": name, "prefix": prefix, "revoked": True}))
        return 0

    print(f"Revoked journal source '{name}' ({prefix})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sol import journal-source",
        description="Manage journal source registrations",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Output as JSON"
    )

    sub = parser.add_subparsers(dest="command")

    p_create = sub.add_parser("create", help="Create a new journal source")
    p_create.add_argument("name", help="Name for the journal source")

    sub.add_parser("list", help="List all registered journal sources")

    p_status = sub.add_parser("status", help="Show journal source status details")
    p_status.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Journal source name (omit for overview)",
    )

    p_revoke = sub.add_parser("revoke", help="Revoke a journal source")
    p_revoke.add_argument("name", help="Journal source name")

    args = setup_cli(parser)

    import convey.state

    convey.state.journal_root = get_journal()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "create": cmd_create,
        "list": cmd_list,
        "status": cmd_status,
        "revoke": cmd_revoke,
    }
    sys.exit(handlers[args.command](args))
