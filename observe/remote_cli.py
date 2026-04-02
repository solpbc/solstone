# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI for remote observer management.

Provides commands for creating, listing, revoking, and checking status
of remote observer registrations. Operates directly on the journal
filesystem — no dependency on the Convey web server.

Usage:
    sol remote create <name>           Create a new remote observer
    sol remote list                    List all registered remotes
    sol remote revoke <name-or-prefix> Revoke a remote registration
    sol remote status [name-or-prefix] Show remote status details
"""

from __future__ import annotations

import argparse
import base64
import datetime
import logging
import secrets
import sys

from apps.remote.utils import (
    find_remote_by_name,
    get_hist_dir,
    get_remotes_dir,
    list_remotes,
    load_history,
    save_remote,
)
from apps.utils import log_app_action
from think.utils import now_ms, setup_cli

logger = logging.getLogger(__name__)

# Key: 256 bits = 32 bytes, URL-safe base64 (same as web API)
KEY_BYTES = 32

# Connected threshold: last_seen within 2 minutes (matches web UI)
CONNECTED_THRESHOLD_MS = 2 * 60 * 1000


def _generate_key() -> str:
    """Generate a URL-safe key for remote authentication."""
    return base64.urlsafe_b64encode(secrets.token_bytes(KEY_BYTES)).decode().rstrip("=")


def _find_remote(identifier: str) -> dict | None:
    """Find a remote by name or key prefix."""
    # Try name first
    remote = find_remote_by_name(identifier)
    if remote:
        return remote

    # Try key prefix (file is named <prefix>.json)
    import json

    remotes_dir = get_remotes_dir()
    remote_path = remotes_dir / f"{identifier}.json"
    if remote_path.exists():
        try:
            with open(remote_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return None


def _status_label(remote: dict) -> str:
    """Get human-readable connection status."""
    if remote.get("revoked", False):
        return "revoked"
    last_seen = remote.get("last_seen")
    if last_seen is None:
        return "disconnected"
    if now_ms() - last_seen < CONNECTED_THRESHOLD_MS:
        return "connected"
    return "disconnected"


def _fmt_bytes(n: int) -> str:
    """Format byte count for display."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.1f} GB"


def _fmt_time(ms: int | None) -> str:
    """Format millisecond timestamp for display."""
    if ms is None:
        return "never"
    dt = datetime.datetime.fromtimestamp(ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M")


# === Subcommands ===


def cmd_create(args: argparse.Namespace) -> int:
    """Create a new remote observer registration."""
    name = args.name

    if find_remote_by_name(name):
        print(f"Error: remote '{name}' already exists", file=sys.stderr)
        return 1

    key = _generate_key()
    remote_data = {
        "key": key,
        "name": name,
        "created_at": now_ms(),
        "last_seen": None,
        "last_segment": None,
        "enabled": True,
        "stats": {
            "segments_received": 0,
            "bytes_received": 0,
        },
    }

    if not save_remote(remote_data):
        print("Error: failed to save remote", file=sys.stderr)
        return 1

    log_app_action(
        app="remote",
        facet=None,
        action="observer_create",
        params={"name": name, "key_prefix": key[:8]},
    )

    print("Remote observer created:")
    print(f"  Name:       {name}")
    print(f"  Prefix:     {key[:8]}")
    print(f"  server url:  (set during server configuration)")
    print(f"  api key:     {key}")
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    """List all registered remotes."""
    remotes = list_remotes()

    if not remotes:
        print("No remotes registered.")
        return 0

    print(
        f"{'Name':<20} {'Prefix':<10} {'Status':<14} "
        f"{'Last Seen':<18} {'Segments':>10} {'Bytes':>12}"
    )
    print("-" * 86)

    for r in remotes:
        name = r.get("name", "")
        prefix = r.get("key", "")[:8]
        status = _status_label(r)
        last_seen = _fmt_time(r.get("last_seen"))
        stats = r.get("stats", {})
        segments = stats.get("segments_received", 0)
        bytes_recv = _fmt_bytes(stats.get("bytes_received", 0))
        print(
            f"{name:<20} {prefix:<10} {status:<14} "
            f"{last_seen:<18} {segments:>10} {bytes_recv:>12}"
        )

    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    """Revoke a remote registration (soft-delete)."""
    identifier = args.identifier

    remote = _find_remote(identifier)
    if not remote:
        print(f"Error: remote '{identifier}' not found", file=sys.stderr)
        return 1

    if remote.get("revoked", False):
        print(f"Remote '{remote.get('name')}' is already revoked.", file=sys.stderr)
        return 1

    name = remote.get("name", "")
    key_prefix = remote.get("key", "")[:8]

    remote["revoked"] = True
    remote["revoked_at"] = now_ms()

    if not save_remote(remote):
        print("Error: failed to save remote", file=sys.stderr)
        return 1

    log_app_action(
        app="remote",
        facet=None,
        action="observer_revoke",
        params={"name": name, "key_prefix": key_prefix},
    )

    print(f"Revoked remote '{name}' ({key_prefix})")
    return 0


def cmd_rename(args: argparse.Namespace) -> int:
    """Rename a remote observer (affects future stream names)."""
    identifier = args.identifier
    new_name = args.new_name

    remote = _find_remote(identifier)
    if not remote:
        print(f"Error: remote '{identifier}' not found", file=sys.stderr)
        return 1

    # Check new name isn't taken
    existing = find_remote_by_name(new_name)
    if existing and existing.get("key") != remote.get("key"):
        print(f"Error: remote '{new_name}' already exists", file=sys.stderr)
        return 1

    old_name = remote.get("name", "")
    if old_name == new_name:
        print(f"Remote is already named '{new_name}'.", file=sys.stderr)
        return 1

    key_prefix = remote.get("key", "")[:8]
    remote["name"] = new_name

    if not save_remote(remote):
        print("Error: failed to save remote", file=sys.stderr)
        return 1

    log_app_action(
        app="remote",
        facet=None,
        action="observer_rename",
        params={"old_name": old_name, "new_name": new_name, "key_prefix": key_prefix},
    )

    print(f"Renamed remote '{old_name}' -> '{new_name}' ({key_prefix})")
    print(f"  Future segments will use stream: {new_name}")
    print(f"  Existing segments remain under stream: {old_name}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show remote status details."""
    if args.identifier:
        return _status_single(args.identifier)
    return _status_all()


def _status_single(identifier: str) -> int:
    """Detailed status for a single remote."""
    remote = _find_remote(identifier)
    if not remote:
        print(f"Error: remote '{identifier}' not found", file=sys.stderr)
        return 1

    name = remote.get("name", "")
    key_prefix = remote.get("key", "")[:8]
    stats = remote.get("stats", {})

    print(f"Remote: {name}")
    print(f"  Prefix:     {key_prefix}")
    print(f"  Status:     {_status_label(remote)}")
    print(f"  Created:    {_fmt_time(remote.get('created_at'))}")
    print(f"  Last seen:  {_fmt_time(remote.get('last_seen'))}")
    if remote.get("revoked"):
        print(f"  Revoked at: {_fmt_time(remote.get('revoked_at'))}")
    print(f"  Segments:   {stats.get('segments_received', 0)}")
    print(f"  Bytes:      {_fmt_bytes(stats.get('bytes_received', 0))}")
    if stats.get("duplicates_rejected"):
        print(f"  Duplicates: {stats['duplicates_rejected']} rejected")

    # Today's sync history
    today = datetime.date.today().strftime("%Y%m%d")
    history = load_history(key_prefix, today)
    if history:
        uploads = [r for r in history if not r.get("type")]
        print(f"\n  Today ({today}): {len(uploads)} segment(s) synced")
        for rec in uploads[-5:]:
            seg = rec.get("segment", "?")
            files = rec.get("files", [])
            total = sum(f.get("size", 0) for f in files)
            ts = _fmt_time(rec.get("ts"))
            print(f"    {seg}  {len(files)} file(s)  {_fmt_bytes(total)}  {ts}")

    # Segment count by recent days
    hist_dir = get_hist_dir(key_prefix, ensure_exists=False)
    if hist_dir.exists():
        day_files = sorted(hist_dir.glob("*.jsonl"), reverse=True)[:7]
        if day_files:
            print("\n  Recent days:")
            for df in day_files:
                day = df.stem
                records = load_history(key_prefix, day)
                day_uploads = [r for r in records if not r.get("type")]
                print(f"    {day}: {len(day_uploads)} segment(s)")

    return 0


def _status_all() -> int:
    """Health overview for all remotes."""
    remotes = list_remotes()

    if not remotes:
        print("No remotes registered.")
        return 0

    connected = sum(1 for r in remotes if _status_label(r) == "connected")
    disconnected = sum(1 for r in remotes if _status_label(r) == "disconnected")
    revoked = sum(1 for r in remotes if _status_label(r) == "revoked")
    total_segments = sum(
        r.get("stats", {}).get("segments_received", 0) for r in remotes
    )
    total_bytes = sum(r.get("stats", {}).get("bytes_received", 0) for r in remotes)

    print(f"Remote observers: {len(remotes)} total")
    print(f"  Connected:    {connected}")
    print(f"  Disconnected: {disconnected}")
    print(f"  Revoked:      {revoked}")
    print(f"  Total segments: {total_segments}")
    print(f"  Total bytes:    {_fmt_bytes(total_bytes)}")

    print(f"\n{'Name':<20} {'Status':<14} {'Last Seen':<18}")
    print("-" * 54)
    for r in remotes:
        name = r.get("name", "")
        status = _status_label(r)
        last_seen = _fmt_time(r.get("last_seen"))
        print(f"{name:<20} {status:<14} {last_seen:<18}")

    return 0


# === Entry point ===


def main() -> None:
    """Entry point for sol remote CLI."""
    parser = argparse.ArgumentParser(
        prog="sol remote",
        description="Manage remote observer registrations",
    )

    sub = parser.add_subparsers(dest="command")

    # create
    p_create = sub.add_parser("create", help="Create a new remote observer")
    p_create.add_argument("name", help="Name for the remote observer")

    # list
    sub.add_parser("list", help="List all registered remotes")

    # rename
    p_rename = sub.add_parser("rename", help="Rename a remote (affects future streams)")
    p_rename.add_argument("identifier", help="Remote name or key prefix")
    p_rename.add_argument("new_name", help="New name for the remote")

    # revoke
    p_revoke = sub.add_parser("revoke", help="Revoke a remote registration")
    p_revoke.add_argument("identifier", help="Remote name or key prefix")

    # status
    p_status = sub.add_parser("status", help="Show remote status details")
    p_status.add_argument(
        "identifier",
        nargs="?",
        default=None,
        help="Remote name or key prefix (omit for overview)",
    )

    args = setup_cli(parser)

    # Bridge journal path to convey.state so apps.utils resolves correctly
    # (setup_cli initializes the journal, but convey.state needs it too)
    import convey.state
    from think.utils import get_journal

    convey.state.journal_root = get_journal()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "create": cmd_create,
        "list": cmd_list,
        "rename": cmd_rename,
        "revoke": cmd_revoke,
        "status": cmd_status,
    }

    sys.exit(handlers[args.command](args))
