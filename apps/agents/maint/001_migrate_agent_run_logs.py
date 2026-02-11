# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Migrate agent run logs from flat to per-agent subdirectory layout.

Changes applied:
- agents/<id>.jsonl -> agents/<name>/<id>.jsonl
- agents/<name>.jsonl symlinks -> agents/<name>.log symlinks
- Build day index files (agents/<day>.jsonl) from migrated data

Use --dry-run to preview without writing changes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from think.utils import get_journal, setup_cli


class MigrationSummary:
    """Mutable counters for migration operations."""

    def __init__(self) -> None:
        self.moved = 0
        self.symlinks_removed = 0
        self.symlinks_created = 0
        self.day_index_entries = 0
        self.skipped = 0
        self.errors = 0


def _read_first_line(path: Path) -> dict | None:
    """Read and parse the first JSON line from a file."""
    try:
        with open(path, "r") as f:
            line = f.readline().strip()
            if line:
                return json.loads(line)
    except (json.JSONDecodeError, IOError):
        pass
    return None


def _read_last_event(path: Path, event_types: set[str]) -> dict | None:
    """Read last few lines to find a specific event type."""
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        for line in reversed(lines[-10:]):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("event") in event_types:
                    return event
            except json.JSONDecodeError:
                continue
    except IOError:
        pass
    return None


def _agent_id_to_day(agent_id: str) -> str:
    """Convert agent ID (ms timestamp) to YYYYMMDD string."""
    try:
        ts = int(agent_id) / 1000
        return datetime.fromtimestamp(ts).strftime("%Y%m%d")
    except (ValueError, OSError):
        return ""


def migrate(agents_dir: Path, dry_run: bool = False) -> MigrationSummary:
    """Migrate flat agent files to per-agent subdirectories."""
    summary = MigrationSummary()

    if not agents_dir.exists():
        return summary

    # Phase 1: Remove old symlinks (*.jsonl that are symlinks)
    for path in sorted(agents_dir.iterdir()):
        if path.is_symlink() and path.suffix == ".jsonl":
            print(f"  Remove symlink: {path.name}")
            if not dry_run:
                path.unlink()
            summary.symlinks_removed += 1

    # Phase 2: Move flat agent files into subdirectories
    # Collect all non-symlink .jsonl files at the root level
    # (exclude day index files which are 8-digit dates)
    latest_per_name: dict[str, tuple[int, str]] = {}  # name -> (agent_id_num, agent_id)
    day_entries: dict[str, list[dict]] = {}  # day -> [summary_dicts]

    for path in sorted(agents_dir.iterdir()):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        if path.suffix != ".jsonl":
            continue
        # Skip day index files (8-digit filenames)
        if len(path.stem) == 8 and path.stem.isdigit():
            continue

        # Determine agent_id and whether active
        stem = path.stem
        is_active = stem.endswith("_active")
        agent_id = stem.replace("_active", "")

        # Read first line to get agent name
        first_line = _read_first_line(path)
        if not first_line:
            print(f"  Skip (unreadable): {path.name}")
            summary.skipped += 1
            continue

        name = first_line.get("name", "default")
        safe_name = name.replace(":", "--")

        # Move to subdirectory
        subdir = agents_dir / safe_name
        new_path = subdir / path.name

        if new_path.exists():
            print(f"  Skip (already exists): {safe_name}/{path.name}")
            summary.skipped += 1
            continue

        print(f"  Move: {path.name} -> {safe_name}/{path.name}")
        if not dry_run:
            subdir.mkdir(parents=True, exist_ok=True)
            path.rename(new_path)
        summary.moved += 1

        # Track latest completed agent per name for symlinks
        if not is_active:
            try:
                agent_id_num = int(agent_id)
            except ValueError:
                summary.skipped += 1
                continue

            if name not in latest_per_name or agent_id_num > latest_per_name[name][0]:
                latest_per_name[name] = (agent_id_num, agent_id)

            # Build day index entry
            start_ts = first_line.get("ts", 0)
            day = first_line.get("day") or _agent_id_to_day(agent_id)
            if day:
                end_event = _read_last_event(
                    new_path if not dry_run else path,
                    {"finish", "error"},
                )
                runtime_seconds = None
                status = "completed"
                if end_event:
                    if end_event.get("event") == "error":
                        status = "error"
                    end_ts = end_event.get("ts", 0)
                    if end_ts and start_ts:
                        runtime_seconds = round((end_ts - start_ts) / 1000.0, 1)

                entry = {
                    "agent_id": agent_id,
                    "name": name,
                    "day": day,
                    "facet": first_line.get("facet"),
                    "ts": start_ts,
                    "status": status,
                    "runtime_seconds": runtime_seconds,
                    "provider": first_line.get("provider"),
                    "model": first_line.get("model"),
                }
                day_entries.setdefault(day, []).append(entry)
                summary.day_index_entries += 1

    # Phase 3: Create new .log symlinks
    for name, (_agent_id_num, agent_id) in latest_per_name.items():
        safe_name = name.replace(":", "--")
        link_path = agents_dir / f"{safe_name}.log"
        target = f"{safe_name}/{agent_id}.jsonl"
        print(f"  Symlink: {safe_name}.log -> {target}")
        if not dry_run:
            from think.runner import _atomic_symlink

            _atomic_symlink(link_path, target)
        summary.symlinks_created += 1

    # Phase 4: Write day index files
    for day, entries in sorted(day_entries.items()):
        day_index_path = agents_dir / f"{day}.jsonl"

        # Append to existing day index (idempotent: skip entries already present)
        existing_ids: set[str] = set()
        if day_index_path.exists() and not dry_run:
            try:
                with open(day_index_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                existing = json.loads(line)
                                existing_id = existing.get("agent_id")
                                if isinstance(existing_id, str):
                                    existing_ids.add(existing_id)
                            except json.JSONDecodeError:
                                continue
            except IOError:
                pass

        new_entries = [e for e in entries if e["agent_id"] not in existing_ids]
        if new_entries:
            print(f"  Day index: {day}.jsonl ({len(new_entries)} entries)")
            if not dry_run:
                with open(day_index_path, "a") as f:
                    for entry in new_entries:
                        f.write(json.dumps(entry) + "\n")

    return summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Migrate agent run logs to per-agent subdirectories"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing"
    )
    args = setup_cli(parser)

    journal_path = Path(get_journal())
    agents_dir = journal_path / "agents"

    if args.dry_run:
        print("[DRY-RUN] No files will be modified.")

    print(f"Migrating agent run logs in: {agents_dir}")
    summary = migrate(agents_dir, dry_run=args.dry_run)

    print("Migration complete")
    print(f"  moved:            {summary.moved}")
    print(f"  symlinks_removed: {summary.symlinks_removed}")
    print(f"  symlinks_created: {summary.symlinks_created}")
    print(f"  day_index_entries:{summary.day_index_entries}")
    print(f"  skipped:          {summary.skipped}")
    print(f"  errors:           {summary.errors}")


if __name__ == "__main__":
    main()
