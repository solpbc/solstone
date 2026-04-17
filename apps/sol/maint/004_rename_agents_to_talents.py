# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Rename journal agents paths to talents paths.

Use --dry-run to preview without writing changes.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from think.utils import day_dirs, get_journal, iter_segments, setup_cli


@dataclass(frozen=True)
class RenameOp:
    src: Path
    dst: Path


class MigrationSummary:
    """Mutable counters for migration operations."""

    def __init__(self) -> None:
        self.moved = 0
        self.skipped = 0
        self.errors = 0


def _discover_ops(journal_path: Path) -> list[RenameOp]:
    ops: list[RenameOp] = []

    def add(src: Path, dst: Path) -> None:
        if src.exists() or os.path.lexists(src):
            ops.append(RenameOp(src=src, dst=dst))

    add(journal_path / "agents", journal_path / "talents")
    add(
        journal_path / "health" / "agents.json",
        journal_path / "health" / "talents.json",
    )
    add(journal_path / "maint" / "agents", journal_path / "maint" / "talents")

    for _day_name, day_abs in sorted(day_dirs().items()):
        day_dir = Path(day_abs)
        add(day_dir / "agents", day_dir / "talents")
        for _stream, _segment, segment_dir in iter_segments(day_dir.name):
            add(segment_dir / "agents", segment_dir / "talents")

    unique_ops: dict[Path, RenameOp] = {}
    for op in ops:
        unique_ops[op.src] = op
    return sorted(
        unique_ops.values(), key=lambda op: (len(op.src.parts), op.src.as_posix())
    )


def _check_collisions(ops: list[RenameOp]) -> list[RenameOp]:
    return [op for op in ops if os.path.lexists(op.dst)]


def _print_summary(summary: MigrationSummary) -> None:
    print("Migration complete")
    print(f"  moved:   {summary.moved}")
    print(f"  skipped: {summary.skipped}")
    print(f"  errors:  {summary.errors}")


def rename_agents_to_talents(*, dry_run: bool) -> MigrationSummary:
    """Rename legacy journal paths from agents to talents."""
    journal_path = Path(get_journal())
    ops = _discover_ops(journal_path)
    summary = MigrationSummary()

    collisions = _check_collisions(ops)
    if collisions:
        print(
            "Collision(s) detected; aborting with no filesystem changes:",
            file=sys.stderr,
        )
        for op in collisions:
            print(f"  {op.src} -> {op.dst}", file=sys.stderr)
        summary.errors = len(collisions)
        _print_summary(summary)
        raise SystemExit(1)

    for op in ops:
        if dry_run:
            print(f"[DRY-RUN] rename {op.src} -> {op.dst}")
            summary.moved += 1
            continue
        try:
            op.dst.parent.mkdir(parents=True, exist_ok=True)
            op.src.rename(op.dst)
            summary.moved += 1
        except FileNotFoundError:
            summary.skipped += 1
        except OSError as exc:
            summary.errors += 1
            print(
                f"[ERROR] rename failed: {op.src} -> {op.dst}: {exc}", file=sys.stderr
            )

    _print_summary(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename journal agents paths to talents."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview renames without writing files.",
    )
    args = setup_cli(parser)

    if args.dry_run:
        print("[DRY-RUN] No files will be modified.")

    summary = rename_agents_to_talents(dry_run=args.dry_run)
    raise SystemExit(1 if summary.errors else 0)


if __name__ == "__main__":
    main()
