# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Rename live journal agents paths to talents."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from solstone.think.utils import day_dirs, get_journal, iter_segments, setup_cli


@dataclass
class RenameSummary:
    discovered: int = 0
    moved: int = 0
    skipped: int = 0
    errors: int = 0
    collisions: int = 0


def discover_moves(journal_path: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Return planned (src, dst) moves and already-migrated destinations."""
    planned: list[tuple[Path, Path]] = []
    skipped: list[Path] = []

    def add_pair(src: Path, dst: Path) -> None:
        if src.exists():
            planned.append((src, dst))
        elif dst.exists():
            skipped.append(dst)

    add_pair(journal_path / "agents", journal_path / "talents")
    add_pair(
        journal_path / "health" / "agents.json",
        journal_path / "health" / "talents.json",
    )

    for day_name, day_abs in sorted(day_dirs().items()):
        day_dir = Path(day_abs)
        if not day_dir.is_dir():
            continue

        add_pair(day_dir / "agents", day_dir / "talents")

        for _stream, _segment, seg_path in iter_segments(day_name):
            add_pair(seg_path / "agents", seg_path / "talents")

    return planned, skipped


def run_migration(
    journal_path: Path, *, dry_run: bool
) -> tuple[RenameSummary, list[tuple[Path, Path]]]:
    """Run or preview the agents->talents path rename."""
    summary = RenameSummary()
    planned, skipped = discover_moves(journal_path)
    summary.discovered = len(planned)
    summary.skipped = len(skipped)

    collisions = [(src, dst) for src, dst in planned if dst.exists()]
    summary.collisions = len(collisions)
    if collisions:
        return summary, collisions

    for src, dst in planned:
        print(f"{'[DRY-RUN] ' if dry_run else ''}move {src} -> {dst}")
        if dry_run:
            continue

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            summary.moved += 1
        except Exception as exc:
            summary.errors += 1
            print(f"[ERROR] move failed: {src} -> {dst}: {exc}")

    if dry_run:
        summary.moved = len(planned)

    return summary, []


def _print_summary(summary: RenameSummary) -> None:
    print("Summary")
    print(f"  discovered: {summary.discovered}")
    print(f"  moved:      {summary.moved}")
    print(f"  skipped:    {summary.skipped}")
    print(f"  errors:     {summary.errors}")
    print(f"  collisions: {summary.collisions}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename live journal agents paths to talents."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned renames without writing files.",
    )
    args = setup_cli(parser)

    journal_path = Path(get_journal())
    summary, collisions = run_migration(journal_path, dry_run=args.dry_run)

    if collisions:
        print("Collision(s) detected; no files were moved:")
        for src, dst in collisions:
            print(f"  {src} -> {dst}")
        _print_summary(summary)
        sys.exit(2)

    _print_summary(summary)
    if summary.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
