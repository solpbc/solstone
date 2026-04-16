# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Migrate root day directories into chronicle/."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from think.utils import CHRONICLE_DIR, DATE_RE, get_journal, setup_cli


@dataclass
class MigrationSummary:
    """Mutable counters for migration operations."""

    moved: int = 0
    skipped: int = 0
    sqlite_deleted: int = 0
    errors: int = 0


def _root_day_dirs(journal_path: Path) -> list[Path]:
    return [
        path
        for path in sorted(journal_path.iterdir())
        if path.is_dir() and DATE_RE.fullmatch(path.name)
    ]


def _sqlite_paths(journal_path: Path) -> list[Path]:
    db_path = journal_path / "indexer" / "journal.sqlite"
    return [
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
    ]


def _validate_end_state(journal_path: Path) -> None:
    if _root_day_dirs(journal_path):
        raise RuntimeError("root day directories remain after chronicle migration")
    if not (journal_path / CHRONICLE_DIR).is_dir():
        raise RuntimeError("chronicle/ missing after chronicle migration")
    remaining_sqlite = [path for path in _sqlite_paths(journal_path) if path.exists()]
    if remaining_sqlite:
        joined = ", ".join(str(path) for path in remaining_sqlite)
        raise RuntimeError(f"sqlite files remain after chronicle migration: {joined}")


def migrate(journal_path: Path, dry_run: bool = False) -> MigrationSummary:
    """Migrate root day directories into chronicle/."""
    summary = MigrationSummary()
    day_dirs = _root_day_dirs(journal_path)
    if not day_dirs:
        print("Nothing to migrate.")
        return summary

    chronicle_dir = journal_path / CHRONICLE_DIR
    if dry_run:
        print("[DRY-RUN] No files will be modified.")
    else:
        chronicle_dir.mkdir(parents=True, exist_ok=True)

    print(f"Migrating chronicle dirs in: {journal_path}")
    for source_day in day_dirs:
        target_day = chronicle_dir / source_day.name
        if target_day.exists():
            print(f"  Skip (already exists): {source_day.name}")
            summary.skipped += 1
            continue

        print(f"  Move: {source_day.name} -> {CHRONICLE_DIR}/{source_day.name}")
        if not dry_run:
            shutil.move(str(source_day), str(target_day))
        summary.moved += 1

    if summary.moved and not dry_run:
        for sqlite_path in _sqlite_paths(journal_path):
            if not sqlite_path.exists():
                continue
            print(f"  Delete: {sqlite_path.relative_to(journal_path)}")
            sqlite_path.unlink()
            summary.sqlite_deleted += 1
        _validate_end_state(journal_path)

    return summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Migrate root day directories into chronicle/"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without writing"
    )
    args = setup_cli(parser)

    summary = migrate(Path(get_journal()), dry_run=args.dry_run)
    if summary.moved == 0 and summary.skipped == 0 and summary.sqlite_deleted == 0:
        return

    print("Migration complete")
    print(f"  moved:         {summary.moved}")
    print(f"  skipped:       {summary.skipped}")
    print(f"  sqlite_deleted:{summary.sqlite_deleted}")
    print(f"  errors:        {summary.errors}")


if __name__ == "__main__":
    main()
