# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Migrate agent output files to the new agents/ directory layout.

Changes applied:
- Segment agent outputs: YYYYMMDD/HHMMSS_LEN/*.{md,json} -> YYYYMMDD/HHMMSS_LEN/agents/**
- Daily faceted outputs: YYYYMMDD/agents/{topic}_{facet}.{ext} -> YYYYMMDD/agents/{facet}/{topic}.{ext}

Use --dry-run to preview without writing changes.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from think.utils import day_dirs, get_journal, iter_segments, setup_cli

KNOWN_SEGMENT_AGENT_JSON = frozenset(
    {"facets.json", "speakers.json", "activity_state.json"}
)


class MigrationSummary:
    """Mutable counters for migration operations."""

    def __init__(self) -> None:
        self.moved = 0
        self.cleaned = 0
        self.skipped = 0
        self.errors = 0


def _move_file(
    src: Path, dst: Path, *, dry_run: bool, summary: MigrationSummary
) -> None:
    """Move one file, cleaning up identical duplicates when dest already exists."""
    if dst.exists():
        # Dest already exists — clean up src if content is identical.
        if src.read_bytes() == dst.read_bytes():
            if dry_run:
                print(f"[DRY-RUN] clean {src} (identical to {dst})")
            else:
                src.unlink()
            summary.cleaned += 1
        else:
            summary.skipped += 1
        return

    if dry_run:
        print(f"[DRY-RUN] move {src} -> {dst}")
        summary.moved += 1
        return

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        summary.moved += 1
    except Exception as exc:
        summary.errors += 1
        print(f"[ERROR] move failed: {src} -> {dst}: {exc}")


def _migrate_segment_outputs(
    segment_dir: Path,
    *,
    facet_names: set[str],
    dry_run: bool,
    summary: MigrationSummary,
) -> None:
    """Move segment-level agent outputs into segment/agents/ layout."""
    agents_dir = segment_dir / "agents"

    # Move segment markdown outputs from segment root to segment/agents/
    for md_file in sorted(segment_dir.glob("*.md")):
        _move_file(md_file, agents_dir / md_file.name, dry_run=dry_run, summary=summary)

    # Move known segment JSON outputs from segment root to segment/agents/
    for json_file in sorted(segment_dir.glob("*.json")):
        name = json_file.name

        if name in KNOWN_SEGMENT_AGENT_JSON:
            _move_file(
                json_file,
                agents_dir / name,
                dry_run=dry_run,
                summary=summary,
            )
            continue

        if name.startswith("activity_state_") and name.endswith(".json"):
            facet = name[len("activity_state_") : -len(".json")]
            if facet in facet_names:
                _move_file(
                    json_file,
                    agents_dir / facet / "activity_state.json",
                    dry_run=dry_run,
                    summary=summary,
                )
                continue

        summary.skipped += 1


def _migrate_daily_faceted_outputs(
    day_dir: Path,
    *,
    facet_names: set[str],
    dry_run: bool,
    summary: MigrationSummary,
) -> None:
    """Move daily faceted files from suffix naming to facet subdirectory naming."""
    agents_dir = day_dir / "agents"
    if not agents_dir.is_dir():
        return

    # Match longest facet names first to avoid partial matches.
    ordered_facets = sorted(facet_names, key=len, reverse=True)

    for file_path in sorted(agents_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix not in (".md", ".json"):
            continue

        stem = file_path.stem
        matched_facet = None
        matched_topic = None

        for facet in ordered_facets:
            suffix = f"_{facet}"
            if stem.endswith(suffix):
                topic = stem[: -len(suffix)]
                if topic:
                    matched_facet = facet
                    matched_topic = topic
                    break

        if matched_facet is None or matched_topic is None:
            summary.skipped += 1
            continue

        dest = agents_dir / matched_facet / f"{matched_topic}{file_path.suffix}"
        _move_file(file_path, dest, dry_run=dry_run, summary=summary)


def migrate_agent_layout(*, dry_run: bool) -> MigrationSummary:
    """Run filesystem migration for all day directories in the active journal."""
    summary = MigrationSummary()
    journal_path = Path(get_journal())

    facets_dir = journal_path / "facets"
    facet_names = (
        {entry.name for entry in facets_dir.iterdir() if entry.is_dir()}
        if facets_dir.is_dir()
        else set()
    )

    for day_name, day_abs in sorted(day_dirs().items()):
        day_dir = Path(day_abs)
        if not day_dir.is_dir():
            continue

        # Segment directories (across all streams)
        for _stream, _seg_key, seg_path in iter_segments(day_name):
            _migrate_segment_outputs(
                seg_path,
                facet_names=facet_names,
                dry_run=dry_run,
                summary=summary,
            )

        # Daily agents/ directory
        _migrate_daily_faceted_outputs(
            day_dir,
            facet_names=facet_names,
            dry_run=dry_run,
            summary=summary,
        )

    print("Migration complete")
    print(f"  moved:   {summary.moved}")
    print(f"  cleaned: {summary.cleaned}")
    print(f"  skipped: {summary.skipped}")
    print(f"  errors:  {summary.errors}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate agent output layout.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview moves without writing files.",
    )
    args = setup_cli(parser)

    if args.dry_run:
        print("[DRY-RUN] No files will be modified.")

    migrate_agent_layout(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
