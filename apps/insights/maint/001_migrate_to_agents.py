# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Migrate journal from insights/ to agents/ directory structure.

This migration:
1. Renames YYYYMMDD/insights/ -> YYYYMMDD/agents/ for all day directories
2. Updates config/journal.json: "insights" key -> "agents" key
3. Renames apps/*/insights/ -> apps/*/agents/ for app-specific outputs
4. Updates source paths in facets/*/events/*.jsonl files

All operations are idempotent - safe to re-run if interrupted or run manually.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from think.utils import get_journal, setup_cli

logger = logging.getLogger(__name__)


def is_day_directory(name: str) -> bool:
    """Check if directory name is YYYYMMDD format."""
    return bool(re.match(r"^\d{8}$", name))


def migrate_day_directories(journal: Path, dry_run: bool) -> int:
    """Rename insights/ -> agents/ in all day directories.

    Returns count of directories renamed.
    """
    count = 0
    for day_dir in sorted(journal.iterdir()):
        if not day_dir.is_dir() or not is_day_directory(day_dir.name):
            continue

        insights_dir = day_dir / "insights"
        agents_dir = day_dir / "agents"

        if insights_dir.is_dir() and not agents_dir.exists():
            if dry_run:
                print(f"  [DRY-RUN] Would rename: {insights_dir} -> {agents_dir}")
            else:
                insights_dir.rename(agents_dir)
                print(f"  Renamed: {insights_dir} -> {agents_dir}")
            count += 1
        elif insights_dir.is_dir() and agents_dir.exists():
            logger.warning(f"Both insights/ and agents/ exist in {day_dir.name}")

    return count


def migrate_config(journal: Path, dry_run: bool) -> bool:
    """Update config/journal.json: "insights" key -> "agents" key.

    Returns True if config was updated.
    """
    config_file = journal / "config" / "journal.json"
    if not config_file.exists():
        return False

    try:
        config = json.loads(config_file.read_text())
    except json.JSONDecodeError as e:
        logger.warning(f"Could not parse config file: {e}")
        return False

    if "insights" not in config:
        return False

    if "agents" in config:
        logger.warning("Both 'insights' and 'agents' keys exist in config")
        return False

    if dry_run:
        print("  [DRY-RUN] Would update config: 'insights' -> 'agents'")
    else:
        config["agents"] = config.pop("insights")
        config_file.write_text(json.dumps(config, indent=2) + "\n")
        print("  Updated config: 'insights' -> 'agents'")

    return True


def migrate_app_directories(journal: Path, dry_run: bool) -> int:
    """Rename apps/*/insights/ -> apps/*/agents/ for app-specific outputs.

    Returns count of directories renamed.
    """
    apps_dir = journal / "apps"
    if not apps_dir.is_dir():
        return 0

    count = 0
    for app_dir in sorted(apps_dir.iterdir()):
        if not app_dir.is_dir():
            continue

        insights_dir = app_dir / "insights"
        agents_dir = app_dir / "agents"

        if insights_dir.is_dir() and not agents_dir.exists():
            if dry_run:
                print(f"  [DRY-RUN] Would rename: {insights_dir} -> {agents_dir}")
            else:
                insights_dir.rename(agents_dir)
                print(f"  Renamed: {insights_dir} -> {agents_dir}")
            count += 1
        elif insights_dir.is_dir() and agents_dir.exists():
            logger.warning(f"Both insights/ and agents/ exist in apps/{app_dir.name}")

    return count


def migrate_event_sources(journal: Path, dry_run: bool) -> int:
    """Update source paths in facets/*/events/*.jsonl files.

    Changes "YYYYMMDD/insights/..." -> "YYYYMMDD/agents/..."

    Returns count of files updated.
    """
    facets_dir = journal / "facets"
    if not facets_dir.is_dir():
        return 0

    count = 0
    pattern = re.compile(r'("source":\s*")(\d{8})/insights/')

    for facet_dir in sorted(facets_dir.iterdir()):
        if not facet_dir.is_dir():
            continue

        events_dir = facet_dir / "events"
        if not events_dir.is_dir():
            continue

        for jsonl_file in sorted(events_dir.glob("*.jsonl")):
            content = jsonl_file.read_text()
            if "/insights/" not in content:
                continue

            new_content = pattern.sub(r"\g<1>\g<2>/agents/", content)
            if new_content != content:
                if dry_run:
                    print(f"  [DRY-RUN] Would update sources in: {jsonl_file}")
                else:
                    jsonl_file.write_text(new_content)
                    print(f"  Updated sources in: {jsonl_file}")
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = setup_cli(parser)

    journal = Path(get_journal())

    print(f"Migrating journal: {journal}")
    if args.dry_run:
        print("[DRY-RUN MODE - No changes will be made]\n")

    # Step 1: Rename day directories
    print("\nStep 1: Renaming insights/ -> agents/ in day directories...")
    day_count = migrate_day_directories(journal, args.dry_run)
    print(
        f"  Total: {day_count} directories {'would be ' if args.dry_run else ''}renamed"
    )

    # Step 2: Update config
    print("\nStep 2: Updating config/journal.json...")
    config_updated = migrate_config(journal, args.dry_run)
    if not config_updated:
        print("  No config changes needed")

    # Step 3: Rename app directories
    print("\nStep 3: Renaming insights/ -> agents/ in app directories...")
    app_count = migrate_app_directories(journal, args.dry_run)
    print(
        f"  Total: {app_count} directories {'would be ' if args.dry_run else ''}renamed"
    )

    # Step 4: Update event sources
    print("\nStep 4: Updating source paths in event files...")
    event_count = migrate_event_sources(journal, args.dry_run)
    print(f"  Total: {event_count} files {'would be ' if args.dry_run else ''}updated")

    # Summary
    print("\n" + "=" * 50)
    if args.dry_run:
        print("DRY-RUN COMPLETE - no changes were made")
    else:
        print("MIGRATION COMPLETE")
        print(f"  - {day_count} day directories renamed")
        print(f"  - {app_count} app directories renamed")
        print(f"  - Config {'updated' if config_updated else 'unchanged'}")
        print(f"  - {event_count} event files updated")

    logger.info(
        f"Migration complete: {day_count} days, {app_count} apps, {event_count} events"
    )


if __name__ == "__main__":
    main()
