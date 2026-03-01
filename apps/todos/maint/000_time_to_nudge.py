# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Migrate todo time field to nudge format.

Scans all facets/*/todos/*.jsonl files and converts any "time" field
to the new "nudge" format (YYYYMMDDTHH:MM), using the filename day.
"""

import argparse
import json
import logging
from pathlib import Path

from think.utils import get_journal, setup_cli

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Migrate todo time → nudge")
    setup_cli(parser)

    journal = Path(get_journal())
    facets_dir = journal / "facets"

    if not facets_dir.is_dir():
        print("No facets directory found.")
        return

    migrated_files = 0
    migrated_entries = 0

    for facet_dir in sorted(facets_dir.iterdir()):
        if not facet_dir.is_dir():
            continue
        todos_dir = facet_dir / "todos"
        if not todos_dir.is_dir():
            continue

        for jsonl_file in sorted(todos_dir.glob("*.jsonl")):
            stem = jsonl_file.stem
            if len(stem) != 8 or not stem.isdigit():
                continue

            day = stem
            lines = jsonl_file.read_text(encoding="utf-8").splitlines()
            new_lines = []
            file_changed = False

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    new_lines.append(line)
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    new_lines.append(line)
                    continue

                if "time" in data and "nudge" not in data:
                    time_val = data.pop("time")
                    data["nudge"] = f"{day}T{time_val}"
                    new_lines.append(json.dumps(data, ensure_ascii=False))
                    file_changed = True
                    migrated_entries += 1
                else:
                    new_lines.append(line)

            if file_changed:
                content = "\n".join(new_lines)
                if new_lines and not content.endswith("\n"):
                    content += "\n"
                jsonl_file.write_text(content, encoding="utf-8")
                migrated_files += 1
                logger.info("Migrated %s", jsonl_file)

    print(f"Migration complete: {migrated_entries} entries in {migrated_files} files.")


if __name__ == "__main__":
    main()
