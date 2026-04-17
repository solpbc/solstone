# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Seed per-journal AGENTS.md, CLAUDE.md, and GEMINI.md files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from think.utils import get_journal, setup_cli


def _symlink_points_to_agents(path: Path) -> bool:
    return path.is_symlink() and path.readlink() == Path("AGENTS.md")


def _repair_symlink(path: Path) -> int:
    if path.exists() and not path.is_symlink():
        print(f"refusing to replace existing non-symlink {path.name}", file=sys.stderr)
        return 1
    if path.is_symlink():
        path.unlink()
        action = "replaced"
    else:
        action = "created"
    path.symlink_to("AGENTS.md")
    print(f"{action} {path.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed journal AGENTS.md symlinks.")
    setup_cli(parser)

    journal = Path(get_journal())
    repo_root = Path(__file__).resolve().parents[3]

    agents_path = journal / "AGENTS.md"
    claude_path = journal / "CLAUDE.md"
    gemini_path = journal / "GEMINI.md"

    if (
        agents_path.exists()
        and _symlink_points_to_agents(claude_path)
        and _symlink_points_to_agents(gemini_path)
    ):
        print("all journal agent files already present")
        return 0

    try:
        journal_md = (repo_root / "docs" / "JOURNAL.md").read_text(encoding="utf-8")

        if not agents_path.exists():
            agents_path.write_text(journal_md, encoding="utf-8")
            print("created AGENTS.md")

        for path in (claude_path, gemini_path):
            if _symlink_points_to_agents(path):
                continue
            status = _repair_symlink(path)
            if status != 0:
                return status
    except OSError as exc:
        print(f"failed to seed journal agent files: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
