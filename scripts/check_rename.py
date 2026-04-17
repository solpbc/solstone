#!/usr/bin/env python3

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

ALLOWLIST = [
    "apps/sol/maint/00[0-3]_*.py",
    "apps/sol/maint/004_rename_agents_to_talents.py",
    "AGENTS.md",
    "think/pipeline_health.py",
    "apps/home/routes.py",
    "tests/test_pipeline_health.py",
    "tests/test_home_yesterdays_processing.py",
    "apps/health/tests/test_call.py",
    "scripts/check_rename.py",
]
SKIP_PARTS = {".git", ".venv", "__pycache__", "node_modules", "journal"}
SKIP_PREFIXES = ("tests/fixtures/journal/",)
RULES = {
    r'(["\'])agents/': "legacy agents path component",
    r"/agents/": "legacy agents path component",
    r"think\.agents": "legacy module path",
    r"sol agents": "legacy CLI reference",
    r'["\']agent\.(dispatch|start|complete|fail)["\']': "legacy emitted event name",
}


def _is_allowed(rel_path: str) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in ALLOWLIST)


def _should_skip(path: Path) -> bool:
    rel_path = path.as_posix()
    if any(part in SKIP_PARTS for part in path.parts):
        return True
    return any(rel_path.startswith(prefix) for prefix in SKIP_PREFIXES)


def main() -> int:
    hits: list[str] = []

    for path in Path(".").rglob("*"):
        if not path.is_file() or _should_skip(path):
            continue

        rel_path = path.as_posix()
        if _is_allowed(rel_path):
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern, reason in RULES.items():
                if re.search(pattern, line):
                    hits.append(f"{rel_path}:{line_number}: {reason}: {line.strip()}")

    if hits:
        print("agents->talents rename gate failed:")
        for hit in hits:
            print(hit)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
