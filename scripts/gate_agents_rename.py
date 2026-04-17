#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIM_FILES = {
    Path("think/pipeline_health.py"),
    Path("apps/home/routes.py"),
}
ALLOWLIST_RE = re.compile(r"^apps/sol/maint/00[0-4]_.+\.py$")
PRODUCTION_PREFIXES = ("think/", "apps/", "talent/", "convey/", "observe/")
SHIM_WINDOW = 20

RULES = [
    (
        "legacy dream emitter",
        re.compile(r'_jsonl_log\(\s*["\']agent\.(fail|dispatch|complete|skip)["\']'),
        None,
    ),
    (
        "legacy callosum emitter",
        re.compile(r'emit\(\s*["\']agent_(started|completed)["\']'),
        None,
    ),
    ("legacy module path", re.compile(r"\bthink\.agents\b"), None),
    ("legacy/new CLI command", re.compile(r"\bsol agents\b|\bsol talents\b"), None),
    ("legacy payload key", re.compile(r'["\']agent_id["\']\s*:'), "production"),
    ("legacy wire event", re.compile(r'["\']agent_updated["\']'), "production"),
    (
        "legacy summary/anomaly key",
        re.compile(
            r'summary\["agents"\]|["\']agent_failure["\']|["\']agents_fired["\']'
        ),
        "production",
    ),
]


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def is_allowed(path: Path) -> bool:
    path_str = path.as_posix()
    if path == Path("AGENTS.md"):
        return True
    if path == Path("tests/test_maint_004_rename.py"):
        return True
    if path == Path("scripts/gate_agents_rename.py"):
        return True
    if ALLOWLIST_RE.match(path_str):
        return True
    return False


def is_production(path: Path) -> bool:
    path_str = path.as_posix()
    return path_str == "sol.py" or path_str.startswith(PRODUCTION_PREFIXES)


def iter_lines(path: Path) -> list[tuple[int, str]]:
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    if path not in SHIM_FILES:
        return list(enumerate(lines, start=1))

    visible: list[tuple[int, str]] = []
    suppress_until = 0
    for line_no, line in enumerate(lines, start=1):
        if line_no <= suppress_until:
            continue
        if "HISTORICAL SHIM:" in line:
            suppress_until = line_no + SHIM_WINDOW
            continue
        visible.append((line_no, line))
    return visible


def main() -> int:
    failures: list[str] = []
    for path in tracked_files():
        if is_allowed(path):
            continue
        if not (ROOT / path).is_file():
            continue
        try:
            lines = iter_lines(path)
        except UnicodeDecodeError:
            continue
        for line_no, line in lines:
            for label, pattern, scope in RULES:
                if scope == "production" and not is_production(path):
                    continue
                if pattern.search(line):
                    failures.append(f"{path}:{line_no}: {label}: {line.strip()}")

    if failures:
        print("agents rename gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("agents rename gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
