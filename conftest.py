# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc
"""Pytest fixture-leak detector.

Captures `git status --porcelain -- tests/fixtures/` at session start and
diffs at session end. Fails the session with a named-path error when tests
leave the fixture tree dirty.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_FIXTURE_ROOT = "tests/fixtures"
_BASELINE: set[tuple[str, str]] | None = None
_GIT_AVAILABLE = True


def _capture_status(repo_root: Path) -> set[tuple[str, str]] | None:
    """Return the set of (status_XY, path) tuples from fixture-tree git status.

    Returns None when git is unavailable or the command fails (e.g. not a git
    repo).
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", _FIXTURE_ROOT],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None

    entries: set[tuple[str, str]] = set()
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        status = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        entries.add((status, path))
    return entries


def _format_leak_message(new_entries: set[tuple[str, str]]) -> str:
    lines = [f"  {status} {path}" for status, path in sorted(new_entries)]
    return (
        "\n"
        "solstone fixture-leak detector: tests left tests/fixtures/ dirty\n"
        + "\n".join(lines)
        + "\n\n"
        "To fix, use one of these isolation mechanisms:\n"
        "  - journal_copy fixture (tests/conftest.py:57) — copies tracked fixtures to tmp_path\n"
        "  - point _SOLSTONE_JOURNAL_OVERRIDE at a tmp_path directly\n"
        "  - mock the subprocess/write path so code never touches tests/fixtures/\n"
        "\n"
        "Prior incidents: f6f382a6, 2996e072\n"
    )


def pytest_sessionstart(session):
    global _BASELINE, _GIT_AVAILABLE

    repo_root = session.config.rootpath
    _BASELINE = _capture_status(repo_root)
    if _BASELINE is None:
        _GIT_AVAILABLE = False
        sys.stderr.write("solstone fixture-leak detector: git unavailable, skipping\n")


def pytest_sessionfinish(session, exitstatus):
    if not _GIT_AVAILABLE or _BASELINE is None:
        return

    repo_root = session.config.rootpath
    current = _capture_status(repo_root)
    if current is None:
        return

    new_entries = current - _BASELINE
    if not new_entries:
        return

    sys.stderr.write(_format_leak_message(new_entries))
    session.exitstatus = 1
