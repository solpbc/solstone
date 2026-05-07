# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Shared isolation harness for API baseline tests and baseline regeneration.

Used by:
- tests/test_api_baselines.py - module-scoped fixtures
- tests/verify_api.py - verify/update CLI mode

Keeps both paths on identical isolation so generated baselines match the test
oracle.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

FROZEN_DATE = "2026-04-15"
FROZEN_TZ_OFFSET = -7


def copytree_tracked(src: Path, dst: Path) -> None:
    """Copy only git-tracked files from src to dst."""
    src = Path(src)
    dst = Path(dst)
    result = subprocess.run(
        ["git", "ls-files", "."],
        cwd=str(src),
        capture_output=True,
        text=True,
        check=True,
    )
    for rel in result.stdout.splitlines():
        if not rel:
            continue
        src_file = src / rel
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if src_file.is_symlink():
            os.symlink(os.readlink(src_file), dst_file)
        else:
            shutil.copy2(src_file, dst_file)


def prepare_isolated_journal(dst: Path) -> Path:
    """Copy the git-tracked fixture journal into dst and return the absolute path."""
    src = Path("tests/fixtures/journal").resolve()
    dst = dst.resolve()
    copytree_tracked(src, dst)
    return dst


@contextmanager
def isolated_app_env(journal: Path) -> Iterator[Path]:
    """Patch env so create_app(journal) is fully isolated."""

    journal = Path(journal).resolve()
    prev_override = os.environ.get("SOLSTONE_JOURNAL")

    os.environ["SOLSTONE_JOURNAL"] = str(journal)
    try:
        yield journal
    finally:
        if prev_override is None:
            os.environ.pop("SOLSTONE_JOURNAL", None)
        else:
            os.environ["SOLSTONE_JOURNAL"] = prev_override


def make_logged_in_test_client(journal: Path):
    """Create a Flask test client with an authenticated session."""
    from solstone.convey import create_app

    app = create_app(journal=str(Path(journal).resolve()))
    app.config["TESTING"] = True
    client = app.test_client()
    with client.session_transaction() as session:
        session["logged_in"] = True
        session.permanent = True
    return client
