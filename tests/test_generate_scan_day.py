# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import os
from pathlib import Path

from think.utils import day_path
from tests.conftest import copytree_tracked

FIXTURES = Path("tests/fixtures")


def copy_day(tmp_path: Path) -> Path:
    os.environ["_SOLSTONE_JOURNAL_OVERRIDE"] = str(tmp_path)
    dest = day_path("20240101")
    src = FIXTURES / "journal" / "20240101"
    copytree_tracked(src, dest)
    agents_dir = dest / "agents"
    agents_dir.mkdir(exist_ok=True)  # Allow existing directory
    (agents_dir / "flow.md").write_text("done")
    return dest


def test_scan_day(tmp_path, monkeypatch):
    mod = importlib.import_module("think.agents")
    day_dir = copy_day(tmp_path)
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    info = mod.scan_day("20240101")
    assert "agents/flow.md" in info["processed"]
    assert "agents/timeline.md" in info["repairable"]

    (day_dir / "agents" / "timeline.md").write_text("done")
    info_after = mod.scan_day("20240101")
    assert "agents/timeline.md" in info_after["processed"]
    assert "agents/timeline.md" not in info_after["repairable"]
