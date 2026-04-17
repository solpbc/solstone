# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import os
from pathlib import Path

from tests.conftest import copytree_tracked
from think.utils import day_path

FIXTURES = Path("tests/fixtures")


def copy_day(tmp_path: Path) -> Path:
    os.environ["_SOLSTONE_JOURNAL_OVERRIDE"] = str(tmp_path)
    dest = day_path("20240101")
    src = FIXTURES / "journal" / "chronicle" / "20240101"
    copytree_tracked(src, dest)
    talents_dir = dest / "talents"
    talents_dir.mkdir(exist_ok=True)  # Allow existing directory
    (talents_dir / "flow.md").write_text("done")
    return dest


def test_scan_day(tmp_path, monkeypatch):
    mod = importlib.import_module("think.talents")
    day_dir = copy_day(tmp_path)
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    info = mod.scan_day("20240101")
    assert "talents/flow.md" in info["processed"]
    assert "talents/timeline.md" in info["repairable"]

    (day_dir / "talents" / "timeline.md").write_text("done")
    info_after = mod.scan_day("20240101")
    assert "talents/timeline.md" in info_after["processed"]
    assert "talents/timeline.md" not in info_after["repairable"]
