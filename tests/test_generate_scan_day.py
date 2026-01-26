# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
import os
import shutil
from pathlib import Path

from think.utils import day_path

FIXTURES = Path("fixtures")


def copy_day(tmp_path: Path) -> Path:
    os.environ["JOURNAL_PATH"] = str(tmp_path)
    dest = day_path("20240101")
    src = FIXTURES / "journal" / "20240101"
    # Copy contents from fixture to the day_path created directory
    for item in src.iterdir():
        if item.is_dir():
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest / item.name)
    insights = dest / "insights"
    insights.mkdir(exist_ok=True)  # Allow existing directory
    (insights / "flow.md").write_text("done")
    return dest


def test_scan_day(tmp_path, monkeypatch):
    mod = importlib.import_module("think.insight")
    day_dir = copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    info = mod.scan_day("20240101")
    assert "insights/flow.md" in info["processed"]
    assert "insights/media.md" in info["repairable"]

    (day_dir / "insights" / "media.md").write_text("done")
    info_after = mod.scan_day("20240101")
    assert "insights/media.md" in info_after["processed"]
    assert "insights/media.md" not in info_after["repairable"]
