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
    return dest


def test_cluster_full(tmp_path, monkeypatch):
    mod = importlib.import_module("think.cluster")
    copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    md, counts = mod.cluster(
        "20240101", sources={"audio": True, "screen": False, "agents": True}
    )
    # Count: audio.jsonl (1) + audio.md (1) + screen.md (1) = 3 entries
    assert counts["audio"] == 1
    assert counts["agents"] == 2  # audio.md + screen.md
    assert "Audio Transcript" in md
    # Now uses insight format: "### {stem} summary"
    assert "### screen summary" in md
    assert "### audio summary" in md


def test_cluster_default_sources(tmp_path, monkeypatch):
    mod = importlib.import_module("think.cluster")
    copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    out, _counts = mod.cluster(
        "20240101", sources={"audio": True, "screen": False, "agents": True}
    )
    # Now uses insight format: "### {stem} summary"
    assert "### screen summary" in out


def test_cluster_range_raw_screen(tmp_path, monkeypatch):
    mod = importlib.import_module("think.cluster")
    copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    out = mod.cluster_range(
        "20240101",
        "123456",
        "123556",
        sources={"audio": True, "screen": True, "agents": False},
    )
    # Range mode with screen=True uses raw screen data.
    assert "### Screen Activity" in out
    assert "IDE with auth.py open" in out
