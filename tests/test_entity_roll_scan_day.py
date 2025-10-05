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


def test_scan_day(tmp_path, monkeypatch):
    mod = importlib.import_module("think.entity_roll")
    day_dir = copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    topics = day_dir / "topics"
    topics.mkdir(exist_ok=True)  # Allow existing directory
    (topics / "knowledge_graph.md").write_text("data")

    info = mod.scan_day("20240101")
    assert "entities.md" in info["processed"]

    (day_dir / "entities.md").unlink()
    info_after = mod.scan_day("20240101")
    assert "topics/knowledge_graph.md" in info_after["repairable"]
