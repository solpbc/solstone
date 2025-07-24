import importlib
import shutil
from pathlib import Path

FIXTURES = Path("fixtures")


def copy_day(tmp_path: Path) -> Path:
    src = FIXTURES / "journal" / "20240101"
    dest = tmp_path / "20240101"
    shutil.copytree(src, dest)
    return dest


def test_scan_day(tmp_path, monkeypatch):
    mod = importlib.import_module("think.entity_roll")
    day_dir = copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    topics = day_dir / "topics"
    topics.mkdir()
    (topics / "knowledge_graph.md").write_text("data")

    info = mod.scan_day("20240101")
    assert "entities.md" in info["processed"]

    (day_dir / "entities.md").unlink()
    info_after = mod.scan_day("20240101")
    assert "topics/knowledge_graph.md" in info_after["repairable"]
