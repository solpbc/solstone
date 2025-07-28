import importlib
import shutil
from pathlib import Path

FIXTURES = Path("fixtures")


def copy_day(tmp_path: Path) -> Path:
    src = FIXTURES / "journal" / "20240101"
    dest = tmp_path / "20240101"
    shutil.copytree(src, dest)
    topics = dest / "topics"
    topics.mkdir()
    (topics / "flow.md").write_text("done")
    return dest


def test_scan_day(tmp_path, monkeypatch):
    mod = importlib.import_module("think.ponder")
    day_dir = copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    info = mod.scan_day("20240101")
    assert "topics/flow.md" in info["processed"]
    assert "topics/media.md" in info["repairable"]

    (day_dir / "topics" / "media.md").write_text("done")
    info_after = mod.scan_day("20240101")
    assert "topics/media.md" in info_after["processed"]
    assert "topics/media.md" not in info_after["repairable"]
