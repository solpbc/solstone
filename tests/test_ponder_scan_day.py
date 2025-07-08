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
    mod = importlib.import_module("think.ponder")
    day_dir = copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    info = mod.scan_day("20240101")
    assert "ponder_day.md" in info["pondered"]
    assert "ponder_media.md" in info["unpondered"]

    (day_dir / "ponder_media.md").write_text("done")
    info_after = mod.scan_day("20240101")
    assert "ponder_media.md" in info_after["pondered"]
    assert "ponder_media.md" not in info_after["unpondered"]
