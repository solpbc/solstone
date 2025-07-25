import importlib
import json
import os
import shutil
from pathlib import Path

FIXTURES = Path("fixtures")


def copy_day(tmp_path: Path) -> Path:
    src = FIXTURES / "journal" / "20240101"
    dest = tmp_path / "20240101"
    shutil.copytree(src, dest)
    return dest


def test_scan_day(tmp_path, monkeypatch):
    mod = importlib.import_module("think.entities")
    day_dir = copy_day(tmp_path)
    cache = {
        "20240101": {
            "file": "20240101/entities.md",
            "mtime": int(os.path.getmtime(day_dir / "entities.md")),
            "entries": [],
        }
    }
    (tmp_path / "entities.json").write_text(json.dumps(cache))

    ent = mod.Entities(str(tmp_path))
    info = ent.scan_day("20240101")
    assert "entities.md" in info["processed"]

    md_path = day_dir / "entities.md"
    os.utime(md_path, (md_path.stat().st_atime, md_path.stat().st_mtime + 1))
    info_after = ent.scan_day("20240101")
    assert "entities.md" in info_after["repairable"]

    totals = ent.scan()
    assert totals == {"processed": 0, "repairable": 1}
