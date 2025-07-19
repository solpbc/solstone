import importlib
import shutil
from pathlib import Path

from PIL import Image

FIXTURES = Path("fixtures")


def copy_journal(tmp_path: Path) -> Path:
    src = FIXTURES / "journal"
    dest = tmp_path / "journal"
    shutil.copytree(src, dest)
    return dest


def test_process_once(tmp_path, monkeypatch):
    mod = importlib.import_module("see.describe")
    journal = copy_journal(tmp_path)
    img = journal / "20240101/123456_monitor_1_diff.png"
    box = journal / "20240101/123456_monitor_1_diff_box.json"
    out = journal / "20240101/123456_monitor_1_diff.json"
    out.unlink()

    monkeypatch.setattr(
        mod.gemini_look,
        "gemini_describe_region",
        lambda *a, **k: {"result": {"ok": True}, "model_used": "m"},
    )
    monkeypatch.setattr(Image, "open", lambda p: Image.new("RGB", (10, 10)))
    desc = mod.Describer(journal)
    desc._process_once(img, box, out)
    assert out.is_file()
    assert out.with_suffix(out.suffix + ".crumb").is_file()
    seen_dir = journal / "20240101" / "seen"
    assert (seen_dir / img.name).is_file()
    assert (seen_dir / box.name).is_file()
