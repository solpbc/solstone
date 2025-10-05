import importlib
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

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


def test_parse_and_group_entries(tmp_path, monkeypatch):
    mod = importlib.import_module("see.reduce")
    day_dir = copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    entries = mod.parse_monitor_files(str(day_dir))
    assert entries and entries[0]["monitor"] == 1
    groups = mod.group_entries(entries)
    assert groups
    start = next(iter(groups))
    assert hasattr(start, "hour")


def test_reduce_day(tmp_path, monkeypatch):
    mod = importlib.import_module("see.reduce")
    day_dir = copy_day(tmp_path)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("prompt")

    def fake_call(md, prompt_text, api_key, debug=False):
        return "summary"

    monkeypatch.setattr(mod, "call_gemini", fake_call)
    monkeypatch.setattr(mod, "load_dotenv", lambda: True)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    prompt_content = SimpleNamespace(text="prompt", path=prompt_path)
    monkeypatch.setattr(mod, "load_prompt", lambda *a, **k: prompt_content)
    mod.reduce_day("20240101")
    out = day_dir / "123000_screen.md"
    crumb = Path(str(out) + ".crumb")
    assert out.read_text() == "summary"
    assert crumb.is_file()


def test_scan_day(tmp_path, monkeypatch):
    mod = importlib.import_module("see.reduce")
    copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    info = mod.scan_day("20240101")
    assert "123456_screen.md" in info["processed"]
    assert "123456_monitor_1_diff.json" in info["repairable"]

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("prompt")

    def fake_call(md, prompt_text, api_key, debug=False):
        return "summary"

    monkeypatch.setattr(mod, "call_gemini", fake_call)
    monkeypatch.setattr(mod, "load_dotenv", lambda: True)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    prompt_content = SimpleNamespace(text="prompt", path=prompt_path)
    monkeypatch.setattr(mod, "load_prompt", lambda *a, **k: prompt_content)
    mod.reduce_day("20240101")

    info_after = mod.scan_day("20240101")
    assert "123000_screen.md" in info_after["processed"]
    assert not info_after["repairable"]


def test_reduce_day_parallel(tmp_path, monkeypatch):
    mod = importlib.import_module("see.reduce")
    day_dir = copy_day(tmp_path)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("prompt")

    # create a second diff file for another time block
    src = day_dir / "123456_monitor_1_diff.json"
    dest = day_dir / "124501_monitor_1_diff.json"
    shutil.copy(src, dest)

    calls: list[int] = []

    def fake_call(md, prompt_text, api_key, debug=False):
        calls.append(1)
        return "summary"

    monkeypatch.setattr(mod, "call_gemini", fake_call)
    monkeypatch.setattr(mod, "load_dotenv", lambda: True)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    prompt_content = SimpleNamespace(text="prompt", path=prompt_path)
    monkeypatch.setattr(mod, "load_prompt", lambda *a, **k: prompt_content)
    mod.reduce_day("20240101", jobs=2)

    assert len(calls) == 2
