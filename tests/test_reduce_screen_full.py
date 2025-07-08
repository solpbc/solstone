import importlib
import shutil
from pathlib import Path
from types import SimpleNamespace

FIXTURES = Path("fixtures")


def copy_day(tmp_path: Path) -> Path:
    src = FIXTURES / "journal" / "20240101"
    dest = tmp_path / "20240101"
    shutil.copytree(src, dest)
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
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt")

    def fake_call(md, prompt_text, api_key, debug=False):
        return "summary", SimpleNamespace(prompt_token_count=1, candidates_token_count=1)

    monkeypatch.setattr(mod, "call_gemini", fake_call)
    monkeypatch.setattr(mod, "load_dotenv", lambda: True)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    mod.reduce_day("20240101", str(prompt))
    out = day_dir / "123000_screen.md"
    crumb = Path(str(out) + ".crumb")
    assert out.read_text() == "summary"
    assert crumb.is_file()
