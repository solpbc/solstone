import importlib
import json
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


def test_ponder_main(tmp_path, monkeypatch):
    mod = importlib.import_module("think.summarize")
    day_dir = copy_day(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt")

    monkeypatch.setattr(
        mod,
        "send_markdown",
        lambda *a, **k: "summary",
    )
    captured = {}

    def fake_send_occurrence(*args, **kwargs):
        captured["extra"] = kwargs.get("extra_instructions")
        return [
            {
                "type": "meeting",
                "start": "00:00:00",
                "end": "00:00:00",
                "title": "t",
                "summary": "s",
                "work": True,
                "participants": [],
                "details": "",
            }
        ]

    monkeypatch.setattr(mod, "send_occurrence", fake_send_occurrence)
    monkeypatch.setattr(mod, "load_dotenv", lambda: True)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["think-ponder", "20240101", "-f", str(prompt)])
    mod.main()

    md = day_dir / "topics" / "prompt.md"
    js = day_dir / "topics" / "prompt.json"
    assert md.read_text() == "summary"
    data = json.loads(js.read_text())
    assert data["day"] == "20240101"
    assert data["occurrences"]
    assert md.with_suffix(md.suffix + ".crumb").is_file()
    assert js.with_suffix(js.suffix + ".crumb").is_file()
    assert captured["extra"] is None


def test_ponder_extra_instructions(tmp_path, monkeypatch):
    mod = importlib.import_module("think.summarize")
    day_dir = copy_day(tmp_path)
    topic_file = Path(mod.__file__).resolve().parent / "topics" / "flow.txt"

    # Remove existing flow.md to ensure mock content is used
    flow_md = day_dir / "topics" / "flow.md"
    if flow_md.exists():
        flow_md.unlink()

    monkeypatch.setattr(
        mod,
        "send_markdown",
        lambda *a, **k: "summary",
    )
    captured = {}

    def fake_send_occurrence(*args, **kwargs):
        captured["extra"] = kwargs.get("extra_instructions")
        return [
            {
                "type": "meeting",
                "start": "00:00:00",
                "end": "00:00:00",
                "title": "t",
                "summary": "s",
                "work": True,
                "participants": [],
                "details": "",
            }
        ]

    monkeypatch.setattr(mod, "send_occurrence", fake_send_occurrence)
    monkeypatch.setattr(mod, "load_dotenv", lambda: True)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["think-ponder", "20240101", "-f", str(topic_file)])
    mod.main()

    md = day_dir / "topics" / "flow.md"
    js = day_dir / "topics" / "flow.json"
    assert md.read_text() == "summary"
    data = json.loads(js.read_text())
    assert data["day"] == "20240101"
    assert data["occurrences"]
    assert captured["extra"]


def test_ponder_skip_occurrences(tmp_path, monkeypatch):
    mod = importlib.import_module("think.summarize")
    day_dir = copy_day(tmp_path)
    topic_file = Path(mod.__file__).resolve().parent / "topics" / "flow.txt"

    # Remove existing flow.md to ensure mock content is used
    flow_md = day_dir / "topics" / "flow.md"
    if flow_md.exists():
        flow_md.unlink()

    def fake_get_topics():
        utils = importlib.import_module("think.utils")
        topics = utils.get_topics()
        topics["flow"]["occurrences"] = False
        return topics

    monkeypatch.setattr(mod, "get_topics", fake_get_topics)
    monkeypatch.setattr(
        mod,
        "send_markdown",
        lambda *a, **k: "summary",
    )
    called = {}

    def fake_send_occurrence(*args, **kwargs):
        called["called"] = True
        return []

    monkeypatch.setattr(mod, "send_occurrence", fake_send_occurrence)
    monkeypatch.setattr(mod, "load_dotenv", lambda: True)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["think-ponder", "20240101", "-f", str(topic_file)])
    mod.main()

    md = day_dir / "topics" / "flow.md"
    js = day_dir / "topics" / "flow.json"
    assert md.read_text() == "summary"
    assert not js.exists()
    assert md.with_suffix(md.suffix + ".crumb").is_file()
    assert not js.with_suffix(js.suffix + ".crumb").exists()
    assert "called" not in called
