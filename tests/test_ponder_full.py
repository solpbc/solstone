import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

FIXTURES = Path("fixtures")


def copy_day(tmp_path: Path) -> Path:
    src = FIXTURES / "journal" / "20240101"
    dest = tmp_path / "20240101"
    shutil.copytree(src, dest)
    return dest


def test_ponder_main(tmp_path, monkeypatch):
    mod = importlib.import_module("think.ponder")
    day_dir = copy_day(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt")

    monkeypatch.setattr(
        mod,
        "send_markdown",
        lambda *a, **k: (
            "summary",
            SimpleNamespace(
                prompt_token_count=1, thoughts_token_count=1, candidates_token_count=1
            ),
        ),
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
    mod = importlib.import_module("think.ponder")
    day_dir = copy_day(tmp_path)
    topic_file = Path(mod.__file__).resolve().parent / "topics" / "flow.txt"

    monkeypatch.setattr(
        mod,
        "send_markdown",
        lambda *a, **k: (
            "summary",
            SimpleNamespace(
                prompt_token_count=1,
                thoughts_token_count=1,
                candidates_token_count=1,
            ),
        ),
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
    mod = importlib.import_module("think.ponder")
    day_dir = copy_day(tmp_path)
    topic_file = Path(mod.__file__).resolve().parent / "topics" / "flow.txt"

    def fake_get_topics():
        utils = importlib.import_module("think.utils")
        topics = utils.get_topics()
        topics["flow"]["occurrences"] = False
        return topics

    monkeypatch.setattr(mod, "get_topics", fake_get_topics)
    monkeypatch.setattr(
        mod,
        "send_markdown",
        lambda *a, **k: (
            "summary",
            SimpleNamespace(
                prompt_token_count=1,
                thoughts_token_count=1,
                candidates_token_count=1,
            ),
        ),
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
