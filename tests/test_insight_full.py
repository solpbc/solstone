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
    mod = importlib.import_module("think.insight")
    day_dir = copy_day(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt")

    monkeypatch.setattr(
        mod,
        "send_markdown",
        lambda *a, **k: "summary",
    )
    captured = {}

    def fake_send_extraction(*args, **kwargs):
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
                "facet": "work",
            }
        ]

    monkeypatch.setattr(mod, "send_extraction", fake_send_extraction)
    monkeypatch.setattr(mod, "load_dotenv", lambda: True)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["think-insight", "20240101", "-f", str(prompt)])
    mod.main()

    md = day_dir / "insights" / "prompt.md"
    assert md.read_text() == "summary"
    # Events now go to facets/{facet}/events/YYYYMMDD.jsonl
    events_file = tmp_path / "facets" / "work" / "events" / "20240101.jsonl"
    assert events_file.exists()
    data = json.loads(events_file.read_text().strip())
    assert data["occurred"] is True
    assert data["topic"] == "prompt"
    # Facet summaries are now always included in extra_instructions
    assert captured["extra"] == "No facets found."


def test_ponder_extra_instructions(tmp_path, monkeypatch):
    mod = importlib.import_module("think.insight")
    day_dir = copy_day(tmp_path)
    insight_file = Path(mod.__file__).resolve().parent / "insights" / "flow.txt"

    # Remove existing flow.md to ensure mock content is used
    flow_md = day_dir / "insights" / "flow.md"
    if flow_md.exists():
        flow_md.unlink()

    monkeypatch.setattr(
        mod,
        "send_markdown",
        lambda *a, **k: "summary",
    )
    captured = {}

    def fake_send_extraction(*args, **kwargs):
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
                "facet": "work",
            }
        ]

    monkeypatch.setattr(mod, "send_extraction", fake_send_extraction)
    monkeypatch.setattr(mod, "load_dotenv", lambda: True)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv", ["think-insight", "20240101", "-f", str(insight_file)]
    )
    mod.main()

    md = day_dir / "insights" / "flow.md"
    assert md.read_text() == "summary"
    # Events now go to facets/{facet}/events/YYYYMMDD.jsonl
    events_file = tmp_path / "facets" / "work" / "events" / "20240101.jsonl"
    assert events_file.exists()
    data = json.loads(events_file.read_text().strip())
    assert data["occurred"] is True
    assert data["topic"] == "flow"
    # Facet summaries are prepended to insight-specific occurrence instructions
    assert captured["extra"]
    assert captured["extra"].startswith("No facets found.")


def test_ponder_skip_occurrences(tmp_path, monkeypatch):
    mod = importlib.import_module("think.insight")
    day_dir = copy_day(tmp_path)
    insight_file = Path(mod.__file__).resolve().parent / "insights" / "flow.txt"

    # Remove existing flow.md to ensure mock content is used
    flow_md = day_dir / "insights" / "flow.md"
    if flow_md.exists():
        flow_md.unlink()

    def fake_get_insights():
        utils = importlib.import_module("think.utils")
        insights = utils.get_insights()
        insights["flow"]["occurrences"] = False
        return insights

    monkeypatch.setattr(mod, "get_insights", fake_get_insights)
    monkeypatch.setattr(
        mod,
        "send_markdown",
        lambda *a, **k: "summary",
    )
    called = {}

    def fake_send_extraction(*args, **kwargs):
        called["called"] = True
        return []

    monkeypatch.setattr(mod, "send_extraction", fake_send_extraction)
    monkeypatch.setattr(mod, "load_dotenv", lambda: True)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv", ["think-insight", "20240101", "-f", str(insight_file)]
    )
    mod.main()

    md = day_dir / "insights" / "flow.md"
    js = day_dir / "insights" / "flow.json"
    assert md.read_text() == "summary"
    assert not js.exists()
    assert "called" not in called
