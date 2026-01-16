# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

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


# Mock result must be >= MIN_EXTRACTION_CHARS (50) to trigger extraction
MOCK_RESULT = "## Meeting Summary\n\nTeam standup at 9am with Alice and Bob discussing project status."


def test_ponder_main(tmp_path, monkeypatch):
    mod = importlib.import_module("think.insight")
    day_dir = copy_day(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt")

    monkeypatch.setattr(
        mod,
        "send_insight",
        lambda *a, **k: MOCK_RESULT,
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
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["think-insight", "20240101", "-f", str(prompt)])
    mod.main()

    md = day_dir / "insights" / "prompt.md"
    assert md.read_text() == MOCK_RESULT
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
        "send_insight",
        lambda *a, **k: MOCK_RESULT,
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
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv", ["think-insight", "20240101", "-f", str(insight_file)]
    )
    mod.main()

    md = day_dir / "insights" / "flow.md"
    assert md.read_text() == MOCK_RESULT
    # Events now go to facets/{facet}/events/YYYYMMDD.jsonl
    events_file = tmp_path / "facets" / "work" / "events" / "20240101.jsonl"
    assert events_file.exists()
    data = json.loads(events_file.read_text().strip())
    assert data["occurred"] is True
    assert data["topic"] == "flow"
    # Facet summaries are prepended to insight-specific occurrence instructions
    assert captured["extra"]
    assert captured["extra"].startswith("No facets found.")


def test_ponder_skip_minimal_content(tmp_path, monkeypatch):
    """Test that extraction is skipped when send_insight returns minimal content."""
    mod = importlib.import_module("think.insight")
    day_dir = copy_day(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt")

    # Return minimal content that should trigger skip
    monkeypatch.setattr(
        mod,
        "send_insight",
        lambda *a, **k: "No meetings detected",
    )
    called = {}

    def fake_send_extraction(*args, **kwargs):
        called["called"] = True
        return []

    monkeypatch.setattr(mod, "send_extraction", fake_send_extraction)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["think-insight", "20240101", "-f", str(prompt)])
    mod.main()

    md = day_dir / "insights" / "prompt.md"
    assert md.read_text() == "No meetings detected"
    # Extraction should NOT have been called due to minimal content
    assert "called" not in called
    # No events file should be created
    events_file = tmp_path / "facets" / "work" / "events" / "20240101.jsonl"
    assert not events_file.exists()


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
        "send_insight",
        lambda *a, **k: MOCK_RESULT,
    )
    called = {}

    def fake_send_extraction(*args, **kwargs):
        called["called"] = True
        return []

    monkeypatch.setattr(mod, "send_extraction", fake_send_extraction)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv", ["think-insight", "20240101", "-f", str(insight_file)]
    )
    mod.main()

    md = day_dir / "insights" / "flow.md"
    js = day_dir / "insights" / "flow.json"
    assert md.read_text() == MOCK_RESULT
    assert not js.exists()
    assert "called" not in called
