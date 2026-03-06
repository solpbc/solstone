# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.importers.gemini — Gemini/Bard activity importer."""

import json
import os
import tempfile
import zipfile
from pathlib import Path

from think.importers.gemini import GeminiImporter, _parse_activity, _strip_html

importer = GeminiImporter()


def _sample_activity(
    prompt: str = "What is Python?",
    response: str = "Python is a programming language.",
    time: str = "2026-01-15T10:30:00Z",
    header: str = "Gemini Apps",
    title: str | None = None,
) -> dict:
    """Build a sample Gemini activity record."""
    act: dict = {
        "header": header,
        "title": title or f"Asked Gemini: {prompt[:40]}",
        "time": time,
        "products": ["Gemini"],
        "subtitles": [{"value": prompt}],
        "safeHtmlItem": [{"html": f"<p>{response}</p>"}],
    }
    return act


def _bard_activity() -> dict:
    return _sample_activity(
        prompt="Tell me a joke",
        response="Why did the chicken cross the road?",
        header="Bard",
        title="Talked to Bard",
    )


# --- Unit tests for helpers ---


def test_strip_html():
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert _strip_html("No tags") == "No tags"
    assert _strip_html("&amp; entities &lt;") == "& entities <"


def test_parse_activity_basic():
    act = _sample_activity()
    entry = _parse_activity(act)
    assert entry is not None
    assert entry["type"] == "ai_chat"
    assert entry["source"] == "gemini"
    assert "Python" in entry["content"]
    assert entry["message_count"] == 2


def test_parse_activity_bard():
    entry = _parse_activity(_bard_activity())
    assert entry is not None
    assert entry.get("variant") == "bard"
    assert entry["source"] == "gemini"


def test_parse_activity_no_content():
    act = {"header": "Gemini Apps", "time": "2026-01-15T10:00:00Z"}
    assert _parse_activity(act) is None


def test_parse_activity_no_time():
    act = _sample_activity()
    del act["time"]
    assert _parse_activity(act) is None


def test_parse_activity_prompt_only():
    act = {
        "header": "Gemini Apps",
        "title": "Asked Gemini",
        "time": "2026-01-15T10:00:00Z",
        "subtitles": [{"value": "What is the meaning of life?"}],
        "products": ["Gemini"],
    }
    entry = _parse_activity(act)
    assert entry is not None
    assert "meaning of life" in entry["content"]
    assert entry["message_count"] == 1


def test_parse_activity_title_cleanup():
    act = _sample_activity(title="Asked Gemini: What is Python?")
    entry = _parse_activity(act)
    assert entry is not None
    assert entry["title"] == "What is Python?"


# --- Detection tests ---


def test_detect_json_file():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump([_sample_activity()], f)
        f.flush()
        try:
            assert importer.detect(Path(f.name)) is True
        finally:
            os.unlink(f.name)


def test_detect_json_wrong_format():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump([{"not": "gemini"}], f)
        f.flush()
        try:
            assert importer.detect(Path(f.name)) is False
        finally:
            os.unlink(f.name)


def test_detect_zip():
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        with zipfile.ZipFile(tmp, "w") as zf:
            data = json.dumps([_sample_activity()])
            zf.writestr("Takeout/My Activity/Gemini Apps/MyActivity.json", data)
        try:
            assert importer.detect(Path(tmp.name)) is True
        finally:
            os.unlink(tmp.name)


def test_detect_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        activity_dir = Path(tmpdir) / "My Activity" / "Gemini Apps"
        activity_dir.mkdir(parents=True)
        (activity_dir / "MyActivity.json").write_text(json.dumps([_sample_activity()]))
        assert importer.detect(Path(tmpdir)) is True


def test_detect_directory_no_activity():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert importer.detect(Path(tmpdir)) is False


# --- Preview tests ---


def test_preview_json():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        activities = [
            _sample_activity(time="2026-01-15T10:00:00Z"),
            _sample_activity(time="2026-02-20T14:00:00Z"),
            _bard_activity(),
        ]
        json.dump(activities, f)
        f.flush()
        try:
            preview = importer.preview(Path(f.name))
            assert preview.item_count == 3
            assert preview.date_range[0] == "20260115"
            assert "Bard" in preview.summary or "bard" in preview.summary.lower()
        finally:
            os.unlink(f.name)


# --- Process tests ---


def test_process_json():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        activities = [
            _sample_activity(time="2026-01-15T10:00:00Z"),
            _sample_activity(
                prompt="How to sort a list?",
                response="Use sorted().",
                time="2026-01-15T14:00:00Z",
            ),
        ]
        json.dump(activities, f)
        f.flush()

        try:
            with tempfile.TemporaryDirectory() as journal:
                os.environ["JOURNAL_PATH"] = journal
                result = importer.process(Path(f.name), Path(journal))
                assert result.entries_written == 2
                assert len(result.files_created) == 1  # same day
                assert result.errors == []

                # Verify JSONL content
                jsonl_path = Path(result.files_created[0])
                assert jsonl_path.exists()
                lines = jsonl_path.read_text().strip().split("\n")
                header = json.loads(lines[0])
                assert header["import"]["source"] == "gemini"
                assert header["entry_count"] == 2

                entry = json.loads(lines[1])
                assert entry["type"] == "ai_chat"
                assert entry["source"] == "gemini"
        finally:
            os.unlink(f.name)
            os.environ.pop("JOURNAL_PATH", None)


def test_process_zip():
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        with zipfile.ZipFile(tmp, "w") as zf:
            activities = [_sample_activity(time="2026-03-01T09:00:00Z")]
            zf.writestr(
                "Takeout/My Activity/Gemini Apps/MyActivity.json",
                json.dumps(activities),
            )
        try:
            with tempfile.TemporaryDirectory() as journal:
                os.environ["JOURNAL_PATH"] = journal
                result = importer.process(Path(tmp.name), Path(journal))
                assert result.entries_written == 1
                assert len(result.files_created) == 1
        finally:
            os.unlink(tmp.name)
            os.environ.pop("JOURNAL_PATH", None)


# --- Registry test ---


def test_registered_in_registry():
    from think.importers.file_importer import FILE_IMPORTER_REGISTRY, get_file_importer

    assert "gemini" in FILE_IMPORTER_REGISTRY
    imp = get_file_importer("gemini")
    assert imp is not None
    assert imp.name == "gemini"
