# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the generator output pipeline.

Tests cover:
- Basic output generation via NDJSON protocol
- Hook invocation with correct context
- Generators without hooks
"""

import importlib
import io
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


# Mock result must be >= MIN_INPUT_CHARS (50) to generate output
MOCK_RESULT = {
    "text": "## Meeting Summary\n\nTeam standup at 9am with Alice and Bob discussing project status.",
    "usage": {"input_tokens": 100, "output_tokens": 50},
}


def run_generator_with_config(mod, config: dict, monkeypatch) -> list[dict]:
    """Run generator with NDJSON config and capture output events."""
    # Mock argv to prevent argparse from seeing pytest args
    monkeypatch.setattr("sys.argv", ["sol"])

    # Mock stdin with config
    stdin_data = json.dumps(config) + "\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin_data))

    # Capture stdout
    captured_output = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured_output)

    # Run main
    mod.main()

    # Parse output events
    events = []
    captured_output.seek(0)
    for line in captured_output:
        line = line.strip()
        if line:
            events.append(json.loads(line))

    return events


def test_generate_output_ndjson(tmp_path, monkeypatch):
    """Test basic output generation via NDJSON protocol."""
    mod = importlib.import_module("think.agents")
    copy_day(tmp_path)

    # Create a test generator in muse directory (with explicit sources)
    muse_dir = Path(mod.__file__).resolve().parent.parent / "muse"
    test_generator = muse_dir / "test_gen.md"
    test_generator.write_text(
        '{\n  "type": "generate",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "instructions": {"system": "journal", "sources": {"audio": true, "screen": true}}\n}\n\nTest prompt'
    )

    try:
        # Mock the underlying generation function in think.models
        import think.models

        monkeypatch.setattr(
            think.models,
            "generate_with_result",
            lambda *a, **k: MOCK_RESULT,
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        config = {
            "name": "test_gen",
            "day": "20240101",
            "output": "md",
            "provider": "google",
            "model": "gemini-2.0-flash",
        }

        events = run_generator_with_config(mod, config, monkeypatch)

        # Should have start and finish events
        assert len(events) >= 2
        assert events[0]["event"] == "start"
        assert events[0]["name"] == "test_gen"

        # Find finish event
        finish_events = [e for e in events if e["event"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["result"] == MOCK_RESULT["text"]

    finally:
        if test_generator.exists():
            test_generator.unlink()


def test_generate_hook_invoked_with_context(tmp_path, monkeypatch):
    """Test that hooks receive correct context including span flag."""
    mod = importlib.import_module("think.agents")
    copy_day(tmp_path)

    # Create the hook file in muse/ directory
    muse_dir = Path(mod.__file__).resolve().parent.parent / "muse"
    hook_file = muse_dir / "test_hook.py"
    hook_file.write_text("""
def post_process(result, context):
    import json
    from pathlib import Path
    # Write context to file for test verification
    out_path = Path(context["output_path"]).parent / "context_captured.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ctx_copy = {
        "day": context.get("day"),
        "segment": context.get("segment"),
        "span": context.get("span_mode"),
        "name": context.get("name"),
        "has_transcript": bool(context.get("transcript")),
        "has_hook": bool(context.get("hook")),  # Frontmatter fields now directly in config
    }
    with open(out_path, "w") as f:
        json.dump(ctx_copy, f)
    return None
""")

    # Create generator with hook (new format, with explicit sources)
    test_generator = muse_dir / "hooked_gen.md"
    test_generator.write_text(
        '{\n  "type": "generate",\n  "title": "Hooked",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"post": "test_hook"},\n  "instructions": {"system": "journal", "sources": {"audio": true, "screen": true}}\n}\n\nTest prompt'
    )

    try:
        # Mock the underlying generation function in think.models
        import think.models

        monkeypatch.setattr(
            think.models,
            "generate_with_result",
            lambda *a, **k: MOCK_RESULT,
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        config = {
            "name": "hooked_gen",
            "day": "20240101",
            "output": "md",
            "provider": "google",
            "model": "gemini-2.0-flash",
        }

        events = run_generator_with_config(mod, config, monkeypatch)

        # Should have start and finish events
        finish_events = [e for e in events if e["event"] == "finish"]
        assert len(finish_events) == 1

        # Read captured context
        captured_path = tmp_path / "20240101" / "agents" / "context_captured.json"
        captured = json.loads(captured_path.read_text())

        assert captured["day"] == "20240101"
        assert captured["segment"] is None
        # span_mode is a bool in the new config structure
        assert captured["span"] is False
        assert captured["name"] == "hooked_gen"
        assert captured["has_transcript"] is True
        assert captured["has_hook"] is True  # Frontmatter fields now directly in config

    finally:
        # Clean up test files
        if hook_file.exists():
            hook_file.unlink()
        if test_generator.exists():
            test_generator.unlink()


def test_generate_without_hook_succeeds(tmp_path, monkeypatch):
    """Test that generators without hooks still work correctly."""
    mod = importlib.import_module("think.agents")
    copy_day(tmp_path)

    # Create generator without hook (with explicit sources)
    muse_dir = Path(mod.__file__).resolve().parent.parent / "muse"
    test_generator = muse_dir / "nohook_gen.md"
    test_generator.write_text(
        '{\n  "type": "generate",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "instructions": {"system": "journal", "sources": {"audio": true, "screen": true}}\n}\n\nNo hook prompt'
    )

    try:
        # Mock the underlying generation function in think.models
        import think.models

        monkeypatch.setattr(
            think.models,
            "generate_with_result",
            lambda *a, **k: MOCK_RESULT,
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        config = {
            "name": "nohook_gen",
            "day": "20240101",
            "output": "md",
            "provider": "google",
            "model": "gemini-2.0-flash",
        }

        events = run_generator_with_config(mod, config, monkeypatch)

        # Should have start and finish events
        assert len(events) >= 2
        finish_events = [e for e in events if e["event"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["result"] == MOCK_RESULT["text"]

    finally:
        if test_generator.exists():
            test_generator.unlink()


def test_generate_error_event_on_missing_generator(tmp_path, monkeypatch):
    """Test that missing generator name emits error event."""
    mod = importlib.import_module("think.agents")
    copy_day(tmp_path)

    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

    config = {
        "name": "nonexistent_generator",
        "day": "20240101",
        "output": "md",
    }

    events = run_generator_with_config(mod, config, monkeypatch)

    # Should have an error event
    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) == 1
    assert "not found" in error_events[0]["error"].lower()


def test_generate_skipped_on_no_input(tmp_path, monkeypatch):
    """Test that generator emits skipped finish when no input."""
    mod = importlib.import_module("think.agents")

    # Create empty day directory (no transcripts)
    os.environ["JOURNAL_PATH"] = str(tmp_path)
    day_dir = day_path("20240101")
    day_dir.mkdir(parents=True, exist_ok=True)

    # Create a test generator (with explicit sources)
    muse_dir = Path(mod.__file__).resolve().parent.parent / "muse"
    test_generator = muse_dir / "empty_gen.md"
    test_generator.write_text(
        '{\n  "type": "generate",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "instructions": {"system": "journal", "sources": {"audio": true, "screen": true}}\n}\n\nTest prompt'
    )

    try:
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        config = {
            "name": "empty_gen",
            "day": "20240101",
            "output": "md",
            "provider": "google",
            "model": "gemini-2.0-flash",
        }

        events = run_generator_with_config(mod, config, monkeypatch)

        # Should have start and finish with skipped
        finish_events = [e for e in events if e["event"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0].get("skipped") == "no_input"

    finally:
        if test_generator.exists():
            test_generator.unlink()


def test_named_hook_resolution(tmp_path, monkeypatch):
    """Test that named hooks are resolved via load_post_hook."""
    from think.muse import load_post_hook

    # Config with named hook (new format)
    config = {"hook": {"post": "occurrence"}}
    hook_fn = load_post_hook(config)

    # Should resolve to muse/occurrence.py and be callable
    assert callable(hook_fn)
