# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the generator output pipeline.

Tests cover:
- Basic output generation and saving
- Hook invocation with correct context
- Named hook resolution
"""

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


# Mock result must be >= MIN_INPUT_CHARS (50) to generate output
MOCK_RESULT = "## Meeting Summary\n\nTeam standup at 9am with Alice and Bob discussing project status."


def test_generate_output(tmp_path, monkeypatch):
    """Test basic output generation saves markdown output."""
    mod = importlib.import_module("think.generate")
    day_dir = copy_day(tmp_path)
    prompt = tmp_path / "prompt.md"
    prompt.write_text('{\n  "schedule": "daily"\n}\n\nprompt')

    monkeypatch.setattr(
        mod,
        "generate_agent_output",
        lambda *a, **k: MOCK_RESULT,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["sol generate", "20240101", "-f", str(prompt)])
    mod.main()

    md = day_dir / "agents" / "prompt.md"
    assert md.read_text() == MOCK_RESULT


def test_generate_hook_invoked_with_context(tmp_path, monkeypatch):
    """Test that hooks receive correct context including multi_segment flag."""
    mod = importlib.import_module("think.generate")
    copy_day(tmp_path)

    # Create generator with hook
    generators_dir = tmp_path / "generators"
    generators_dir.mkdir()

    prompt_file = generators_dir / "hooked.md"
    prompt_file.write_text(
        '{\n  "title": "Hooked",\n  "schedule": "daily",\n  "hook": "test_hook"\n}\n\nTest prompt'
    )

    # Create the hook file in muse/ directory
    hooks_dir = Path(mod.__file__).resolve().parent.parent / "muse"
    hook_file = hooks_dir / "test_hook.py"
    hook_file.write_text("""
def process(result, context):
    import json
    from pathlib import Path
    # Write context to file for test verification
    out_path = Path(context["output_path"]).parent / "context_captured.json"
    ctx_copy = {
        "day": context.get("day"),
        "segment": context.get("segment"),
        "multi_segment": context.get("multi_segment"),
        "name": context.get("name"),
        "has_transcript": bool(context.get("transcript")),
        "has_meta": bool(context.get("meta")),
    }
    with open(out_path, "w") as f:
        json.dump(ctx_copy, f)
    return None
""")

    try:
        monkeypatch.setattr(
            mod,
            "generate_agent_output",
            lambda *a, **k: MOCK_RESULT,
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
        monkeypatch.setattr(
            "sys.argv", ["sol generate", "20240101", "-f", str(prompt_file)]
        )
        mod.main()

        # Read captured context
        captured_path = tmp_path / "20240101" / "agents" / "context_captured.json"
        captured = json.loads(captured_path.read_text())

        assert captured["day"] == "20240101"
        assert captured["segment"] is None
        assert captured["multi_segment"] is False
        assert captured["name"] == "hooked"
        assert captured["has_transcript"] is True
        assert captured["has_meta"] is True

    finally:
        # Clean up test hook
        if hook_file.exists():
            hook_file.unlink()


def test_generate_without_hook_succeeds(tmp_path, monkeypatch):
    """Test that generators without hooks still work correctly."""
    mod = importlib.import_module("think.generate")
    day_dir = copy_day(tmp_path)

    # Create generator without hook
    prompt = tmp_path / "nohook.md"
    prompt.write_text('{\n  "schedule": "daily"\n}\n\nNo hook prompt')

    monkeypatch.setattr(
        mod,
        "generate_agent_output",
        lambda *a, **k: MOCK_RESULT,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["sol generate", "20240101", "-f", str(prompt)])
    mod.main()

    md = day_dir / "agents" / "nohook.md"
    assert md.read_text() == MOCK_RESULT


def test_named_hook_resolution(tmp_path, monkeypatch):
    """Test that named hooks are resolved from muse/{hook}.py."""
    utils = importlib.import_module("think.utils")

    # Create generator with named hook
    generator_file = tmp_path / "test_generator.md"
    generator_file.write_text(
        '{\n  "title": "Test",\n  "hook": "occurrence"\n}\n\nTest prompt'
    )

    meta = utils._load_prompt_metadata(generator_file)

    # Should resolve to muse/occurrence.py
    assert "hook_path" in meta
    assert meta["hook_path"].endswith("occurrence.py")
    assert "muse/occurrence.py" in meta["hook_path"].replace("\\", "/")
