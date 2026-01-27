# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the generator output hooks system.

Tests cover:
- Hook loading and validation
- Hook invocation via NDJSON protocol
- Hook error handling
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
    for item in src.iterdir():
        if item.is_dir():
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest / item.name)
    return dest


MOCK_RESULT = {
    "text": "## Original Result\n\nThis is the original output content.",
    "usage": {"input_tokens": 100, "output_tokens": 50},
}


def run_generator_with_config(mod, config: dict, monkeypatch) -> list[dict]:
    """Run generator with NDJSON config and capture output events."""
    # Mock argv to prevent argparse from seeing pytest args
    monkeypatch.setattr("sys.argv", ["sol"])

    stdin_data = json.dumps(config) + "\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin_data))

    captured_output = io.StringIO()
    monkeypatch.setattr("sys.stdout", captured_output)

    mod.main()

    events = []
    captured_output.seek(0)
    for line in captured_output:
        line = line.strip()
        if line:
            events.append(json.loads(line))

    return events


def test_load_output_hook_success(tmp_path):
    """Test loading a valid hook with process function."""
    utils = importlib.import_module("think.utils")

    hook_file = tmp_path / "test_hook.py"
    hook_file.write_text("""
def process(result, context):
    return result + "\\n\\n## Added by hook"
""")

    process_func = utils.load_output_hook(hook_file)
    assert callable(process_func)

    # Test the hook transforms content
    output = process_func("Original", {"day": "20240101"})
    assert output == "Original\n\n## Added by hook"


def test_load_output_hook_missing_process(tmp_path):
    """Test that hook without process function raises ValueError."""
    utils = importlib.import_module("think.utils")

    hook_file = tmp_path / "bad_hook.py"
    hook_file.write_text("""
def other_function():
    pass
""")

    try:
        utils.load_output_hook(hook_file)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must define a 'process' function" in str(e)


def test_load_output_hook_process_not_callable(tmp_path):
    """Test that hook with non-callable process raises ValueError."""
    utils = importlib.import_module("think.utils")

    hook_file = tmp_path / "bad_hook.py"
    hook_file.write_text("""
process = "not a function"
""")

    try:
        utils.load_output_hook(hook_file)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "'process' must be callable" in str(e)


def test_prompt_metadata_includes_hook_path(tmp_path):
    """Test that _load_prompt_metadata detects .py hook file."""
    utils = importlib.import_module("think.utils")

    # Create prompt file with frontmatter
    md_file = tmp_path / "test_generator.md"
    md_file.write_text('{\n  "title": "Test",\n  "color": "#ff0000"\n}\n\nTest prompt')

    hook_file = tmp_path / "test_generator.py"
    hook_file.write_text("def process(r, c): return r")

    meta = utils._load_prompt_metadata(md_file)

    assert meta["path"] == str(md_file)
    assert meta["hook_path"] == str(hook_file)
    assert meta["title"] == "Test"


def test_prompt_metadata_no_hook(tmp_path):
    """Test that _load_prompt_metadata works without hook file."""
    utils = importlib.import_module("think.utils")

    md_file = tmp_path / "test_generator.md"
    md_file.write_text("Test prompt")

    meta = utils._load_prompt_metadata(md_file)

    assert meta["path"] == str(md_file)
    assert "hook_path" not in meta


def test_output_hook_invocation(tmp_path, monkeypatch):
    """Test that agents.py invokes hook and uses transformed result."""
    mod = importlib.import_module("think.agents")
    copy_day(tmp_path)

    # Create generator with hook in muse directory
    muse_dir = Path(mod.__file__).resolve().parent.parent / "muse"

    prompt_file = muse_dir / "hooked_test.md"
    prompt_file.write_text(
        '{\n  "title": "Hooked",\n  "schedule": "daily",\n  "output": "md"\n}\n\nTest prompt'
    )

    hook_file = muse_dir / "hooked_test.py"
    hook_file.write_text("""
def process(result, context):
    # Verify context has expected fields
    assert "day" in context
    assert "transcript" in context
    assert "name" in context
    return result + "\\n\\n## Hook was here"
""")

    try:
        monkeypatch.setattr(
            mod,
            "generate_agent_output",
            lambda *a, **k: (
                MOCK_RESULT if k.get("return_result") else MOCK_RESULT["text"]
            ),
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        config = {
            "name": "hooked_test",
            "day": "20240101",
            "output": "md",
            "provider": "google",
            "model": "gemini-2.0-flash",
        }

        events = run_generator_with_config(mod, config, monkeypatch)

        # Find finish event
        finish_events = [e for e in events if e["event"] == "finish"]
        assert len(finish_events) == 1

        content = finish_events[0]["result"]
        assert "## Original Result" in content
        assert "## Hook was here" in content

    finally:
        if hook_file.exists():
            hook_file.unlink()
        if prompt_file.exists():
            prompt_file.unlink()


def test_output_hook_returns_none(tmp_path, monkeypatch):
    """Test that hook returning None uses original result."""
    mod = importlib.import_module("think.agents")
    copy_day(tmp_path)

    muse_dir = Path(mod.__file__).resolve().parent.parent / "muse"

    prompt_file = muse_dir / "noop_test.md"
    prompt_file.write_text(
        '{\n  "title": "Noop",\n  "schedule": "daily",\n  "output": "md"\n}\n\nTest prompt'
    )

    hook_file = muse_dir / "noop_test.py"
    hook_file.write_text("""
def process(result, context):
    return None  # Signal to use original
""")

    try:
        monkeypatch.setattr(
            mod,
            "generate_agent_output",
            lambda *a, **k: (
                MOCK_RESULT if k.get("return_result") else MOCK_RESULT["text"]
            ),
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        config = {
            "name": "noop_test",
            "day": "20240101",
            "output": "md",
            "provider": "google",
            "model": "gemini-2.0-flash",
        }

        events = run_generator_with_config(mod, config, monkeypatch)

        finish_events = [e for e in events if e["event"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["result"] == MOCK_RESULT["text"]

    finally:
        if hook_file.exists():
            hook_file.unlink()
        if prompt_file.exists():
            prompt_file.unlink()


def test_output_hook_error_fallback(tmp_path, monkeypatch):
    """Test that hook errors fall back to original result."""
    mod = importlib.import_module("think.agents")
    copy_day(tmp_path)

    muse_dir = Path(mod.__file__).resolve().parent.parent / "muse"

    prompt_file = muse_dir / "broken_test.md"
    prompt_file.write_text(
        '{\n  "title": "Broken",\n  "schedule": "daily",\n  "output": "md"\n}\n\nTest prompt'
    )

    hook_file = muse_dir / "broken_test.py"
    hook_file.write_text("""
def process(result, context):
    raise RuntimeError("Hook exploded!")
""")

    try:
        monkeypatch.setattr(
            mod,
            "generate_agent_output",
            lambda *a, **k: (
                MOCK_RESULT if k.get("return_result") else MOCK_RESULT["text"]
            ),
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        config = {
            "name": "broken_test",
            "day": "20240101",
            "output": "md",
            "provider": "google",
            "model": "gemini-2.0-flash",
        }

        # Should not raise, should fall back gracefully
        events = run_generator_with_config(mod, config, monkeypatch)

        finish_events = [e for e in events if e["event"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["result"] == MOCK_RESULT["text"]

    finally:
        if hook_file.exists():
            hook_file.unlink()
        if prompt_file.exists():
            prompt_file.unlink()


def test_named_hook_resolution_takes_precedence(tmp_path):
    """Test that named hooks via 'hook' field take precedence over co-located .py files."""
    utils = importlib.import_module("think.utils")

    # Create prompt file with named hook
    md_file = tmp_path / "test_generator.md"
    md_file.write_text(
        '{\n  "title": "Test",\n  "hook": "occurrence"\n}\n\nTest prompt'
    )

    # Also create a co-located .py file that would normally be picked up
    colocated_hook = tmp_path / "test_generator.py"
    colocated_hook.write_text("def process(r, c): return 'colocated'")

    meta = utils._load_prompt_metadata(md_file)

    # Should resolve to named hook, not co-located
    assert "hook_path" in meta
    assert meta["hook_path"].endswith("occurrence.py")
    assert "muse/occurrence.py" in meta["hook_path"].replace("\\", "/")


def test_named_hook_nonexistent_falls_through(tmp_path):
    """Test that nonexistent named hooks fall back to co-located .py files."""
    utils = importlib.import_module("think.utils")

    # Create prompt file with nonexistent named hook
    md_file = tmp_path / "test_generator.md"
    md_file.write_text(
        '{\n  "title": "Test",\n  "hook": "nonexistent_hook_xyz"\n}\n\nTest prompt'
    )

    # Create a co-located .py file
    colocated_hook = tmp_path / "test_generator.py"
    colocated_hook.write_text("def process(r, c): return 'colocated'")

    meta = utils._load_prompt_metadata(md_file)

    # Named hook doesn't exist, so no hook_path should be set
    assert "hook_path" not in meta
