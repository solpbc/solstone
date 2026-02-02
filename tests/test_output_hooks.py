# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the generator output hooks system.

Tests cover:
- Hook loading and validation via load_post_hook
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


def test_load_post_hook_success(tmp_path):
    """Test loading a valid hook with post_process function."""
    agents = importlib.import_module("think.agents")

    hook_file = tmp_path / "test_hook.py"
    hook_file.write_text("""
def post_process(result, context):
    return result + "\\n\\n## Added by hook"
""")

    # Config with explicit path
    config = {"hook": {"post": str(hook_file)}}
    hook_fn = agents.load_post_hook(config)
    assert callable(hook_fn)

    # Test the hook transforms content
    output = hook_fn("Original", {"day": "20240101"})
    assert output == "Original\n\n## Added by hook"


def test_load_post_hook_missing_post_process(tmp_path):
    """Test that hook without post_process function raises ValueError."""
    agents = importlib.import_module("think.agents")

    hook_file = tmp_path / "bad_hook.py"
    hook_file.write_text("""
def other_function():
    pass
""")

    config = {"hook": {"post": str(hook_file)}}
    try:
        agents.load_post_hook(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must define a 'post_process' function" in str(e)


def test_load_post_hook_not_callable(tmp_path):
    """Test that hook with non-callable post_process raises ValueError."""
    agents = importlib.import_module("think.agents")

    hook_file = tmp_path / "bad_hook.py"
    hook_file.write_text("""
post_process = "not a function"
""")

    config = {"hook": {"post": str(hook_file)}}
    try:
        agents.load_post_hook(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "'post_process' must be callable" in str(e)


def test_load_post_hook_no_hook_config():
    """Test that missing hook config returns None."""
    agents = importlib.import_module("think.agents")

    assert agents.load_post_hook({}) is None
    assert agents.load_post_hook({"hook": {}}) is None
    assert agents.load_post_hook({"hook": {"pre": "something"}}) is None


def test_load_post_hook_named_resolution():
    """Test that named hooks resolve to muse/{name}.py."""
    agents = importlib.import_module("think.agents")

    # occurrence.py exists in muse/
    config = {"hook": {"post": "occurrence"}}
    hook_fn = agents.load_post_hook(config)
    assert callable(hook_fn)


def test_load_post_hook_file_not_found(tmp_path):
    """Test that nonexistent hook file raises ImportError."""
    agents = importlib.import_module("think.agents")

    config = {"hook": {"post": str(tmp_path / "nonexistent.py")}}
    try:
        agents.load_post_hook(config)
        assert False, "Should have raised ImportError"
    except ImportError as e:
        assert "not found" in str(e)


def test_prompt_metadata_no_hook_path(tmp_path):
    """Test that _load_prompt_metadata no longer sets hook_path."""
    utils = importlib.import_module("think.utils")

    md_file = tmp_path / "test_generator.md"
    md_file.write_text(
        '{\n  "title": "Test",\n  "hook": {"post": "entities"}\n}\n\nTest prompt'
    )

    # Create a co-located .py file
    hook_file = tmp_path / "test_generator.py"
    hook_file.write_text("def post_process(r, c): return r")

    meta = utils._load_prompt_metadata(md_file)

    # hook_path should no longer be set (hooks are loaded via load_post_hook)
    assert "hook_path" not in meta
    assert meta["path"] == str(md_file)
    assert meta["title"] == "Test"


def test_output_hook_invocation(tmp_path, monkeypatch):
    """Test that agents.py invokes hook and uses transformed result."""
    mod = importlib.import_module("think.agents")
    copy_day(tmp_path)

    # Create generator with hook in muse directory
    muse_dir = Path(mod.__file__).resolve().parent.parent / "muse"

    prompt_file = muse_dir / "hooked_test.md"
    prompt_file.write_text(
        '{\n  "title": "Hooked",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"post": "hooked_test"},\n  "instructions": {"system": "journal", "sources": {"audio": true, "screen": true}}\n}\n\nTest prompt'
    )

    hook_file = muse_dir / "hooked_test.py"
    hook_file.write_text("""
def post_process(result, context):
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
        '{\n  "title": "Noop",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"post": "noop_test"},\n  "instructions": {"system": "journal", "sources": {"audio": true, "screen": true}}\n}\n\nTest prompt'
    )

    hook_file = muse_dir / "noop_test.py"
    hook_file.write_text("""
def post_process(result, context):
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
        '{\n  "title": "Broken",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"post": "broken_test"},\n  "instructions": {"system": "journal", "sources": {"audio": true, "screen": true}}\n}\n\nTest prompt'
    )

    hook_file = muse_dir / "broken_test.py"
    hook_file.write_text("""
def post_process(result, context):
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


def test_build_hook_context():
    """Test that build_hook_context creates correct context."""
    agents = importlib.import_module("think.agents")

    config = {
        "name": "test_gen",
        "agent_id": "123456",
        "provider": "google",
        "model": "gemini-2.0-flash",
        "prompt": "test prompt",
        "output": "md",
        "day": "20240101",
        "segment": "120000_3600",
    }

    context = agents.build_hook_context(
        config,
        transcript="test transcript",
        output_path="/tmp/test.md",
        span=False,
    )

    assert context["name"] == "test_gen"
    assert context["agent_id"] == "123456"
    assert context["provider"] == "google"
    assert context["model"] == "gemini-2.0-flash"
    assert context["prompt"] == "test prompt"
    assert context["output_format"] == "md"
    assert context["day"] == "20240101"
    assert context["segment"] == "120000_3600"
    assert context["transcript"] == "test transcript"
    assert context["output_path"] == "/tmp/test.md"
    assert context["span"] is False
    assert context["meta"] == config


def test_run_post_hook_transforms_result():
    """Test that run_post_hook applies transformation."""
    agents = importlib.import_module("think.agents")

    def hook(result, context):
        return result.upper()

    context = agents.build_hook_context({"name": "test"})
    output = agents.run_post_hook("hello world", context, hook)

    assert output == "HELLO WORLD"


def test_run_post_hook_none_keeps_original():
    """Test that run_post_hook keeps original when hook returns None."""
    agents = importlib.import_module("think.agents")

    def hook(result, context):
        return None

    context = agents.build_hook_context({"name": "test"})
    output = agents.run_post_hook("original", context, hook)

    assert output == "original"


def test_run_post_hook_error_keeps_original():
    """Test that run_post_hook keeps original on error."""
    agents = importlib.import_module("think.agents")

    def hook(result, context):
        raise RuntimeError("boom")

    context = agents.build_hook_context({"name": "test"})
    output = agents.run_post_hook("original", context, hook)

    assert output == "original"


# =============================================================================
# Pre-hook Tests
# =============================================================================


def test_load_pre_hook_success(tmp_path):
    """Test loading a valid hook with pre_process function."""
    agents = importlib.import_module("think.agents")

    hook_file = tmp_path / "test_pre_hook.py"
    hook_file.write_text("""
def pre_process(context):
    return {"prompt": context["prompt"] + " [modified]"}
""")

    config = {"hook": {"pre": str(hook_file)}}
    hook_fn = agents.load_pre_hook(config)
    assert callable(hook_fn)

    # Test the hook returns modifications
    result = hook_fn({"prompt": "original"})
    assert result == {"prompt": "original [modified]"}


def test_load_pre_hook_missing_pre_process(tmp_path):
    """Test that hook without pre_process function raises ValueError."""
    agents = importlib.import_module("think.agents")

    hook_file = tmp_path / "bad_hook.py"
    hook_file.write_text("""
def other_function():
    pass
""")

    config = {"hook": {"pre": str(hook_file)}}
    try:
        agents.load_pre_hook(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must define a 'pre_process' function" in str(e)


def test_load_pre_hook_not_callable(tmp_path):
    """Test that hook with non-callable pre_process raises ValueError."""
    agents = importlib.import_module("think.agents")

    hook_file = tmp_path / "bad_hook.py"
    hook_file.write_text("""
pre_process = "not a function"
""")

    config = {"hook": {"pre": str(hook_file)}}
    try:
        agents.load_pre_hook(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "'pre_process' must be callable" in str(e)


def test_load_pre_hook_no_hook_config():
    """Test that missing hook config returns None."""
    agents = importlib.import_module("think.agents")

    assert agents.load_pre_hook({}) is None
    assert agents.load_pre_hook({"hook": {}}) is None
    assert agents.load_pre_hook({"hook": {"post": "something"}}) is None


def test_load_pre_hook_file_not_found(tmp_path):
    """Test that nonexistent hook file raises ImportError."""
    agents = importlib.import_module("think.agents")

    config = {"hook": {"pre": str(tmp_path / "nonexistent.py")}}
    try:
        agents.load_pre_hook(config)
        assert False, "Should have raised ImportError"
    except ImportError as e:
        assert "not found" in str(e)


def test_build_pre_hook_context():
    """Test that build_pre_hook_context creates correct context."""
    agents = importlib.import_module("think.agents")

    config = {
        "name": "test_gen",
        "agent_id": "123456",
        "provider": "google",
        "model": "gemini-2.0-flash",
        "prompt": "test prompt",
        "system_instruction": "be helpful",
        "user_instruction": "answer questions",
        "extra_context": "extra info",
        "output": "md",
        "day": "20240101",
        "segment": "120000_3600",
    }

    context = agents.build_pre_hook_context(
        config,
        transcript="test transcript",
        output_path="/tmp/test.md",
        span=False,
    )

    assert context["name"] == "test_gen"
    assert context["agent_id"] == "123456"
    assert context["provider"] == "google"
    assert context["model"] == "gemini-2.0-flash"
    assert context["prompt"] == "test prompt"
    assert context["system_instruction"] == "be helpful"
    assert context["user_instruction"] == "answer questions"
    assert context["extra_context"] == "extra info"
    assert context["output_format"] == "md"
    assert context["day"] == "20240101"
    assert context["segment"] == "120000_3600"
    assert context["transcript"] == "test transcript"
    assert context["output_path"] == "/tmp/test.md"
    assert context["span"] is False
    assert context["meta"] == config


def test_run_pre_hook_returns_modifications():
    """Test that run_pre_hook returns modifications dict."""
    agents = importlib.import_module("think.agents")

    def hook(context):
        return {"prompt": "modified prompt", "transcript": "modified transcript"}

    context = agents.build_pre_hook_context({"name": "test", "prompt": "original"})
    result = agents.run_pre_hook(context, hook)

    assert result == {"prompt": "modified prompt", "transcript": "modified transcript"}


def test_run_pre_hook_none_returns_none():
    """Test that run_pre_hook returns None when hook returns None."""
    agents = importlib.import_module("think.agents")

    def hook(context):
        return None

    context = agents.build_pre_hook_context({"name": "test"})
    result = agents.run_pre_hook(context, hook)

    assert result is None


def test_run_pre_hook_error_returns_none():
    """Test that run_pre_hook returns None on error."""
    agents = importlib.import_module("think.agents")

    def hook(context):
        raise RuntimeError("boom")

    context = agents.build_pre_hook_context({"name": "test"})
    result = agents.run_pre_hook(context, hook)

    assert result is None


def test_pre_hook_invocation(tmp_path, monkeypatch):
    """Test that agents.py invokes pre-hook and uses modified inputs."""
    mod = importlib.import_module("think.agents")
    copy_day(tmp_path)

    muse_dir = Path(mod.__file__).resolve().parent.parent / "muse"

    prompt_file = muse_dir / "prehooked_test.md"
    prompt_file.write_text(
        '{\n  "title": "Prehooked",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"pre": "prehooked_test"},\n  "instructions": {"system": "journal", "sources": {"audio": true, "screen": true}}\n}\n\nOriginal prompt'
    )

    hook_file = muse_dir / "prehooked_test.py"
    hook_file.write_text("""
def pre_process(context):
    # Verify context has expected fields
    assert "transcript" in context
    assert "prompt" in context
    assert "system_instruction" in context
    # Modify the prompt
    return {"prompt": context["prompt"] + " [pre-processed]"}
""")

    try:
        # Track what generate_agent_output receives
        received_args = {}

        def mock_generate(*args, **kwargs):
            received_args["transcript"] = args[0]
            received_args["prompt"] = args[1]
            return MOCK_RESULT if kwargs.get("return_result") else MOCK_RESULT["text"]

        monkeypatch.setattr(mod, "generate_agent_output", mock_generate)
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        config = {
            "name": "prehooked_test",
            "day": "20240101",
            "output": "md",
            "provider": "google",
            "model": "gemini-2.0-flash",
        }

        events = run_generator_with_config(mod, config, monkeypatch)

        # Verify pre-hook modified the prompt
        assert "[pre-processed]" in received_args["prompt"]

        # Verify generator still completed successfully
        finish_events = [e for e in events if e["event"] == "finish"]
        assert len(finish_events) == 1

    finally:
        if hook_file.exists():
            hook_file.unlink()
        if prompt_file.exists():
            prompt_file.unlink()


def test_both_pre_and_post_hooks(tmp_path, monkeypatch):
    """Test that both pre and post hooks can be configured together."""
    mod = importlib.import_module("think.agents")
    copy_day(tmp_path)

    muse_dir = Path(mod.__file__).resolve().parent.parent / "muse"

    prompt_file = muse_dir / "both_hooks_test.md"
    prompt_file.write_text(
        '{\n  "title": "Both Hooks",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"pre": "both_hooks_test", "post": "both_hooks_test"},\n  "instructions": {"system": "journal", "sources": {"audio": true, "screen": true}}\n}\n\nOriginal prompt'
    )

    hook_file = muse_dir / "both_hooks_test.py"
    hook_file.write_text("""
def pre_process(context):
    return {"prompt": context["prompt"] + " [pre]"}

def post_process(result, context):
    return result + "\\n\\n[post]"
""")

    try:
        received_args = {}

        def mock_generate(*args, **kwargs):
            received_args["prompt"] = args[1]
            return MOCK_RESULT if kwargs.get("return_result") else MOCK_RESULT["text"]

        monkeypatch.setattr(mod, "generate_agent_output", mock_generate)
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))

        config = {
            "name": "both_hooks_test",
            "day": "20240101",
            "output": "md",
            "provider": "google",
            "model": "gemini-2.0-flash",
        }

        events = run_generator_with_config(mod, config, monkeypatch)

        # Verify pre-hook modified the prompt
        assert "[pre]" in received_args["prompt"]

        # Verify post-hook modified the result
        finish_events = [e for e in events if e["event"] == "finish"]
        assert len(finish_events) == 1
        assert "[post]" in finish_events[0]["result"]

    finally:
        if hook_file.exists():
            hook_file.unlink()
        if prompt_file.exists():
            prompt_file.unlink()
