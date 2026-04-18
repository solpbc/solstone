# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the generator output hooks system.

Tests cover:
- Hook loading and validation via load_post_hook / load_pre_hook
- Hook invocation via NDJSON protocol
- Hook error handling
"""

import importlib
import io
import json
import logging
import os
from pathlib import Path

import talent.occurrence as occurrence
from tests.conftest import copytree_tracked
from think.hooks import write_events_jsonl
from think.talent import load_post_hook, load_pre_hook
from think.talents import _apply_template_vars
from think.utils import day_path

FIXTURES = Path("tests/fixtures")


def copy_day(tmp_path: Path) -> Path:
    os.environ["_SOLSTONE_JOURNAL_OVERRIDE"] = str(tmp_path)
    dest = day_path("20240101")
    src = FIXTURES / "journal" / "chronicle" / "20240101"
    copytree_tracked(src, dest)
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
    hook_file = tmp_path / "test_hook.py"
    hook_file.write_text("""
def post_process(result, context):
    return result + "\\n\\n## Added by hook"
""")

    # Config with explicit path
    config = {"hook": {"post": str(hook_file)}}
    hook_fn = load_post_hook(config)
    assert callable(hook_fn)

    # Test the hook transforms content
    output = hook_fn("Original", {"day": "20240101"})
    assert output == "Original\n\n## Added by hook"


def test_load_post_hook_missing_post_process(tmp_path):
    """Test that hook without post_process function raises ValueError."""
    hook_file = tmp_path / "bad_hook.py"
    hook_file.write_text("""
def other_function():
    pass
""")

    config = {"hook": {"post": str(hook_file)}}
    try:
        load_post_hook(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must define a 'post_process' function" in str(e)


def test_load_post_hook_not_callable(tmp_path):
    """Test that hook with non-callable post_process raises ValueError."""
    hook_file = tmp_path / "bad_hook.py"
    hook_file.write_text("""
post_process = "not a function"
""")

    config = {"hook": {"post": str(hook_file)}}
    try:
        load_post_hook(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "'post_process' must be callable" in str(e)


def test_load_post_hook_no_hook_config():
    """Test that missing hook config returns None."""
    assert load_post_hook({}) is None
    assert load_post_hook({"hook": {}}) is None
    assert load_post_hook({"hook": {"pre": "something"}}) is None


def test_load_post_hook_named_resolution():
    """Test that named hooks resolve to talent/{name}.py."""
    # occurrence.py exists in talent/
    config = {"hook": {"post": "occurrence"}}
    hook_fn = load_post_hook(config)
    assert callable(hook_fn)


def test_load_post_hook_file_not_found(tmp_path):
    """Test that nonexistent hook file raises ImportError."""
    config = {"hook": {"post": str(tmp_path / "nonexistent.py")}}
    try:
        load_post_hook(config)
        assert False, "Should have raised ImportError"
    except ImportError as e:
        assert "not found" in str(e)


def test_prompt_metadata_no_hook_path(tmp_path):
    """Test that _load_prompt_metadata no longer sets hook_path."""
    talent = importlib.import_module("think.talent")

    md_file = tmp_path / "test_generator.md"
    md_file.write_text(
        '{\n  "title": "Test",\n  "hook": {"post": "entities"}\n}\n\nTest prompt'
    )

    # Create a co-located .py file
    hook_file = tmp_path / "test_generator.py"
    hook_file.write_text("def post_process(r, c): return r")

    meta = talent._load_prompt_metadata(md_file)

    # hook_path should no longer be set (hooks are loaded via load_post_hook)
    assert "hook_path" not in meta
    assert meta["path"] == str(md_file)
    assert meta["title"] == "Test"


def test_output_hook_invocation(tmp_path, monkeypatch):
    """Test that agents.py invokes hook and uses transformed result."""
    mod = importlib.import_module("think.talents")
    copy_day(tmp_path)

    # Use tmp_path as talent directory to avoid polluting real talent/
    import think.talent

    monkeypatch.setattr(think.talent, "TALENT_DIR", tmp_path)

    prompt_file = tmp_path / "hooked_test.md"
    prompt_file.write_text(
        '{\n  "type": "generate",\n  "title": "Hooked",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"post": "hooked_test"},\n  "load": {"transcripts": true, "percepts": true}\n}\n\nTest prompt'
    )

    hook_file = tmp_path / "hooked_test.py"
    hook_file.write_text("""
def post_process(result, context):
    # Verify context has expected fields
    assert "day" in context
    assert "transcript" in context
    assert "name" in context
    return result + "\\n\\n## Hook was here"
""")

    # Mock the underlying generation function in think.models
    import think.models

    monkeypatch.setattr(
        think.models,
        "generate_with_result",
        lambda *a, **k: MOCK_RESULT,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

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


def test_output_hook_returns_none(tmp_path, monkeypatch):
    """Test that hook returning None uses original result."""
    mod = importlib.import_module("think.talents")
    copy_day(tmp_path)

    import think.talent

    monkeypatch.setattr(think.talent, "TALENT_DIR", tmp_path)

    prompt_file = tmp_path / "noop_test.md"
    prompt_file.write_text(
        '{\n  "type": "generate",\n  "title": "Noop",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"post": "noop_test"},\n  "load": {"transcripts": true, "percepts": true}\n}\n\nTest prompt'
    )

    hook_file = tmp_path / "noop_test.py"
    hook_file.write_text("""
def post_process(result, context):
    return None  # Signal to use original
""")

    # Mock the underlying generation function in think.models
    import think.models

    monkeypatch.setattr(
        think.models,
        "generate_with_result",
        lambda *a, **k: MOCK_RESULT,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

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


def test_output_hook_error_fallback(tmp_path, monkeypatch):
    """Test that hook errors fall back to original result."""
    mod = importlib.import_module("think.talents")
    copy_day(tmp_path)

    import think.talent

    monkeypatch.setattr(think.talent, "TALENT_DIR", tmp_path)

    prompt_file = tmp_path / "broken_test.md"
    prompt_file.write_text(
        '{\n  "type": "generate",\n  "title": "Broken",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"post": "broken_test"},\n  "load": {"transcripts": true, "percepts": true}\n}\n\nTest prompt'
    )

    hook_file = tmp_path / "broken_test.py"
    hook_file.write_text("""
def post_process(result, context):
    raise RuntimeError("Hook exploded!")
""")

    # Mock the underlying generation function in think.models
    import think.models

    monkeypatch.setattr(
        think.models,
        "generate_with_result",
        lambda *a, **k: MOCK_RESULT,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

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


def test_occurrence_post_process_drops_meeting_with_26_participants(
    monkeypatch, caplog
):
    """Test megameeting occurrences are dropped before writing."""
    captured = {}
    participants = [f"Person {i}" for i in range(26)]

    def mock_generate(**kwargs):
        return json.dumps(
            [
                {
                    "type": "meeting",
                    "title": "All Hands",
                    "summary": "Large meeting",
                    "work": True,
                    "participants": participants,
                    "facet": "capulet",
                    "details": "",
                }
            ]
        )

    def mock_write_events_jsonl(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(occurrence, "generate", mock_generate)
    monkeypatch.setattr(
        occurrence, "compute_output_source", lambda context: "source.md"
    )
    monkeypatch.setattr(occurrence, "write_events_jsonl", mock_write_events_jsonl)
    caplog.set_level(logging.WARNING)

    result = occurrence.post_process(
        "x" * 60,
        {
            "name": "meetings",
            "day": "20240101",
            "meta": {},
            "output_path": "ignored",
        },
    )

    assert result is None
    assert captured["events"] == []
    assert "Dropping megameeting occurrence" in caplog.text
    assert "All Hands" in caplog.text
    assert "meetings" in caplog.text
    assert "26" in caplog.text


def test_occurrence_post_process_keeps_meeting_with_25_participants(
    monkeypatch, caplog
):
    """Test meetings at the participant threshold are preserved."""
    captured = {}
    event = {
        "type": "meeting",
        "title": "Planning",
        "summary": "Planning meeting",
        "work": True,
        "participants": [f"Person {i}" for i in range(25)],
        "facet": "capulet",
        "details": "",
    }

    monkeypatch.setattr(occurrence, "generate", lambda **kwargs: json.dumps([event]))
    monkeypatch.setattr(
        occurrence, "compute_output_source", lambda context: "source.md"
    )
    monkeypatch.setattr(
        occurrence,
        "write_events_jsonl",
        lambda **kwargs: captured.update(kwargs) or [],
    )
    caplog.set_level(logging.WARNING)

    result = occurrence.post_process(
        "x" * 60,
        {
            "name": "meetings",
            "day": "20240101",
            "meta": {},
            "output_path": "ignored",
        },
    )

    assert result is None
    assert captured["events"] == [event]
    assert "Dropping megameeting occurrence" not in caplog.text


def test_occurrence_post_process_keeps_non_meeting_with_large_participants_list(
    monkeypatch, caplog
):
    """Test non-meeting events are not filtered by participant count."""
    captured = {}
    event = {
        "type": "message",
        "title": "Inbox review",
        "summary": "Reviewed messages",
        "work": True,
        "participants": [f"Person {i}" for i in range(100)],
        "facet": "capulet",
        "details": "",
    }

    monkeypatch.setattr(occurrence, "generate", lambda **kwargs: json.dumps([event]))
    monkeypatch.setattr(
        occurrence, "compute_output_source", lambda context: "source.md"
    )
    monkeypatch.setattr(
        occurrence,
        "write_events_jsonl",
        lambda **kwargs: captured.update(kwargs) or [],
    )
    caplog.set_level(logging.WARNING)

    result = occurrence.post_process(
        "x" * 60,
        {
            "name": "timeline",
            "day": "20240101",
            "meta": {},
            "output_path": "ignored",
        },
    )

    assert result is None
    assert captured["events"] == [event]
    assert "Dropping megameeting occurrence" not in caplog.text


def test_write_events_jsonl_skips_trailing_comma_facet(journal_copy, caplog):
    """Test invalid trailing punctuation facets are rejected."""
    caplog.set_level(logging.WARNING)

    written = write_events_jsonl(
        events=[
            {
                "type": "message",
                "title": "Chat",
                "summary": "Sent a chat",
                "work": True,
                "participants": [],
                "facet": "kognova,",
                "details": "",
            }
        ],
        agent="timeline",
        occurred=True,
        source_output="20240101/agents/timeline.md",
        capture_day="20240101",
    )

    assert written == []
    assert "Skipping event with unknown facet" in caplog.text
    assert "kognova," in caplog.text
    assert "timeline" in caplog.text
    assert "20240101/agents/timeline.md" in caplog.text
    assert not (journal_copy / "facets" / "kognova," / "events").exists()


def test_write_events_jsonl_skips_unknown_person_facet(journal_copy, caplog):
    """Test unknown person-like facet names are rejected."""
    caplog.set_level(logging.WARNING)

    written = write_events_jsonl(
        events=[
            {
                "type": "message",
                "title": "Chat",
                "summary": "Sent a chat",
                "work": True,
                "participants": [],
                "facet": "Person",
                "details": "",
            }
        ],
        agent="timeline",
        occurred=True,
        source_output="20240101/agents/timeline.md",
        capture_day="20240101",
    )

    assert written == []
    assert "Skipping event with unknown facet" in caplog.text
    assert "Person" in caplog.text
    assert not (journal_copy / "facets" / "Person" / "events").exists()


def test_write_events_jsonl_skips_mixed_case_known_facet(journal_copy, caplog):
    """Test mixed-case facet values are rejected when not exact registry matches."""
    caplog.set_level(logging.WARNING)

    written = write_events_jsonl(
        events=[
            {
                "type": "message",
                "title": "Chat",
                "summary": "Sent a chat",
                "work": True,
                "participants": [],
                "facet": "Capulet",
                "details": "",
            }
        ],
        agent="timeline",
        occurred=True,
        source_output="20240101/agents/timeline.md",
        capture_day="20240101",
    )

    assert written == []
    assert "Skipping event with unknown facet" in caplog.text
    assert "Capulet" in caplog.text
    assert not (journal_copy / "facets" / "Capulet" / "events").exists()


def test_write_events_jsonl_writes_valid_registry_facet(journal_copy):
    """Test valid registry facets are written normally."""
    written = write_events_jsonl(
        events=[
            {
                "type": "message",
                "title": "Chat",
                "summary": "Sent a chat",
                "work": True,
                "participants": [],
                "facet": "capulet",
                "details": "",
            }
        ],
        agent="timeline",
        occurred=True,
        source_output="20240101/agents/timeline.md",
        capture_day="20240101",
    )

    jsonl_path = journal_copy / "facets" / "capulet" / "events" / "20240101.jsonl"

    assert written == [jsonl_path]
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    assert rows[0]["facet"] == "capulet"
    assert rows[0]["agent"] == "timeline"
    assert rows[0]["source"] == "20240101/agents/timeline.md"


def test_write_events_jsonl_skips_empty_facet(journal_copy, caplog):
    """Test missing facets are skipped and logged."""
    caplog.set_level(logging.WARNING)

    written = write_events_jsonl(
        events=[
            {
                "type": "message",
                "title": "Chat",
                "summary": "Sent a chat",
                "work": True,
                "participants": [],
                "details": "",
            }
        ],
        agent="timeline",
        occurred=True,
        source_output="20240101/agents/timeline.md",
        capture_day="20240101",
    )

    assert written == []
    assert "Skipping event with unknown facet" in caplog.text
    assert "timeline" in caplog.text
    assert not (journal_copy / "facets" / "" / "events").exists()


# =============================================================================
# Pre-hook Tests
# =============================================================================


def test_load_pre_hook_success(tmp_path):
    """Test loading a valid hook with pre_process function."""
    hook_file = tmp_path / "test_pre_hook.py"
    hook_file.write_text("""
def pre_process(context):
    return {"prompt": context["prompt"] + " [modified]"}
""")

    config = {"hook": {"pre": str(hook_file)}}
    hook_fn = load_pre_hook(config)
    assert callable(hook_fn)

    # Test the hook returns modifications
    result = hook_fn({"prompt": "original"})
    assert result == {"prompt": "original [modified]"}


def test_load_pre_hook_missing_pre_process(tmp_path):
    """Test that hook without pre_process function raises ValueError."""
    hook_file = tmp_path / "bad_hook.py"
    hook_file.write_text("""
def other_function():
    pass
""")

    config = {"hook": {"pre": str(hook_file)}}
    try:
        load_pre_hook(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must define a 'pre_process' function" in str(e)


def test_load_pre_hook_not_callable(tmp_path):
    """Test that hook with non-callable pre_process raises ValueError."""
    hook_file = tmp_path / "bad_hook.py"
    hook_file.write_text("""
pre_process = "not a function"
""")

    config = {"hook": {"pre": str(hook_file)}}
    try:
        load_pre_hook(config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "'pre_process' must be callable" in str(e)


def test_load_pre_hook_no_hook_config():
    """Test that missing hook config returns None."""
    assert load_pre_hook({}) is None
    assert load_pre_hook({"hook": {}}) is None
    assert load_pre_hook({"hook": {"post": "something"}}) is None


def test_load_pre_hook_file_not_found(tmp_path):
    """Test that nonexistent hook file raises ImportError."""
    config = {"hook": {"pre": str(tmp_path / "nonexistent.py")}}
    try:
        load_pre_hook(config)
        assert False, "Should have raised ImportError"
    except ImportError as e:
        assert "not found" in str(e)


def test_pre_hook_invocation(tmp_path, monkeypatch):
    """Test that agents.py invokes pre-hook and uses modified inputs."""
    mod = importlib.import_module("think.talents")
    copy_day(tmp_path)

    import think.talent

    monkeypatch.setattr(think.talent, "TALENT_DIR", tmp_path)

    prompt_file = tmp_path / "prehooked_test.md"
    prompt_file.write_text(
        '{\n  "type": "generate",\n  "title": "Prehooked",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"pre": "prehooked_test"},\n  "load": {"transcripts": true, "percepts": true}\n}\n\nOriginal prompt'
    )

    hook_file = tmp_path / "prehooked_test.py"
    hook_file.write_text("""
def pre_process(context):
    # Verify context has expected fields
    assert "transcript" in context
    assert "prompt" in context
    assert "user_instruction" in context
    # Modify the prompt
    return {"prompt": context["prompt"] + " [pre-processed]"}
""")

    # Track what generate_with_result receives
    received_kwargs = {}

    def mock_generate(*args, **kwargs):
        received_kwargs.update(kwargs)
        received_kwargs["contents"] = args[0] if args else kwargs.get("contents")
        return MOCK_RESULT

    import think.models

    monkeypatch.setattr(think.models, "generate_with_result", mock_generate)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    config = {
        "name": "prehooked_test",
        "day": "20240101",
        "output": "md",
        "provider": "google",
        "model": "gemini-2.0-flash",
    }

    events = run_generator_with_config(mod, config, monkeypatch)

    # Verify pre-hook modified the prompt - check in contents
    contents = received_kwargs.get("contents", [])
    # The prompt should contain [pre-processed]
    prompt_found = any("[pre-processed]" in str(c) for c in contents)
    assert prompt_found, f"Expected [pre-processed] in contents: {contents}"

    # Verify generator still completed successfully
    finish_events = [e for e in events if e["event"] == "finish"]
    assert len(finish_events) == 1


# ---- Pre-hook Template Vars Tests ----


def test_template_vars_basic_substitution():
    """Test basic template var substitution in user_instruction."""
    config = {"user_instruction": "Hello $name"}

    _apply_template_vars(config, {"name": "world"})

    assert config["user_instruction"] == "Hello world"


def test_template_vars_auto_capitalize():
    """Test auto-capitalized template key/value expansion."""
    config = {"prompt": "$greeting and $Greeting"}

    _apply_template_vars(config, {"greeting": "hello"})

    assert config["prompt"] == "hello and Hello"


def test_template_vars_all_fields():
    """Test substitution applies to all supported text fields."""
    config = {
        "user_instruction": "$x",
        "transcript": "$x",
        "prompt": "$x",
    }

    _apply_template_vars(config, {"x": "replaced"})

    assert config["user_instruction"] == "replaced"
    assert config["transcript"] == "replaced"
    assert config["prompt"] == "replaced"


def test_template_vars_no_system_instruction():
    """Test substitution does not touch system_instruction."""
    config = {
        "system_instruction": "$x",
        "user_instruction": "$x",
    }

    _apply_template_vars(config, {"x": "val"})

    assert config["system_instruction"] == "$x"
    assert config["user_instruction"] == "val"


def test_template_vars_empty_string_value():
    """Test empty string values substitute without errors."""
    config = {"prompt": "before $tag after"}

    _apply_template_vars(config, {"tag": ""})

    assert config["prompt"] == "before  after"


def test_template_vars_dollar_in_value_not_reprocessed():
    """Test substituted dollar signs in values are not reprocessed."""
    config = {"prompt": "$var"}

    _apply_template_vars(config, {"var": "costs $100"})

    assert config["prompt"] == "costs $100"


def test_template_vars_missing_placeholder_left_alone():
    """Test missing placeholders remain unchanged."""
    config = {"prompt": "$defined and $undefined"}

    _apply_template_vars(config, {"defined": "yes"})

    assert config["prompt"] == "yes and $undefined"


def test_template_vars_empty_field_skipped():
    """Test empty or missing supported fields are skipped safely."""
    config = {"prompt": ""}

    _apply_template_vars(config, {"x": "val"})
    assert config["prompt"] == ""

    config = {}
    _apply_template_vars(config, {"x": "val"})
    assert "prompt" not in config


def test_template_vars_popped_from_modifications():
    """Test template_vars are applied and not copied into config."""
    config = {}
    modifications = {
        "user_instruction": "Hello $name",
        "template_vars": {"name": "world"},
    }

    template_vars = modifications.pop("template_vars", None)
    for key, value in modifications.items():
        config[key] = value
    if template_vars:
        _apply_template_vars(config, template_vars)

    assert config["user_instruction"] == "Hello world"
    assert "template_vars" not in config


def test_pre_hook_template_vars_integration(tmp_path, monkeypatch):
    """Test pre-hook template_vars reach the model as substituted text."""
    mod = importlib.import_module("think.talents")
    copy_day(tmp_path)

    import think.talent

    monkeypatch.setattr(think.talent, "TALENT_DIR", tmp_path)

    prompt_file = tmp_path / "prehook_template_vars.md"
    prompt_file.write_text(
        '{\n  "type": "generate",\n  "title": "Prehook Template Vars",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"pre": "prehook_template_vars"},\n  "load": {"transcripts": true, "percepts": true}\n}\n\nTalk about $topic'
    )

    hook_file = tmp_path / "prehook_template_vars.py"
    hook_file.write_text("""
def pre_process(context):
    return {"template_vars": {"topic": "weather"}}
""")

    received_kwargs = {}

    def mock_generate(*args, **kwargs):
        received_kwargs.update(kwargs)
        received_kwargs["contents"] = args[0] if args else kwargs.get("contents")
        return MOCK_RESULT

    import think.models

    monkeypatch.setattr(think.models, "generate_with_result", mock_generate)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    config = {
        "name": "prehook_template_vars",
        "day": "20240101",
        "output": "md",
        "provider": "google",
        "model": "gemini-2.0-flash",
    }

    events = run_generator_with_config(mod, config, monkeypatch)

    contents = received_kwargs.get("contents", [])
    prompt_found = any("weather" in str(c) for c in contents)
    assert prompt_found, f"Expected weather in contents: {contents}"

    finish_events = [e for e in events if e["event"] == "finish"]
    assert len(finish_events) == 1


def test_pre_hook_template_vars_with_field_mods(tmp_path, monkeypatch):
    """Test pre-hook can return field mods and template_vars together."""
    mod = importlib.import_module("think.talents")
    copy_day(tmp_path)

    import think.talent

    monkeypatch.setattr(think.talent, "TALENT_DIR", tmp_path)

    prompt_file = tmp_path / "prehook_template_with_mods.md"
    prompt_file.write_text(
        '{\n  "type": "generate",\n  "title": "Prehook Template With Mods",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"pre": "prehook_template_with_mods"},\n  "load": {"transcripts": true, "percepts": true}\n}\n\nOriginal prompt'
    )

    hook_file = tmp_path / "prehook_template_with_mods.py"
    hook_file.write_text("""
def pre_process(context):
    return {
        "user_instruction": "Talk about $topic",
        "template_vars": {"topic": "music"},
    }
""")

    received_kwargs = {}

    def mock_generate(*args, **kwargs):
        received_kwargs.update(kwargs)
        received_kwargs["contents"] = args[0] if args else kwargs.get("contents")
        return MOCK_RESULT

    import think.models

    monkeypatch.setattr(think.models, "generate_with_result", mock_generate)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    config = {
        "name": "prehook_template_with_mods",
        "day": "20240101",
        "output": "md",
        "provider": "google",
        "model": "gemini-2.0-flash",
    }

    events = run_generator_with_config(mod, config, monkeypatch)

    contents = received_kwargs.get("contents", [])
    prompt_found = any("Talk about music" in str(c) for c in contents)
    assert prompt_found, f"Expected Talk about music in contents: {contents}"

    finish_events = [e for e in events if e["event"] == "finish"]
    assert len(finish_events) == 1


def test_both_pre_and_post_hooks(tmp_path, monkeypatch):
    """Test that both pre and post hooks can be configured together."""
    mod = importlib.import_module("think.talents")
    copy_day(tmp_path)

    import think.talent

    monkeypatch.setattr(think.talent, "TALENT_DIR", tmp_path)

    prompt_file = tmp_path / "both_hooks_test.md"
    prompt_file.write_text(
        '{\n  "type": "generate",\n  "title": "Both Hooks",\n  "schedule": "daily",\n  "priority": 10,\n  "output": "md",\n  "hook": {"pre": "both_hooks_test", "post": "both_hooks_test"},\n  "load": {"transcripts": true, "percepts": true}\n}\n\nOriginal prompt'
    )

    hook_file = tmp_path / "both_hooks_test.py"
    hook_file.write_text("""
def pre_process(context):
    return {"prompt": context["prompt"] + " [pre]"}

def post_process(result, context):
    return result + "\\n\\n[post]"
""")

    received_kwargs = {}

    def mock_generate(*args, **kwargs):
        received_kwargs.update(kwargs)
        received_kwargs["contents"] = args[0] if args else kwargs.get("contents")
        return MOCK_RESULT

    import think.models

    monkeypatch.setattr(think.models, "generate_with_result", mock_generate)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))

    config = {
        "name": "both_hooks_test",
        "day": "20240101",
        "output": "md",
        "provider": "google",
        "model": "gemini-2.0-flash",
    }

    events = run_generator_with_config(mod, config, monkeypatch)

    # Verify pre-hook modified the prompt - check in contents
    contents = received_kwargs.get("contents", [])
    prompt_found = any("[pre]" in str(c) for c in contents)
    assert prompt_found, f"Expected [pre] in contents: {contents}"

    # Verify post-hook modified the result
    finish_events = [e for e in events if e["event"] == "finish"]
    assert len(finish_events) == 1
    assert "[post]" in finish_events[0]["result"]
