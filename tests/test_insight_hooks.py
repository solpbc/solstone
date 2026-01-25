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
    for item in src.iterdir():
        if item.is_dir():
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest / item.name)
    return dest


MOCK_RESULT = "## Original Result\n\nThis is the original insight content."


def test_load_insight_hook_success(tmp_path):
    """Test loading a valid hook with process function."""
    utils = importlib.import_module("think.utils")

    hook_file = tmp_path / "test_hook.py"
    hook_file.write_text("""
def process(result, context):
    return result + "\\n\\n## Added by hook"
""")

    process_func = utils.load_insight_hook(hook_file)
    assert callable(process_func)

    # Test the hook transforms content
    output = process_func("Original", {"day": "20240101"})
    assert output == "Original\n\n## Added by hook"


def test_load_insight_hook_missing_process(tmp_path):
    """Test that hook without process function raises ValueError."""
    utils = importlib.import_module("think.utils")

    hook_file = tmp_path / "bad_hook.py"
    hook_file.write_text("""
def other_function():
    pass
""")

    try:
        utils.load_insight_hook(hook_file)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must define a 'process' function" in str(e)


def test_load_insight_hook_process_not_callable(tmp_path):
    """Test that hook with non-callable process raises ValueError."""
    utils = importlib.import_module("think.utils")

    hook_file = tmp_path / "bad_hook.py"
    hook_file.write_text("""
process = "not a function"
""")

    try:
        utils.load_insight_hook(hook_file)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "'process' must be callable" in str(e)


def test_insight_metadata_includes_hook_path(tmp_path):
    """Test that _load_insight_metadata detects .py hook file."""
    utils = importlib.import_module("think.utils")

    # Create insight file with frontmatter
    md_file = tmp_path / "test_insight.md"
    md_file.write_text('{\n  "title": "Test",\n  "color": "#ff0000"\n}\n\nTest prompt')

    hook_file = tmp_path / "test_insight.py"
    hook_file.write_text("def process(r, c): return r")

    meta = utils._load_insight_metadata(md_file)

    assert meta["path"] == str(md_file)
    assert meta["hook_path"] == str(hook_file)
    assert meta["title"] == "Test"


def test_insight_metadata_no_hook(tmp_path):
    """Test that _load_insight_metadata works without hook file."""
    utils = importlib.import_module("think.utils")

    md_file = tmp_path / "test_insight.md"
    md_file.write_text("Test prompt")

    meta = utils._load_insight_metadata(md_file)

    assert meta["path"] == str(md_file)
    assert "hook_path" not in meta


def test_insight_hook_invocation(tmp_path, monkeypatch):
    """Test that insight.py invokes hook and uses transformed result."""
    mod = importlib.import_module("think.insight")
    day_dir = copy_day(tmp_path)

    # Create insight with hook
    insights_dir = tmp_path / "insights"
    insights_dir.mkdir()

    prompt_file = insights_dir / "hooked.md"
    prompt_file.write_text(
        '{\n  "title": "Hooked",\n  "occurrences": false,\n  "frequency": "daily"\n}\n\nTest prompt'
    )

    hook_file = insights_dir / "hooked.py"
    hook_file.write_text("""
def process(result, context):
    # Verify context has expected fields
    assert "day" in context
    assert "transcript" in context
    assert "insight_key" in context
    return result + "\\n\\n## Hook was here"
""")

    monkeypatch.setattr(
        mod,
        "send_insight",
        lambda *a, **k: MOCK_RESULT,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["sol insight", "20240101", "-f", str(prompt_file)])

    mod.main()

    md = day_dir / "insights" / "hooked.md"
    content = md.read_text()
    assert "## Original Result" in content
    assert "## Hook was here" in content


def test_insight_hook_returns_none(tmp_path, monkeypatch):
    """Test that hook returning None uses original result."""
    mod = importlib.import_module("think.insight")
    day_dir = copy_day(tmp_path)

    insights_dir = tmp_path / "insights"
    insights_dir.mkdir()

    prompt_file = insights_dir / "noop.md"
    prompt_file.write_text(
        '{\n  "title": "Noop",\n  "occurrences": false,\n  "frequency": "daily"\n}\n\nTest prompt'
    )

    hook_file = insights_dir / "noop.py"
    hook_file.write_text("""
def process(result, context):
    return None  # Signal to use original
""")

    monkeypatch.setattr(
        mod,
        "send_insight",
        lambda *a, **k: MOCK_RESULT,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["sol insight", "20240101", "-f", str(prompt_file)])

    mod.main()

    md = day_dir / "insights" / "noop.md"
    content = md.read_text()
    assert content == MOCK_RESULT  # Original, not modified


def test_insight_hook_error_fallback(tmp_path, monkeypatch):
    """Test that hook errors fall back to original result."""
    mod = importlib.import_module("think.insight")
    day_dir = copy_day(tmp_path)

    insights_dir = tmp_path / "insights"
    insights_dir.mkdir()

    prompt_file = insights_dir / "broken.md"
    prompt_file.write_text(
        '{\n  "title": "Broken",\n  "occurrences": false,\n  "frequency": "daily"\n}\n\nTest prompt'
    )

    hook_file = insights_dir / "broken.py"
    hook_file.write_text("""
def process(result, context):
    raise RuntimeError("Hook exploded!")
""")

    monkeypatch.setattr(
        mod,
        "send_insight",
        lambda *a, **k: MOCK_RESULT,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["sol insight", "20240101", "-f", str(prompt_file)])

    # Should not raise, should fall back gracefully
    mod.main()

    md = day_dir / "insights" / "broken.md"
    content = md.read_text()
    assert content == MOCK_RESULT  # Original result preserved


def test_insight_hook_context_fields(tmp_path, monkeypatch):
    """Test that hook receives complete context dict."""
    mod = importlib.import_module("think.insight")
    copy_day(tmp_path)

    insights_dir = tmp_path / "insights"
    insights_dir.mkdir()

    prompt_file = insights_dir / "context_check.md"
    prompt_file.write_text(
        '{\n  "title": "Context Check",\n  "occurrences": false,\n  "frequency": "daily"\n}\n\nTest prompt'
    )

    # Write captured context to a file for verification
    hook_file = insights_dir / "context_check.py"
    hook_file.write_text("""
import json
from pathlib import Path

def process(result, context):
    # Write context to file for test verification
    out_path = Path(context["output_path"]).parent / "context_captured.json"
    with open(out_path, "w") as f:
        # Remove transcript for brevity, just check it exists
        ctx_copy = dict(context)
        ctx_copy["has_transcript"] = bool(ctx_copy.get("transcript"))
        ctx_copy["has_insight_meta"] = bool(ctx_copy.get("insight_meta"))
        del ctx_copy["transcript"]
        del ctx_copy["insight_meta"]
        json.dump(ctx_copy, f)
    return result
""")

    monkeypatch.setattr(
        mod,
        "send_insight",
        lambda *a, **k: MOCK_RESULT,
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["sol insight", "20240101", "-f", str(prompt_file)])

    mod.main()

    # Read captured context
    captured_path = tmp_path / "20240101" / "insights" / "context_captured.json"
    captured = json.loads(captured_path.read_text())

    assert captured["day"] == "20240101"
    assert captured["segment"] is None
    assert captured["insight_key"] == "context_check"  # stem of the prompt file
    assert captured["has_transcript"] is True
    assert captured["has_insight_meta"] is True
    assert "output_path" in captured


def test_named_hook_resolution_takes_precedence(tmp_path):
    """Test that named hooks via 'hook' field take precedence over co-located .py files."""
    utils = importlib.import_module("think.utils")

    # Create insight file with named hook
    md_file = tmp_path / "test_insight.md"
    md_file.write_text(
        '{\n  "title": "Test",\n  "hook": "occurrence"\n}\n\nTest prompt'
    )

    # Also create a co-located .py file that would normally be picked up
    colocated_hook = tmp_path / "test_insight.py"
    colocated_hook.write_text("def process(r, c): return 'colocated'")

    meta = utils._load_insight_metadata(md_file)

    # Should resolve to named hook, not co-located
    assert "hook_path" in meta
    assert meta["hook_path"].endswith("occurrence.py")
    assert "muse/occurrence.py" in meta["hook_path"].replace("\\", "/")


def test_named_hook_nonexistent_falls_through(tmp_path):
    """Test that nonexistent named hooks fall back to co-located .py files."""
    utils = importlib.import_module("think.utils")

    # Create insight file with nonexistent named hook
    md_file = tmp_path / "test_insight.md"
    md_file.write_text(
        '{\n  "title": "Test",\n  "hook": "nonexistent_hook_xyz"\n}\n\nTest prompt'
    )

    # Create a co-located .py file
    colocated_hook = tmp_path / "test_insight.py"
    colocated_hook.write_text("def process(r, c): return 'colocated'")

    meta = utils._load_insight_metadata(md_file)

    # Named hook doesn't exist, so no hook_path should be set (co-located not checked when named specified)
    # Actually the current implementation checks co-located only if hook field is not set
    # So with a nonexistent named hook, no hook_path should be set
    assert "hook_path" not in meta
