# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the sol muse CLI."""

import json

import pytest

from think.muse_cli import (
    _collect_configs,
    _format_output_path,
    _format_tags,
    _format_tools,
    _scan_variables,
    json_output,
    list_prompts,
    show_prompt,
)


def test_collect_configs_returns_prompts():
    """All configs include known system prompts."""
    configs = _collect_configs(include_disabled=True)
    assert "flow" in configs
    assert "activity" in configs
    assert "default" in configs


def test_collect_configs_excludes_disabled_by_default():
    """Disabled prompts are excluded unless include_disabled is set."""
    without = _collect_configs(include_disabled=False)
    with_disabled = _collect_configs(include_disabled=True)
    assert len(with_disabled) >= len(without)

    # files.md is disabled by default
    disabled_keys = set(with_disabled.keys()) - set(without.keys())
    assert len(disabled_keys) > 0


def test_collect_configs_filter_schedule():
    """Schedule filter returns only matching prompts."""
    daily = _collect_configs(schedule="daily", include_disabled=True)
    for key, info in daily.items():
        assert info.get("schedule") == "daily", f"{key} should be daily"

    segment = _collect_configs(schedule="segment", include_disabled=True)
    for key, info in segment.items():
        assert info.get("schedule") == "segment", f"{key} should be segment"

    # No overlap
    assert not set(daily.keys()) & set(segment.keys())


def test_collect_configs_filter_source():
    """Source filter returns only matching prompts."""
    system = _collect_configs(source="system", include_disabled=True)
    for key, info in system.items():
        assert info.get("source") == "system", f"{key} should be system"

    app = _collect_configs(source="app", include_disabled=True)
    for key, info in app.items():
        assert info.get("source") == "app", f"{key} should be app"


def test_format_tags_hook():
    """Format tags shows hook and disabled status."""
    # Dict-based hook format
    assert _format_tags({"hook": {"post": "occurrence"}}) == "hook:post=occurrence"
    assert _format_tags({"hook": {"pre": "prep"}}) == "hook:pre=prep"
    assert (
        _format_tags({"hook": {"pre": "prep", "post": "process"}})
        == "hook:pre=prep,post=process"
    )

    assert _format_tags({}) == ""
    assert "disabled" in _format_tags({"disabled": True})

    # Hook + disabled combined
    tags = _format_tags({"hook": {"post": "occurrence"}, "disabled": True})
    assert "hook:post=occurrence" in tags
    assert "disabled" in tags


def test_format_output_path_segment():
    """Output path for segment-scheduled prompts."""
    assert _format_output_path("activity", {"schedule": "segment", "output": "md"}) == (
        "<segment>/activity.md"
    )
    assert _format_output_path(
        "speakers", {"schedule": "segment", "output": "json"}
    ) == ("<segment>/speakers.json")


def test_format_output_path_daily():
    """Output path for daily-scheduled prompts."""
    assert _format_output_path("flow", {"schedule": "daily", "output": "md"}) == (
        "<day>/agents/flow.md"
    )
    assert _format_output_path("schedule", {"schedule": "daily", "output": "json"}) == (
        "<day>/agents/schedule.json"
    )


def test_format_output_path_unscheduled():
    """Unscheduled prompts with output go to agents/."""
    assert _format_output_path("importer", {"output": "md"}) == (
        "<day>/agents/importer.md"
    )


def test_format_output_path_no_output():
    """Prompts without output field return dash."""
    assert _format_output_path("default", {"tools": "journal"}) == "-"
    assert _format_output_path("joke_bot", {}) == "-"


def test_format_output_path_app_namespaced():
    """App-namespaced prompts use underscore prefix in filename."""
    # This tests the get_output_topic integration
    assert _format_output_path(
        "entities:entities", {"schedule": "daily", "output": "md"}
    ) == ("<day>/agents/_entities_entities.md")


def test_format_tools():
    """Format tools extracts tools field or returns dash."""
    assert _format_tools({"tools": "journal, todo"}) == "journal, todo"
    assert _format_tools({"tools": ["journal", "todo"]}) == "journal, todo"
    assert _format_tools({}) == "-"
    assert _format_tools({"output": "md"}) == "-"


def test_scan_variables():
    """Variable scanning finds template variables in prompt body."""
    assert "name" in _scan_variables("Hello $name, welcome")
    assert "daily_preamble" in _scan_variables("$daily_preamble\n\n# Title")
    assert _scan_variables("No variables here") == []
    # Deduplicates
    result = _scan_variables("$foo and $bar and $foo again")
    assert result == ["foo", "bar"]


def test_list_prompts_output(capsys):
    """List view outputs expected groups and prompts with column layout."""
    list_prompts()
    output = capsys.readouterr().out

    # Column header
    assert "NAME" in output
    assert "TITLE" in output
    assert "OUTPUT" in output
    assert "TOOLS" in output
    assert "TAGS" in output

    # Group headers
    assert "segment:" in output
    assert "daily:" in output

    # Prompt names
    assert "activity" in output
    assert "flow" in output

    # Output path column shows path patterns
    assert "<segment>/activity.md" in output
    assert "<day>/agents/flow.md" in output

    # Tools column is present
    assert "TOOLS" in output


def test_list_prompts_schedule_filter(capsys):
    """Schedule filter shows only matching group."""
    list_prompts(schedule="segment")
    output = capsys.readouterr().out

    assert "activity" in output
    # Should not show daily-only prompts
    # (but don't assert group headers since they're suppressed with filter)


def test_list_prompts_disabled_shown(capsys):
    """--disabled includes disabled prompts."""
    list_prompts(include_disabled=True)
    output = capsys.readouterr().out

    # files.md is disabled, should appear
    assert "files" in output


def test_show_prompt_known(capsys):
    """Detail view shows expected fields for a known prompt."""
    show_prompt("flow")
    output = capsys.readouterr().out

    assert "muse/flow.md" in output
    assert "title:" in output
    assert "schedule:" in output
    assert "daily" in output
    assert "hook:" in output
    assert "occurrence" in output
    assert "variables:" in output
    assert "$daily_preamble" in output
    assert "body:" in output
    assert "lines" in output


def test_show_prompt_not_found(capsys):
    """Detail view exits with error for unknown prompt."""
    with pytest.raises(SystemExit):
        show_prompt("nonexistent_prompt_xyz")

    output = capsys.readouterr().err
    assert "not found" in output.lower()


def test_json_output_format(capsys):
    """JSON output produces valid JSONL with file field."""
    json_output()
    output = capsys.readouterr().out

    lines = [x for x in output.strip().splitlines() if x.strip()]
    assert len(lines) > 0

    for line in lines:
        record = json.loads(line)
        assert "file" in record, f"Missing 'file' key in: {line}"
        assert record["file"].endswith(".md")


def test_json_output_contains_known_prompts(capsys):
    """JSON output includes known prompts with expected fields."""
    json_output(include_disabled=True)
    output = capsys.readouterr().out

    records = [json.loads(x) for x in output.strip().splitlines() if x.strip()]
    files = {r["file"] for r in records}
    assert any("flow.md" in f for f in files)
    assert any("activity.md" in f for f in files)

    # Check a specific record has expected fields
    flow = next(r for r in records if "flow.md" in r["file"])
    assert "title" in flow
    assert "schedule" in flow


def test_json_output_schedule_filter(capsys):
    """JSON output respects schedule filter."""
    json_output(schedule="segment")
    output = capsys.readouterr().out

    records = [json.loads(x) for x in output.strip().splitlines() if x.strip()]
    for r in records:
        assert r.get("schedule") == "segment", f"Expected segment: {r}"


def test_show_prompt_as_json(capsys):
    """Detail view with --json outputs single JSONL record."""
    show_prompt("flow", as_json=True)
    output = capsys.readouterr().out

    lines = [x for x in output.strip().splitlines() if x.strip()]
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["file"].endswith("flow.md")
    assert "title" in record
    assert "schedule" in record
    # Should not contain expanded instruction text
    assert "system_instruction" not in record


def test_truncate_content():
    """Content truncation works correctly."""
    from think.muse_cli import _truncate_content

    # Short content not truncated
    short = "line1\nline2\nline3"
    result, omitted = _truncate_content(short, max_lines=10)
    assert result == short
    assert omitted == 0

    # Long content truncated
    long = "\n".join(f"line{i}" for i in range(200))
    result, omitted = _truncate_content(long, max_lines=100)
    assert omitted == 100
    assert "lines omitted" in result
    assert "line0" in result  # First lines kept
    assert "line199" in result  # Last lines kept


def test_yesterday():
    """Yesterday helper returns correct format."""
    from think.muse_cli import _yesterday

    result = _yesterday()
    assert len(result) == 8
    assert result.isdigit()


def test_show_prompt_context_segment_validation(capsys):
    """Segment-scheduled prompts require --segment."""
    from think.muse_cli import show_prompt_context

    with pytest.raises(SystemExit):
        show_prompt_context("activity", day="20260101")

    output = capsys.readouterr().err
    assert "segment-scheduled" in output.lower()


def test_show_prompt_context_multi_facet_validation(capsys):
    """Multi-facet prompts require --facet."""
    from think.muse_cli import show_prompt_context

    with pytest.raises(SystemExit):
        show_prompt_context("entities:entities")

    output = capsys.readouterr().err
    assert "multi-facet" in output.lower()


def test_show_prompt_context_day_format_validation(capsys):
    """Day argument must be YYYYMMDD format."""
    from think.muse_cli import show_prompt_context

    # Too short
    with pytest.raises(SystemExit):
        show_prompt_context("flow", day="2026")

    output = capsys.readouterr().err
    assert "invalid --day format" in output.lower()

    # Non-numeric
    with pytest.raises(SystemExit):
        show_prompt_context("flow", day="abcdefgh")

    output = capsys.readouterr().err
    assert "invalid --day format" in output.lower()
