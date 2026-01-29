# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the sol muse CLI."""

import json

import pytest

from think.muse_cli import (
    _collect_configs,
    _property_tags,
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


def test_property_tags_output():
    """Property tags show output, tools, hook."""
    assert _property_tags({"output": "md"}) == "output:md"
    assert _property_tags({"tools": "journal, todo"}) == "tools:journal, todo"

    # New dict-based hook format
    assert _property_tags({"hook": {"post": "occurrence"}}) == "hook:post=occurrence"
    assert _property_tags({"hook": {"pre": "prep"}}) == "hook:pre=prep"
    assert (
        _property_tags({"hook": {"pre": "prep", "post": "process"}})
        == "hook:pre=prep,post=process"
    )

    tags = _property_tags({"output": "md", "hook": {"post": "occurrence"}})
    assert "output:md" in tags
    assert "hook:post=occurrence" in tags

    assert _property_tags({}) == ""
    assert "disabled" in _property_tags({"disabled": True})


def test_property_tags_tools_list():
    """Property tags handle tools as a list."""
    tags = _property_tags({"tools": ["journal", "todo"]})
    assert tags == "tools:journal,todo"


def test_scan_variables():
    """Variable scanning finds template variables in prompt body."""
    assert "name" in _scan_variables("Hello $name, welcome")
    assert "daily_preamble" in _scan_variables("$daily_preamble\n\n# Title")
    assert _scan_variables("No variables here") == []
    # Deduplicates
    result = _scan_variables("$foo and $bar and $foo again")
    assert result == ["foo", "bar"]


def test_list_prompts_output(capsys):
    """List view outputs expected groups and prompts."""
    list_prompts()
    output = capsys.readouterr().out

    assert "segment:" in output
    assert "daily:" in output
    assert "activity" in output
    assert "flow" in output


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
