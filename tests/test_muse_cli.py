# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the sol muse CLI."""

import json

import pytest

from think.muse_cli import (
    _collect_configs,
    _format_bytes,
    _format_cost,
    _format_tags,
    _parse_run_stats,
    _scan_variables,
    json_output,
    list_prompts,
    log_run,
    logs_runs,
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

    activity = _collect_configs(schedule="activity", include_disabled=True)
    for key, info in activity.items():
        assert info.get("schedule") == "activity", f"{key} should be activity"

    # decisions is activity-scheduled
    assert "decisions" in activity


def test_collect_configs_filter_source():
    """Source filter returns only matching prompts."""
    system = _collect_configs(source="system", include_disabled=True)
    for key, info in system.items():
        assert info.get("source") == "system", f"{key} should be system"

    app = _collect_configs(source="app", include_disabled=True)
    for key, info in app.items():
        assert info.get("source") == "app", f"{key} should be app"


def test_format_tags_hook():
    """Format tags shows compact output, hook, disabled, and FAIL tags."""
    # Output format tags
    assert _format_tags({"output": "md"}) == "md"
    assert _format_tags({"output": "json"}) == "json"
    assert _format_tags({}) == ""

    # Hook tags (compact, no =name suffix)
    assert _format_tags({"hook": {"post": "occurrence"}}) == "post"
    assert _format_tags({"hook": {"pre": "prep"}}) == "pre"
    assert _format_tags({"hook": {"pre": "prep", "post": "process"}}) == "pre post"

    # Disabled
    assert _format_tags({"disabled": True}) == "disabled"

    # FAIL tag
    assert _format_tags({}, failed=True) == "FAIL"
    assert _format_tags({"output": "md"}, failed=True) == "md FAIL"

    # Combined: output + hooks + disabled + FAIL
    tags = _format_tags(
        {"output": "md", "hook": {"post": "occurrence"}, "disabled": True},
        failed=True,
    )
    assert tags == "md post disabled FAIL"


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
    assert "LAST RUN" in output
    assert "TAGS" in output
    assert "OUTPUT" not in output

    # Group headers
    assert "segment:" in output
    assert "daily:" in output
    assert "activity:" in output

    # Prompt names
    assert "activity" in output
    assert "flow" in output

    # Last run column is present
    assert "LAST RUN" in output


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


def test_logs_runs_default(capsys):
    """Logs shows recent runs from fixture day-index files."""
    logs_runs()
    output = capsys.readouterr().out

    # Should have runs from both fixture days
    assert "default" in output
    assert "flow" in output
    assert "activity" in output
    assert "entities" in output
    # Error run should show ✗
    assert "\u2717" in output
    # Completed runs should show ✓
    assert "\u2713" in output


def test_logs_runs_filter_agent(capsys):
    """Logs filters to a specific agent."""
    logs_runs(agent="default")
    output = capsys.readouterr().out

    lines = [line for line in output.strip().splitlines() if line.strip()]
    # fixture has 2 "default" runs in 20231114.jsonl
    assert len(lines) == 2
    for line in lines:
        assert "default" in line
    # Should NOT contain other agents
    assert "flow" not in output
    assert "activity" not in output


def test_logs_runs_count_limit(capsys):
    """Logs respects count limit."""
    logs_runs(count=2)
    output = capsys.readouterr().out

    lines = [line for line in output.strip().splitlines() if line.strip()]
    assert len(lines) == 2


def test_logs_runs_no_results(capsys):
    """Logs with unknown agent produces empty output."""
    logs_runs(agent="nonexistent_agent_xyz")
    output = capsys.readouterr().out
    assert output.strip() == ""


def test_logs_runs_new_columns(capsys):
    """Logs output includes enriched columns for runs with JSONL files."""
    logs_runs()
    output = capsys.readouterr().out
    lines = [line for line in output.strip().splitlines() if line.strip()]

    # Find the line for agent_id 1700000000001 (has JSONL file)
    enriched_line = None
    for line in lines:
        if "1700000000001" in line:
            enriched_line = line
            break
    assert enriched_line is not None

    # Should have numeric event/tool counts (not "-")
    # The fixture has 7 events total, 6 non-request, 1 tool_start
    assert "  6  " in enriched_line  # events
    assert "  1  " in enriched_line  # tools

    # Lines without JSONL files should show "-" for enriched columns
    # (most lines lack JSONL files)
    dash_count = sum(1 for line in lines if "  -  " in line)
    assert dash_count > 0


def test_parse_run_stats():
    """Parse run stats extracts correct counts from fixture JSONL."""
    from pathlib import Path

    jsonl = Path("tests/fixtures/journal/agents/default/1700000000001.jsonl")
    stats = _parse_run_stats(jsonl)
    assert stats["event_count"] == 6  # all except request
    assert stats["tool_count"] == 1  # one tool_start
    assert stats["model"] == "gpt-4o"
    assert stats["usage"] == {"input_tokens": 150, "output_tokens": 80}
    assert stats["request"] is not None
    assert stats["request"]["prompt"] == "Search for meetings about project updates"


def test_parse_run_stats_error():
    """Parse run stats handles error run JSONL correctly."""
    from pathlib import Path

    jsonl = Path("tests/fixtures/journal/agents/flow/1700000000002.jsonl")
    stats = _parse_run_stats(jsonl)
    assert stats["event_count"] == 2  # start + error (not request)
    assert stats["tool_count"] == 0
    assert stats["model"] == "claude-3-haiku"
    assert stats["usage"] is None


def test_format_bytes():
    """Byte formatting produces human-readable strings."""
    assert _format_bytes(0) == "0"
    assert _format_bytes(500) == "500"
    assert _format_bytes(999) == "999"
    assert _format_bytes(1000) == "1.0K"
    assert _format_bytes(1200) == "1.2K"
    assert _format_bytes(34000) == "34.0K"
    assert _format_bytes(1500000) == "1.5M"


def test_format_cost():
    """Cost formatting shows rounded cents."""
    assert _format_cost(None) == "-"
    assert _format_cost(0.0) == "0¢"
    assert _format_cost(0.001) == "<1¢"
    assert _format_cost(0.02) == "2¢"
    assert _format_cost(0.10) == "10¢"
    assert _format_cost(1.50) == "150¢"


def test_log_run_default(capsys):
    """Log run shows one-line-per-event output."""
    log_run("1700000000001")
    output = capsys.readouterr().out
    lines = output.strip().splitlines()

    # Fixture has 7 events
    assert len(lines) == 7

    # Each line should be ≤100 chars
    for line in lines:
        assert len(line) <= 100, f"Line too long ({len(line)}): {line}"

    # Check event type labels appear
    full_output = output
    assert "request" in full_output
    assert "start" in full_output
    assert "think" in full_output
    assert "tool" in full_output
    assert "tool_end" in full_output
    assert "updated" in full_output
    assert "finish" in full_output


def test_log_run_json(capsys):
    """Log run --json outputs raw JSONL."""
    log_run("1700000000001", json_mode=True)
    output = capsys.readouterr().out
    lines = [line for line in output.strip().splitlines() if line.strip()]

    assert len(lines) == 7
    # Each line should be valid JSON
    for line in lines:
        parsed = json.loads(line)
        assert "event" in parsed


def test_log_run_full(capsys):
    """Log run --full shows expanded content with escaped newlines."""
    log_run("1700000000001", full=True)
    output = capsys.readouterr().out

    # The thinking event in the fixture has actual newlines in "content"
    # In --full mode, these should appear as literal \n
    assert "\\n" in output

    # Lines can exceed 100 chars in full mode
    lines = output.strip().splitlines()
    assert len(lines) == 7


def test_log_run_missing():
    """Log run with unknown ID exits with error."""
    with pytest.raises(SystemExit):
        log_run("nonexistent_id_12345")


def test_log_run_error_run(capsys):
    """Log run displays error events correctly."""
    log_run("1700000000002")
    output = capsys.readouterr().out
    lines = output.strip().splitlines()
    assert len(lines) == 3  # request, start, error
    assert "error" in output
    assert "Rate limit" in output


def test_show_prompt_context_activity_requires_facet(capsys):
    """Activity-scheduled prompts require --facet."""
    from think.muse_cli import show_prompt_context

    with pytest.raises(SystemExit):
        show_prompt_context("decisions", day="20260214")

    output = capsys.readouterr().err
    assert "activity-scheduled" in output.lower()
    assert "--facet" in output


def test_show_prompt_context_activity_requires_activity_id(capsys):
    """Activity-scheduled prompts require --activity and list available IDs."""
    from think.muse_cli import show_prompt_context

    with pytest.raises(SystemExit):
        show_prompt_context("decisions", day="20260214", facet="full-featured")

    output = capsys.readouterr().err
    assert "--activity" in output
    assert "coding_093000_300" in output
    assert "meeting_140000_300" in output


def test_show_prompt_context_activity_not_found(capsys):
    """Activity-scheduled prompt with unknown activity ID errors."""
    from think.muse_cli import show_prompt_context

    with pytest.raises(SystemExit):
        show_prompt_context(
            "decisions",
            day="20260214",
            facet="full-featured",
            activity="nonexistent_999",
        )

    output = capsys.readouterr().err
    assert "not found" in output.lower()


def test_list_prompts_activity_group(capsys):
    """List view includes activity group with decisions agent."""
    list_prompts()
    output = capsys.readouterr().out

    assert "activity:" in output
    assert "decisions" in output
