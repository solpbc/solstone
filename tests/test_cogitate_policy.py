# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import re
import stat

import tomllib

from think import cogitate_policy


def test_resolve_read_scope_defaults_to_current_day_chronicle():
    assert cogitate_policy.resolve_read_scope({}, "20260427") == ["chronicle/20260427"]


def test_resolve_read_scope_expands_override_placeholders():
    assert cogitate_policy.resolve_read_scope(
        {"read_scope": ["chronicle/<day>", "chronicle/<day-2>", "facets"]},
        "20260427",
    ) == ["chronicle/20260427", "chronicle/20260425", "facets"]


def test_resolve_read_scope_span_is_inclusive():
    assert cogitate_policy.resolve_read_scope(
        {"read_scope_span": 2},
        "20260427",
    ) == ["chronicle/20260425", "chronicle/20260426", "chronicle/20260427"]


def test_build_per_task_policy_appends_read_tool_rules(tmp_path, monkeypatch):
    policy_dir = tmp_path / "policies"
    journal = tmp_path / "journal"
    journal.mkdir()
    base_policy = tmp_path / "base.toml"
    base_policy.write_text("# base\n", encoding="utf-8")
    monkeypatch.setattr(cogitate_policy, "_POLICY_DIR", policy_dir)
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))

    path = cogitate_policy.build_per_task_policy(
        "morning_briefing",
        {
            "read_scope": [
                "chronicle/<day>",
                "facets",
                "entities",
                "imports",
                "health",
                "identity",
            ]
        },
        "20260427",
        0,
        base_policy,
    )

    assert path.parent == policy_dir
    assert path.name.startswith("morning_briefing-20260427-")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    rules = parsed["rule"]
    assert len(rules) == 8

    escaped_root = re.escape(str(journal).rstrip("/") + "/")
    prefix = f"(?:{escaped_root}|\\./)?"
    scope = "chronicle/20260427|facets|entities|imports|health|identity"
    expected_args = [
        f'"file_path":"{prefix}(?:{scope})(?:/|")',
        f'"dir_path":"{prefix}(?:{scope})(?:/|")',
        f'"(?:pattern|path)":"{prefix}(?:{scope})(?:/|[^"]*")',
        f'"(?:path|dir_path|include|pattern)":"{prefix}(?:{scope})(?:/|[^"]*")',
    ]

    tools = ["read_file", "list_directory", "glob", "grep_search"]
    for index, tool_name in enumerate(tools):
        allow = rules[index * 2]
        deny = rules[index * 2 + 1]
        assert allow == {
            "toolName": tool_name,
            "argsPattern": expected_args[index],
            "decision": "allow",
            "priority": 260,
        }
        assert deny == {
            "toolName": tool_name,
            "decision": "deny",
            "priority": 160,
        }


def test_build_per_task_policy_file_scope_has_exact_boundary(tmp_path, monkeypatch):
    policy_dir = tmp_path / "policies"
    journal = tmp_path / "journal"
    journal.mkdir()
    base_policy = tmp_path / "base.toml"
    base_policy.write_text("# base\n", encoding="utf-8")
    monkeypatch.setattr(cogitate_policy, "_POLICY_DIR", policy_dir)
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))

    path = cogitate_policy.build_per_task_policy(
        "awareness_tender",
        {"read_scope": ["stats.json"]},
        "20260427",
        0,
        base_policy,
    )

    rules = tomllib.loads(path.read_text(encoding="utf-8"))["rule"]
    pattern = rules[0]["argsPattern"]
    assert re.search(pattern, '{"file_path":"stats.json"}')
    assert not re.search(pattern, '{"file_path":"stats.jsonfoo"}')
