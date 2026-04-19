# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.talent module."""

import json
from pathlib import Path

import pytest

from think import talent as talent_module
from think.talent import (
    _validate_cwd,
    get_talent,
    get_talent_filter,
    source_is_enabled,
    source_is_required,
)


def test_source_is_enabled_bool():
    """Test source_is_enabled with bool values."""
    assert source_is_enabled(True) is True
    assert source_is_enabled(False) is False


def test_source_is_enabled_required_string():
    """Test source_is_enabled with 'required' string."""
    assert source_is_enabled("required") is True


def test_source_is_enabled_dict():
    """Test source_is_enabled with dict values for agents source."""
    assert source_is_enabled({"entities": True, "meetings": False}) is True
    assert source_is_enabled({"entities": "required", "meetings": False}) is True
    assert source_is_enabled({"entities": False, "meetings": False}) is False
    assert source_is_enabled({}) is False


def test_source_is_required_bool():
    """Test source_is_required with bool values."""
    assert source_is_required(True) is False
    assert source_is_required(False) is False


def test_source_is_required_string():
    """Test source_is_required with 'required' string."""
    assert source_is_required("required") is True


def test_source_is_required_dict():
    """Test source_is_required with dict values."""
    assert source_is_required({"entities": "required", "meetings": False}) is True
    assert source_is_required({"entities": True, "meetings": False}) is False
    assert source_is_required({}) is False


def test_get_agent_filter_bool():
    """Test get_talent_filter with bool values."""
    assert get_talent_filter(True) is None
    assert get_talent_filter(False) == {}


def test_get_agent_filter_required_string():
    """Test get_talent_filter with 'required' string."""
    assert get_talent_filter("required") is None


def test_get_agent_filter_dict():
    """Test get_talent_filter with dict values."""
    filter_dict = {"entities": True, "meetings": "required", "flow": False}
    assert get_talent_filter(filter_dict) == filter_dict
    assert get_talent_filter({}) == {}


def test_validate_cwd_defaults_cogitate_to_journal():
    assert _validate_cwd(None, "cogitate", "test-agent") == "journal"


def test_validate_cwd_accepts_repo():
    assert _validate_cwd("repo", "cogitate", "test-agent") == "repo"


def test_validate_cwd_accepts_journal():
    assert _validate_cwd("journal", "cogitate", "test-agent") == "journal"


def test_validate_cwd_rejects_generate_with_cwd():
    with pytest.raises(
        ValueError,
        match="Prompt 'test-agent' sets 'cwd' but cwd is only valid for type: cogitate",
    ):
        _validate_cwd("journal", "generate", "test-agent")


def test_validate_cwd_rejects_invalid_value():
    with pytest.raises(
        ValueError,
        match="Prompt 'test-agent' has invalid 'cwd' value 'home'",
    ):
        _validate_cwd("home", "cogitate", "test-agent")


def test_get_agent_normalizes_cwd_for_cogitate():
    config = get_talent("chat")
    assert config["cwd"] == "journal"


def test_get_agent_preserves_repo_cwd_for_coder():
    config = get_talent("coder")
    assert config["cwd"] == "repo"


def _write_talent_file(tmp_path: Path, name: str, metadata: dict) -> Path:
    md_path = tmp_path / f"{name}.md"
    md_path.write_text(
        f"{json.dumps(metadata, indent=2)}\n\nTest prompt\n",
        encoding="utf-8",
    )
    return md_path


def _write_schema_file(path: Path, schema: dict) -> None:
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def test_schema_absent_no_json_schema_key(tmp_path, monkeypatch):
    monkeypatch.setattr(talent_module, "TALENT_DIR", tmp_path)
    _write_talent_file(
        tmp_path, "schema_absent", {"type": "generate", "output": "json"}
    )

    config = get_talent("schema_absent")

    assert "json_schema" not in config
    assert "schema" not in config


def test_schema_loads_valid_file(tmp_path, monkeypatch):
    monkeypatch.setattr(talent_module, "TALENT_DIR", tmp_path)
    schema = {"type": "object", "properties": {"value": {"type": "string"}}}
    _write_schema_file(tmp_path / "schema.json", schema)
    _write_talent_file(
        tmp_path,
        "schema_valid",
        {"type": "generate", "output": "json", "schema": "schema.json"},
    )

    config = get_talent("schema_valid")

    assert config["json_schema"] == schema
    assert "schema" not in config


def test_schema_absolute_path_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(talent_module, "TALENT_DIR", tmp_path)
    _write_talent_file(
        tmp_path,
        "schema_absolute",
        {"type": "generate", "output": "json", "schema": "/etc/passwd"},
    )

    with pytest.raises(
        ValueError,
        match=r"talent schema_absolute: schema path must be relative: /etc/passwd",
    ):
        get_talent("schema_absolute")


def test_schema_parent_traversal_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(talent_module, "TALENT_DIR", tmp_path)
    _write_talent_file(
        tmp_path,
        "schema_parent",
        {"type": "generate", "output": "json", "schema": "../escape.json"},
    )

    with pytest.raises(
        ValueError,
        match=r"talent schema_parent: schema path must not contain '\.\.': \.\./escape\.json",
    ):
        get_talent("schema_parent")


def test_schema_symlink_escape_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(talent_module, "TALENT_DIR", tmp_path)
    outside_schema = tmp_path.parent / "outside_schema.json"
    _write_schema_file(outside_schema, {"type": "object"})
    try:
        (tmp_path / "schema.json").symlink_to(outside_schema)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable on this filesystem: {exc}")

    _write_talent_file(
        tmp_path,
        "schema_symlink",
        {"type": "generate", "output": "json", "schema": "schema.json"},
    )

    with pytest.raises(
        ValueError,
        match=r"talent schema_symlink: schema path escapes talent directory:",
    ):
        get_talent("schema_symlink")


def test_schema_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(talent_module, "TALENT_DIR", tmp_path)
    _write_talent_file(
        tmp_path,
        "schema_missing",
        {"type": "generate", "output": "json", "schema": "missing.json"},
    )

    with pytest.raises(
        FileNotFoundError,
        match=r"talent schema_missing: schema file not found: .*missing\.json",
    ):
        get_talent("schema_missing")


def test_schema_malformed_json(tmp_path, monkeypatch):
    monkeypatch.setattr(talent_module, "TALENT_DIR", tmp_path)
    schema_path = tmp_path / "broken.json"
    schema_path.write_text("{\n", encoding="utf-8")
    _write_talent_file(
        tmp_path,
        "schema_malformed",
        {"type": "generate", "output": "json", "schema": "broken.json"},
    )

    with pytest.raises(
        ValueError,
        match=r"talent schema_malformed: schema file is not valid JSON: .*broken\.json",
    ):
        get_talent("schema_malformed")


def test_schema_invalid_schema_draft(tmp_path, monkeypatch):
    monkeypatch.setattr(talent_module, "TALENT_DIR", tmp_path)
    _write_schema_file(tmp_path / "invalid_schema.json", {"type": 3})
    _write_talent_file(
        tmp_path,
        "schema_invalid",
        {"type": "generate", "output": "json", "schema": "invalid_schema.json"},
    )

    with pytest.raises(
        ValueError,
        match=(
            r"talent schema_invalid: schema file is not a valid JSON Schema: "
            r".*invalid_schema\.json"
        ),
    ):
        get_talent("schema_invalid")


def test_schema_not_string(tmp_path, monkeypatch):
    monkeypatch.setattr(talent_module, "TALENT_DIR", tmp_path)
    _write_talent_file(
        tmp_path,
        "schema_not_string",
        {"type": "generate", "output": "json", "schema": 42},
    )

    with pytest.raises(
        ValueError,
        match=r"talent schema_not_string: schema must be a string, got int: 42",
    ):
        get_talent("schema_not_string")
