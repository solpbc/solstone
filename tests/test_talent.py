# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.talent module."""

import pytest

from think.talent import (
    _validate_cwd,
    get_agent,
    get_agent_filter,
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
    """Test get_agent_filter with bool values."""
    assert get_agent_filter(True) is None
    assert get_agent_filter(False) == {}


def test_get_agent_filter_required_string():
    """Test get_agent_filter with 'required' string."""
    assert get_agent_filter("required") is None


def test_get_agent_filter_dict():
    """Test get_agent_filter with dict values."""
    filter_dict = {"entities": True, "meetings": "required", "flow": False}
    assert get_agent_filter(filter_dict) == filter_dict
    assert get_agent_filter({}) == {}


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
    config = get_agent("chat")
    assert config["cwd"] == "journal"


def test_get_agent_preserves_repo_cwd_for_coder():
    config = get_agent("coder")
    assert config["cwd"] == "repo"
