# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.talent module."""

from think.talent import get_agent_filter, source_is_enabled, source_is_required


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
