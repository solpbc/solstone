# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for insights app routes."""

from apps.insights.routes import _build_topic_map, _format_label


def test_format_label_simple():
    """Test formatting simple insight keys."""
    assert _format_label("activity") == "Activity"
    assert _format_label("flow") == "Flow"
    assert _format_label("knowledge_graph") == "Knowledge Graph"


def test_format_label_app_namespaced():
    """Test formatting app-namespaced insight keys."""
    assert _format_label("chat:sentiment") == "Chat: Sentiment"
    assert _format_label("my_app:weekly_summary") == "My App: Weekly Summary"


def test_build_topic_map():
    """Test topic map builds correctly from discovered insights."""
    topic_map = _build_topic_map()

    # Should have system insights
    assert "flow" in topic_map
    assert topic_map["flow"]["key"] == "flow"
    assert topic_map["flow"]["meta"].get("source") == "system"


def test_cost_conditional_logic():
    """Test the cost value logic: None when zero, value when positive."""
    # This tests the pattern used in insights_day():
    # cost = cost_data["cost"] if cost_data["cost"] > 0 else None

    def get_cost(cost_data):
        return cost_data["cost"] if cost_data["cost"] > 0 else None

    # Zero cost should return None
    assert get_cost({"cost": 0.0}) is None
    assert get_cost({"cost": 0}) is None

    # Positive cost should return the value
    assert get_cost({"cost": 0.0234}) == 0.0234
    assert get_cost({"cost": 0.0001}) == 0.0001


def test_get_usage_cost_imported():
    """Test that get_usage_cost is available in the routes module."""
    from apps.insights import routes

    # Verify the function is imported and callable
    assert hasattr(routes, "get_usage_cost")
    assert callable(routes.get_usage_cost)

    # Verify it returns expected structure for non-existent day
    result = routes.get_usage_cost("19000101", context="muse.system.test")
    assert "cost" in result
    assert "requests" in result
    assert "tokens" in result
