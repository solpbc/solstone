# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for agents conversation history functions."""

from think.agents import format_tool_summary, parse_agent_events_to_turns


def test_format_tool_summary():
    """Test format_tool_summary with various inputs."""
    # Empty list returns empty string
    assert format_tool_summary([]) == ""

    # Single tool call
    result = format_tool_summary(
        [{"name": "fetch_data", "args": {"url": "https://example.com"}}]
    )
    assert result == "\n\nTools used: fetch_data(url='https://example.com')"

    # Multiple tool calls
    result = format_tool_summary(
        [
            {"name": "fetch_data", "args": {"url": "https://example.com"}},
            {"name": "process", "args": {"id": 123, "mode": "fast"}},
        ]
    )
    assert "fetch_data(url='https://example.com')" in result
    assert "process(id=123, mode='fast')" in result
    assert "Tools used:" in result

    # Long arg values are truncated at 50 chars
    long_value = "x" * 100
    result = format_tool_summary([{"name": "test", "args": {"data": long_value}}])
    assert len(result) < len(long_value) + 50  # Truncated
    assert "data=" in result


def test_parse_agent_events_to_turns_basic(monkeypatch):
    """Test parsing a simple conversation."""

    # Mock read_agent_events to return basic conversation
    def mock_read_events(conversation_id):
        return [
            {"event": "start", "prompt": "Hello, how are you?"},
            {"event": "finish", "result": "I'm doing well, thank you!"},
        ]

    monkeypatch.setattr("think.cortex_client.read_agent_events", mock_read_events)

    turns = parse_agent_events_to_turns("test-conversation")

    assert len(turns) == 2
    assert turns[0] == {"role": "user", "content": "Hello, how are you?"}
    assert turns[1] == {"role": "assistant", "content": "I'm doing well, thank you!"}


def test_parse_agent_events_to_turns_with_tools(monkeypatch):
    """Test parsing conversation with tool usage."""

    # Mock read_agent_events to return conversation with tools
    def mock_read_events(conversation_id):
        return [
            {"event": "start", "prompt": "What's the weather?"},
            {"event": "tool_start", "tool": "get_weather", "args": {"location": "NYC"}},
            {"event": "finish", "result": "It's sunny in NYC."},
        ]

    monkeypatch.setattr("think.cortex_client.read_agent_events", mock_read_events)

    turns = parse_agent_events_to_turns("test-conversation")

    assert len(turns) == 2
    assert turns[0] == {"role": "user", "content": "What's the weather?"}
    assert turns[1]["role"] == "assistant"
    assert "It's sunny in NYC." in turns[1]["content"]
    assert "Tools used:" in turns[1]["content"]
    assert "get_weather(location='NYC')" in turns[1]["content"]


def test_parse_agent_events_to_turns_incomplete(monkeypatch):
    """Test that incomplete assistant responses are skipped."""

    # Mock read_agent_events with incomplete turn (no finish)
    def mock_read_events(conversation_id):
        return [
            {"event": "start", "prompt": "First question"},
            {"event": "finish", "result": "First answer"},
            {"event": "start", "prompt": "Second question"},
            # No finish event for second turn - assistant response is incomplete
        ]

    monkeypatch.setattr("think.cortex_client.read_agent_events", mock_read_events)

    turns = parse_agent_events_to_turns("test-conversation")

    # Should include user messages but skip incomplete assistant response
    assert (
        len(turns) == 3
    )  # First user + first assistant + second user (no second assistant)
    assert turns[0] == {"role": "user", "content": "First question"}
    assert turns[1] == {"role": "assistant", "content": "First answer"}
    assert turns[2] == {"role": "user", "content": "Second question"}
    # No assistant response for second turn since it's incomplete


def test_parse_agent_events_to_turns_not_found(monkeypatch, caplog):
    """Test handling of missing conversation log."""

    # Mock read_agent_events to raise FileNotFoundError
    def mock_read_events(conversation_id):
        raise FileNotFoundError("Log not found")

    monkeypatch.setattr("think.cortex_client.read_agent_events", mock_read_events)

    turns = parse_agent_events_to_turns("missing-conversation")

    # Should return empty list
    assert turns == []

    # Should log warning
    assert "Cannot continue from missing-conversation: log not found" in caplog.text


def test_parse_agent_events_to_turns_multi_turn(monkeypatch):
    """Test parsing back-and-forth multi-turn conversation."""

    # Mock read_agent_events with multiple turns
    def mock_read_events(conversation_id):
        return [
            {"event": "start", "prompt": "First question"},
            {"event": "finish", "result": "First answer"},
            {"event": "start", "prompt": "Follow-up question"},
            {"event": "tool_start", "tool": "search", "args": {"query": "test"}},
            {"event": "tool_start", "tool": "fetch", "args": {"id": 123}},
            {"event": "finish", "result": "Second answer with tool usage"},
            {"event": "start", "prompt": "Final question"},
            {"event": "finish", "result": "Final answer"},
        ]

    monkeypatch.setattr("think.cortex_client.read_agent_events", mock_read_events)

    turns = parse_agent_events_to_turns("test-conversation")

    # Should have 6 turns total (3 user + 3 assistant)
    assert len(turns) == 6

    # Check first turn
    assert turns[0] == {"role": "user", "content": "First question"}
    assert turns[1] == {"role": "assistant", "content": "First answer"}

    # Check second turn with tools
    assert turns[2] == {"role": "user", "content": "Follow-up question"}
    assert turns[3]["role"] == "assistant"
    assert "Second answer with tool usage" in turns[3]["content"]
    assert "Tools used:" in turns[3]["content"]
    assert "search(query='test')" in turns[3]["content"]
    assert "fetch(id=123)" in turns[3]["content"]

    # Check third turn
    assert turns[4] == {"role": "user", "content": "Final question"}
    assert turns[5] == {"role": "assistant", "content": "Final answer"}
