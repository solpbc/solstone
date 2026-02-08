# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for Gemini CLI subprocess provider (_translate_gemini)."""

from think.providers.cli import ThinkingAggregator
from think.providers.google import _translate_gemini
from think.providers.shared import JSONEventCallback


class TestTranslateGemini:
    """Tests for _translate_gemini event translation."""

    def _make_callback(self):
        """Create a callback that records emitted events."""
        events = []
        cb = JSONEventCallback(events.append)
        return cb, events

    def _make_aggregator(self, cb):
        """Create a ThinkingAggregator with the given callback."""
        return ThinkingAggregator(cb, model="gemini-2.5-flash")

    def test_init_returns_session_id(self):
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        result = _translate_gemini(
            {
                "type": "init",
                "timestamp": 1000,
                "session_id": "sess-abc",
                "model": "gemini-2.5-flash",
            },
            agg,
            cb,
        )
        assert result == "sess-abc"
        assert events == []  # init should not emit events

    def test_user_message_ignored(self):
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        result = _translate_gemini(
            {"type": "message", "role": "user", "content": "Hello"},
            agg,
            cb,
        )
        assert result is None
        assert events == []

    def test_assistant_delta_accumulates(self):
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        _translate_gemini(
            {
                "type": "message",
                "role": "assistant",
                "delta": True,
                "content": "Hello ",
            },
            agg,
            cb,
        )
        _translate_gemini(
            {"type": "message", "role": "assistant", "delta": True, "content": "world"},
            agg,
            cb,
        )
        assert agg.flush_as_result() == "Hello world"
        assert events == []  # deltas don't emit events directly

    def test_tool_use_flushes_thinking_and_emits_start(self):
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        # Accumulate some text first
        agg.accumulate("I'll use a tool now.")
        event = {
            "type": "tool_use",
            "timestamp": 2000,
            "tool_name": "read_file",
            "tool_id": "tool-1",
            "parameters": {"path": "/tmp/test.txt"},
        }
        _translate_gemini(event, agg, cb)
        # Should have emitted thinking + tool_start
        assert len(events) == 2
        assert events[0]["event"] == "thinking"
        assert events[0]["summary"] == "I'll use a tool now."
        assert events[1]["event"] == "tool_start"
        assert events[1]["tool"] == "read_file"
        assert events[1]["call_id"] == "tool-1"
        assert events[1]["args"] == {"path": "/tmp/test.txt"}
        assert events[1]["raw"] == [event]

    def test_tool_use_no_thinking_if_buffer_empty(self):
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        event = {
            "type": "tool_use",
            "timestamp": 2000,
            "tool_name": "read_file",
            "tool_id": "tool-1",
            "parameters": {},
        }
        _translate_gemini(event, agg, cb)
        # Only tool_start, no thinking event
        assert len(events) == 1
        assert events[0]["event"] == "tool_start"

    def test_tool_result_emits_end(self):
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        event = {
            "type": "tool_result",
            "timestamp": 3000,
            "tool_id": "tool-1",
            "status": "success",
            "output": "file contents here",
        }
        _translate_gemini(event, agg, cb)
        assert len(events) == 1
        assert events[0]["event"] == "tool_end"
        assert events[0]["call_id"] == "tool-1"
        assert events[0]["result"] == "file contents here"
        assert events[0]["raw"] == [event]

    def test_result_stores_usage(self):
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        usage = {}
        event = {
            "type": "result",
            "timestamp": 5000,
            "status": "success",
            "stats": {
                "total_tokens": 1500,
                "input_tokens": 1000,
                "output_tokens": 500,
                "cached": 200,
                "duration_ms": 3000,
                "tool_calls": 2,
            },
        }
        _translate_gemini(event, agg, cb, usage)
        assert events == []  # result should NOT emit events
        assert usage["input_tokens"] == 1000
        assert usage["output_tokens"] == 500
        assert usage["total_tokens"] == 1500
        assert usage["cached_tokens"] == 200
        assert usage["duration_ms"] == 3000

    def test_result_no_stats(self):
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        usage = {}
        _translate_gemini(
            {"type": "result", "timestamp": 5000, "status": "success"},
            agg,
            cb,
            usage,
        )
        assert usage == {}

    def test_unknown_event_type_ignored(self):
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        result = _translate_gemini(
            {"type": "unknown_type", "data": "whatever"},
            agg,
            cb,
        )
        assert result is None
        assert events == []

    def test_full_sequence(self):
        """Integration test: process a full sequence of Gemini JSONL events."""
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        usage = {}

        sequence = [
            {
                "type": "init",
                "timestamp": 100,
                "session_id": "sess-42",
                "model": "gemini-2.5-flash",
            },
            {"type": "message", "role": "user", "content": "Analyze this file"},
            {
                "type": "message",
                "role": "assistant",
                "delta": True,
                "content": "I'll read the file. ",
            },
            {
                "type": "tool_use",
                "timestamp": 200,
                "tool_name": "read_file",
                "tool_id": "t1",
                "parameters": {"path": "test.py"},
            },
            {
                "type": "tool_result",
                "timestamp": 300,
                "tool_id": "t1",
                "status": "success",
                "output": "print('hello')",
            },
            {
                "type": "message",
                "role": "assistant",
                "delta": True,
                "content": "The file contains ",
            },
            {
                "type": "message",
                "role": "assistant",
                "delta": True,
                "content": "a print statement.",
            },
            {
                "type": "result",
                "timestamp": 500,
                "status": "success",
                "stats": {
                    "total_tokens": 100,
                    "input_tokens": 60,
                    "output_tokens": 40,
                },
            },
        ]

        session_ids = []
        for ev in sequence:
            sid = _translate_gemini(ev, agg, cb, usage)
            if sid:
                session_ids.append(sid)

        # Session ID captured from init
        assert session_ids == ["sess-42"]

        # Events: thinking, tool_start, tool_end
        assert len(events) == 3
        assert events[0]["event"] == "thinking"
        assert "read the file" in events[0]["summary"]
        assert events[1]["event"] == "tool_start"
        assert events[1]["tool"] == "read_file"
        assert events[2]["event"] == "tool_end"
        assert events[2]["call_id"] == "t1"

        # Final result text in aggregator
        result = agg.flush_as_result()
        assert result == "The file contains a print statement."

        # Usage populated
        assert usage["total_tokens"] == 100
