# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for Gemini CLI subprocess provider (_translate_gemini)."""

import asyncio
import importlib
from unittest.mock import AsyncMock, patch

from think.providers.cli import ThinkingAggregator
from think.providers.google import _translate_gemini
from think.providers.shared import JSONEventCallback


def _google_provider():
    return importlib.import_module("think.providers.google")


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
        assert events == []

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
        assert events == []

    def test_tool_use_flushes_thinking_and_emits_start(self):
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        agg.accumulate("I'll use a tool now.")
        event = {
            "type": "tool_use",
            "timestamp": 2000,
            "tool_name": "read_file",
            "tool_id": "tool-1",
            "parameters": {"path": "/tmp/test.txt"},
        }
        _translate_gemini(event, agg, cb)
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
        assert len(events) == 1
        assert events[0]["event"] == "tool_start"

    def test_tool_result_includes_tool_name(self):
        """tool_end should include tool name from preceding tool_use."""
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        pending = {}
        _translate_gemini(
            {
                "type": "tool_use",
                "tool_name": "read_file",
                "tool_id": "t1",
                "parameters": {"path": "x.py"},
            },
            agg,
            cb,
            pending_tools=pending,
        )
        events.clear()  # ignore tool_start
        _translate_gemini(
            {
                "type": "tool_result",
                "tool_id": "t1",
                "status": "success",
                "output": "contents",
            },
            agg,
            cb,
            pending_tools=pending,
        )
        assert len(events) == 1
        assert events[0]["event"] == "tool_end"
        assert events[0]["tool"] == "read_file"
        assert events[0]["args"] == {"path": "x.py"}
        assert events[0]["result"] == "contents"
        assert events[0]["call_id"] == "t1"

    def test_tool_result_without_pending(self):
        """tool_end without pending_tools still works, tool is empty."""
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        _translate_gemini(
            {
                "type": "tool_result",
                "tool_id": "t1",
                "status": "success",
                "output": "data",
            },
            agg,
            cb,
        )
        assert len(events) == 1
        assert events[0]["event"] == "tool_end"
        assert events[0]["tool"] == ""

    def test_result_stores_usage(self):
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        usage = {}
        _translate_gemini(
            {
                "type": "result",
                "status": "success",
                "stats": {
                    "total_tokens": 1500,
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "cached": 200,
                    "duration_ms": 3000,
                    "tool_calls": 2,
                },
            },
            agg,
            cb,
            usage,
        )
        assert events == []
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
            {"type": "result", "status": "success"},
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
        """Process a full sequence of Gemini JSONL events."""
        cb, events = self._make_callback()
        agg = self._make_aggregator(cb)
        usage = {}
        pending = {}

        sequence = [
            {
                "type": "init",
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
                "tool_name": "read_file",
                "tool_id": "t1",
                "parameters": {"path": "test.py"},
            },
            {
                "type": "tool_result",
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
            sid = _translate_gemini(ev, agg, cb, usage, pending)
            if sid:
                session_ids.append(sid)

        assert session_ids == ["sess-42"]

        # Events: thinking, tool_start, tool_end
        assert len(events) == 3
        assert events[0]["event"] == "thinking"
        assert "read the file" in events[0]["summary"]
        assert events[1]["event"] == "tool_start"
        assert events[1]["tool"] == "read_file"
        assert events[2]["event"] == "tool_end"
        assert events[2]["tool"] == "read_file"
        assert events[2]["call_id"] == "t1"

        # Final result text in aggregator
        result = agg.flush_as_result()
        assert result == "The file contains a print statement."

        assert usage["total_tokens"] == 100


class TestRunCogitateCommand:
    """Tests for run_cogitate command construction."""

    def _mock_runner(self):
        """Create a MockCLIRunner that captures the command."""

        class MockCLIRunner:
            last_instance = None

            def __init__(self, **kwargs):
                self.cmd = kwargs["cmd"]
                self.prompt_text = kwargs["prompt_text"]
                self.cli_session_id = "test-session"
                self.run = AsyncMock(return_value="result")
                MockCLIRunner.last_instance = self

        return MockCLIRunner

    def test_yolo_mode_with_sol_call_allowed(self):
        provider = _google_provider()
        MockCLIRunner = self._mock_runner()
        with patch("think.providers.google.CLIRunner", MockCLIRunner):
            asyncio.run(
                provider.run_cogitate(
                    {"prompt": "hello", "model": "gemini-2.5-flash"}, lambda e: None
                )
            )
        cmd = MockCLIRunner.last_instance.cmd
        assert "--yolo" in cmd
        assert cmd[cmd.index("--allowed-tools") + 1] == "run_shell_command(sol call)"
