# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import asyncio
import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests.conftest import setup_google_genai_stub
from think.models import GEMINI_FLASH
from think.providers.google import (
    _extract_finish_reason,
    _format_completion_message,
)


async def run_main(mod, argv, stdin_data=None):
    sys.argv = argv
    if stdin_data:
        import io

        sys.stdin = io.StringIO(stdin_data)
    await mod.main_async()


def make_mock_process(stdout_lines, return_code=0):
    """Create a mock asyncio subprocess for CLI tests."""

    class MockStdout:
        def __init__(self, lines):
            self._lines = [line.encode() + b"\n" for line in lines]
            self._index = 0

        async def readline(self):
            if self._index >= len(self._lines):
                return b""
            line = self._lines[self._index]
            self._index += 1
            return line

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index >= len(self._lines):
                raise StopAsyncIteration
            line = self._lines[self._index]
            self._index += 1
            return line

    class MockStderr:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    process = AsyncMock()
    process.stdout = MockStdout(stdout_lines)
    process.stderr = MockStderr()
    process.stdin = AsyncMock()
    process.stdin.write = lambda data: None
    process.stdin.close = lambda: None
    process.wait = AsyncMock(return_value=return_code)
    return process


def test_google_main(monkeypatch, tmp_path, capsys):
    setup_google_genai_stub(monkeypatch, with_thinking=False)
    sys.modules.pop("think.providers.google", None)
    importlib.reload(importlib.import_module("think.providers.google"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setattr(
        "think.providers.cli.shutil.which",
        lambda name: "/usr/bin/gemini" if name == "gemini" else None,
    )

    stdout_lines = [
        json.dumps(
            {
                "type": "init",
                "timestamp": 100,
                "session_id": "sess-test",
                "model": "gemini-2.5-flash",
            }
        ),
        json.dumps(
            {
                "type": "message",
                "role": "assistant",
                "delta": True,
                "content": "ok",
            }
        ),
        json.dumps(
            {
                "type": "result",
                "timestamp": 200,
                "status": "success",
                "stats": {
                    "total_tokens": 10,
                    "input_tokens": 5,
                    "output_tokens": 5,
                },
            }
        ),
    ]
    process = make_mock_process(stdout_lines)
    monkeypatch.setattr(
        "think.providers.cli.asyncio.create_subprocess_exec",
        AsyncMock(return_value=process),
    )

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "provider": "google",
            "model": GEMINI_FLASH,
            "tools": ["search_insights"],
        }
    )
    asyncio.run(run_main(mod, ["sol agents"], stdin_data=ndjson_input))

    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[0]["event"] == "start"
    assert isinstance(events[0]["ts"], int)
    assert "hello" in events[0]["prompt"]
    assert events[0]["name"] == "default"
    assert events[0]["model"] == GEMINI_FLASH
    assert events[-1]["event"] == "finish"
    assert isinstance(events[-1]["ts"], int)
    assert events[-1]["result"] == "ok"

    # Journal logging is now handled by cortex, not by agents directly
    # So we don't check for journal files here


def test_google_cli_not_found_error(monkeypatch, tmp_path, capsys):
    setup_google_genai_stub(monkeypatch, with_thinking=False)

    sys.modules.pop("think.providers.google", None)
    importlib.reload(importlib.import_module("think.providers.google"))
    mod = importlib.reload(importlib.import_module("think.agents"))

    journal = tmp_path / "journal"
    journal.mkdir()

    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    monkeypatch.setattr("think.providers.cli.shutil.which", lambda _name: None)

    ndjson_input = json.dumps(
        {
            "prompt": "hello",
            "provider": "google",
            "model": GEMINI_FLASH,
            "tools": ["search_insights"],
        }
    )
    asyncio.run(run_main(mod, ["sol agents"], stdin_data=ndjson_input))

    # Check stdout for error event
    out_lines = capsys.readouterr().out.strip().splitlines()
    events = [json.loads(line) for line in out_lines]
    assert events[-1]["event"] == "error"
    assert isinstance(events[-1]["ts"], int)
    error_message = events[-1]["error"].lower()
    assert "gemini" in error_message
    assert "not found" in error_message
    assert "trace" in events[-1]


# ---------------------------------------------------------------------------
# Tests for finish reason extraction and formatting
# ---------------------------------------------------------------------------


def test_extract_finish_reason_with_enum():
    """Test extracting finish_reason from enum-style response."""

    class MockEnum:
        name = "STOP"

    candidate = SimpleNamespace(finish_reason=MockEnum())
    response = SimpleNamespace(candidates=[candidate])
    assert _extract_finish_reason(response) == "STOP"


def test_extract_finish_reason_with_string():
    """Test extracting finish_reason when it's already a string."""
    candidate = SimpleNamespace(finish_reason="MAX_TOKENS")
    response = SimpleNamespace(candidates=[candidate])
    assert _extract_finish_reason(response) == "MAX_TOKENS"


def test_extract_finish_reason_no_candidates():
    """Test extracting finish_reason when no candidates exist."""
    response = SimpleNamespace(candidates=[])
    assert _extract_finish_reason(response) is None

    response = SimpleNamespace()
    assert _extract_finish_reason(response) is None


def test_format_completion_message_stop_with_tools():
    """Test message for STOP with tool calls."""
    msg = _format_completion_message("STOP", had_tool_calls=True)
    assert msg == "Completed via tools."


def test_format_completion_message_stop_no_tools():
    """Test message for STOP without tool calls."""
    msg = _format_completion_message("STOP", had_tool_calls=False)
    assert msg == "Completed."


def test_format_completion_message_max_tokens():
    """Test message for MAX_TOKENS finish reason."""
    msg = _format_completion_message("MAX_TOKENS", had_tool_calls=False)
    assert msg == "Reached token limit."


def test_format_completion_message_safety():
    """Test message for safety-related finish reasons."""
    msg = _format_completion_message("SAFETY", had_tool_calls=False)
    assert msg == "Blocked by safety filters."

    msg = _format_completion_message("PROHIBITED_SAFETY", had_tool_calls=False)
    assert msg == "Blocked by safety filters."


def test_format_completion_message_tool_errors():
    """Test message for tool-related error finish reasons."""
    msg = _format_completion_message("UNEXPECTED_TOOL_CALL", had_tool_calls=True)
    assert msg == "Tool execution incomplete."

    msg = _format_completion_message("MALFORMED_FUNCTION_CALL", had_tool_calls=False)
    assert msg == "Tool execution incomplete."


def test_format_completion_message_unknown():
    """Test message for unknown finish reasons."""
    msg = _format_completion_message("SOME_NEW_REASON", had_tool_calls=False)
    assert msg == "Completed (some_new_reason)."


def test_format_completion_message_none():
    """Test message when finish_reason is None."""
    msg = _format_completion_message(None, had_tool_calls=False)
    assert msg == "Completed (unknown)."
