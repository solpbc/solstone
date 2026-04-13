# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for think.providers.cli — CLI subprocess runner infrastructure."""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, mock_open, patch

import pytest

from think.providers.cli import (
    CLIRunner,
    ThinkingAggregator,
    assemble_prompt,
    build_cogitate_env,
)
from think.providers.shared import JSONEventCallback, safe_raw

# ---------------------------------------------------------------------------
# assemble_prompt
# ---------------------------------------------------------------------------


class TestAssemblePrompt:
    def test_all_fields(self):
        config = {
            "transcript": "Speaker A: hello",
            "extra_context": "Today is Monday",
            "user_instruction": "Summarize the transcript",
            "prompt": "What happened?",
            "system_instruction": "You are a helpful assistant",
        }
        body, system = assemble_prompt(config)
        assert "Speaker A: hello" in body
        assert "Today is Monday" in body
        assert "Summarize the transcript" in body
        assert "What happened?" in body
        assert system == "You are a helpful assistant"
        # Parts joined with double newlines
        assert body.count("\n\n") == 3

    def test_prompt_only(self):
        config = {"prompt": "hello"}
        body, system = assemble_prompt(config)
        assert body == "hello"
        assert system is None

    def test_empty_config(self):
        body, system = assemble_prompt({})
        assert body == ""
        assert system is None

    def test_skips_empty_values(self):
        config = {
            "transcript": "",
            "extra_context": None,
            "user_instruction": "Do something",
            "prompt": "Go",
        }
        body, system = assemble_prompt(config)
        assert body == "Do something\n\nGo"
        assert system is None

    def test_system_instruction_empty_string(self):
        config = {"prompt": "test", "system_instruction": ""}
        _, system = assemble_prompt(config)
        assert system is None


# ---------------------------------------------------------------------------
# ThinkingAggregator
# ---------------------------------------------------------------------------


class TestThinkingAggregator:
    def _make_aggregator(self):
        """Create aggregator with event capture."""
        events = []
        cb = JSONEventCallback(events.append)
        agg = ThinkingAggregator(cb, model="test-model")
        return agg, events

    def test_accumulate_and_flush_as_thinking(self):
        agg, events = self._make_aggregator()
        agg.accumulate("hello ")
        agg.accumulate("world")
        agg.flush_as_thinking(raw_events=[{"type": "message"}])

        assert len(events) == 1
        assert events[0]["event"] == "thinking"
        assert events[0]["summary"] == "hello world"
        assert events[0]["model"] == "test-model"
        assert events[0]["raw"] == [{"type": "message"}]

    def test_flush_thinking_empty_buffer_is_noop(self):
        agg, events = self._make_aggregator()
        agg.flush_as_thinking()
        assert len(events) == 0

    def test_flush_thinking_whitespace_only_is_noop(self):
        agg, events = self._make_aggregator()
        agg.accumulate("   ")
        agg.flush_as_thinking()
        assert len(events) == 0

    def test_flush_as_result(self):
        agg, events = self._make_aggregator()
        agg.accumulate("final answer")
        result = agg.flush_as_result()
        assert result == "final answer"
        # No events emitted for result flush
        assert len(events) == 0
        # Buffer is cleared
        assert agg.flush_as_result() == ""

    def test_multiple_thinking_flushes(self):
        """Simulate text -> tool -> text -> tool -> text pattern."""
        agg, events = self._make_aggregator()

        # First text chunk (before first tool call)
        agg.accumulate("Let me check...")
        agg.flush_as_thinking()

        # Second text chunk (between tool calls)
        agg.accumulate("Now let me verify...")
        agg.flush_as_thinking()

        # Final text (the result)
        agg.accumulate("The answer is 42")
        result = agg.flush_as_result()

        assert len(events) == 2
        assert events[0]["summary"] == "Let me check..."
        assert events[1]["summary"] == "Now let me verify..."
        assert result == "The answer is 42"

    def test_has_content(self):
        agg, _ = self._make_aggregator()
        assert not agg.has_content
        agg.accumulate("x")
        assert agg.has_content
        agg.flush_as_result()
        assert not agg.has_content

    def test_no_raw_events(self):
        agg, events = self._make_aggregator()
        agg.accumulate("thinking")
        agg.flush_as_thinking()
        assert "raw" not in events[0]

    def test_strips_whitespace(self):
        agg, events = self._make_aggregator()
        agg.accumulate("  padded  ")
        agg.flush_as_thinking()
        assert events[0]["summary"] == "padded"


class _MockStderr:
    """Async iterator yielding pre-set stderr lines."""

    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        return line


class _MockStdout:
    """Async iterator yielding pre-set stdout lines, with readline support."""

    def __init__(self, lines: list[bytes]):
        self._lines = lines
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


def _make_process(stdout_lines, stderr_lines, return_code):
    """Create a mock process with given stdout/stderr/exit code."""
    process = AsyncMock()
    process.stdout = _MockStdout(stdout_lines)
    process.stderr = _MockStderr(stderr_lines)
    process.stdin = AsyncMock()
    process.stdin.write = lambda _data: None
    process.stdin.close = lambda: None
    process.kill = lambda: None
    process.wait = AsyncMock(return_value=return_code)
    return process


class TestCLIRunnerExitCode:
    """Tests for CLIRunner handling of non-zero exit codes."""

    def test_nonzero_exit_no_output_raises(self):
        """CLI exits with error and no result → RuntimeError with stderr."""
        events = []
        callback = JSONEventCallback(events.append)
        aggregator = ThinkingAggregator(callback, model="test-model")

        process = _make_process(
            stdout_lines=[],
            stderr_lines=[b"TerminalQuotaError: quota exhausted\n"],
            return_code=1,
        )

        runner = CLIRunner(
            cmd=["fakecli", "--json"],
            prompt_text="test",
            translate=lambda _e, _a, _c: None,
            callback=callback,
            aggregator=aggregator,
        )

        with (
            patch(
                "think.providers.cli.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
            patch("think.providers.cli.shutil.which", return_value="/usr/bin/fakecli"),
            pytest.raises(RuntimeError, match="quota exhausted"),
        ):
            asyncio.run(runner.run())

        # CLIRunner should NOT emit error events — that's the caller's job
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) == 0

    def test_nonzero_exit_with_output_returns_result(self):
        """CLI exits with error but produced output → return result + warning."""
        events = []
        callback = JSONEventCallback(events.append)
        aggregator = ThinkingAggregator(callback, model="test-model")

        # translate accumulates text from stdout events
        def translate(event, agg, cb):
            if event.get("type") == "text":
                agg.accumulate(event["content"])
            return None

        process = _make_process(
            stdout_lines=[b'{"type": "text", "content": "The answer is 42"}\n'],
            stderr_lines=[b"Warning: something went wrong\n"],
            return_code=1,
        )

        runner = CLIRunner(
            cmd=["fakecli", "--json"],
            prompt_text="test",
            translate=translate,
            callback=callback,
            aggregator=aggregator,
        )

        with (
            patch(
                "think.providers.cli.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
            patch("think.providers.cli.shutil.which", return_value="/usr/bin/fakecli"),
        ):
            result = asyncio.run(runner.run())

        assert result == "The answer is 42"
        warning_events = [e for e in events if e.get("event") == "warning"]
        assert len(warning_events) == 1
        assert "code 1" in warning_events[0]["message"]
        assert "something went wrong" in warning_events[0]["stderr"]

    def test_zero_exit_empty_result_ok(self):
        """CLI exits 0 with no output → return empty string, no error."""
        events = []
        callback = JSONEventCallback(events.append)
        aggregator = ThinkingAggregator(callback, model="test-model")

        process = _make_process(
            stdout_lines=[],
            stderr_lines=[],
            return_code=0,
        )

        runner = CLIRunner(
            cmd=["fakecli", "--json"],
            prompt_text="test",
            translate=lambda _e, _a, _c: None,
            callback=callback,
            aggregator=aggregator,
        )

        with (
            patch(
                "think.providers.cli.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
            patch("think.providers.cli.shutil.which", return_value="/usr/bin/fakecli"),
        ):
            result = asyncio.run(runner.run())

        assert result == ""
        assert not [e for e in events if e.get("event") in ("error", "warning")]

    def test_env_dict_used_directly_without_merge(self):
        events = []
        callback = JSONEventCallback(events.append)
        aggregator = ThinkingAggregator(callback, model="test-model")
        provided_env = {"PATH": "/custom/bin"}
        sentinel_key = "CLIRUNNER_TEST_LEAK"
        captured_env = None

        async def create_subprocess_exec(*args, **kwargs):
            nonlocal captured_env
            captured_env = kwargs["env"]
            return _make_process([], [], 0)

        runner = CLIRunner(
            cmd=["fakecli", "--json"],
            prompt_text="test",
            translate=lambda _e, _a, _c: None,
            callback=callback,
            aggregator=aggregator,
            env=provided_env,
        )

        os.environ[sentinel_key] = "should-not-leak"
        try:
            with (
                patch(
                    "think.providers.cli.asyncio.create_subprocess_exec",
                    AsyncMock(side_effect=create_subprocess_exec),
                ),
                patch(
                    "think.providers.cli.shutil.which", return_value="/usr/bin/fakecli"
                ),
            ):
                asyncio.run(runner.run())
        finally:
            os.environ.pop(sentinel_key, None)

        assert captured_env == provided_env
        assert captured_env is provided_env
        assert sentinel_key not in captured_env


class TestCLIRunnerFirstEventTimeout:
    def test_first_event_timeout_includes_stderr(self):
        events = []
        callback = JSONEventCallback(events.append)
        aggregator = ThinkingAggregator(callback, model="test-model")

        class HangingStdout:
            async def readline(self):
                future = asyncio.get_running_loop().create_future()
                return await future

        process = _make_process([], [b"Please authenticate first\n"], 0)
        process.stdout = HangingStdout()  # Override with hanging version

        runner = CLIRunner(
            cmd=["fakecli", "--json"],
            prompt_text="test prompt",
            translate=lambda _event, _agg, _cb: None,
            callback=callback,
            aggregator=aggregator,
            timeout=5,
            first_event_timeout=0.1,
        )

        with (
            patch(
                "think.providers.cli.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
            patch("think.providers.cli.shutil.which", return_value="/usr/bin/fakecli"),
            pytest.raises(RuntimeError) as exc_info,
        ):
            asyncio.run(runner.run())

        message = str(exc_info.value)
        assert "authenticate" in message.lower()
        assert "Check that the CLI tool is installed and authenticated." in message

        error_events = [event for event in events if event.get("event") == "error"]
        assert len(error_events) == 1
        assert "Please authenticate first" in error_events[0]["error"]


_OVERSIZE = object()  # sentinel for oversize line in _MockStdoutWithOversize


class _MockStdoutWithOversize:
    """Stdout mock that raises LimitOverrunError on a specific readline() call."""

    def __init__(self, lines: list):
        # lines entries are either bytes or the sentinel OVERSIZE
        self._lines = lines
        self._index = 0
        self._draining_oversize = False

    async def readline(self):
        if self._draining_oversize:
            self._draining_oversize = False
            return b"x" * 1024 * 1024 + b"\n"
        if self._index >= len(self._lines):
            return b""
        entry = self._lines[self._index]
        self._index += 1
        if entry is _OVERSIZE:
            self._draining_oversize = True
            raise asyncio.LimitOverrunError(
                "Separator is not found, and chunk exceed the limit", 1024 * 1024
            )
        return entry

    async def readexactly(self, n: int) -> bytes:
        return b"x" * n

    def __aiter__(self):
        return self

    async def __anext__(self):
        val = await self.readline()
        if val == b"":
            raise StopAsyncIteration
        return val


class TestCLIRunnerOversizedOutput:
    """CLIRunner recovers from LimitOverrunError in the stdout loop."""

    def test_oversized_line_emits_tool_end_and_continues(self):
        """Oversize line → synthetic tool_end emitted + subsequent line processed."""
        import json

        normal_line_1 = json.dumps({"event": "text", "text": "hello"}).encode() + b"\n"
        normal_line_2 = json.dumps({"event": "text", "text": "world"}).encode() + b"\n"

        events = []
        callback = JSONEventCallback(events.append)
        aggregator = ThinkingAggregator(callback, model="test-model")

        process = AsyncMock()
        process.stdout = _MockStdoutWithOversize(
            [
                normal_line_1,
                _OVERSIZE,
                normal_line_2,
            ]
        )
        process.stderr = _MockStderr([])
        process.stdin = AsyncMock()
        process.stdin.write = lambda _data: None
        process.stdin.close = lambda: None
        process.kill = lambda: None
        process.wait = AsyncMock(return_value=0)

        # translate just forwards text events as-is
        def translate(event_data, agg, cb):
            if event_data.get("event") == "text":
                cb.emit({"event": "text", "text": event_data["text"]})
            return None

        runner = CLIRunner(
            cmd=["fakecli", "--json"],
            prompt_text="test",
            translate=translate,
            callback=callback,
            aggregator=aggregator,
        )

        with (
            patch(
                "think.providers.cli.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
            patch("think.providers.cli.shutil.which", return_value="/usr/bin/fakecli"),
        ):
            asyncio.run(runner.run())

        event_types = [e["event"] for e in events]
        # tool_end should be emitted
        assert "tool_end" in event_types, f"Expected tool_end in events: {events}"

        # the tool_end result should indicate truncation
        tool_end_events = [e for e in events if e["event"] == "tool_end"]
        assert len(tool_end_events) == 1
        assert "truncated" in tool_end_events[0]["result"]

        # the normal line after the oversize error should also be processed
        text_events = [e for e in events if e["event"] == "text"]
        texts = [e["text"] for e in text_events]
        assert "world" in texts, f"Expected 'world' in text events: {texts}"


# ---------------------------------------------------------------------------
# safe_raw
# ---------------------------------------------------------------------------


class TestSafeRaw:
    def test_small_event_returned_unchanged(self):
        events = [{"type": "tool_use", "tool_name": "read_file", "tool_id": "t1"}]
        assert safe_raw(events) is events

    def test_large_event_trimmed(self):
        big_output = "x" * 20_000
        events = [
            {
                "type": "tool_result",
                "tool_id": "t1",
                "output": big_output,
                "extra_field": "value",
            }
        ]
        result = safe_raw(events)
        assert result is not events
        # Should keep only structural keys
        assert result[0] == {"type": "tool_result", "tool_id": "t1"}
        # Last element is the trimmed metadata
        meta = result[-1]["_raw_trimmed"]
        assert meta["limit"] == 16_384
        assert meta["original_bytes"] > 16_384

    def test_custom_limit(self):
        events = [{"type": "message", "content": "a" * 200}]
        # Under custom limit
        assert safe_raw(events, limit=1024) is events
        # Over custom limit
        result = safe_raw(events, limit=50)
        assert result is not events
        assert result[-1]["_raw_trimmed"]["limit"] == 50

    def test_structural_keys_preserved(self):
        events = [
            {
                "type": "tool_use",
                "id": "abc",
                "tool_id": "t1",
                "tool_name": "search",
                "role": "assistant",
                "event_type": "message",
                "timestamp": 12345,
                "big_content": "z" * 20_000,
            }
        ]
        result = safe_raw(events)
        kept = result[0]
        assert kept == {
            "type": "tool_use",
            "id": "abc",
            "tool_id": "t1",
            "tool_name": "search",
            "role": "assistant",
            "event_type": "message",
            "timestamp": 12345,
        }

    def test_multiple_events(self):
        events = [
            {"type": "msg", "data": "a" * 10_000},
            {"type": "msg", "data": "b" * 10_000},
        ]
        result = safe_raw(events)
        assert len(result) == 3  # 2 trimmed events + 1 metadata
        assert result[0] == {"type": "msg"}
        assert result[1] == {"type": "msg"}
        assert "_raw_trimmed" in result[2]


# ---------------------------------------------------------------------------
# build_cogitate_env
# ---------------------------------------------------------------------------


class TestBuildCogitateEnv:
    """Tests for build_cogitate_env — API key stripping for CLI subprocesses."""

    def test_default_strips_key(self):
        """No auth config → default platform mode → key removed."""
        config = {"providers": {}}
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret"}, clear=False),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("ANTHROPIC_API_KEY")
        assert "ANTHROPIC_API_KEY" not in env

    def test_explicit_platform_strips_key(self):
        """auth.anthropic = "platform" → key removed."""
        config = {"providers": {"auth": {"anthropic": "platform"}}}
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret"}, clear=False),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("ANTHROPIC_API_KEY")
        assert "ANTHROPIC_API_KEY" not in env

    def test_api_key_mode_preserves_key(self):
        """auth.anthropic = "api_key" → key preserved."""
        config = {"providers": {"auth": {"anthropic": "api_key"}}}
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-secret"}, clear=False),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("ANTHROPIC_API_KEY")
        assert env["ANTHROPIC_API_KEY"] == "sk-secret"

    def test_missing_auth_section_strips_key(self):
        """No providers section at all → safe default, key removed."""
        config = {}
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai"}, clear=False),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("OPENAI_API_KEY")
        assert "OPENAI_API_KEY" not in env

    def test_other_env_vars_preserved(self):
        """Non-API-key vars are never stripped."""
        config = {"providers": {}}
        with (
            patch.dict(
                os.environ,
                {"ANTHROPIC_API_KEY": "sk-secret", "HOME": "/home/test"},
                clear=False,
            ),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("ANTHROPIC_API_KEY")
        assert env["HOME"] == "/home/test"

    def test_key_not_in_env_is_harmless(self):
        """Stripping a key that doesn't exist doesn't error."""
        config = {"providers": {}}
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert "GOOGLE_API_KEY" not in env

    def test_per_provider_independence(self):
        """Each provider's auth mode is independent."""
        config = {
            "providers": {
                "auth": {
                    "anthropic": "api_key",
                    "openai": "platform",
                }
            }
        }
        with (
            patch.dict(
                os.environ,
                {"ANTHROPIC_API_KEY": "sk-ant", "OPENAI_API_KEY": "sk-oai"},
                clear=False,
            ),
            patch("think.utils.get_config", return_value=config),
        ):
            ant_env = build_cogitate_env("ANTHROPIC_API_KEY")
            oai_env = build_cogitate_env("OPENAI_API_KEY")
        assert ant_env["ANTHROPIC_API_KEY"] == "sk-ant"
        assert "OPENAI_API_KEY" not in oai_env

    def test_vertex_backend_sets_env_vars(self):
        """Vertex backend sets GOOGLE_GENAI_USE_VERTEXAI (no API key for vertex)."""
        config = {
            "providers": {
                "google_backend": "vertex",
                "auth": {"google": "platform"},
            }
        }
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
        assert "GOOGLE_API_KEY" not in env

    def test_vertex_backend_with_sa_creds(self):
        """Vertex with SA credentials sets GOOGLE_APPLICATION_CREDENTIALS, removes API key."""
        config = {
            "providers": {
                "google_backend": "vertex",
                "vertex_credentials": "/tmp/fake-sa.json",
                "auth": {"google": "platform"},
            }
        }
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True),
            patch("think.utils.get_config", return_value=config),
            patch("os.path.exists", return_value=True),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
        assert env["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/fake-sa.json"
        assert "GOOGLE_API_KEY" not in env

    def test_aistudio_backend_no_vertex_env_vars(self):
        """AI Studio backend does not set Vertex env vars."""
        config = {
            "providers": {
                "google_backend": "aistudio",
                "auth": {"google": "api_key"},
            }
        }
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert "GOOGLE_GENAI_USE_VERTEXAI" not in env
        assert env["GOOGLE_API_KEY"] == "gk-test"

    def test_auto_backend_detects_vertex(self):
        """Auto backend with Vertex detection sets env vars."""
        config = {"providers": {"auth": {"google": "platform"}}}
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True),
            patch("think.utils.get_config", return_value=config),
            patch("think.providers.google._detect_backend", return_value="vertex"),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
        assert "GOOGLE_API_KEY" not in env

    def test_non_google_key_unaffected_by_vertex(self):
        """Vertex logic only applies to GOOGLE_API_KEY."""
        config = {
            "providers": {
                "google_backend": "vertex",
                "auth": {"anthropic": "api_key"},
            }
        }
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant"}, clear=True),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("ANTHROPIC_API_KEY")
        assert "GOOGLE_GENAI_USE_VERTEXAI" not in env
        assert env["ANTHROPIC_API_KEY"] == "sk-ant"

    def test_vertex_backend_sets_project_and_location(self):
        """Vertex backend exposes project context for Gemini CLI subprocesses."""
        config = {
            "providers": {
                "google_backend": "vertex",
                "vertex_credentials": "/tmp/fake-sa.json",
                "auth": {"google": "platform"},
            }
        }
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True),
            patch("think.utils.get_config", return_value=config),
            patch("os.path.exists", return_value=True),
            patch(
                "builtins.open",
                mock_open(
                    read_data='{"type": "service_account", "project_id": "my-gcp-project"}'
                ),
            ),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert env["GOOGLE_GENAI_USE_VERTEXAI"] == "true"
        assert env["GOOGLE_APPLICATION_CREDENTIALS"] == "/tmp/fake-sa.json"
        assert env["GOOGLE_CLOUD_PROJECT"] == "my-gcp-project"
        assert env["GOOGLE_CLOUD_LOCATION"] == "global"
        assert "GOOGLE_API_KEY" not in env

    def test_vertex_backend_missing_creds_no_project(self):
        """Vertex backend still sets location without explicit SA credentials."""
        config = {
            "providers": {
                "google_backend": "vertex",
                "auth": {"google": "platform"},
            }
        }
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert "GOOGLE_CLOUD_PROJECT" not in env
        assert env["GOOGLE_CLOUD_LOCATION"] == "global"

    def test_vertex_backend_invalid_sa_json_no_project(self):
        """Invalid SA JSON logs and skips project configuration."""
        config = {
            "providers": {
                "google_backend": "vertex",
                "vertex_credentials": "/tmp/fake-sa.json",
                "auth": {"google": "platform"},
            }
        }
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True),
            patch("think.utils.get_config", return_value=config),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data="not json")),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert "GOOGLE_CLOUD_PROJECT" not in env
        assert env["GOOGLE_CLOUD_LOCATION"] == "global"

    def test_vertex_backend_sa_missing_project_id(self):
        """Missing project_id in SA JSON leaves project env unset."""
        config = {
            "providers": {
                "google_backend": "vertex",
                "vertex_credentials": "/tmp/fake-sa.json",
                "auth": {"google": "platform"},
            }
        }
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True),
            patch("think.utils.get_config", return_value=config),
            patch("os.path.exists", return_value=True),
            patch(
                "builtins.open",
                mock_open(
                    read_data=(
                        '{"type": "service_account", "client_email": "bot@example.com"}'
                    )
                ),
            ),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert "GOOGLE_CLOUD_PROJECT" not in env
        assert env["GOOGLE_CLOUD_LOCATION"] == "global"

    def test_aistudio_clears_project_and_location(self):
        """AI Studio clears inherited Vertex project context."""
        config = {
            "providers": {
                "google_backend": "aistudio",
                "auth": {"google": "api_key"},
            }
        }
        with (
            patch.dict(
                os.environ,
                {
                    "GOOGLE_API_KEY": "gk-test",
                    "GOOGLE_CLOUD_LOCATION": "us-central1",
                    "GOOGLE_CLOUD_PROJECT": "inherited-proj",
                },
                clear=True,
            ),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert "GOOGLE_CLOUD_PROJECT" not in env
        assert "GOOGLE_CLOUD_LOCATION" not in env

    def test_vertex_backend_sets_system_settings_path(self):
        """Vertex backend exposes the Gemini CLI system settings path."""
        config = {
            "providers": {
                "google_backend": "vertex",
                "auth": {"google": "platform"},
            }
        }
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True),
            patch("think.utils.get_config", return_value=config),
            patch("think.utils.get_journal", return_value="/fake/journal"),
            patch.object(Path, "exists", return_value=True),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert (
            env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"]
            == "/fake/journal/.config/gemini-vertex-settings.json"
        )

    def test_aistudio_backend_no_system_settings_path(self):
        """AI Studio backend does not set Gemini CLI system settings."""
        config = {
            "providers": {
                "google_backend": "aistudio",
                "auth": {"google": "api_key"},
            }
        }
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert "GEMINI_CLI_SYSTEM_SETTINGS_PATH" not in env

    def test_aistudio_clears_inherited_system_settings_path(self):
        """AI Studio clears inherited Gemini CLI system settings."""
        config = {
            "providers": {
                "google_backend": "aistudio",
                "auth": {"google": "api_key"},
            }
        }
        with (
            patch.dict(
                os.environ,
                {
                    "GOOGLE_API_KEY": "gk-test",
                    "GEMINI_CLI_SYSTEM_SETTINGS_PATH": "/old/settings.json",
                },
                clear=True,
            ),
            patch("think.utils.get_config", return_value=config),
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert "GEMINI_CLI_SYSTEM_SETTINGS_PATH" not in env

    def test_vertex_writes_settings_file_when_absent(self):
        """Vertex backend creates Gemini CLI system settings when missing."""
        config = {
            "providers": {
                "google_backend": "vertex",
                "auth": {"google": "platform"},
            }
        }
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True),
            patch("think.utils.get_config", return_value=config),
            patch("think.utils.get_journal", return_value="/fake/journal"),
            patch.object(Path, "exists", return_value=False),
            patch("os.makedirs") as mock_mkdirs,
            patch("builtins.open", mock_open()) as mock_file,
            patch("os.chmod") as mock_chmod,
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        written = "".join(call.args[0] for call in mock_file().write.call_args_list)
        assert env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] == (
            "/fake/journal/.config/gemini-vertex-settings.json"
        )
        assert mock_mkdirs.called
        assert mock_file.called
        assert json.loads(written) == {
            "security": {"auth": {"selectedType": "vertex-ai"}}
        }
        mock_chmod.assert_called_once_with(
            "/fake/journal/.config/gemini-vertex-settings.json", 0o600
        )

    def test_vertex_skips_settings_write_when_exists(self):
        """Vertex backend does not rewrite existing Gemini CLI settings."""
        config = {
            "providers": {
                "google_backend": "vertex",
                "auth": {"google": "platform"},
            }
        }
        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "gk-test"}, clear=True),
            patch("think.utils.get_config", return_value=config),
            patch("think.utils.get_journal", return_value="/fake/journal"),
            patch.object(Path, "exists", return_value=True),
            patch("builtins.open", mock_open()) as mock_file,
        ):
            env = build_cogitate_env("GOOGLE_API_KEY")
        assert env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] == (
            "/fake/journal/.config/gemini-vertex-settings.json"
        )
        mock_file.assert_not_called()
