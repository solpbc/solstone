# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""CLI subprocess runner for AI provider tool agents.

Spawns provider CLI tools (claude, codex, gemini) in JSON streaming mode
and translates their JSONL output into our standard Event format.

Each provider module implements a translate() function that converts
provider-specific JSONL events into our Event TypedDicts. The CLIRunner
handles subprocess lifecycle, stdin piping, and event emission.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable

from think.providers.shared import JSONEventCallback, safe_raw
from think.utils import now_ms

LOG = logging.getLogger("think.providers.cli")

_PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Prompt Assembly
# ---------------------------------------------------------------------------


def assemble_prompt(config: dict[str, Any]) -> tuple[str, str | None]:
    """Combine config fields into a single prompt string and system instruction.

    Joins transcript, extra_context, user_instruction, and prompt with
    double newlines. Returns the system_instruction separately for CLIs
    that support --system-prompt (Claude); callers for other CLIs should
    prepend it to the prompt body.

    Args:
        config: Agent config dict with prompt, transcript, etc.

    Returns:
        Tuple of (prompt_body, system_instruction).
        system_instruction may be None.
    """
    parts = []
    for key in ("transcript", "extra_context", "user_instruction", "prompt"):
        value = config.get(key)
        if value:
            parts.append(value)

    prompt_body = "\n\n".join(parts) if parts else ""
    system_instruction = config.get("system_instruction") or None
    return prompt_body, system_instruction


# ---------------------------------------------------------------------------
# Thinking Aggregator
# ---------------------------------------------------------------------------


class ThinkingAggregator:
    """Buffers assistant text between tool calls for thinking/result classification.

    All assistant text that arrives between tool calls is treated as "thinking".
    Only the final text after all tool activity completes is the "result".

    Usage:
        agg = ThinkingAggregator(callback, model)
        # As text arrives:
        agg.accumulate("some text")
        # When a tool_start arrives, flush buffered text as thinking:
        agg.flush_as_thinking(raw_events=[...])
        # When done (no more tool calls), get the final result:
        result = agg.flush_as_result()
    """

    def __init__(
        self,
        callback: JSONEventCallback,
        model: str | None = None,
    ) -> None:
        self._buffer: list[str] = []
        self._callback = callback
        self._model = model

    def accumulate(self, text: str) -> None:
        """Add text to the buffer."""
        if text:
            self._buffer.append(text)

    def flush_as_thinking(self, raw_events: list[dict[str, Any]] | None = None) -> None:
        """Emit buffered text as a thinking event and clear the buffer.

        Does nothing if the buffer is empty.
        """
        text = "".join(self._buffer).strip()
        self._buffer.clear()
        if not text:
            return

        event: dict[str, Any] = {
            "event": "thinking",
            "summary": text,
            "ts": now_ms(),
        }
        if self._model:
            event["model"] = self._model
        if raw_events:
            event["raw"] = safe_raw(raw_events)
        self._callback.emit(event)

    def flush_as_result(self) -> str:
        """Return buffered text as the final result and clear the buffer."""
        text = "".join(self._buffer).strip()
        self._buffer.clear()
        return text

    @property
    def has_content(self) -> bool:
        """Whether the buffer has any content."""
        return bool(self._buffer)


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------


class CLIRunner:
    """Spawn a CLI subprocess and translate its JSONL output to our events.

    The runner pipes a prompt to stdin, reads JSONL from stdout line by line,
    and calls a provider-specific translate function for each line.

    Args:
        cmd: Command to run (e.g., ["claude", "-p", "-", ...]).
        prompt_text: Text to pipe to stdin.
        translate: Function that receives (raw_event_dict, aggregator, callback)
            and emits our Event types. Must return the cli_session_id from the
            init event (or None for non-init events).
        callback: JSONEventCallback for emitting events.
        aggregator: ThinkingAggregator for text buffering.
        cwd: Working directory for the subprocess. Defaults to project root.
        env: Optional environment overrides (merged with os.environ).
        timeout: Subprocess timeout in seconds. Default 600.
        first_event_timeout: Timeout for first stdout line in seconds. Default 30.
    """

    def __init__(
        self,
        cmd: list[str],
        prompt_text: str,
        translate: Callable[
            [dict[str, Any], ThinkingAggregator, JSONEventCallback],
            str | None,
        ],
        callback: JSONEventCallback,
        aggregator: ThinkingAggregator,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 600,
        first_event_timeout: int = 30,
    ) -> None:
        self.cmd = cmd
        self.prompt_text = prompt_text
        self.translate = translate
        self.callback = callback
        self.aggregator = aggregator
        self.cwd = cwd or _PROJECT_ROOT
        self.env = env
        self.timeout = timeout
        self.first_event_timeout = first_event_timeout
        self._timed_out_waiting_for_first_event = False
        self.cli_session_id: str | None = None

    async def run(self) -> str:
        """Spawn the CLI process, stream events, and return the final result.

        Returns:
            The final result text from the agent.

        Raises:
            RuntimeError: If the CLI binary is not found or process fails.
        """
        binary = self.cmd[0]
        if not shutil.which(binary):
            raise RuntimeError(
                f"CLI tool '{binary}' not found. "
                f"Install it and ensure it's on PATH."
            )

        import os

        proc_env = os.environ.copy()
        if self.env:
            proc_env.update(self.env)

        LOG.info("Spawning CLI: %s (cwd=%s)", " ".join(self.cmd), self.cwd)

        process = await asyncio.create_subprocess_exec(
            *self.cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024,  # 1 MB – tool results can exceed the 64 KB default
            cwd=str(self.cwd),
            env=proc_env,
        )

        # Pipe prompt to stdin and close
        if process.stdin:
            process.stdin.write(self.prompt_text.encode("utf-8"))
            process.stdin.close()

        # Read stdout line by line, translate each JSONL event
        stderr_lines: list[str] = []

        async def _read_stderr() -> None:
            if not process.stderr:
                return
            async for raw_line in process.stderr:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    stderr_lines.append(line)
                    LOG.debug("[%s stderr] %s", binary, line)

        stderr_task = asyncio.create_task(_read_stderr())
        self._timed_out_waiting_for_first_event = False

        try:
            await asyncio.wait_for(
                self._process_stdout(process),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            timeout_seconds = (
                self.first_event_timeout
                if self._timed_out_waiting_for_first_event
                else self.timeout
            )
            LOG.error("CLI process timed out after %ss, killing", timeout_seconds)
            process.kill()
            await stderr_task
            stderr_tail = "\n".join(stderr_lines[-20:])
            error_message = (
                f"CLI process timed out after {timeout_seconds}s. "
                f"Stderr tail:\n{stderr_tail}\n"
                "Check that the CLI tool is installed and authenticated."
            )
            self.callback.emit(
                {
                    "event": "error",
                    "error": error_message,
                    "ts": now_ms(),
                }
            )
            raise RuntimeError(error_message)
        finally:
            # Wait for stderr reader to finish
            if not stderr_task.done():
                await stderr_task

        # Wait for process to exit
        return_code = await process.wait()
        result = self.aggregator.flush_as_result()

        if return_code != 0:
            stderr_text = "\n".join(stderr_lines[-20:])  # Last 20 lines
            if result:
                # CLI failed but produced output — warn and return what we got
                LOG.warning(
                    "CLI process exited with code %d but produced output. "
                    "Stderr: %s",
                    return_code,
                    stderr_text,
                )
                self.callback.emit(
                    {
                        "event": "warning",
                        "message": f"CLI exited with code {return_code}",
                        "stderr": stderr_text,
                        "ts": now_ms(),
                    }
                )
            else:
                # CLI failed with no output — this is an error.
                # Don't emit error event here; the caller's exception
                # handler is responsible for error event emission.
                LOG.error(
                    "CLI process exited with code %d: %s",
                    return_code,
                    stderr_text,
                )
                raise RuntimeError(
                    f"CLI process exited with code {return_code}. "
                    f"Stderr: {stderr_text}"
                )

        return result

    async def _process_stdout(self, process: asyncio.subprocess.Process) -> None:
        """Read and translate JSONL lines from stdout."""
        if not process.stdout:
            return

        def _process_line(raw_line: bytes) -> None:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                return

            try:
                event_data = json.loads(line)
            except json.JSONDecodeError:
                LOG.warning("Non-JSON stdout line: %s", line[:200])
                return

            try:
                session_id = self.translate(event_data, self.aggregator, self.callback)
                if session_id:
                    self.cli_session_id = session_id
            except Exception:
                LOG.exception("Error translating CLI event: %s", line[:200])

        try:
            first_line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=self.first_event_timeout,
            )
        except asyncio.TimeoutError:
            self._timed_out_waiting_for_first_event = True
            raise
        if not first_line:
            return
        _process_line(first_line)

        async for raw_line in process.stdout:
            _process_line(raw_line)


# ---------------------------------------------------------------------------
# CLI Binary Check
# ---------------------------------------------------------------------------


def check_cli_binary(name: str) -> str:
    """Check that a CLI binary is available on PATH.

    Args:
        name: Binary name (e.g., "claude", "codex", "gemini").

    Returns:
        The full path to the binary.

    Raises:
        RuntimeError: If the binary is not found.
    """
    path = shutil.which(name)
    if not path:
        raise RuntimeError(
            f"CLI tool '{name}' not found on PATH. "
            f"Install it and ensure it's accessible."
        )
    return path


__all__ = [
    "CLIRunner",
    "ThinkingAggregator",
    "assemble_prompt",
    "check_cli_binary",
]
