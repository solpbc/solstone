#!/usr/bin/env python3
"""Unified process spawning and lifecycle management utilities.

All subprocess output is automatically logged to:
    {JOURNAL_PATH}/{YYYYMMDD}/health/{process_name}.log

Logs automatically roll over at midnight for long-running processes.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_journal_path() -> Path:
    """Get JOURNAL_PATH from environment."""
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")
    return Path(journal)


def _current_day() -> str:
    """Get current day in YYYYMMDD format."""
    return datetime.now().strftime("%Y%m%d")


def _day_health_log_path(day: str, name: str) -> Path:
    """Build path to day health log.

    Returns: {JOURNAL_PATH}/{day}/health/{name}.log
    """
    return _get_journal_path() / day / "health" / f"{name}.log"


def _format_log_line(prefix: str, stream: str, line: str) -> str:
    """Format log line with ISO timestamp and labels.

    Args:
        prefix: Process identifier (e.g., "observer" or "describe:file.webm")
        stream: "stdout" or "stderr"
        line: Output line from process

    Returns:
        Formatted line: "2024-11-01T10:30:45 [prefix:stream] line\\n"
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    clean_line = line.rstrip("\n")
    return f"{timestamp} [{prefix}:{stream}] {clean_line}\n"


class DailyLogWriter:
    """Thread-safe log writer that automatically rolls over at midnight.

    Writes to: {JOURNAL_PATH}/{YYYYMMDD}/health/{name}.log

    When the day changes, automatically closes old file and opens new file.
    """

    def __init__(self, name: str):
        self._name = name
        self._lock = threading.Lock()
        self._current_day = _current_day()
        self._fh = self._open_log()

    def _open_log(self):
        """Open log file for current day."""
        log_path = _day_health_log_path(self._current_day, self._name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        """Write message to log, handling day rollover."""
        with self._lock:
            # Check for day change
            day_now = _current_day()
            if day_now != self._current_day:
                # Close old log
                if not self._fh.closed:
                    self._fh.close()
                # Open new log
                self._current_day = day_now
                self._fh = self._open_log()

            # Write and flush
            self._fh.write(message)
            self._fh.flush()

    def close(self) -> None:
        """Close log file."""
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    @property
    def path(self) -> Path:
        """Get current log file path."""
        return _day_health_log_path(self._current_day, self._name)


@dataclass
class ManagedProcess:
    """Subprocess wrapper with automatic output logging and lifecycle management.

    All output is automatically logged to:
        {JOURNAL_PATH}/{YYYYMMDD}/health/{name}.log

    Logs roll over automatically at midnight for long-running processes.
    """

    process: subprocess.Popen
    name: str
    log_writer: DailyLogWriter
    cmd: list[str]
    _threads: list[threading.Thread]

    @classmethod
    def spawn(
        cls,
        cmd: list[str],
        *,
        name: str | None = None,
        log_name: str | None = None,
        env: dict | None = None,
    ) -> "ManagedProcess":
        """Spawn process with automatic output logging to daily health directory.

        Args:
            cmd: Command and arguments
            name: Process name for logging (defaults to cmd[0] basename)
            log_name: Override log filename base (defaults to name)
            env: Optional environment variables (inherits parent env if not provided)

        Returns:
            ManagedProcess instance

        Raises:
            RuntimeError: If process fails to spawn

        Example:
            managed = ManagedProcess.spawn(
                ["observe-gnome", "-v"],
                name="observer",
            )
            # Output automatically logs to: journal/{today}/health/observer.log

            # Or with custom log filename:
            managed = ManagedProcess.spawn(
                ["think-importer", "file.txt"],
                name="importer",
                log_name="1730476800123",  # Logs to: 1730476800123.log
            )
        """
        if name is None:
            name = Path(cmd[0]).name

        log_writer = DailyLogWriter(log_name or name)

        logger.info(f"Starting {name}: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            log_writer.close()
            raise RuntimeError(f"Failed to spawn {name}: {exc}") from exc

        logger.info(f"Started {name} with PID {proc.pid}")

        # Start output streaming threads
        def stream_output(pipe, stream_label: str):
            if pipe is None:
                return
            with pipe:
                for line in pipe:
                    formatted = _format_log_line(name, stream_label, line)
                    log_writer.write(formatted)

        threads = [
            threading.Thread(
                target=stream_output,
                args=(proc.stdout, "stdout"),
                daemon=True,
            ),
            threading.Thread(
                target=stream_output,
                args=(proc.stderr, "stderr"),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        return cls(
            process=proc,
            name=name,
            log_writer=log_writer,
            cmd=list(cmd),
            _threads=threads,
        )

    def wait(self, timeout: float | None = None) -> int:
        """Wait for process completion, return exit code.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Exit code

        Raises:
            subprocess.TimeoutExpired: If timeout exceeded
        """
        return self.process.wait(timeout=timeout)

    def poll(self) -> int | None:
        """Check if process has terminated.

        Returns:
            Exit code if terminated, None if still running
        """
        return self.process.poll()

    def is_running(self) -> bool:
        """Check if process is still running."""
        return self.process.poll() is None

    def terminate(self, timeout: float = 15) -> int:
        """Gracefully terminate process with automatic escalation.

        This method handles the full termination sequence in ONE CALL:
        1. Send SIGTERM (graceful shutdown request)
        2. Wait up to `timeout` seconds for process to exit
        3. If still alive, send SIGKILL (force kill)
        4. Wait for final cleanup (max 1 second)
        5. Return exit code

        Args:
            timeout: Seconds to wait after SIGTERM before SIGKILL (default: 15)

        Returns:
            Exit code (may be negative for signals, e.g., -15 for SIGTERM)

        Example:
            exit_code = managed.terminate(timeout=10)  # One call, blocks until dead
        """
        logger.debug(f"Terminating {self.name} (PID {self.pid})...")
        try:
            self.process.terminate()
            exit_code = self.process.wait(timeout=timeout)
            logger.debug(f"{self.name} terminated gracefully with code {exit_code}")
            return exit_code
        except subprocess.TimeoutExpired:
            logger.warning(
                f"{self.name} did not terminate after {timeout}s, force killing..."
            )
            self.process.kill()
            exit_code = self.process.wait(timeout=1)
            logger.debug(f"{self.name} killed with code {exit_code}")
            return exit_code

    def cleanup(self) -> None:
        """Wait for output threads to finish and close log file.

        Call this after process exits to clean up resources.
        """
        for thread in self._threads:
            thread.join(timeout=1)
        self.log_writer.close()

    @property
    def pid(self) -> int:
        """Process ID."""
        return self.process.pid

    @property
    def returncode(self) -> int | None:
        """Return code if process has exited, None otherwise."""
        return self.process.returncode


def run_task(
    cmd: list[str],
    *,
    name: str | None = None,
    log_name: str | None = None,
    timeout: float | None = None,
    env: dict | None = None,
) -> tuple[bool, int]:
    """Run a task to completion with automatic logging (blocking).

    Spawns process, waits for completion, cleans up resources.
    Output is automatically logged to: {JOURNAL_PATH}/{YYYYMMDD}/health/{name}.log

    Args:
        cmd: Command and arguments
        name: Process name for logging (defaults to cmd[0] basename)
        log_name: Override log filename base (defaults to name)
        timeout: Optional timeout in seconds
        env: Optional environment variables

    Returns:
        (success, exit_code) tuple where success = (exit_code == 0)

    Example:
        success, code = run_task(
            ["think-summarize", "20241101"],
            name="summarize",
            timeout=300,
        )
        if not success:
            logger.error(f"Summarize failed with code {code}")

        # Or with custom log filename:
        success, code = run_task(
            ["think-importer", "file.txt"],
            name="importer",
            log_name="1730476800123",  # Logs to: 1730476800123.log
        )
    """
    if name is None:
        name = Path(cmd[0]).name

    managed = ManagedProcess.spawn(cmd, name=name, log_name=log_name, env=env)
    try:
        exit_code = managed.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error(f"{name} timed out after {timeout}s, terminating...")
        exit_code = managed.terminate()
    finally:
        managed.cleanup()

    if exit_code != 0:
        logger.warning(f"{name} exited with code {exit_code}")

    return (exit_code == 0, exit_code)
