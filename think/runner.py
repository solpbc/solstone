#!/usr/bin/env python3
"""Unified process spawning and lifecycle management utilities.

All subprocess output is automatically logged to:
    {JOURNAL_PATH}/{YYYYMMDD}/health/{ref}_{process_name}.log

Where process_name is derived from cmd[0] basename, and ref is a unique correlation ID.

Symlinks provide stable access paths:
    {JOURNAL_PATH}/{YYYYMMDD}/health/{process_name}.log (day-level symlink)
    {JOURNAL_PATH}/health/{process_name}.log (journal-level symlink)

Logs automatically roll over at midnight for long-running processes.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from think.callosum import CallosumConnection

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


def _day_health_log_path(day: str, ref: str, name: str) -> Path:
    """Build path to day health log.

    Returns: {JOURNAL_PATH}/{day}/health/{ref}_{name}.log
    """
    return _get_journal_path() / day / "health" / f"{ref}_{name}.log"


def _atomic_symlink(link_path: Path, target: str) -> None:
    """Create or update symlink atomically.

    Args:
        link_path: Path where symlink should be created
        target: Target path (can be relative or absolute)
    """
    link_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_link = link_path.with_suffix(f".tmp{os.getpid()}")
    try:
        tmp_link.symlink_to(target)
        tmp_link.replace(link_path)
    finally:
        # Clean up temp file if it still exists
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink(missing_ok=True)


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

    Writes to: {JOURNAL_PATH}/{YYYYMMDD}/health/{ref}_{name}.log

    Creates and maintains symlinks:
    - {JOURNAL_PATH}/{YYYYMMDD}/health/{name}.log -> {ref}_{name}.log (day-level)
    - {JOURNAL_PATH}/health/{name}.log -> {YYYYMMDD}/health/{ref}_{name}.log (journal-level)

    When the day changes, automatically closes old file, opens new file, and updates symlinks.
    """

    def __init__(self, ref: str, name: str):
        self._ref = ref
        self._name = name
        self._lock = threading.Lock()
        self._current_day = _current_day()
        self._fh = self._open_log()
        self._update_symlinks()

    def _open_log(self):
        """Open log file for current day."""
        log_path = _day_health_log_path(self._current_day, self._ref, self._name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path.open("a", encoding="utf-8")

    def _update_symlinks(self) -> None:
        """Update day-level and journal-level symlinks to point to current log."""
        journal = _get_journal_path()
        day_health = journal / self._current_day / "health"
        log_filename = f"{self._ref}_{self._name}.log"

        # Day-level symlink: {YYYYMMDD}/health/{name}.log -> {ref}_{name}.log
        day_symlink = day_health / f"{self._name}.log"
        _atomic_symlink(day_symlink, log_filename)

        # Journal-level symlink: health/{name}.log -> ../{YYYYMMDD}/health/{ref}_{name}.log
        # Relative from journal/health/ to journal/{YYYYMMDD}/health/
        journal_symlink = journal / "health" / f"{self._name}.log"
        relative_target = f"../{self._current_day}/health/{log_filename}"
        _atomic_symlink(journal_symlink, relative_target)

    def write(self, message: str) -> None:
        """Write message to log, handling day rollover."""
        with self._lock:
            # Check for day change
            day_now = _current_day()
            if day_now != self._current_day:
                # Close old log
                if not self._fh.closed:
                    self._fh.close()
                # Open new log for new day
                self._current_day = day_now
                self._fh = self._open_log()
                # Update symlinks to point to new day's file
                self._update_symlinks()

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
        return _day_health_log_path(self._current_day, self._ref, self._name)


@dataclass
class ManagedProcess:
    """Subprocess wrapper with automatic output logging and lifecycle management.

    All output is automatically logged to:
        {JOURNAL_PATH}/{YYYYMMDD}/health/{ref}_{name}.log

    Where name is derived from cmd[0] basename, and ref is a unique correlation ID.

    Symlinks are automatically created and maintained:
        {JOURNAL_PATH}/{YYYYMMDD}/health/{name}.log -> {ref}_{name}.log (day-level)
        {JOURNAL_PATH}/health/{name}.log -> {YYYYMMDD}/health/{ref}_{name}.log (journal-level)

    Logs roll over automatically at midnight for long-running processes.

    Process lifecycle events are broadcast via Callosum logs tract.
    """

    process: subprocess.Popen
    name: str
    log_writer: DailyLogWriter
    cmd: list[str]
    _threads: list[threading.Thread]
    ref: str
    _start_time: float
    _callosum: CallosumConnection | None
    _owns_callosum: bool = True

    @classmethod
    def spawn(
        cls,
        cmd: list[str],
        *,
        env: dict | None = None,
        ref: str | None = None,
        callosum: CallosumConnection | None = None,
    ) -> "ManagedProcess":
        """Spawn process with automatic output logging to daily health directory.

        Args:
            cmd: Command and arguments
            env: Optional environment variables (inherits parent env if not provided)
            ref: Optional correlation ID (auto-generated if not provided)
            callosum: Optional shared CallosumConnection (creates new one if not provided)

        Returns:
            ManagedProcess instance

        Raises:
            RuntimeError: If process fails to spawn

        Example:
            managed = ManagedProcess.spawn(["observe-gnome", "-v"])
            # Logs to: {JOURNAL}/{YYYYMMDD}/health/{ref}_observe-gnome.log
            # Symlinks: {YYYYMMDD}/health/observe-gnome.log (day-level)
            #           health/observe-gnome.log (journal-level)

            # With explicit correlation ID:
            managed = ManagedProcess.spawn(
                ["think-indexer", "--rescan"],
                ref="1730476800000",
            )
            # Logs to: {JOURNAL}/{YYYYMMDD}/health/1730476800000_think-indexer.log
        """
        # Derive name from command basename
        name = Path(cmd[0]).name

        # Generate correlation ID (use provided ref, else timestamp)
        ref = ref if ref else str(int(time.time() * 1000))
        start_time = time.time()

        # Use provided callosum or create new one
        owns_callosum = callosum is None
        if owns_callosum:
            callosum = CallosumConnection()
            callosum.start()

        log_writer = DailyLogWriter(ref, name)

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
            if owns_callosum and callosum:
                callosum.stop()
            raise RuntimeError(f"Failed to spawn {name}: {exc}") from exc

        logger.info(f"Started {name} with PID {proc.pid}")

        # Emit exec event
        if callosum:
            callosum.emit(
                "logs",
                "exec",
                ref=ref,
                name=name,
                pid=proc.pid,
                cmd=list(cmd),
                log_path=str(log_writer.path),
            )

        # Start output streaming threads
        def stream_output(pipe, stream_label: str):
            if pipe is None:
                return
            with pipe:
                for line in pipe:
                    formatted = _format_log_line(name, stream_label, line)
                    log_writer.write(formatted)

                    # Emit line event
                    if callosum:
                        callosum.emit(
                            "logs",
                            "line",
                            ref=ref,
                            name=name,
                            pid=proc.pid,
                            stream=stream_label,
                            line=line.rstrip("\n"),
                        )

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
            ref=ref,
            _start_time=start_time,
            _callosum=callosum,
            _owns_callosum=owns_callosum,
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

        # Emit exit event
        if self._callosum:
            duration_ms = int((time.time() - self._start_time) * 1000)
            self._callosum.emit(
                "logs",
                "exit",
                ref=self.ref,
                name=self.name,
                pid=self.pid,
                exit_code=self.returncode,
                duration_ms=duration_ms,
                cmd=self.cmd,
                log_path=str(self.log_writer.path),
            )
            # Only stop callosum if we created it (not shared)
            if self._owns_callosum:
                self._callosum.stop()

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
    timeout: float | None = None,
    env: dict | None = None,
    ref: str | None = None,
    callosum: CallosumConnection | None = None,
) -> tuple[bool, int]:
    """Run a task to completion with automatic logging (blocking).

    Spawns process, waits for completion, cleans up resources.
    Output is automatically logged to: {JOURNAL_PATH}/{YYYYMMDD}/health/{ref}_{name}.log
    where name is derived from cmd[0] basename.

    Args:
        cmd: Command and arguments
        timeout: Optional timeout in seconds
        env: Optional environment variables
        ref: Optional correlation ID (auto-generated if not provided)
        callosum: Optional shared CallosumConnection (creates new one if not provided)

    Returns:
        (success, exit_code) tuple where success = (exit_code == 0)

    Example:
        success, code = run_task(
            ["think-insight", "20241101", "-f", "flow"],
            timeout=300,
        )
        # Logs to: {JOURNAL}/{YYYYMMDD}/health/{ref}_think-insight.log

        # With explicit correlation ID:
        success, code = run_task(
            ["think-indexer", "--rescan"],
            ref="1730476800000",
        )
        # Logs to: {JOURNAL}/{YYYYMMDD}/health/1730476800000_think-indexer.log
    """
    managed = ManagedProcess.spawn(cmd, env=env, ref=ref, callosum=callosum)
    try:
        exit_code = managed.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error(f"{managed.name} timed out after {timeout}s, terminating...")
        exit_code = managed.terminate()
    finally:
        managed.cleanup()

    if exit_code != 0:
        logger.warning(f"{managed.name} exited with code {exit_code}")

    return (exit_code == 0, exit_code)
