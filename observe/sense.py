#!/usr/bin/env python3
"""File-based processor dispatcher for observe subsystem.

Watches day directories for new files and spawns appropriate handler processes,
capturing their stdout/stderr to log files like supervisor.py does for runners.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from think.utils import day_path, setup_cli

logger = logging.getLogger(__name__)


class ProcessLogWriter:
    """Thread-safe writer that appends process output to a log file."""

    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = log_path
        self._lock = threading.Lock()
        self._fh = log_path.open("a", encoding="utf-8")

    def write(self, message: str):
        with self._lock:
            self._fh.write(message)
            self._fh.flush()

    def close(self):
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    @property
    def path(self) -> Path:
        return self._log_path


class HandlerProcess:
    """Manages a running handler subprocess."""

    def __init__(
        self,
        file_path: Path,
        process: subprocess.Popen,
        handler_name: str,
        logger: ProcessLogWriter,
    ):
        self.file_path = file_path
        self.process = process
        self.handler_name = handler_name
        self.logger = logger
        self.threads: List[threading.Thread] = []

    def cleanup(self):
        for thread in self.threads:
            thread.join(timeout=1)
        self.logger.close()


def _format_log_line(handler_name: str, file_name: str, stream: str, line: str) -> str:
    """Format log line with timestamp, handler, file, and stream label."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    clean_line = line.rstrip("\n")
    return f"{timestamp} [{handler_name}:{file_name}:{stream}] {clean_line}\n"


def _stream_output(
    pipe,
    handler_name: str,
    file_name: str,
    stream_label: str,
    logger: ProcessLogWriter,
):
    """Stream process output to log file."""
    if pipe is None:
        return
    with pipe:
        for line in pipe:
            logger.write(_format_log_line(handler_name, file_name, stream_label, line))


class FileSensor:
    """Pattern-based file watcher that spawns handler processes."""

    def __init__(self, journal_dir: Path):
        self.journal_dir = journal_dir

        # Registry: {glob_pattern: (handler_name, command_template)}
        self.handlers: Dict[str, tuple[str, List[str]]] = {}

        # Track running processes: {file_path: HandlerProcess}
        self.running: Dict[Path, HandlerProcess] = {}
        self.lock = threading.RLock()

        self.observer: Optional[Observer] = None
        self.current_day: Optional[str] = None
        self.running_flag = True

    def register(self, pattern: str, handler_name: str, command: List[str]):
        """
        Register a handler for a file pattern.

        Args:
            pattern: Glob pattern (e.g., "*.webm", "*_raw.flac")
            handler_name: Name for logging (e.g., "describe", "transcribe")
            command: Command list where "{file}" will be replaced with file path
        """
        self.handlers[pattern] = (handler_name, command)
        logger.info(f"Registered handler '{handler_name}' for pattern '{pattern}'")

    def _match_pattern(self, file_path: Path) -> Optional[tuple[str, List[str]]]:
        """Check if file matches any registered pattern."""
        # Ignore files in subdirectories (heard/, trash/)
        # Expected structure: journal_dir/YYYYMMDD/file.ext (2 parts from journal_dir)
        # Reject: journal_dir/YYYYMMDD/heard/file.ext (3+ parts from journal_dir)
        try:
            rel_path = file_path.relative_to(self.journal_dir)
            if len(rel_path.parts) != 2:
                return None
        except ValueError:
            # File not under journal directory
            return None

        for pattern, handler_info in self.handlers.items():
            if file_path.match(pattern):
                return handler_info
        return None

    def _spawn_handler(self, file_path: Path, handler_name: str, command: List[str]):
        """Spawn a handler process for the file."""
        with self.lock:
            # Skip if already processing this file
            if file_path in self.running:
                logger.debug(f"File {file_path.name} already being processed")
                return

        # Replace {file} placeholder with actual file path
        cmd = [str(file_path) if arg == "{file}" else arg for arg in command]

        # Create log file in day-specific health directory: YYYYMMDD/health/sense_{handler_name}.log
        day_health_dir = file_path.parent / "health"
        log_writer = ProcessLogWriter(day_health_dir / f"sense_{handler_name}.log")

        logger.info(f"Spawning {handler_name} for {file_path.name}: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            logger.error(f"Failed to spawn {handler_name} for {file_path.name}: {exc}")
            log_writer.close()
            return

        handler_proc = HandlerProcess(file_path, proc, handler_name, log_writer)

        # Start output streaming threads
        threads = [
            threading.Thread(
                target=_stream_output,
                args=(proc.stdout, handler_name, file_path.name, "stdout", log_writer),
                daemon=True,
            ),
            threading.Thread(
                target=_stream_output,
                args=(proc.stderr, handler_name, file_path.name, "stderr", log_writer),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        handler_proc.threads = threads

        with self.lock:
            self.running[file_path] = handler_proc

        # Monitor process completion in background
        threading.Thread(
            target=self._monitor_completion,
            args=(handler_proc,),
            daemon=True,
        ).start()

    def _monitor_completion(self, handler_proc: HandlerProcess):
        """Monitor handler process and cleanup when done."""
        exit_code = handler_proc.process.wait()

        if exit_code == 0:
            logger.info(
                f"{handler_proc.handler_name} completed successfully for {handler_proc.file_path.name}"
            )

            # If describe completed successfully, run reduce
            if handler_proc.handler_name == "describe":
                self._run_reduce(handler_proc.file_path)
        else:
            logger.error(
                f"{handler_proc.handler_name} failed with exit code {exit_code} for {handler_proc.file_path.name}"
            )

        handler_proc.cleanup()

        with self.lock:
            if handler_proc.file_path in self.running:
                del self.running[handler_proc.file_path]

    def _run_reduce(self, video_path: Path):
        """Run reduce on the video file after describe completes."""
        cmd = ["observe-reduce", str(video_path)]
        logger.info(f"Running reduce for {video_path.name}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"Reduce completed successfully for {video_path.name}")
            else:
                logger.warning(
                    f"Reduce failed for {video_path.name} (exit code {result.returncode})"
                )
                if result.stderr:
                    logger.debug(f"Reduce stderr: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Reduce timed out for {video_path.name}")
        except Exception as exc:
            logger.warning(f"Reduce failed for {video_path.name}: {exc}")

    def _handle_file(self, file_path: Path):
        """Route file to appropriate handler."""
        # Small delay to ensure file is fully written
        time.sleep(0.1)

        if not file_path.exists():
            return

        handler_info = self._match_pattern(file_path)
        if handler_info:
            handler_name, command = handler_info
            self._spawn_handler(file_path, handler_name, command)

    def start(self):
        """Start watching for new files with day rollover."""

        class SensorEventHandler(FileSystemEventHandler):
            def __init__(self, sensor):
                self.sensor = sensor

            def on_created(self, event):
                if not event.is_directory:
                    self.sensor._handle_file(Path(event.src_path))

        event_handler = SensorEventHandler(self)

        while self.running_flag:
            today_str = datetime.now().strftime("%Y%m%d")
            day_dir = day_path()

            if day_dir.exists() and (self.current_day != today_str):
                if self.observer:
                    logger.info("Day rollover, stopping old observer")
                    self.observer.stop()
                    self.observer.join()

                self.observer = Observer()
                self.observer.schedule(event_handler, str(day_dir), recursive=False)
                self.observer.start()
                self.current_day = today_str
                logger.info(f"Watching {day_dir}")

            time.sleep(1)

    def stop(self):
        """Stop watching and cleanup running processes."""
        self.running_flag = False
        if self.observer:
            self.observer.stop()
            self.observer.join()

        # Gracefully terminate any running handler processes
        with self.lock:
            running_handlers = list(self.running.values())

        if running_handlers:
            logger.info(f"Terminating {len(running_handlers)} running handler(s)...")

        for handler_proc in running_handlers:
            try:
                # Send SIGTERM for graceful shutdown
                handler_proc.process.terminate()
                logger.debug(
                    f"Sent SIGTERM to {handler_proc.handler_name} for {handler_proc.file_path.name}"
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to terminate {handler_proc.handler_name} for {handler_proc.file_path.name}: {exc}"
                )

        # Wait up to 5 seconds for processes to terminate gracefully
        import signal

        deadline = time.time() + 5
        for handler_proc in running_handlers:
            try:
                timeout = max(0.1, deadline - time.time())
                handler_proc.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if still running
                logger.warning(
                    f"Force killing {handler_proc.handler_name} for {handler_proc.file_path.name}"
                )
                handler_proc.process.kill()
                handler_proc.process.wait()

            # Cleanup threads and log files
            handler_proc.cleanup()

        # Clear running dict
        with self.lock:
            self.running.clear()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Unified observe file processor")
    args = setup_cli(parser)

    journal = Path(os.getenv("JOURNAL_PATH", ""))
    if not journal.is_dir():
        parser.error("JOURNAL_PATH not set or invalid")

    sensor = FileSensor(journal)

    # Register handlers
    sensor.register("*_screen.webm", "describe", ["observe-describe", "{file}"])
    sensor.register("*_screen.mp4", "describe", ["observe-describe", "{file}"])
    sensor.register("*_raw.flac", "transcribe", ["observe-transcribe", "{file}"])
    sensor.register("*_raw.m4a", "transcribe", ["observe-transcribe", "{file}"])

    logger.info("Starting observe sensor...")
    try:
        sensor.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sensor.stop()


if __name__ == "__main__":
    main()
