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

from think.runner import ManagedProcess as RunnerManagedProcess
from think.runner import run_task
from think.utils import day_path, setup_cli

logger = logging.getLogger(__name__)


class HandlerProcess:
    """Manages a running handler subprocess with RunnerManagedProcess."""

    def __init__(self, file_path: Path, managed: RunnerManagedProcess):
        self.file_path = file_path
        self.managed = managed
        self.handler_name = managed.name
        self.process = managed.process

    def cleanup(self):
        self.managed.cleanup()


class FileSensor:
    """Pattern-based file watcher that spawns handler processes."""

    def __init__(self, journal_dir: Path, verbose: bool = False, debug: bool = False):
        self.journal_dir = journal_dir
        self.verbose = verbose
        self.debug = debug

        # Registry: {glob_pattern: (handler_name, command_template)}
        self.handlers: Dict[str, tuple[str, List[str]]] = {}

        # Track running processes: {file_path: HandlerProcess}
        self.running: Dict[Path, HandlerProcess] = {}
        self.lock = threading.RLock()

        # Queue for describe requests (only one describe runs at a time)
        self.describe_queue: List[Path] = []
        self.describe_running = False

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

            # Queue describe requests to ensure only one runs at a time
            if handler_name == "describe":
                if self.describe_running:
                    if file_path not in self.describe_queue:
                        self.describe_queue.append(file_path)
                        logger.info(
                            f"Queueing {file_path.name} for describe (queue size: {len(self.describe_queue)})"
                        )
                    return
                self.describe_running = True

        # Replace {file} placeholder with actual file path
        cmd = [str(file_path) if arg == "{file}" else arg for arg in command]

        # Add verbose/debug flags if set
        if self.debug:
            cmd.append("-d")
        elif self.verbose:
            cmd.append("-v")

        # Use unified runner to spawn process with automatic logging
        logger.info(f"Spawning {handler_name} for {file_path.name}: {' '.join(cmd)}")

        try:
            managed = RunnerManagedProcess.spawn(
                cmd, name=f"sense_{handler_name}:{file_path.name}"
            )
        except RuntimeError as exc:
            logger.error(str(exc))
            # Release describe lock if this was a describe handler
            if handler_name == "describe":
                with self.lock:
                    self.describe_running = False
            return

        handler_proc = HandlerProcess(file_path, managed)

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

        # Process next queued describe if this was a describe handler
        if handler_proc.handler_name == "describe":
            next_file = None
            with self.lock:
                self.describe_running = False
                if self.describe_queue:
                    next_file = self.describe_queue.pop(0)
                    logger.info(
                        f"Starting queued describe for {next_file.name} ({len(self.describe_queue)} remaining)"
                    )

            if next_file:
                handler_info = self._match_pattern(next_file)
                if handler_info:
                    handler_name, command = handler_info
                    self._spawn_handler(next_file, handler_name, command)

    def _run_reduce(self, video_path: Path):
        """Run reduce on the video file after describe completes."""
        jsonl_path = video_path.parent / f"{video_path.stem}.jsonl"
        cmd = ["observe-reduce", str(jsonl_path)]

        # Add verbose/debug flags if set
        if self.debug:
            cmd.append("-d")
        elif self.verbose:
            cmd.append("-v")

        logger.info(f"Running reduce for {video_path.name}")

        # Use unified runner with automatic logging and timeout
        success, exit_code = run_task(
            cmd, name=f"reduce:{video_path.name}", timeout=300  # 5 minute timeout
        )

        if success:
            logger.info(f"Reduce completed successfully for {video_path.name}")
        else:
            logger.warning(
                f"Reduce failed for {video_path.name} (exit code {exit_code})"
            )

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

            def on_moved(self, event):
                if not event.is_directory:
                    self.sensor._handle_file(Path(event.dest_path))

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

    def process_day(self, day: str, max_jobs: int = 1):
        """Process all matching unprocessed files from a specific day directory.

        Files are considered unprocessed if the source media file has not been
        moved to seen/ or heard/ subdirectories. This approach handles incomplete
        processing gracefully by re-running even if output files exist.

        Also finds JSONL files without corresponding MD files and runs reduce on them.

        Args:
            day: Day in YYYYMMDD format
            max_jobs: Maximum number of concurrent processing jobs
        """
        day_dir = day_path(day)
        if not day_dir.exists():
            logger.error(f"Day directory not found: {day_dir}")
            return

        # Find all matching unprocessed files (not yet moved to seen/heard)
        to_process = []
        for file_path in day_dir.iterdir():
            if file_path.is_file():
                handler_info = self._match_pattern(file_path)
                if handler_info:
                    handler_name, command = handler_info
                    to_process.append((file_path, handler_name, command))

        # Find incomplete reduces (JSONL files without corresponding MD)
        for file_path in day_dir.glob("*_screen.jsonl"):
            md_path = file_path.parent / f"{file_path.stem}.md"
            if not md_path.exists():
                # Register reduce as a handler task
                to_process.append((file_path, "reduce", ["observe-reduce", "{file}"]))

        if not to_process:
            logger.info(f"No unprocessed files found in {day_dir}")
            return

        # Count files by extension
        ext_counts = {}
        for file_path, handler_name, command in to_process:
            ext = file_path.suffix.lower()  # e.g., ".webm", ".flac", ".jsonl"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        # Format breakdown: "21 files (10 .webm, 8 .flac, 3 .jsonl)"
        breakdown = ", ".join(
            f"{count} {ext}" for ext, count in sorted(ext_counts.items())
        )
        logger.info(
            f"Found {len(to_process)} unprocessed files to process ({breakdown})"
        )

        # Process with concurrency limit using semaphore
        semaphore = threading.Semaphore(max_jobs)
        completion_events = {}

        def process_with_limit(file_path, handler_name, command):
            """Process a single file with semaphore-controlled concurrency."""
            with semaphore:
                self._spawn_handler(file_path, handler_name, command)
                # Wait for this specific file to complete
                while file_path in self.running:
                    time.sleep(0.5)
                # Signal completion
                completion_events[file_path].set()

        # Spawn all handlers (semaphore controls concurrency)
        threads = []
        for file_path, handler_name, command in to_process:
            completion_events[file_path] = threading.Event()
            thread = threading.Thread(
                target=process_with_limit,
                args=(file_path, handler_name, command),
                daemon=False,
            )
            thread.start()
            threads.append(thread)

        # Wait for all to complete
        for thread in threads:
            thread.join()

        logger.info("Batch processing complete")


def scan_day(day_dir: Path) -> dict[str, list[str]]:
    """Scan a day directory for processed and unprocessed files.

    Args:
        day_dir: Path to day directory (YYYYMMDD)

    Returns:
        Dictionary with:
        - "processed": List of JSONL output files (*_audio.jsonl, *_screen.jsonl)
        - "unprocessed": List of unprocessed source media files
    """
    # Find processed output files
    processed = sorted(p.name for p in day_dir.glob("*_audio.jsonl"))
    processed.extend(sorted(p.name for p in day_dir.glob("*_screen.jsonl")))

    # Find unprocessed source media (still in day root)
    unprocessed = []
    unprocessed.extend(sorted(p.name for p in day_dir.glob("*_raw.flac")))
    unprocessed.extend(sorted(p.name for p in day_dir.glob("*_raw.m4a")))
    unprocessed.extend(sorted(p.name for p in day_dir.glob("*_screen.webm")))
    unprocessed.extend(sorted(p.name for p in day_dir.glob("*_screen.mp4")))
    unprocessed.extend(sorted(p.name for p in day_dir.glob("*_screen.mov")))

    return {"processed": processed, "unprocessed": unprocessed}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Unified observe file processor")
    parser.add_argument(
        "--day",
        type=str,
        help="Process files from specific day (YYYYMMDD format) instead of watching",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Max concurrent processing jobs when using --day (default: 1)",
    )
    args = setup_cli(parser)

    journal = Path(os.getenv("JOURNAL_PATH", ""))
    if not journal.is_dir():
        parser.error("JOURNAL_PATH not set or invalid")

    sensor = FileSensor(journal, verbose=args.verbose, debug=args.debug)

    # Register handlers
    sensor.register("*_screen.webm", "describe", ["observe-describe", "{file}"])
    sensor.register("*_screen.mp4", "describe", ["observe-describe", "{file}"])
    sensor.register("*_screen.mov", "describe", ["observe-describe", "{file}"])
    sensor.register("*_raw.flac", "transcribe", ["observe-transcribe", "{file}"])
    sensor.register("*_raw.m4a", "transcribe", ["observe-transcribe", "{file}"])

    if args.day:
        # Batch mode: process specific day
        logger.info(
            f"Processing files from day {args.day} with {args.jobs} concurrent jobs"
        )
        sensor.process_day(args.day, max_jobs=args.jobs)
    else:
        # Watch mode: monitor for new files
        logger.info("Starting observe sensor in watch mode...")
        try:
            sensor.start()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            sensor.stop()


if __name__ == "__main__":
    main()
