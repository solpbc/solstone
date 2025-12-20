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

from think.callosum import CallosumConnection
from think.runner import ManagedProcess as RunnerManagedProcess
from think.utils import day_path, setup_cli

logger = logging.getLogger(__name__)


class HandlerProcess:
    """Manages a running handler subprocess with RunnerManagedProcess."""

    def __init__(
        self, file_path: Path, managed: RunnerManagedProcess, handler_name: str
    ):
        self.file_path = file_path
        self.managed = managed
        self.process = managed.process
        self.handler_name = handler_name

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
        # Each entry is (file_path, queued_at_timestamp)
        self.describe_queue: List[tuple[Path, float]] = []
        self.current_describe_process: Optional[HandlerProcess] = None

        self.observer: Optional[Observer] = None
        self.current_day: Optional[str] = None
        self.running_flag = True

        # Callosum connection for emitting detected events
        self.callosum: Optional[CallosumConnection] = None

        # Track last status emission time
        self.last_status_emit = 0.0

        # Track segment processing: {segment_key: {pending_files}}
        self.segment_files: Dict[str, set[Path]] = {}
        # Track segment start times: {segment_key: start_timestamp}
        self.segment_start_time: Dict[str, float] = {}

    def register(self, pattern: str, handler_name: str, command: List[str]):
        """
        Register a handler for a file pattern.

        Args:
            pattern: Glob pattern (e.g., "*.webm", "*.flac")
            handler_name: Name for logging (e.g., "describe", "transcribe")
            command: Command list where "{file}" will be replaced with file path
        """
        self.handlers[pattern] = (handler_name, command)
        logger.info(f"Registered handler '{handler_name}' for pattern '{pattern}'")

    def _match_pattern(self, file_path: Path) -> Optional[tuple[str, List[str]]]:
        """Check if file matches any registered pattern."""
        # Ignore hidden files (temp recordings with dot prefix)
        if file_path.name.startswith("."):
            return None

        # Ignore files in subdirectories (segments, trash/)
        # Expected structure: journal_dir/YYYYMMDD/file.ext (2 parts from journal_dir)
        # Reject: journal_dir/YYYYMMDD/HHMMSS/file.ext (3+ parts from journal_dir)
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

            # Register file for segment tracking
            from think.utils import segment_key

            segment = segment_key(file_path.name)
            if segment:
                if segment not in self.segment_files:
                    self.segment_files[segment] = set()
                    self.segment_start_time[segment] = time.time()
                self.segment_files[segment].add(file_path)

            # Queue describe requests to ensure only one runs at a time
            if handler_name == "describe":
                if self.current_describe_process is not None:
                    # Check if file already queued (compare just paths)
                    queued_paths = [p for p, _ in self.describe_queue]
                    if file_path not in queued_paths:
                        self.describe_queue.append((file_path, time.time()))
                        logger.info(
                            f"Queueing {file_path.name} for describe (queue size: {len(self.describe_queue)})"
                        )
                    return

        # Generate correlation ID for this handler run
        ref = str(int(time.time() * 1000))

        # Emit detected event with file and ref
        if self.callosum:
            try:
                rel_file = file_path.relative_to(self.journal_dir)
            except ValueError:
                rel_file = file_path

            self.callosum.emit(
                "observe",
                "detected",
                file=str(rel_file),
                handler=handler_name,
                ref=ref,
            )

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
            managed = RunnerManagedProcess.spawn(cmd, ref=ref, callosum=self.callosum)
        except RuntimeError as exc:
            logger.error(str(exc))
            # Release describe lock if this was a describe handler
            if handler_name == "describe":
                with self.lock:
                    self.current_describe_process = None
            return

        handler_proc = HandlerProcess(file_path, managed, handler_name)

        with self.lock:
            self.running[file_path] = handler_proc
            if handler_name == "describe":
                self.current_describe_process = handler_proc

        # Monitor process completion in background
        threading.Thread(
            target=self._monitor_completion,
            args=(handler_proc,),
            daemon=True,
        ).start()

    def _monitor_completion(self, handler_proc: HandlerProcess):
        """Monitor handler process and cleanup when done."""
        try:
            exit_code = handler_proc.process.wait()

            if exit_code == 0:
                logger.info(
                    f"Handler completed successfully for {handler_proc.file_path.name}"
                )

                # Check if segment is fully observed
                self._check_segment_observed(handler_proc.file_path)
            else:
                logger.error(
                    f"{handler_proc.handler_name} failed for {handler_proc.file_path.name} "
                    f"with exit code {exit_code} - see log {handler_proc.managed.ref}.log"
                )

            handler_proc.cleanup()

            with self.lock:
                if handler_proc.file_path in self.running:
                    del self.running[handler_proc.file_path]

        except Exception as exc:
            logger.error(
                f"Unexpected error in monitor thread for {handler_proc.file_path.name}: {exc}",
                exc_info=True,
            )
        finally:
            # Always process next queued describe if this was a describe handler
            if handler_proc is self.current_describe_process:
                self._process_next_describe()

    def _process_next_describe(self):
        """Process next queued describe task."""
        with self.lock:
            self.current_describe_process = None
            if self.describe_queue:
                next_file, queued_at = self.describe_queue.pop(0)
                logger.info(
                    f"Starting queued describe for {next_file.name} ({len(self.describe_queue)} remaining)"
                )
                handler_info = self._match_pattern(next_file)
                if handler_info:
                    handler_name, command = handler_info
                    self._spawn_handler(next_file, handler_name, command)

    def _check_segment_observed(self, file_path: Path):
        """Check if all files for this segment have completed processing."""
        from think.utils import segment_key

        segment = segment_key(file_path.name)
        if not segment:
            return

        with self.lock:
            if segment in self.segment_files:
                self.segment_files[segment].discard(file_path)

                # If no more pending files, emit observed event
                if not self.segment_files[segment]:
                    # Calculate processing duration
                    duration = int(time.time() - self.segment_start_time[segment])

                    if self.callosum:
                        self.callosum.emit(
                            "observe",
                            "observed",
                            segment=segment,
                            duration=duration,
                        )
                    logger.info(f"Segment fully observed: {segment} ({duration}s)")

                    # Cleanup
                    del self.segment_files[segment]
                    del self.segment_start_time[segment]

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

    def _emit_status(self):
        """Emit observe.status event with current processing state (only when active)."""
        if not self.callosum:
            return

        with self.lock:
            # Check if there's any activity to report
            if not self.running and not self.describe_queue:
                return  # Nothing active, don't emit

            # Build status object
            status = {}

            # Get journal path for relative paths
            journal_path = os.getenv("JOURNAL_PATH", "")

            # Collect describe info
            describe_running = None
            describe_queued = []

            if self.current_describe_process is not None:
                handler_proc = self.current_describe_process
                try:
                    rel_file = (
                        str(handler_proc.file_path.relative_to(journal_path))
                        if journal_path
                        else str(handler_proc.file_path)
                    )
                except ValueError:
                    rel_file = str(handler_proc.file_path)

                describe_running = {
                    "file": rel_file,
                    "ref": handler_proc.managed.ref,
                }

            # Get queued describes with age
            now = time.time()
            for file_path, queued_at in self.describe_queue:
                try:
                    rel_file = (
                        str(file_path.relative_to(journal_path))
                        if journal_path
                        else str(file_path)
                    )
                except ValueError:
                    rel_file = str(file_path)

                describe_queued.append(
                    {"file": rel_file, "age_seconds": int(now - queued_at)}
                )

            # Add describe section if any activity
            if describe_running or describe_queued:
                status["describe"] = {}
                if describe_running:
                    status["describe"]["running"] = describe_running
                if describe_queued:
                    status["describe"]["queued"] = describe_queued

            # Collect transcribe info (any running handler that's not describe)
            transcribe_running = []
            for file_path, handler_proc in self.running.items():
                if handler_proc is not self.current_describe_process:
                    try:
                        rel_file = (
                            str(file_path.relative_to(journal_path))
                            if journal_path
                            else str(file_path)
                        )
                    except ValueError:
                        rel_file = str(file_path)

                    transcribe_running.append(
                        {"file": rel_file, "ref": handler_proc.managed.ref}
                    )

            # Add transcribe section if any activity
            if transcribe_running:
                status["transcribe"] = {"running": transcribe_running}

            # Only emit if we have something to report
            if status:
                self.callosum.emit("observe", "status", **status)

    def start(self):
        """Start watching for new files with day rollover."""

        # Start Callosum connection for emitting detected events
        self.callosum = CallosumConnection()
        self.callosum.start()

        class SensorEventHandler(FileSystemEventHandler):
            def __init__(self, sensor):
                self.sensor = sensor

            def on_created(self, event):
                if not event.is_directory:
                    path = Path(event.src_path)
                    # Ignore hidden files (temp recordings)
                    if not path.name.startswith("."):
                        self.sensor._handle_file(path)

            def on_moved(self, event):
                if not event.is_directory:
                    path = Path(event.dest_path)
                    # Ignore hidden files (temp recordings)
                    if not path.name.startswith("."):
                        self.sensor._handle_file(path)

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

            # Emit status every 5 seconds if there's activity
            now = time.time()
            if now - self.last_status_emit >= 5:
                self._emit_status()
                self.last_status_emit = now

            time.sleep(1)

    def stop(self):
        """Stop watching and cleanup running processes."""
        self.running_flag = False
        if self.observer:
            self.observer.stop()
            self.observer.join()

        # Stop Callosum connection
        if self.callosum:
            self.callosum.stop()

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
        moved to segments (HHMMSS/). This approach handles incomplete
        processing gracefully by re-running even if output files exist.

        Args:
            day: Day in YYYYMMDD format
            max_jobs: Maximum number of concurrent processing jobs
        """
        day_dir = day_path(day)
        if not day_dir.exists():
            logger.error(f"Day directory not found: {day_dir}")
            return

        # Find all matching unprocessed files (not yet moved to segments)
        to_process = []
        for file_path in day_dir.iterdir():
            if file_path.is_file():
                handler_info = self._match_pattern(file_path)
                if handler_info:
                    handler_name, command = handler_info
                    to_process.append((file_path, handler_name, command))

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
        - "processed": List of JSONL output files in segments (HHMMSS/audio.jsonl, HHMMSS/screen.jsonl)
        - "unprocessed": List of unprocessed source media files in day root
    """
    # Find processed output files in segments (HHMMSS/)
    from think.utils import segment_key

    processed = []
    for segment in day_dir.iterdir():
        if segment.is_dir() and segment_key(segment.name):
            # Check for audio.jsonl and split audio files
            for audio_file in segment.glob("*audio.jsonl"):
                processed.append(f"{segment.name}/{audio_file.name}")
            # Check for screen.jsonl
            screen_jsonl = segment / "screen.jsonl"
            if screen_jsonl.exists():
                processed.append(f"{segment.name}/screen.jsonl")

    processed.sort()

    # Find unprocessed source media (still in day root, not yet moved to segments)
    # Match by extension only - any descriptive suffix is allowed
    unprocessed = []
    unprocessed.extend(sorted(p.name for p in day_dir.glob("*.flac")))
    unprocessed.extend(sorted(p.name for p in day_dir.glob("*.m4a")))
    unprocessed.extend(sorted(p.name for p in day_dir.glob("*.webm")))
    unprocessed.extend(sorted(p.name for p in day_dir.glob("*.mp4")))
    unprocessed.extend(sorted(p.name for p in day_dir.glob("*.mov")))

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

    # Register handlers - match by extension, ignore descriptive suffix
    # Audio files: any HHMMSS_*.flac or HHMMSS_*.m4a in day root
    sensor.register("*.flac", "transcribe", ["observe-transcribe", "{file}"])
    sensor.register("*.m4a", "transcribe", ["observe-transcribe", "{file}"])

    # Video files: any HHMMSS_*.webm, HHMMSS_*.mp4, HHMMSS_*.mov in day root
    sensor.register("*.webm", "describe", ["observe-describe", "{file}"])
    sensor.register("*.mp4", "describe", ["observe-describe", "{file}"])
    sensor.register("*.mov", "describe", ["observe-describe", "{file}"])

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
