#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Event-based processor dispatcher for observe subsystem.

Listens for observe.observing Callosum events and spawns appropriate handler
processes, capturing their stdout/stderr to log files like supervisor.py does
for runners. Batch mode (--day) uses file-based scanning for historical days.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from observe.utils import VIDEO_EXTENSIONS
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
    """Event-driven sensor that spawns handler processes for media files."""

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

        self.running_flag = True

        # Callosum connection for receiving events and emitting status
        self.callosum: Optional[CallosumConnection] = None

        # Track last status emission time
        self.last_status_emit = 0.0

        # Track segment processing: {segment_key: {pending_files}}
        self.segment_files: Dict[str, set[Path]] = {}
        # Track segment start times: {segment_key: start_timestamp}
        self.segment_start_time: Dict[str, float] = {}
        # Track segment day: {segment_key: day_string}
        self.segment_day: Dict[str, str] = {}
        # Track batch origin: {segment_key: True} for segments from batch mode
        self.segment_batch: Dict[str, bool] = {}
        # Track remote origin: {segment_key: remote_name} for remote observer segments
        self.segment_remote: Dict[str, str] = {}

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

        # Files should be in segment directories: journal_dir/YYYYMMDD/HHMMSS_LEN/file.ext
        # Expected structure: 3 parts from journal_dir
        try:
            rel_path = file_path.relative_to(self.journal_dir)
            if len(rel_path.parts) != 3:
                return None
        except ValueError:
            # File not under journal directory
            return None

        for pattern, handler_info in self.handlers.items():
            if file_path.match(pattern):
                return handler_info
        return None

    def _spawn_handler(
        self,
        file_path: Path,
        handler_name: str,
        command: List[str],
        day: Optional[str] = None,
        batch: bool = False,
        segment: Optional[str] = None,
        remote: Optional[str] = None,
    ):
        """Spawn a handler process for the file.

        Files are expected to be in segment directories: YYYYMMDD/HHMMSS_LEN/file.ext

        Args:
            file_path: Path to the file to process (in segment directory)
            handler_name: Name of the handler (e.g., "describe", "transcribe")
            command: Command template with {file} placeholder
            day: Day string (YYYYMMDD), extracted from path if not provided
            batch: Whether this is from batch processing mode
            segment: Segment key, extracted from path if not provided
            remote: Remote name for REMOTE_NAME env var
        """
        # Extract day and segment from path: journal_dir/YYYYMMDD/HHMMSS_LEN/file.ext
        try:
            rel_path = file_path.relative_to(self.journal_dir)
            if len(rel_path.parts) >= 2:
                if day is None:
                    day = rel_path.parts[0]
                if segment is None:
                    segment = rel_path.parts[1]
        except ValueError:
            pass

        with self.lock:
            # Skip if already processing this file
            if file_path in self.running:
                logger.debug(f"File {file_path.name} already being processed")
                return

            # Register file for segment tracking (segment already extracted above)
            if segment:
                if segment not in self.segment_files:
                    self.segment_files[segment] = set()
                    self.segment_start_time[segment] = time.time()
                    if day:
                        self.segment_day[segment] = day
                    # Track batch origin for observed event
                    if batch:
                        self.segment_batch[segment] = True
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

            event_fields = {
                "file": str(rel_file),
                "handler": handler_name,
                "ref": ref,
            }
            if day:
                event_fields["day"] = day
            if segment:
                event_fields["segment"] = segment
            if remote:
                event_fields["remote"] = remote
            self.callosum.emit("observe", "detected", **event_fields)

        # Replace {file} placeholder with actual file path
        cmd = [str(file_path) if arg == "{file}" else arg for arg in command]

        # Add verbose/debug flags if set
        if self.debug:
            cmd.append("-d")
        elif self.verbose:
            cmd.append("-v")

        # Use unified runner to spawn process with automatic logging
        logger.info(f"Spawning {handler_name} for {file_path.name}: {' '.join(cmd)}")

        # Build environment with remote context for handlers
        env = os.environ.copy()
        if remote:
            env["REMOTE_NAME"] = remote

        try:
            managed = RunnerManagedProcess.spawn(
                cmd, ref=ref, callosum=self.callosum, env=env
            )
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
        from observe.utils import get_segment_key

        segment = get_segment_key(file_path)
        if not segment:
            return

        with self.lock:
            if segment in self.segment_files:
                self.segment_files[segment].discard(file_path)

                # If no more pending files, emit observed event
                if not self.segment_files[segment]:
                    # Calculate processing duration
                    duration = int(time.time() - self.segment_start_time[segment])
                    day = self.segment_day.get(segment)
                    batch = self.segment_batch.get(segment, False)
                    remote = self.segment_remote.get(segment)

                    if self.callosum:
                        event_fields = {
                            "segment": segment,
                            "day": day,
                            "duration": duration,
                        }
                        if batch:
                            event_fields["batch"] = True
                        if remote:
                            event_fields["remote"] = remote
                        self.callosum.emit("observe", "observed", **event_fields)
                    logger.info(
                        f"Segment fully observed: {day}/{segment} ({duration}s)"
                    )

                    # Cleanup
                    del self.segment_files[segment]
                    del self.segment_start_time[segment]
                    if segment in self.segment_day:
                        del self.segment_day[segment]
                    if segment in self.segment_batch:
                        del self.segment_batch[segment]
                    if segment in self.segment_remote:
                        del self.segment_remote[segment]

    def _handle_file(
        self,
        file_path: Path,
        segment: Optional[str] = None,
        remote: Optional[str] = None,
    ):
        """Route file to appropriate handler.

        Args:
            file_path: Path to the file to process
            segment: Optional segment key for tracking
            remote: Optional remote name for REMOTE_NAME env var
        """
        if not file_path.exists():
            logger.warning(f"File not found, skipping: {file_path}")
            return

        handler_info = self._match_pattern(file_path)
        if handler_info:
            handler_name, command = handler_info
            self._spawn_handler(
                file_path, handler_name, command, segment=segment, remote=remote
            )

    def _handle_callosum_message(self, message: Dict[str, Any]):
        """Handle incoming Callosum messages, filtering for observe.observing events."""
        tract = message.get("tract")
        event = message.get("event")

        if tract != "observe" or event != "observing":
            return

        # Extract event fields
        day = message.get("day")
        segment = message.get("segment")
        files = message.get("files", [])
        remote = message.get("remote")  # Optional: set for remote observer uploads

        if not day or not segment or not files:
            logger.warning(
                f"Invalid observing event: missing day/segment/files: {message}"
            )
            return

        logger.info(f"Received observing event: {day}/{segment} ({len(files)} files)")

        # Build full paths for all files in this segment
        # Files are in segment directories: YYYYMMDD/HHMMSS_LEN/filename
        segment_dir = self.journal_dir / day / segment
        file_paths = [segment_dir / filename for filename in files]

        # Pre-register segment tracking with complete file list
        # This ensures segment completion is tracked correctly even if some files
        # don't match patterns or fail to process
        with self.lock:
            if segment not in self.segment_files:
                self.segment_files[segment] = set()
                self.segment_start_time[segment] = time.time()
                self.segment_day[segment] = day
                if remote:
                    self.segment_remote[segment] = remote
            for file_path in file_paths:
                # Only track files that will be processed (match a pattern)
                if self._match_pattern(file_path):
                    self.segment_files[segment].add(file_path)

        # Process each file (pass segment context for env vars)
        for file_path in file_paths:
            self._handle_file(file_path, segment=segment, remote=remote)

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
        """Start listening for observe.observing Callosum events."""

        # Start Callosum connection with callback for receiving events
        self.callosum = CallosumConnection()
        self.callosum.start(callback=self._handle_callosum_message)
        logger.info("Listening for observe.observing events via Callosum")

        while self.running_flag:

            # Emit status every 5 seconds if there's activity
            now = time.time()
            if now - self.last_status_emit >= 5:
                self._emit_status()
                self.last_status_emit = now

            time.sleep(1)

    def stop(self):
        """Stop listening and cleanup running processes."""
        self.running_flag = False

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

        Files are in segment directories (HHMMSS_LEN/). A file is considered
        unprocessed if it has no corresponding .jsonl output file.

        Args:
            day: Day in YYYYMMDD format
            max_jobs: Maximum number of concurrent processing jobs
        """
        from think.utils import segment_key

        day_dir = day_path(day)
        if not day_dir.exists():
            logger.error(f"Day directory not found: {day_dir}")
            return

        # Find all matching unprocessed files in segment directories
        to_process = []
        for segment_dir in day_dir.iterdir():
            if not segment_dir.is_dir() or not segment_key(segment_dir.name):
                continue

            for file_path in segment_dir.iterdir():
                if not file_path.is_file():
                    continue

                # Check if output JSONL exists (already processed)
                output_path = file_path.with_suffix(".jsonl")
                if output_path.exists():
                    continue

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

        def process_with_limit(file_path, handler_name, command, day):
            """Process a single file with semaphore-controlled concurrency."""
            with semaphore:
                self._spawn_handler(
                    file_path, handler_name, command, day=day, batch=True
                )
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
                args=(file_path, handler_name, command, day),
                daemon=False,
            )
            thread.start()
            threads.append(thread)

        # Wait for all to complete
        for thread in threads:
            thread.join()

        logger.info("Batch processing complete")


def scan_day(day_dir: Path) -> dict:
    """Scan a day directory for processed and unprocessed files.

    Files are in segment directories (HHMMSS_LEN/). A file is considered
    processed if it has a corresponding .jsonl output file.

    Args:
        day_dir: Path to day directory (YYYYMMDD)

    Returns:
        Dictionary with:
        - "processed": List of JSONL output files in segments (HHMMSS_LEN/audio.jsonl, etc)
        - "unprocessed": List of unprocessed source media files in segments
        - "pending_segments": Count of unique segments with pending files
    """
    from think.utils import segment_key

    processed = []
    unprocessed = []
    pending_segment_keys = set()

    if not day_dir.exists():
        return {"processed": [], "unprocessed": [], "pending_segments": 0}

    for segment in day_dir.iterdir():
        if not segment.is_dir() or not segment_key(segment.name):
            continue

        # Check each file in the segment
        for file_path in segment.iterdir():
            if not file_path.is_file():
                continue

            # JSONL files are outputs
            if file_path.suffix == ".jsonl":
                processed.append(f"{segment.name}/{file_path.name}")
                continue

            # Check if media file has corresponding JSONL (processed)
            if (
                file_path.suffix.lower() in VIDEO_EXTENSIONS
                or file_path.suffix.lower()
                in (
                    ".flac",
                    ".m4a",
                )
            ):
                output_path = file_path.with_suffix(".jsonl")
                if not output_path.exists():
                    unprocessed.append(f"{segment.name}/{file_path.name}")
                    pending_segment_keys.add(segment.name)

    processed.sort()
    unprocessed.sort()

    return {
        "processed": processed,
        "unprocessed": unprocessed,
        "pending_segments": len(pending_segment_keys),
    }


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

    # Register handlers - match by extension
    # Audio files in segment directories
    sensor.register("*.flac", "transcribe", ["observe-transcribe", "{file}"])
    sensor.register("*.m4a", "transcribe", ["observe-transcribe", "{file}"])

    # Video files in segment directories
    for ext in VIDEO_EXTENSIONS:
        sensor.register(f"*{ext}", "describe", ["observe-describe", "{file}"])

    if args.day:
        # Batch mode: process specific day
        logger.info(
            f"Processing files from day {args.day} with {args.jobs} concurrent jobs"
        )
        sensor.process_day(args.day, max_jobs=args.jobs)
    else:
        # Event mode: listen for Callosum events
        logger.info("Starting observe sensor in event mode...")
        try:
            sensor.start()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            sensor.stop()


if __name__ == "__main__":
    main()
