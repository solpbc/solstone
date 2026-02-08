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
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from observe.utils import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from think.callosum import CallosumConnection
from think.runner import ManagedProcess as RunnerManagedProcess
from think.utils import day_path, get_journal, now_ms, setup_cli

logger = logging.getLogger(__name__)


class QueuedItem:
    """Item in a handler queue with context for deferred processing."""

    __slots__ = ("file_path", "queued_at", "remote", "meta")

    def __init__(
        self,
        file_path: Path,
        remote: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.file_path = file_path
        self.queued_at = time.time()
        self.remote = remote
        self.meta = meta


class HandlerQueue:
    """Queue for serializing handler execution (one at a time).

    Ensures only one handler process runs at a time for resource-intensive
    operations like describe (GPU) or transcribe (memory/API constraints).
    """

    def __init__(self, name: str):
        self.name = name
        self.queue: List[QueuedItem] = []
        self.current_process: Optional["HandlerProcess"] = None

    def can_start(self) -> bool:
        """Returns True if no handler is currently running."""
        return self.current_process is None

    def enqueue(
        self,
        file_path: Path,
        remote: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add file to queue if not already present. Returns True if queued."""
        queued_paths = [item.file_path for item in self.queue]
        if file_path not in queued_paths:
            self.queue.append(QueuedItem(file_path, remote, meta))
            return True
        return False

    def set_current(self, proc: "HandlerProcess") -> None:
        """Set the currently running handler process."""
        self.current_process = proc

    def clear_current(self) -> None:
        """Clear the current process reference."""
        self.current_process = None

    def pop_next(self) -> Optional[QueuedItem]:
        """Pop and return next queued item, or None if empty."""
        if self.queue:
            return self.queue.pop(0)
        return None

    def queue_size(self) -> int:
        """Return number of items in queue."""
        return len(self.queue)


class HandlerProcess:
    """Manages a running handler subprocess with RunnerManagedProcess."""

    def __init__(
        self,
        file_path: Path,
        managed: RunnerManagedProcess,
        handler_name: str,
        cpu_fallback: bool = False,
    ):
        self.file_path = file_path
        self.managed = managed
        self.process = managed.process
        self.handler_name = handler_name
        self.cpu_fallback = (
            cpu_fallback  # True if this is a CPU retry after GPU failure
        )
        self.started_at = time.time()

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

        # Serialized handler queues (only one process at a time per handler type)
        self.handler_queues: Dict[str, HandlerQueue] = {
            "describe": HandlerQueue("describe"),
            "transcribe": HandlerQueue("transcribe"),
        }

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
        # Track handler errors per segment: {segment_key: [error_strings]}
        self.segment_errors: Dict[str, list[str]] = {}

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
        meta: Optional[Dict[str, Any]] = None,
        cpu_fallback: bool = False,
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
            meta: Optional metadata dict (facet, setting, host, platform, etc.)
                  to pass to handlers via SEGMENT_META env var
            cpu_fallback: If True, this is a retry after GPU failure (adds --cpu,
                          skips tracking/events since already done on first attempt)
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

        # Skip tracking/queueing for CPU fallback (already done on first attempt)
        if not cpu_fallback:
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

                # Check if this handler uses serialized execution
                handler_queue = self.handler_queues.get(handler_name)
                if handler_queue and not handler_queue.can_start():
                    if handler_queue.enqueue(file_path, remote=remote, meta=meta):
                        logger.info(
                            f"Queueing {file_path.name} for {handler_name} "
                            f"(queue size: {handler_queue.queue_size()})"
                        )
                    return

        # Generate correlation ID for this handler run
        ref = str(now_ms())

        # Emit detected event with file and ref (skip for CPU fallback)
        if self.callosum and not cpu_fallback:
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

        # Add --cpu flag for CPU fallback retry
        if cpu_fallback:
            cmd.append("--cpu")

        # Add verbose/debug flags if set
        if self.debug:
            cmd.append("-d")
        elif self.verbose:
            cmd.append("-v")

        # Use unified runner to spawn process with automatic logging
        fallback_note = " (CPU fallback)" if cpu_fallback else ""
        logger.info(
            f"Spawning {handler_name}{fallback_note} for {file_path.name}: {' '.join(cmd)}"
        )

        # Build environment with segment and remote context for handlers
        env = os.environ.copy()
        if segment:
            env["SEGMENT_KEY"] = segment
        if remote:
            env["REMOTE_NAME"] = remote
        if meta:
            env["SEGMENT_META"] = json.dumps(meta)

        try:
            managed = RunnerManagedProcess.spawn(
                cmd, ref=ref, callosum=self.callosum, env=env
            )
        except RuntimeError as exc:
            logger.error(str(exc))
            # Release handler queue lock if this handler uses serialized execution
            handler_queue = self.handler_queues.get(handler_name)
            if handler_queue:
                with self.lock:
                    handler_queue.clear_current()
            return

        handler_proc = HandlerProcess(
            file_path, managed, handler_name, cpu_fallback=cpu_fallback
        )

        with self.lock:
            self.running[file_path] = handler_proc
            # Track as current process if using serialized execution
            handler_queue = self.handler_queues.get(handler_name)
            if handler_queue:
                handler_queue.set_current(handler_proc)

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
            elapsed = time.time() - handler_proc.started_at

            if exit_code == 0:
                logger.info(
                    f"Handler completed successfully for {handler_proc.file_path.name} "
                    f"({elapsed:.1f}s)"
                )

                # Check if segment is fully observed
                self._check_segment_observed(handler_proc.file_path)
            elif (
                exit_code == 134
                and handler_proc.handler_name == "transcribe"
                and not handler_proc.cpu_fallback
            ):
                # Exit 134 = SIGABRT, often from cuDNN/CUDA failure
                # Retry transcribe with --cpu flag
                logger.warning(
                    f"Transcribe crashed (exit 134, likely GPU/cuDNN issue) for "
                    f"{handler_proc.file_path.name}, retrying with --cpu"
                )
                handler_proc.cleanup()
                with self.lock:
                    if handler_proc.file_path in self.running:
                        del self.running[handler_proc.file_path]
                # Respawn with CPU fallback
                self._spawn_handler(
                    handler_proc.file_path,
                    "transcribe",
                    ["sol", "transcribe", "{file}"],
                    cpu_fallback=True,
                )
                return  # Skip normal cleanup, we're retrying
            else:
                # Show journal-relative log path for easier debugging
                try:
                    log_rel = handler_proc.managed.log_writer.path.relative_to(
                        self.journal_dir
                    )
                except ValueError:
                    log_rel = handler_proc.managed.log_writer.path
                logger.error(
                    f"{handler_proc.handler_name} failed for {handler_proc.file_path.name} "
                    f"with exit code {exit_code} ({elapsed:.1f}s) - see log {log_rel}"
                )

                # Mark file as done so segment can still complete
                self._check_segment_observed(
                    handler_proc.file_path,
                    error=f"{handler_proc.handler_name} exit {exit_code}",
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
            # Process next queued item if this handler uses serialized execution
            handler_queue = self.handler_queues.get(handler_proc.handler_name)
            if handler_queue and handler_proc is handler_queue.current_process:
                self._process_next_queued(handler_queue)

    def _process_next_queued(self, handler_queue: HandlerQueue):
        """Process next queued task for a serialized handler."""
        with self.lock:
            handler_queue.clear_current()
            item = handler_queue.pop_next()
            if item:
                logger.info(
                    f"Starting queued {handler_queue.name} for {item.file_path.name} "
                    f"({handler_queue.queue_size()} remaining)"
                )
                handler_info = self._match_pattern(item.file_path)
                if handler_info:
                    handler_name, command = handler_info
                    self._spawn_handler(
                        item.file_path,
                        handler_name,
                        command,
                        remote=item.remote,
                        meta=item.meta,
                    )

    def _emit_segment_observed(self, segment: str, note: str = ""):
        """Emit observe.observed event and cleanup segment tracking.

        Must be called while holding self.lock.

        Args:
            segment: Segment key (HHMMSS_LEN format)
            note: Optional note for log message (e.g., "no handlers")
        """
        duration = int(time.time() - self.segment_start_time[segment])
        day = self.segment_day.get(segment)
        batch = self.segment_batch.get(segment, False)
        remote = self.segment_remote.get(segment)
        errors = self.segment_errors.get(segment)

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
            if errors:
                event_fields["error"] = True
                event_fields["errors"] = errors
            self.callosum.emit("observe", "observed", **event_fields)

        if errors:
            logger.warning(
                f"Segment observed with errors: {day}/{segment} ({duration}s) - {errors}"
            )
        else:
            note_str = f" ({note})" if note else ""
            logger.info(
                f"Segment fully observed{note_str}: {day}/{segment} ({duration}s)"
            )

        # Cleanup segment tracking
        del self.segment_files[segment]
        del self.segment_start_time[segment]
        if segment in self.segment_day:
            del self.segment_day[segment]
        if segment in self.segment_batch:
            del self.segment_batch[segment]
        if segment in self.segment_remote:
            del self.segment_remote[segment]
        if segment in self.segment_errors:
            del self.segment_errors[segment]

    def _check_segment_observed(self, file_path: Path, error: str | None = None):
        """Check if all files for this segment have completed processing.

        Args:
            file_path: Path to the file that finished processing
            error: Optional error string if the handler failed
        """
        from observe.utils import get_segment_key

        segment = get_segment_key(file_path)
        if not segment:
            return

        with self.lock:
            if segment in self.segment_files:
                if error:
                    if segment not in self.segment_errors:
                        self.segment_errors[segment] = []
                    self.segment_errors[segment].append(error)

                self.segment_files[segment].discard(file_path)

                # If no more pending files, emit observed event
                if not self.segment_files[segment]:
                    self._emit_segment_observed(segment)

    def _handle_file(
        self,
        file_path: Path,
        segment: Optional[str] = None,
        remote: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """Route file to appropriate handler.

        Args:
            file_path: Path to the file to process
            segment: Optional segment key for tracking
            remote: Optional remote name for REMOTE_NAME env var
            meta: Optional metadata dict for SEGMENT_META env var
        """
        if not file_path.exists():
            logger.warning(f"File not found, skipping: {file_path}")
            return

        handler_info = self._match_pattern(file_path)
        if handler_info:
            handler_name, command = handler_info
            self._spawn_handler(
                file_path,
                handler_name,
                command,
                segment=segment,
                remote=remote,
                meta=meta,
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
        meta = message.get("meta")  # Optional: metadata dict (facet, setting, etc.)

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
            self._handle_file(file_path, segment=segment, remote=remote, meta=meta)

        # If no files matched any handler patterns, emit observed immediately
        # (e.g., tmux-only segments with just .jsonl files)
        with self.lock:
            if segment in self.segment_files and not self.segment_files[segment]:
                self._emit_segment_observed(segment, note="no handlers")

    def _emit_status(self):
        """Emit observe.status event with current processing state (only when active)."""
        if not self.callosum:
            return

        with self.lock:
            # Check if there's any activity to report
            has_queued = any(q.queue_size() > 0 for q in self.handler_queues.values())
            if not self.running and not has_queued:
                return  # Nothing active, don't emit

            # Build status object
            status = {}

            # Get journal path for relative paths
            journal_path = get_journal()
            now = time.time()

            # Build status for each serialized handler queue
            for handler_name, handler_queue in self.handler_queues.items():
                handler_status = {}

                # Current running process
                if handler_queue.current_process is not None:
                    handler_proc = handler_queue.current_process
                    try:
                        rel_file = str(handler_proc.file_path.relative_to(journal_path))
                    except ValueError:
                        rel_file = str(handler_proc.file_path)

                    handler_status["running"] = {
                        "file": rel_file,
                        "ref": handler_proc.managed.ref,
                    }

                # Queued items with age
                if handler_queue.queue_size() > 0:
                    queued_list = []
                    for item in handler_queue.queue:
                        try:
                            rel_file = str(item.file_path.relative_to(journal_path))
                        except ValueError:
                            rel_file = str(item.file_path)

                        queued_list.append(
                            {"file": rel_file, "age_seconds": int(now - item.queued_at)}
                        )
                    handler_status["queued"] = queued_list

                # Add section if any activity for this handler
                if handler_status:
                    status[handler_name] = handler_status

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

    def process_day(
        self, day: str, max_jobs: int = 1, segment_filter: Optional[str] = None
    ):
        """Process all matching unprocessed files from a specific day directory.

        Files are in segment directories (HHMMSS_LEN/). A file is considered
        unprocessed if it has no corresponding .jsonl output file.

        Args:
            day: Day in YYYYMMDD format
            max_jobs: Maximum number of concurrent processing jobs
            segment_filter: Optional segment key to filter (HHMMSS_LEN format)
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

            # Apply segment filter if specified
            if segment_filter and segment_dir.name != segment_filter:
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


def delete_outputs(
    day_dir: Path,
    reprocess_type: str,
    segment_filter: Optional[str] = None,
    dry_run: bool = False,
) -> list[Path]:
    """Delete existing output files to force reprocessing.

    Args:
        day_dir: Path to day directory (YYYYMMDD)
        reprocess_type: Type of outputs to delete ("screen", "audio", or "all")
        segment_filter: Optional segment key to filter (HHMMSS_LEN format)
        dry_run: If True, don't delete, just return what would be deleted

    Returns:
        List of paths that were (or would be) deleted
    """
    from think.utils import segment_key

    deleted = []

    if not day_dir.exists():
        return deleted

    for segment in day_dir.iterdir():
        if not segment.is_dir() or not segment_key(segment.name):
            continue

        # Apply segment filter if specified
        if segment_filter and segment.name != segment_filter:
            continue

        for file_path in segment.iterdir():
            if not file_path.is_file() or file_path.suffix != ".jsonl":
                continue

            stem = file_path.stem.lower()

            # Determine if this output matches the reprocess type
            should_delete = False
            if reprocess_type == "all":
                # Delete all outputs that have a corresponding source file
                # Check for video source
                for ext in VIDEO_EXTENSIONS:
                    if (segment / f"{file_path.stem}{ext}").exists():
                        should_delete = True
                        break
                # Check for audio source
                for ext in AUDIO_EXTENSIONS:
                    if (segment / f"{file_path.stem}{ext}").exists():
                        should_delete = True
                        break
            elif reprocess_type == "screen":
                # Screen outputs end with _screen or are just screen
                if stem.endswith("_screen") or stem == "screen":
                    should_delete = True
            elif reprocess_type == "audio":
                # Audio outputs end with _audio or are just audio
                if stem.endswith("_audio") or stem == "audio":
                    should_delete = True

            if should_delete:
                deleted.append(file_path)
                if not dry_run:
                    file_path.unlink()
                    logger.info(f"Deleted: {file_path.relative_to(day_dir.parent)}")

    return deleted


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
                or file_path.suffix.lower() in AUDIO_EXTENSIONS
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
    parser.add_argument(
        "--reprocess",
        type=str,
        choices=["screen", "audio", "all"],
        help="Delete existing outputs and reprocess (requires --day)",
    )
    parser.add_argument(
        "--segment",
        type=str,
        help="Filter to specific segment (HHMMSS_LEN format, requires --day)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted/processed without making changes",
    )
    args = setup_cli(parser)

    journal = Path(get_journal())

    # Validate argument combinations
    if args.reprocess and not args.day:
        parser.error("--reprocess requires --day")
    if args.segment and not args.day:
        parser.error("--segment requires --day")
    if args.dry_run and not args.reprocess:
        parser.error("--dry-run requires --reprocess")

    # Validate segment format if provided
    if args.segment:
        from think.utils import segment_key

        if not segment_key(args.segment):
            parser.error(f"--segment must be HHMMSS_LEN format, got: {args.segment}")

    sensor = FileSensor(journal, verbose=args.verbose, debug=args.debug)

    # Register handlers - match by extension
    # Audio files in segment directories
    for ext in AUDIO_EXTENSIONS:
        sensor.register(f"*{ext}", "transcribe", ["sol", "transcribe", "{file}"])

    # Video files in segment directories
    for ext in VIDEO_EXTENSIONS:
        sensor.register(f"*{ext}", "describe", ["sol", "describe", "{file}"])

    if args.day:
        day_dir = day_path(args.day)

        # Handle reprocess mode
        if args.reprocess:
            deleted = delete_outputs(
                day_dir,
                args.reprocess,
                segment_filter=args.segment,
                dry_run=args.dry_run,
            )

            if args.dry_run:
                if deleted:
                    logger.info(f"Would delete {len(deleted)} output file(s):")
                    for path in deleted:
                        logger.info(f"  {path.relative_to(journal)}")
                else:
                    logger.info("No files to delete")
                return
            else:
                logger.info(f"Deleted {len(deleted)} output file(s)")

        # Batch mode: process specific day
        segment_msg = f" (segment: {args.segment})" if args.segment else ""
        logger.info(
            f"Processing files from day {args.day}{segment_msg} "
            f"with {args.jobs} concurrent jobs"
        )
        sensor.process_day(args.day, max_jobs=args.jobs, segment_filter=args.segment)
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
