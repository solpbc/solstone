# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from desktop_notifier import DesktopNotifier, Urgency

from observe.sync import check_remote_health
from think.callosum import CallosumConnection, CallosumServer
from think.runner import ManagedProcess as RunnerManagedProcess
from think.utils import (
    find_available_port,
    get_journal,
    get_journal_info,
    now_ms,
    setup_cli,
)

DEFAULT_THRESHOLD = 60
CHECK_INTERVAL = 30

# Global shutdown flag
shutdown_requested = False
# Supervisor identity (set in main() once ref is assigned)
_supervisor_ref: str | None = None
_supervisor_start: float | None = None


class CallosumLogHandler(logging.Handler):
    """Logging handler that emits log records as callosum ``logs`` tract events.

    Silently drops events on any error — callosum mirroring is best-effort.
    """

    def __init__(self, conn: CallosumConnection, ref: str):
        super().__init__()
        self._conn = conn
        self._ref = ref
        self._pid = os.getpid()
        self._emitting = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._emitting:
            return
        self._emitting = True
        try:
            self._conn.emit(
                "logs",
                "line",
                ref=self._ref,
                name="supervisor",
                pid=self._pid,
                stream="log",
                line=self.format(record),
            )
        except Exception:
            pass
        finally:
            self._emitting = False


# Desktop notification system
_notifier: DesktopNotifier | None = None
_notification_ids: dict[tuple, str] = {}  # Maps alert_key -> notification_id


class AlertManager:
    """Manages alerts with exponential backoff and notification clearing."""

    def __init__(self, initial_backoff: int = 60, max_backoff: int = 3600):
        self._state: dict[tuple, tuple[float, int]] = {}  # {key: (last_time, backoff)}
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff

    async def alert_if_ready(self, key: tuple, message: str) -> bool:
        """Send alert with exponential backoff. Returns True if sent."""
        now = time.time()

        if key in self._state:
            last_time, backoff = self._state[key]
            if now - last_time >= backoff:
                await send_notification(message, alert_key=key)
                new_backoff = min(backoff * 2, self._max_backoff)
                self._state[key] = (now, new_backoff)
                logging.info(f"Alert sent, next backoff: {new_backoff}s")
                return True
            else:
                remaining = int(backoff - (now - last_time))
                logging.info(f"Suppressing alert, next in {remaining}s")
                return False
        else:
            await send_notification(message, alert_key=key)
            self._state[key] = (now, self._initial_backoff)
            return True

    async def clear(self, key: tuple) -> None:
        """Clear alert state and notification."""
        if key in self._state:
            del self._state[key]
        await clear_notification(key)

    def clear_matching(self, predicate) -> None:
        """Clear alert states matching predicate."""
        self._state = {k: v for k, v in self._state.items() if not predicate(k, v)}


class TaskQueue:
    """Manages on-demand task execution with per-command serialization.

    Tasks are serialized by command name - only one task per command runs at a time.
    Additional requests for the same command are queued (deduped by exact cmd match).
    Multiple callers requesting the same work have their refs coalesced so all get
    notified when the task completes.

    The lock only protects state mutations, never held during I/O operations.
    """

    def __init__(self, on_queue_change: callable = None):
        """Initialize task queue.

        Args:
            on_queue_change: Optional callback(cmd_name, running_ref, queue_entries)
                            called after queue state changes. Called outside lock.
        """
        self._running: dict[str, str] = {}  # command_name -> ref of running task
        self._queues: dict[str, list] = {}  # command_name -> list of {refs, cmd} dicts
        self._active: dict[str, RunnerManagedProcess] = {}  # ref -> process
        self._lock = threading.Lock()
        self._on_queue_change = on_queue_change

    @staticmethod
    def get_command_name(cmd: list[str]) -> str:
        """Extract command name from cmd array for queue serialization.

        For 'sol X' commands, returns X. Otherwise returns cmd[0] basename.
        """
        if cmd and cmd[0] == "sol" and len(cmd) > 1:
            return cmd[1]
        return Path(cmd[0]).name if cmd else "unknown"

    def _notify_queue_change(self, cmd_name: str) -> None:
        """Notify listener of queue state change (called outside lock)."""
        if not self._on_queue_change:
            return

        with self._lock:
            queue = list(self._queues.get(cmd_name, []))
            running_ref = self._running.get(cmd_name)

        self._on_queue_change(cmd_name, running_ref, queue)

    def submit(
        self,
        cmd: list[str],
        ref: str | None = None,
        callosum: CallosumConnection = None,
    ) -> str | None:
        """Submit a task for execution.

        If no task of this command type is running, starts immediately.
        Otherwise queues (deduped by exact cmd match, refs coalesced).

        Args:
            cmd: Command to execute
            ref: Optional caller-provided ref for tracking
            callosum: CallosumConnection for event emission

        Returns:
            ref if task was started/queued, None if already tracked (no change)
        """
        ref = ref or str(now_ms())
        cmd_name = self.get_command_name(cmd)

        should_notify = False
        should_start = False

        with self._lock:
            if cmd_name in self._running:
                # Command already running - queue or coalesce
                queue = self._queues.setdefault(cmd_name, [])
                existing = next((q for q in queue if q["cmd"] == cmd), None)
                if existing:
                    if ref not in existing["refs"]:
                        existing["refs"].append(ref)
                        logging.info(
                            f"Added ref {ref} to queued task {cmd_name} "
                            f"(refs: {len(existing['refs'])})"
                        )
                        should_notify = True
                    else:
                        logging.debug(f"Ref already tracked for queued task: {ref}")
                        return None
                else:
                    queue.append({"refs": [ref], "cmd": cmd})
                    logging.info(
                        f"Queued task {cmd_name}: {' '.join(cmd)} ref={ref} "
                        f"(queue: {len(queue)})"
                    )
                    should_notify = True
            else:
                # Not running - mark as running and start
                self._running[cmd_name] = ref
                should_start = True

        # Notify outside lock
        if should_notify:
            self._notify_queue_change(cmd_name)
            return ref

        # Start task outside lock
        if should_start:
            threading.Thread(
                target=self._run_task,
                args=([ref], cmd, cmd_name, callosum),
                daemon=True,
            ).start()
            return ref

        return None

    def _run_task(
        self,
        refs: list[str],
        cmd: list[str],
        cmd_name: str,
        parent_callosum: CallosumConnection = None,
    ) -> None:
        """Execute a task and handle completion.

        Args:
            refs: List of refs to notify on completion
            cmd: Command to execute
            cmd_name: Command name for queue management
            parent_callosum: Optional parent callosum (not used, task creates own)
        """
        callosum = CallosumConnection()
        managed = None
        primary_ref = refs[0]
        service = cmd_name

        try:
            callosum.start()
            logging.info(f"Starting task {primary_ref}: {' '.join(cmd)}")

            managed = RunnerManagedProcess.spawn(
                cmd, ref=primary_ref, callosum=callosum
            )
            self._active[primary_ref] = managed

            callosum.emit(
                "supervisor",
                "started",
                service=service,
                pid=managed.pid,
                ref=primary_ref,
            )

            exit_code = managed.wait()

            for ref in refs:
                callosum.emit(
                    "supervisor",
                    "stopped",
                    service=service,
                    pid=managed.pid,
                    ref=ref,
                    exit_code=exit_code,
                )

            if exit_code == 0:
                logging.info(f"Task {primary_ref} finished successfully")
            else:
                logging.warning(f"Task {primary_ref} failed with exit code {exit_code}")

        except Exception as e:
            logging.exception(f"Task {primary_ref} encountered exception: {e}")
            for ref in refs:
                callosum.emit(
                    "supervisor",
                    "stopped",
                    service=service,
                    pid=managed.pid if managed else 0,
                    ref=ref,
                    exit_code=-1,
                )
        finally:
            if managed:
                managed.cleanup()
            self._active.pop(primary_ref, None)
            callosum.stop()
            self._process_next(cmd_name)

    def _process_next(self, cmd_name: str) -> None:
        """Process next queued task after completion."""
        next_cmd = None
        refs = None

        with self._lock:
            queue = self._queues.get(cmd_name, [])
            if queue:
                entry = queue.pop(0)
                refs = entry["refs"]
                next_cmd = entry["cmd"]
                self._running[cmd_name] = refs[0]
                logging.info(
                    f"Dequeued task {cmd_name}: {' '.join(next_cmd)} refs={refs} "
                    f"(remaining: {len(queue)})"
                )
            else:
                self._running.pop(cmd_name, None)

        # Notify and spawn outside lock
        self._notify_queue_change(cmd_name)
        if next_cmd:
            threading.Thread(
                target=self._run_task,
                args=(refs, next_cmd, cmd_name, None),
                daemon=True,
            ).start()

    def cancel(self, ref: str) -> bool:
        """Cancel a running task.

        Returns:
            True if task was found and terminated, False otherwise
        """
        if ref not in self._active:
            logging.warning(f"Cannot cancel task {ref}: not found")
            return False

        managed = self._active[ref]
        if not managed.is_running():
            logging.debug(f"Task {ref} already finished")
            return False

        logging.info(f"Cancelling task {ref}...")
        managed.terminate()
        return True

    def get_status(self, ref: str) -> dict:
        """Get status of a task."""
        if ref not in self._active:
            return {"status": "not_found"}

        managed = self._active[ref]
        return {
            "status": "running" if managed.is_running() else "finished",
            "pid": managed.pid,
            "returncode": managed.returncode,
            "log_path": str(managed.log_writer.path),
            "cmd": managed.cmd,
        }

    def collect_task_status(self) -> list[dict]:
        """Collect status of all running tasks for supervisor status."""
        now = time.time()
        tasks = []
        for ref, managed in self._active.items():
            if managed.is_running():
                duration = int(now - managed._start_time)
                cmd_name = TaskQueue.get_command_name(managed.cmd)
                tasks.append(
                    {
                        "ref": ref,
                        "name": cmd_name,
                        "duration_seconds": duration,
                    }
                )
        return tasks


# Global task queue instance (initialized in main())
_task_queue: TaskQueue | None = None

# Global supervisor callosum connection for event emissions
_supervisor_callosum: CallosumConnection | None = None

# Global reference to managed processes for restart control
_managed_procs: list[ManagedProcess] = []

# Global reference to in-process Callosum server
_callosum_server: CallosumServer | None = None
_callosum_thread: threading.Thread | None = None

# Restart request tracking for SIGKILL enforcement
_restart_requests: dict[str, tuple[float, subprocess.Popen]] = {}

# Observe status state for health monitoring (updated from observe.status events)
# Health is now simple: if observer is running, it sends status events.
# If it has problems, it exits and gets restarted (fail-fast model).
_observe_status_state: dict = {
    "last_ts": 0.0,  # Timestamp of last observe.status event
    "ever_received": False,  # Whether we've received at least one status event
}

# Track whether observer was started (for health check conditioning)
_observer_enabled: bool = True

# Track whether running in remote mode (upload-only, no local processing)
_is_remote_mode: bool = False

# State for daily processing (dream runs in background, agents wait for completion)
_daily_state = {
    "dream_running": False,  # True while dream subprocess is active
    "dream_completed": False,  # True after dream finishes (reset each day)
    "last_day": None,  # Track which day we last processed
    "start_time": 0,  # When daily processing started (for duration tracking)
}

# Timeout before flushing stale segments (seconds)
FLUSH_TIMEOUT = 3600

# State for segment flush (close out dangling agent state after inactivity)
_flush_state: dict = {
    "last_segment_ts": 0.0,  # Wall-clock time of last observe.observed event
    "day": None,  # Day of last observed segment
    "segment": None,  # Last observed segment key
    "flushed": False,  # Whether flush has already run for current segment
}


def _get_journal_path() -> Path:
    return Path(get_journal())


class RestartPolicy:
    """Track restart attempts and compute backoff delays."""

    _SCHEDULE = (0, 1, 5)

    def __init__(self) -> None:
        self.attempts = 0
        self.last_start = 0.0

    def record_start(self) -> None:
        self.last_start = time.time()

    def reset_attempts(self) -> None:
        self.attempts = 0

    def next_delay(self) -> int:
        delay = self._SCHEDULE[min(self.attempts, len(self._SCHEDULE) - 1)]
        self.attempts += 1
        return delay


_RESTART_POLICIES: dict[str, RestartPolicy] = {}


def _get_restart_policy(name: str) -> RestartPolicy:
    return _RESTART_POLICIES.setdefault(name, RestartPolicy())


@dataclass
class ManagedProcess:
    """Wrapper around RunnerManagedProcess for restart policy tracking."""

    process: subprocess.Popen
    name: str
    logger: RunnerManagedProcess  # Actually stores the runner's log_writer
    cmd: list[str]
    restart: bool = False
    threads: list[threading.Thread] = field(default_factory=list)
    ref: str = ""

    def cleanup(self) -> None:
        for thread in self.threads:
            thread.join(timeout=1)
        self.logger.close()


def _launch_process(
    name: str,
    cmd: list[str],
    *,
    restart: bool = False,
    ref: str | None = None,
) -> ManagedProcess:
    # NOTE: All child processes should include -v for verbose logging by default.
    # This ensures their output is captured in logs for debugging.
    """Launch process with automatic output logging and restart policy tracking."""
    policy: RestartPolicy | None = None
    if restart:
        policy = _get_restart_policy(name)

    # Generate ref if not provided
    ref = ref if ref else str(now_ms())

    # Use unified runner to spawn process (share supervisor's callosum)
    try:
        managed = RunnerManagedProcess.spawn(
            cmd, ref=ref, callosum=_supervisor_callosum
        )
    except RuntimeError as exc:
        logging.error(str(exc))
        raise

    if policy:
        policy.record_start()

    # Emit started event
    if _supervisor_callosum:
        _supervisor_callosum.emit(
            "supervisor",
            "started",
            service=name,
            pid=managed.process.pid,
            ref=managed.ref,
        )

    # Wrap in ManagedProcess for restart tracking
    return ManagedProcess(
        process=managed.process,
        name=name,
        logger=managed.log_writer,
        cmd=list(cmd),
        restart=restart,
        threads=managed._threads,
        ref=managed.ref,
    )


def check_health(threshold: int = DEFAULT_THRESHOLD) -> list[str]:
    """Return a list of stale heartbeat names based on observe.status events.

    Health model is simple: if observer is running, it sends status events.
    If it has problems, it exits and supervisor restarts it (fail-fast).

    Returns ["hear", "see"] if no status received within threshold,
    empty list otherwise. During startup grace period (before first
    status event received), returns empty list to avoid false alerts.

    When observer is disabled (--no-observers), always returns empty list
    since there's no local capture to monitor.
    """
    # Skip health checks if observer was not started
    if not _observer_enabled:
        return []

    # Grace period: don't alert until we've received at least one status event
    if not _observe_status_state["ever_received"]:
        return []

    now = time.time()
    last_ts = _observe_status_state["last_ts"]

    # If no recent status, observer is not running - both stale
    if now - last_ts > threshold:
        return ["hear", "see"]

    # Receiving status means observer is healthy
    return []


def _get_notifier() -> DesktopNotifier:
    """Get or create the global desktop notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = DesktopNotifier(app_name="solstone Supervisor")
    return _notifier


async def send_notification(message: str, alert_key: tuple | None = None) -> None:
    """Send a desktop notification with ``message``.

    Args:
        message: The notification message to display
        alert_key: Optional key to track this notification for later clearing
    """
    try:
        notifier = _get_notifier()
        notification_id = await notifier.send(
            title="solstone Supervisor",
            message=message,
            urgency=Urgency.Critical,
        )

        # Store notification ID if we have an alert key
        if alert_key and notification_id:
            _notification_ids[alert_key] = notification_id
            logging.debug(f"Stored notification {notification_id} for key {alert_key}")

    except Exception as exc:  # pragma: no cover - system issues
        logging.error("Failed to send notification: %s", exc)


async def clear_notification(alert_key: tuple) -> None:
    """Clear a notification by its alert key.

    Args:
        alert_key: The key used when the notification was sent
    """
    if alert_key not in _notification_ids:
        return

    try:
        notifier = _get_notifier()
        notification_id = _notification_ids[alert_key]
        await notifier.clear(notification_id)
        del _notification_ids[alert_key]
        logging.debug(f"Cleared notification for key {alert_key}")

    except Exception as exc:  # pragma: no cover - system issues
        logging.error("Failed to clear notification: %s", exc)


def _get_command_name(cmd: list[str]) -> str:
    """Extract command name from cmd array for queue serialization.

    For 'sol X' commands, returns X. Otherwise returns cmd[0] basename.
    """
    return TaskQueue.get_command_name(cmd)


def _emit_queue_event(cmd_name: str, running_ref: str, queue: list) -> None:
    """Emit supervisor.queue event with current queue state for a command.

    This is the callback passed to TaskQueue for queue change notifications.
    """
    if not _supervisor_callosum:
        return

    _supervisor_callosum.emit(
        "supervisor",
        "queue",
        command=cmd_name,
        running=running_ref,
        queued=len(queue),
        queue=queue,
    )


def _handle_task_request(message: dict) -> None:
    """Handle incoming task request from Callosum."""
    if message.get("tract") != "supervisor" or message.get("event") != "request":
        return

    cmd = message.get("cmd")
    if not cmd:
        logging.error(f"Invalid task request: missing cmd: {message}")
        return

    ref = message.get("ref")
    if _task_queue:
        _task_queue.submit(cmd, ref)


def _handle_supervisor_request(message: dict) -> None:
    """Handle incoming supervisor control messages."""
    if message.get("tract") != "supervisor" or message.get("event") != "restart":
        return

    service = message.get("service")
    if not service:
        logging.error("Invalid restart request: missing service")
        return
    if service == "supervisor":
        logging.debug("Ignoring restart request for supervisor itself")
        return

    # Find the process
    for proc in _managed_procs:
        if proc.name == service:
            # Check if process is still running
            if proc.process.poll() is not None:
                # Already exited - ignore, supervision loop will auto-restart
                logging.debug(
                    f"Ignoring restart for {service}: already exited, awaiting auto-restart"
                )
                return

            logging.info(f"Restart requested for {service}, sending SIGINT...")

            # Emit restarting event
            if _supervisor_callosum:
                _supervisor_callosum.emit(
                    "supervisor",
                    "restarting",
                    service=service,
                    pid=proc.process.pid,
                    ref=proc.ref,
                )

            # Send SIGINT to trigger graceful shutdown
            try:
                proc.process.send_signal(signal.SIGINT)
                # Track restart request for SIGKILL enforcement
                _restart_requests[service] = (time.time(), proc.process)
            except Exception as e:
                logging.error(f"Failed to send SIGINT to {service}: {e}")
            return

    logging.warning(f"Cannot restart {service}: not found in managed processes")


def get_task_status(ref: str) -> dict:
    """Get status of a task.

    Args:
        ref: Task correlation ID

    Returns:
        Dict with status info, or {"status": "not_found"} if task doesn't exist
    """
    if _task_queue:
        return _task_queue.get_status(ref)
    return {"status": "not_found"}


def collect_status(procs: list[ManagedProcess]) -> dict:
    """Collect current supervisor status for broadcasting."""
    now = time.time()

    # Running services
    services = []
    running_names = set()
    for proc in procs:
        if proc.process.poll() is None:  # Still running
            policy = _get_restart_policy(proc.name)
            uptime = int(now - policy.last_start) if policy.last_start else 0
            services.append(
                {
                    "name": proc.name,
                    "ref": proc.ref,
                    "pid": proc.process.pid,
                    "uptime_seconds": uptime,
                }
            )
            running_names.add(proc.name)

    # Prepend supervisor itself
    if _supervisor_ref and _supervisor_start:
        services.insert(
            0,
            {
                "name": "supervisor",
                "ref": _supervisor_ref,
                "pid": os.getpid(),
                "uptime_seconds": int(now - _supervisor_start),
            },
        )

    # Crashed services (in restart backoff)
    crashed = []
    for name, policy in _RESTART_POLICIES.items():
        if name not in running_names and policy.attempts > 0:
            crashed.append(
                {
                    "name": name,
                    "restart_attempts": policy.attempts,
                }
            )

    # Running tasks
    tasks = _task_queue.collect_task_status() if _task_queue else []

    # Stale heartbeats
    stale = check_health()

    return {
        "services": services,
        "crashed": crashed,
        "tasks": tasks,
        "stale_heartbeats": stale,
    }


def start_observer() -> ManagedProcess:
    """Launch platform-detected observer with output logging."""
    return _launch_process("observer", ["sol", "observer", "-v"], restart=True)


def start_sense() -> ManagedProcess:
    """Launch sol sense with output logging."""
    return _launch_process("sense", ["sol", "sense", "-v"], restart=True)


def start_sync(remote_url: str) -> ManagedProcess:
    """Launch sol sync with output logging.

    Args:
        remote_url: Remote ingest URL for sync service
    """
    return _launch_process(
        "sync", ["sol", "sync", "-v", "--remote", remote_url], restart=True
    )


def start_callosum_in_process() -> CallosumServer:
    """Start Callosum message bus server in-process.

    Runs the server in a background thread and waits for socket to be ready.

    Returns:
        CallosumServer instance
    """
    global _callosum_server, _callosum_thread

    server = CallosumServer()
    _callosum_server = server

    # Pre-delete stale socket to avoid race condition where the ready check
    # passes due to an old socket file before the server thread deletes it
    socket_path = server.socket_path
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    # Start server in background thread (server.start() is blocking)
    thread = threading.Thread(target=server.start, daemon=False, name="callosum-server")
    thread.start()
    _callosum_thread = thread

    # Wait for socket to be ready (with timeout)
    for _ in range(50):  # Wait up to 500ms
        if socket_path.exists():
            logging.info(f"Callosum server started on {socket_path}")
            return server
        time.sleep(0.01)

    raise RuntimeError("Callosum server failed to create socket within 500ms")


def stop_callosum_in_process() -> None:
    """Stop the in-process Callosum server."""
    global _callosum_server, _callosum_thread

    if _callosum_server:
        logging.info("Stopping Callosum server...")
        _callosum_server.stop()

    if _callosum_thread:
        _callosum_thread.join(timeout=5)
        if _callosum_thread.is_alive():
            logging.warning("Callosum server thread did not stop cleanly")

    _callosum_server = None
    _callosum_thread = None


def start_cortex_server() -> ManagedProcess:
    """Launch the Cortex WebSocket API server."""
    cmd = ["sol", "cortex", "-v"]
    return _launch_process("cortex", cmd, restart=True)


def start_convey_server(
    verbose: bool, debug: bool = False, port: int = 0
) -> tuple[ManagedProcess, int]:
    """Launch the Convey web application with optional verbose and debug logging.

    Returns:
        Tuple of (ManagedProcess, resolved_port) where resolved_port is the
        actual port being used (auto-selected if port was 0).
    """
    # Resolve port 0 to an available port before launching
    resolved_port = port if port != 0 else find_available_port()

    cmd = ["sol", "convey", "--port", str(resolved_port)]
    if debug:
        cmd.append("-d")
    elif verbose:
        cmd.append("-v")
    return _launch_process("convey", cmd, restart=True), resolved_port


def check_runner_exits(procs: list[ManagedProcess]) -> list[ManagedProcess]:
    """Return managed processes that have exited."""

    exited: list[ManagedProcess] = []
    for managed in procs:
        if managed.process.poll() is not None:
            exited.append(managed)
    return exited


async def handle_runner_exits(
    procs: list[ManagedProcess],
    alert_mgr: AlertManager,
) -> None:
    """Check for and handle exited processes with restart policy."""
    exited = check_runner_exits(procs)
    if not exited:
        return

    exited_names = [managed.name for managed in exited]
    msg = f"Runner process exited: {', '.join(sorted(exited_names))}"
    logging.error(msg)
    exit_key = ("runner_exit", tuple(sorted(exited_names)))

    await alert_mgr.alert_if_ready(exit_key, msg)

    for managed in exited:
        # Clear any pending restart request for this service
        _restart_requests.pop(managed.name, None)

        returncode = managed.process.returncode
        logging.info("%s exited with code %s", managed.name, returncode)

        # Emit stopped event
        if _supervisor_callosum:
            _supervisor_callosum.emit(
                "supervisor",
                "stopped",
                service=managed.name,
                pid=managed.process.pid,
                ref=managed.ref,
                exit_code=returncode,
            )

        # Remove from procs list
        try:
            index = procs.index(managed)
        except ValueError:
            index = None

        if index is not None:
            procs.pop(index)
        else:
            try:
                procs.remove(managed)
            except ValueError:
                pass

        managed.cleanup()

        # Handle restart if needed
        if managed.restart and not shutdown_requested:
            policy = _get_restart_policy(managed.name)
            uptime = time.time() - policy.last_start if policy.last_start else 0
            if uptime >= 60:
                policy.reset_attempts()
            delay = policy.next_delay()
            if delay:
                logging.info("Waiting %ss before restarting %s", delay, managed.name)
                for _ in range(delay):
                    if shutdown_requested:
                        break
                    await asyncio.sleep(1)
            if shutdown_requested:
                continue
            logging.info("Restarting %s...", managed.name)
            try:
                new_proc = _launch_process(
                    managed.name,
                    managed.cmd,
                    restart=True,
                )
            except Exception as exc:
                logging.exception("Failed to restart %s: %s", managed.name, exc)
                continue

            insert_at = index if index is not None else len(procs)
            procs.insert(insert_at, new_proc)
            logging.info("Restarted %s after exit code %s", managed.name, returncode)
            # Clear the notification now that process has restarted
            await alert_mgr.clear(exit_key)
        else:
            logging.info("Not restarting %s", managed.name)


async def handle_health_checks(
    last_check: float,
    interval: int,
    threshold: int,
    alert_mgr: AlertManager,
    prev_stale: set[str],
) -> tuple[float, set[str]]:
    """Perform periodic health checks. Returns (new_last_check, new_prev_stale)."""
    now = time.time()
    if now - last_check < interval:
        return last_check, prev_stale

    stale = check_health(threshold)
    stale_set = set(stale)

    recovered = sorted(prev_stale - stale_set)
    for name in recovered:
        logging.info("%s heartbeat recovered", name)
        # Clear notifications for recovered heartbeats
        stale_key = ("stale", tuple(sorted(prev_stale)))
        await alert_mgr.clear(stale_key)

    if stale_set:
        msg = f"Journaling offline: {', '.join(sorted(stale_set))}"
        logging.warning(msg)

        stale_key = ("stale", tuple(sorted(stale_set)))

        # Clear any previous stale notifications with different keys
        for key in list(_notification_ids.keys()):
            if key[0] == "stale" and key != stale_key:
                await clear_notification(key)

        await alert_mgr.alert_if_ready(stale_key, msg)

        # Retain only alert state entries still relevant
        alert_mgr.clear_matching(
            lambda k, v: k[0] == "stale" and not set(k[1]).issubset(stale_set)
        )
    else:
        if prev_stale:
            logging.info("Heartbeat OK")
        # Clear alert state for stale services when they recover
        alert_mgr.clear_matching(lambda k, v: k[0] == "stale")

    return now, stale_set


def _run_daily_processing(day: str) -> None:
    """Run complete daily processing via sol dream.

    dream now handles both generators and agent execution, so we just
    invoke it with --force and let it manage the full pipeline.

    Args:
        day: Target day in YYYYMMDD format
    """
    from think.runner import run_task

    logging.info(f"Starting daily processing for {day}...")
    success, exit_code, log_path = run_task(
        ["sol", "dream", "-v", "--day", day, "--force"],
        callosum=_supervisor_callosum,
    )

    # Update state on completion
    _daily_state["dream_running"] = False

    if success:
        logging.info(f"Daily processing completed for {day}")
        _daily_state["dream_completed"] = True
    else:
        logging.error(
            f"Daily processing failed for {day} with exit code {exit_code}, "
            f"see {log_path}"
        )


def handle_daily_tasks() -> None:
    """Check for day change and spawn daily dream if needed (non-blocking).

    Dream only triggers when the day actually changes during runtime (at midnight).
    The supervisor initializes last_day on startup, so restarts don't trigger dream.
    Scheduled agents are spawned after dream completes successfully.

    Skipped in remote mode (no local data to process).
    """
    # Remote mode: no local processing, data is on the server
    if _is_remote_mode:
        return

    today = datetime.now().date()

    # Only trigger when day actually changes (at midnight)
    if today != _daily_state["last_day"]:
        # The day that just ended is what we process
        prev_day = _daily_state["last_day"]

        # Guard against None (e.g., module reloaded without going through main())
        if prev_day is None:
            logging.warning("Daily state not initialized, skipping daily processing")
            _daily_state["last_day"] = today
            return

        prev_day_str = prev_day.strftime("%Y%m%d")

        # Update state for new day
        _daily_state["last_day"] = today
        _daily_state["dream_completed"] = False

        # Don't start new dream if one is already running (edge case)
        if _daily_state["dream_running"]:
            logging.warning(
                f"Day changed to {today} but dream already running, skipping {prev_day_str}"
            )
            return

        # Flush any dangling segment state from the previous day before daily dream
        if not _flush_state["flushed"] and _flush_state["day"] == prev_day_str:
            _check_segment_flush(force=True)

        logging.info(
            f"Day changed to {today}, starting daily processing for {prev_day_str}"
        )

        # Spawn processing in background thread with target day
        _daily_state["dream_running"] = True
        _daily_state["start_time"] = time.time()
        threading.Thread(
            target=_run_daily_processing, args=(prev_day_str,), daemon=True
        ).start()


def _handle_segment_observed(message: dict) -> None:
    """Handle segment completion events (from live observation or imports).

    Spawns sol dream in segment mode, which handles both generators and
    segment agents. Also updates flush state to track segment recency.
    """
    if message.get("tract") != "observe" or message.get("event") != "observed":
        return

    segment = message.get("segment")  # e.g., "163045_300"
    if not segment:
        logging.warning("observed event missing segment field")
        return

    # Use day from event payload, fallback to today (for live observation)
    day = message.get("day") or datetime.now().strftime("%Y%m%d")
    stream = message.get("stream")

    # Update flush state — new segment resets the flush timer
    _flush_state["last_segment_ts"] = time.time()
    _flush_state["day"] = day
    _flush_state["segment"] = segment
    _flush_state["stream"] = stream
    _flush_state["flushed"] = False

    logging.info(f"Segment observed: {day}/{segment}, spawning processing...")

    # Run dream in segment mode (handles both generators and agents)
    threading.Thread(
        target=_run_segment_processing,
        args=(day, segment, stream),
        daemon=True,
    ).start()


def _run_segment_processing(day: str, segment: str, stream: str | None = None) -> None:
    """Run sol dream for a specific segment."""
    from think.runner import run_task

    logging.info(f"Starting segment processing: {day}/{segment}")
    cmd = ["sol", "dream", "-v", "--day", day, "--segment", segment]
    if stream:
        cmd.extend(["--stream", stream])
    success, exit_code, log_path = run_task(
        cmd,
        callosum=_supervisor_callosum,
    )

    if success:
        logging.info(f"Segment processing completed: {day}/{segment}")
    else:
        logging.error(
            f"Segment processing failed with exit code {exit_code}: "
            f"{day}/{segment}, see {log_path}"
        )


def _check_segment_flush(force: bool = False) -> None:
    """Check if the last observed segment needs flushing.

    If no new segments have arrived within FLUSH_TIMEOUT seconds, runs
    ``sol dream --flush`` on the last segment to let flush-enabled agents
    close out dangling state (e.g., end active activities).

    Args:
        force: Skip timeout check (used at day boundary to flush
               before daily dream regardless of elapsed time).

    Skipped in remote mode (no local processing).
    """
    if _is_remote_mode:
        return

    last_ts = _flush_state["last_segment_ts"]
    if not last_ts or _flush_state["flushed"]:
        return

    if not force and time.time() - last_ts < FLUSH_TIMEOUT:
        return

    day = _flush_state["day"]
    segment = _flush_state["segment"]
    if not day or not segment:
        return

    _flush_state["flushed"] = True

    stream = _flush_state.get("stream")
    cmd = ["sol", "dream", "-v", "--day", day, "--segment", segment, "--flush"]
    if stream:
        cmd.extend(["--stream", stream])
    if _task_queue:
        _task_queue.submit(cmd)
        logging.info(f"Queued segment flush: {day}/{segment}")
    else:
        logging.warning(
            "No task queue available for segment flush: %s/%s", day, segment
        )


def _handle_observe_status(message: dict) -> None:
    """Handle observe.status events for health monitoring.

    Just tracks that we received a status event. The observer is responsible
    for exiting if it's unhealthy (fail-fast model), so receiving status
    means it's working.
    """
    if message.get("tract") != "observe" or message.get("event") != "status":
        return

    _observe_status_state["last_ts"] = time.time()
    _observe_status_state["ever_received"] = True


def _handle_segment_event_log(message: dict) -> None:
    """Log observe, dream, and activity events with day+segment to segment/events.jsonl.

    Any observe, dream, or activity tract message with both day and segment fields
    gets logged to JOURNAL_PATH/day/segment/events.jsonl if that directory exists.
    """
    if message.get("tract") not in {"observe", "dream", "activity"}:
        return

    day = message.get("day")
    segment = message.get("segment")

    if not day or not segment:
        return

    stream = message.get("stream")

    try:
        journal_path = _get_journal_path()

        if stream:
            segment_dir = journal_path / day / stream / segment
        else:
            segment_dir = journal_path / day / segment

        # Only log if segment directory exists
        if not segment_dir.is_dir():
            return

        events_file = segment_dir / "events.jsonl"

        # Append event as JSON line
        with open(events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    except Exception as e:
        logging.debug(f"Failed to log segment event: {e}")


def _handle_activity_recorded(message: dict) -> None:
    """Queue a per-activity dream task when an activity is recorded.

    Listens for activity.recorded events and submits a queued dream task
    for per-activity agent processing (serialized via TaskQueue).
    """
    if message.get("tract") != "activity" or message.get("event") != "recorded":
        return

    record_id = message.get("id")
    facet = message.get("facet")
    day = message.get("day")

    if not record_id or not facet or not day:
        logging.warning("activity.recorded event missing required fields")
        return

    cmd = ["sol", "dream", "--activity", record_id, "--facet", facet, "--day", day]

    if _task_queue:
        _task_queue.submit(cmd)
        logging.info(f"Queued activity dream: {record_id} for #{facet}")
    else:
        logging.warning("No task queue available for activity dream: %s", record_id)


def _handle_callosum_message(message: dict) -> None:
    """Dispatch incoming Callosum messages to appropriate handlers."""
    _handle_task_request(message)
    _handle_supervisor_request(message)
    _handle_segment_observed(message)
    _handle_observe_status(message)
    _handle_activity_recorded(message)
    _handle_segment_event_log(message)


async def supervise(
    *,
    threshold: int = DEFAULT_THRESHOLD,
    interval: int = CHECK_INTERVAL,
    daily: bool = True,
    procs: list[ManagedProcess] | None = None,
) -> None:
    """Monitor health via Callosum events and alert when stale.

    Health is derived from observe.status events (see check_health()).
    Main supervision loop runs at 1-second intervals for responsiveness.
    Subsystems manage their own timing (health checks every interval seconds,
    scheduled agents check continuously but only advance when ready).
    """
    alert_mgr = AlertManager()
    last_health_check = 0.0
    last_status_emit = 0.0
    prev_stale: set[str] = set()

    try:
        while (
            not shutdown_requested
        ):  # pragma: no cover - loop checked via unit tests by patching
            # Check for restart timeouts (enforce SIGKILL after 15s)
            for service, (start_time, proc) in list(_restart_requests.items()):
                if proc.poll() is not None:  # Already exited
                    _restart_requests.pop(service, None)
                elif time.time() - start_time > 15:
                    logging.warning(
                        f"{service} did not exit within 15s after SIGINT, sending SIGKILL"
                    )
                    try:
                        proc.kill()
                    except Exception as e:
                        logging.error(f"Failed to kill {service}: {e}")
                    # Don't delete here - let handle_runner_exits clean up

            # Check for runner exits first (immediate alert)
            if procs:
                await handle_runner_exits(procs, alert_mgr)

            # Check health periodically (interval-based timing)
            last_health_check, prev_stale = await handle_health_checks(
                last_health_check, interval, threshold, alert_mgr, prev_stale
            )

            # Emit status every 5 seconds
            now = time.time()
            if now - last_status_emit >= 5:
                if _supervisor_callosum and procs:
                    try:
                        status = collect_status(procs)
                        _supervisor_callosum.emit("supervisor", "status", **status)
                    except Exception as e:
                        logging.debug(f"Status emission failed: {e}")
                last_status_emit = now

            # Check for segment flush (non-blocking, submits via task queue)
            _check_segment_flush()

            # Check for daily processing (non-blocking, spawns dream in background)
            if daily:
                handle_daily_tasks()

            # Sleep 1 second before next iteration (responsive to shutdown)
            await asyncio.sleep(1)
    finally:
        pass  # Callosum cleanup happens in main()


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor journaling health")
    parser.add_argument(
        "port",
        nargs="?",
        type=int,
        default=0,
        help="Convey port (0 = auto-select available port)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help="Seconds before heartbeat considered stale",
    )
    parser.add_argument(
        "--interval", type=int, default=CHECK_INTERVAL, help="Polling interval seconds"
    )
    parser.add_argument(
        "--no-observers",
        action="store_true",
        help="Do not start local observer (sense still runs for remote/imports)",
    )
    parser.add_argument(
        "--no-daily",
        action="store_true",
        help="Disable daily processing run at midnight",
    )
    parser.add_argument(
        "--no-cortex",
        action="store_true",
        help="Do not start the Cortex server (run it manually for debugging)",
    )
    parser.add_argument(
        "--no-convey",
        action="store_true",
        help="Do not start the Convey web application",
    )
    parser.add_argument(
        "--remote",
        type=str,
        help="Remote mode: sync to server URL instead of local processing",
    )
    return parser


def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    if not shutdown_requested:  # Only log once
        shutdown_requested = True
        logging.info("Shutdown requested, cleaning up...")
    raise KeyboardInterrupt


def main() -> None:
    parser = parse_args()

    # Capture journal info BEFORE setup_cli() loads .env and pollutes os.environ
    journal_info = get_journal_info()

    args = setup_cli(parser)

    journal_path = _get_journal_path()

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_path = journal_path / "health" / "supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=log_level,
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        format="%(asctime)s [supervisor:log] %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    if args.verbose or args.debug:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logging.getLogger().addHandler(console_handler)

    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Show journal path and source on startup
    path, source = journal_info
    print(f"Journal: {path} (from {source})")
    logging.info("Supervisor starting...")

    global _managed_procs, _supervisor_callosum, _observer_enabled, _is_remote_mode
    global _task_queue
    procs: list[ManagedProcess] = []

    # Remote mode: run sync instead of local sense/observer
    _is_remote_mode = bool(args.remote)
    _observer_enabled = not args.no_observers

    # Start Callosum in-process first - it's the message bus that other services depend on
    try:
        start_callosum_in_process()
    except RuntimeError as e:
        logging.error(f"Failed to start Callosum server: {e}")
        parser.error(f"Failed to start Callosum server: {e}")
        return

    # Connect supervisor's Callosum client to capture startup events from other services
    try:
        _supervisor_callosum = CallosumConnection()
        _supervisor_callosum.start(callback=_handle_callosum_message)
        logging.info("Supervisor connected to Callosum")
    except Exception as e:
        logging.warning(f"Failed to start Callosum connection: {e}")

    # Mirror supervisor log output to callosum logs tract (best-effort)
    supervisor_ref = str(now_ms())
    global _supervisor_ref, _supervisor_start
    _supervisor_ref = supervisor_ref
    _supervisor_start = time.time()
    if _supervisor_callosum:
        try:
            handler = CallosumLogHandler(_supervisor_callosum, supervisor_ref)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            logging.getLogger().addHandler(handler)
        except Exception:
            pass

    # Initialize task queue with callosum event callback
    _task_queue = TaskQueue(on_queue_change=_emit_queue_event)

    # Now start other services (their startup events will be captured)
    if _is_remote_mode:
        # Remote mode: verify remote server is reachable before starting sync
        logging.info("Remote mode: checking server connectivity...")
        success, message = check_remote_health(args.remote)
        if not success:
            logging.error(f"Remote health check failed: {message}")
            stop_callosum_in_process()
            parser.error(f"Remote server not available: {message}")
        logging.info(f"Remote server verified: {message}")
        procs.append(start_sync(args.remote))
        # Observer runs unless disabled
        if not args.no_observers:
            procs.append(start_observer())
    else:
        # Local mode: sense handles file processing
        procs.append(start_sense())
        # Observer only runs if not disabled (local capture)
        if not args.no_observers:
            procs.append(start_observer())
    # Cortex and Convey only run in local mode (remote has no data to serve)
    convey_port = None
    if not _is_remote_mode and not args.no_cortex:
        procs.append(start_cortex_server())
    if not _is_remote_mode and not args.no_convey:
        proc, convey_port = start_convey_server(
            verbose=args.verbose, debug=args.debug, port=args.port
        )
        procs.append(proc)

    # Make procs accessible to restart handler
    _managed_procs = procs

    # Initialize daily state to today - dream only triggers at midnight when day changes
    _daily_state["last_day"] = datetime.now().date()

    # Show Convey URL if running
    if convey_port:
        print(f"Convey: http://localhost:{convey_port}/")

    logging.info(f"Started {len(procs)} processes, entering supervision loop")
    daily_enabled = not args.no_daily and not _is_remote_mode
    if daily_enabled:
        logging.info("Daily processing scheduled for midnight")

    try:
        asyncio.run(
            supervise(
                threshold=args.threshold,
                interval=args.interval,
                daily=daily_enabled,
                procs=procs if procs else None,
            )
        )
    except KeyboardInterrupt:
        logging.info("Caught KeyboardInterrupt, shutting down...")
    finally:
        logging.info("Stopping all processes...")
        print(
            "\nShutting down gracefully (this may take up to 15 seconds)...", flush=True
        )
        # Shut down managed processes in reverse order to respect dependencies
        for managed in reversed(procs):
            name = managed.name
            proc = managed.process
            logging.info(f"Stopping {name}...")
            print(f"  Stopping {name}...", end="", flush=True)
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=15)
                print(" done", flush=True)
            except subprocess.TimeoutExpired:
                logging.warning(f"{name} did not terminate gracefully, killing...")
                print(" timeout, forcing kill...", flush=True)
                try:
                    proc.kill()
                    proc.wait(timeout=1)
                except Exception:
                    pass
            managed.cleanup()

        # Disconnect supervisor's Callosum connection
        if _supervisor_callosum:
            _supervisor_callosum.stop()
            logging.info("Supervisor disconnected from Callosum")

        # Stop in-process Callosum server last
        stop_callosum_in_process()

        logging.info("Supervisor shutdown complete.")
        print("Shutdown complete.", flush=True)


if __name__ == "__main__":
    main()
