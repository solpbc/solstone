from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from desktop_notifier import DesktopNotifier, Urgency

from muse.cortex_client import cortex_request
from think.callosum import CallosumConnection, CallosumServer
from think.facets import get_facets
from think.runner import ManagedProcess as RunnerManagedProcess
from think.utils import get_agents, setup_cli

DEFAULT_THRESHOLD = 60
CHECK_INTERVAL = 30

# Global shutdown flag
shutdown_requested = False

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


# State for scheduled agent execution
_scheduled_state = {
    "pending_groups": [],  # List of (priority, [(persona_id, config, yesterday)])
    "active_files": [],  # List of Path objects for current priority group
    "start_time": 0,  # When current group started
}

# State for task execution
_task_state = {
    "running_tasks": set(),  # Set of refs currently running
    "lock": threading.Lock(),  # Lock for thread-safe task state access
}

# Active task processes (ref -> ManagedProcess)
_active_tasks: dict[str, RunnerManagedProcess] = {}

# Global supervisor callosum connection for event emissions
_supervisor_callosum: CallosumConnection | None = None

# Global reference to managed processes for restart control
_managed_procs: list[ManagedProcess] = []

# Global reference to in-process Callosum server
_callosum_server: CallosumServer | None = None
_callosum_thread: threading.Thread | None = None

# Restart request tracking for SIGKILL enforcement
_restart_requests: dict[str, tuple[float, subprocess.Popen]] = {}


def _get_journal_path() -> Path:
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")
    return Path(journal)


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
    ref = ref if ref else str(int(time.time() * 1000))

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
    """Return a list of stale heartbeat names."""
    now = time.time()
    stale: list[str] = []
    health_dir = _get_journal_path() / "health"
    for name in ("see", "hear"):
        path = health_dir / f"{name}.up"
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            stale.append(name)
            continue
        if now - mtime > threshold:
            stale.append(name)
    return stale


def _get_notifier() -> DesktopNotifier:
    """Get or create the global desktop notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = DesktopNotifier(app_name="Sunstone Supervisor")
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
            title="Sunstone Supervisor",
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


async def run_subprocess_task(name: str, cmd: list[str]) -> bool:
    """Run a subprocess task while mirroring output to a dedicated log.

    Runs the subprocess in a thread to avoid blocking the async event loop.

    Args:
        name: Display name for the task
        cmd: Command and arguments to execute

    Returns:
        True when the subprocess exits successfully.
    """

    def _blocking_run():
        start = time.time()
        try:
            managed = RunnerManagedProcess.spawn(cmd, callosum=_supervisor_callosum)
            return_code = managed.wait()
        finally:
            managed.cleanup()

        duration = int(time.time() - start)
        logging.info(f"{name} finished in {duration} seconds")
        return return_code == 0

    return await asyncio.to_thread(_blocking_run)


async def run_dream() -> bool:
    """Run ``think.dream`` while mirroring output to a dedicated log."""
    return await run_subprocess_task("dream", ["think-dream", "-v"])


def spawn_scheduled_agents() -> None:
    """Prepare scheduled agents grouped by priority for sequential execution."""
    try:
        # Calculate yesterday's date
        yesterday = (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")

        agents = get_agents()

        # Group agents by priority
        priority_groups: dict[int, list[tuple[str, dict]]] = {}
        for persona_id, config in agents.items():
            if config.get("schedule") == "daily":
                priority = config.get("priority", 50)
                priority_groups.setdefault(priority, []).append((persona_id, config))

        # Store sorted groups in state for sequential processing
        _scheduled_state["pending_groups"] = [
            (
                priority,
                [(persona_id, config, yesterday) for persona_id, config in agents_list],
            )
            for priority, agents_list in sorted(priority_groups.items())
        ]
        _scheduled_state["active_files"] = []
        _scheduled_state["start_time"] = 0

        total_agents = sum(
            len(agents_list) for _, agents_list in _scheduled_state["pending_groups"]
        )
        logging.info(
            f"Prepared {len(_scheduled_state['pending_groups'])} priority groups "
            f"with {total_agents} total agents"
        )
    except Exception as e:
        logging.error(f"Failed to prepare scheduled agents: {e}")


def _run_full_rescan() -> None:
    """Run full index rescan in background thread."""
    from think.runner import run_task

    logging.info("Starting full index rescan after daily tasks completed")
    success, exit_code = run_task(
        ["think-indexer", "--rescan-full"], callosum=_supervisor_callosum
    )

    if success:
        logging.info("Full index rescan completed")
    else:
        logging.error(f"Full index rescan failed with exit code {exit_code}")


async def check_scheduled_agents() -> None:
    """Check and advance scheduled agent execution (non-blocking).

    Called from the main supervise loop to incrementally process priority groups.
    Each priority group completes before the next begins. When all groups complete,
    triggers a full index rescan.
    """
    state = _scheduled_state

    # Nothing to do if no pending groups and no active agents
    if not state["pending_groups"] and not state["active_files"]:
        return

    # Check if current priority group is done
    if state["active_files"]:
        all_done = not any(f.exists() for f in state["active_files"])
        timed_out = (
            time.time() - state["start_time"]
        ) > 600  # 10 minute timeout per group

        if all_done:
            logging.info("Priority group completed")
        elif timed_out:
            # List unfinished agents
            unfinished = [
                f.stem.replace("_active", "")
                for f in state["active_files"]
                if f.exists()
            ]
            unfinished_str = ", ".join(unfinished) if unfinished else "none"
            logging.warning(
                f"Priority group timed out after 600s, proceeding to next group. "
                f"Unfinished agents: {unfinished_str}"
            )
        else:
            return  # Still running, check again next iteration

        # Group finished (either completed or timed out)
        state["active_files"] = []
        if not state["pending_groups"]:
            # All daily tasks complete - run full rescan
            threading.Thread(target=_run_full_rescan, daemon=True).start()

    # Check for shutdown before starting next group
    if shutdown_requested:
        state["pending_groups"] = []
        state["active_files"] = []
        return

    # Spawn next priority group
    if state["pending_groups"]:
        priority, agents_list = state["pending_groups"].pop(0)
        logging.info(f"Starting priority {priority} agents ({len(agents_list)} agents)")

        # Get agents directory for tracking active files
        agents_dir = _get_journal_path() / "agents"

        active_files = []
        for persona_id, config, yesterday in agents_list:
            try:
                # Check if this is a multi-facet agent
                if config.get("multi_facet"):
                    facets = get_facets()
                    # Filter out muted facets for automated runs
                    enabled_facets = {
                        k: v for k, v in facets.items() if not v.get("muted", False)
                    }
                    muted_count = len(facets) - len(enabled_facets)
                    if muted_count > 0:
                        muted_names = [
                            k for k, v in facets.items() if v.get("muted", False)
                        ]
                        logging.info(
                            f"Skipping {muted_count} muted facet(s) for {persona_id}: "
                            f"{', '.join(muted_names)}"
                        )
                    for facet_name in enabled_facets.keys():
                        logging.info(f"Spawning {persona_id} for facet: {facet_name}")
                        agent_id = cortex_request(
                            prompt=f"You are processing facet '{facet_name}' for yesterday ({yesterday}), use get_facet('{facet_name}') to load the correct context before starting.",
                            persona=persona_id,
                        )
                        active_files.append(agents_dir / f"{agent_id}_active.jsonl")
                        logging.info(
                            f"Started {persona_id} for {facet_name} (ID: {agent_id})"
                        )
                else:
                    # Regular single-instance agent
                    agent_id = cortex_request(
                        prompt=f"Running daily scheduled task for {persona_id}, yesterday was {yesterday}.",
                        persona=persona_id,
                    )
                    active_files.append(agents_dir / f"{agent_id}_active.jsonl")
                    logging.info(f"Started {persona_id} agent (ID: {agent_id})")
            except Exception as e:
                logging.error(f"Failed to spawn {persona_id}: {e}")

        state["active_files"] = active_files
        state["start_time"] = time.time()


def _handle_task_request(message: dict) -> None:
    """Handle incoming task request from Callosum."""
    # Filter for supervisor tract and request event
    if message.get("tract") != "supervisor" or message.get("event") != "request":
        return

    ref = message.get("ref")
    cmd = message.get("cmd")

    if not cmd:
        logging.error(f"Invalid task request: missing cmd: {message}")
        return

    # Generate ref if not provided
    ref = ref if ref else str(int(time.time() * 1000))

    # Check if task is already running
    with _task_state["lock"]:
        if ref in _task_state["running_tasks"]:
            logging.debug(f"Task {ref} already running, skipping duplicate")
            return
        _task_state["running_tasks"].add(ref)

    # Spawn task in background thread
    threading.Thread(
        target=_run_task,
        args=(ref, cmd),
        daemon=True,
    ).start()


def _run_task(ref: str, cmd: list[str]) -> None:
    """Execute a task and broadcast events to Callosum."""
    callosum = CallosumConnection()
    managed = None

    # Extract service name from command
    service = Path(cmd[0]).name if cmd else "unknown"

    try:
        # Start Callosum connection (auto-connects in background)
        callosum.start()

        logging.info(f"Starting task {ref}: {' '.join(cmd)}")

        # Spawn process and track it (share callosum for logs events)
        managed = RunnerManagedProcess.spawn(cmd, ref=ref, callosum=callosum)
        _active_tasks[ref] = managed

        # Emit started event (runner already emits logs/exec, this is supervisor-level)
        callosum.emit(
            "supervisor", "started", service=service, pid=managed.pid, ref=ref
        )

        # Wait for completion (blocks)
        exit_code = managed.wait()

        # Emit stopped event
        callosum.emit(
            "supervisor",
            "stopped",
            service=service,
            pid=managed.pid,
            ref=ref,
            exit_code=exit_code,
        )

        if exit_code == 0:
            logging.info(f"Task {ref} finished successfully")
        else:
            logging.warning(f"Task {ref} failed with exit code {exit_code}")

    except Exception as e:
        logging.exception(f"Task {ref} encountered exception: {e}")
        # Still emit stopped event with error code
        if managed:
            callosum.emit(
                "supervisor",
                "stopped",
                service=service,
                pid=managed.pid if managed else 0,
                ref=ref,
                exit_code=-1,
            )
    finally:
        # Cleanup managed process
        if managed:
            managed.cleanup()
        _active_tasks.pop(ref, None)

        # Remove from running tasks
        with _task_state["lock"]:
            _task_state["running_tasks"].discard(ref)

        callosum.stop()


def _handle_supervisor_request(message: dict) -> None:
    """Handle incoming supervisor control messages."""
    if message.get("tract") != "supervisor" or message.get("event") != "restart":
        return

    service = message.get("service")
    if not service:
        logging.error("Invalid restart request: missing service")
        return

    # Find and signal the process
    for proc in _managed_procs:
        if proc.name == service and proc.process.poll() is None:
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

    logging.warning(f"Cannot restart {service}: not found or not running")


def cancel_task(ref: str) -> bool:
    """Cancel a running task.

    Args:
        ref: Task correlation ID

    Returns:
        True if task was found and terminated, False otherwise
    """
    if ref not in _active_tasks:
        logging.warning(f"Cannot cancel task {ref}: not found")
        return False

    managed = _active_tasks[ref]
    if not managed.is_running():
        logging.debug(f"Task {ref} already finished")
        return False

    logging.info(f"Cancelling task {ref}...")
    managed.terminate()
    return True


def get_task_status(ref: str) -> dict:
    """Get status of a task.

    Args:
        ref: Task correlation ID

    Returns:
        Dict with status info, or {"status": "not_found"} if task doesn't exist
    """
    if ref not in _active_tasks:
        return {"status": "not_found"}

    managed = _active_tasks[ref]
    return {
        "status": "running" if managed.is_running() else "finished",
        "pid": managed.pid,
        "returncode": managed.returncode,
        "log_path": str(managed.log_writer.path),
        "cmd": managed.cmd,
    }


def list_running_tasks() -> list[str]:
    """List all currently running task correlation IDs.

    Returns:
        List of refs for tasks that are still running
    """
    return [ref for ref, managed in _active_tasks.items() if managed.is_running()]


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
    tasks = []
    for ref, managed in _active_tasks.items():
        if managed.is_running():
            duration = int(now - managed._start_time)
            # Extract command name (first element of cmd)
            cmd_name = managed.cmd[0] if managed.cmd else "unknown"
            tasks.append(
                {
                    "ref": ref,
                    "name": cmd_name,
                    "duration_seconds": duration,
                }
            )

    # Stale heartbeats
    stale = check_health()

    return {
        "services": services,
        "crashed": crashed,
        "tasks": tasks,
        "stale_heartbeats": stale,
    }


def start_observers() -> list[ManagedProcess]:
    """Launch observe-gnome and observe-sense with output logging."""
    procs: list[ManagedProcess] = []
    commands = {
        "observer": ["observe-gnome", "-v"],
        "sense": ["observe-sense", "-v"],
    }
    for name, cmd in commands.items():
        procs.append(_launch_process(name, cmd, restart=True))
    return procs


def start_callosum_in_process() -> CallosumServer:
    """Start Callosum message bus server in-process.

    Runs the server in a background thread and waits for socket to be ready.

    Returns:
        CallosumServer instance
    """
    global _callosum_server, _callosum_thread

    server = CallosumServer()
    _callosum_server = server

    # Start server in background thread (server.start() is blocking)
    thread = threading.Thread(target=server.start, daemon=False, name="callosum-server")
    thread.start()
    _callosum_thread = thread

    # Wait for socket to be ready (with timeout)
    socket_path = server.socket_path
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
    cmd = ["muse-cortex", "-v"]
    return _launch_process("cortex", cmd, restart=True)


def start_convey_server(verbose: bool, debug: bool = False) -> ManagedProcess:
    """Launch the Convey web application with optional verbose and debug logging."""

    cmd = ["convey"]
    if debug:
        cmd.append("-d")
    elif verbose:
        cmd.append("-v")
    return _launch_process("convey", cmd, restart=True)


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


async def handle_daily_tasks(last_day: datetime.date) -> datetime.date:
    """Run daily processing (dream + scheduled agents). Returns new last_day."""
    today = datetime.now().date()
    if today != last_day:
        if await run_dream():
            spawn_scheduled_agents()
        return today
    return last_day


def _handle_segment_observed(message: dict) -> None:
    """Handle segment completion events from observe tract."""
    if message.get("tract") != "observe" or message.get("event") != "observed":
        return

    segment = message.get("segment")  # e.g., "163045_300"
    if not segment:
        logging.warning("observed event missing segment field")
        return

    # Extract day from current date (segment observed on same day)
    day = datetime.now().strftime("%Y%m%d")

    logging.info(f"Segment observed: {day}/{segment}, spawning processing...")

    # Spawn agents configured for segment schedule
    agents = get_agents()
    for persona_id, config in agents.items():
        if config.get("schedule") == "segment":
            try:
                cortex_request(
                    prompt=f"Processing segment {segment} from {day}. Use available tools to analyze this specific recording window.",
                    persona=persona_id,
                )
                logging.info(f"Spawned segment agent: {persona_id}")
            except Exception as e:
                logging.error(f"Failed to spawn {persona_id}: {e}")

    # Run dream in segment mode (async, non-blocking)
    threading.Thread(
        target=_run_segment_dream,
        args=(day, segment),
        daemon=True,
    ).start()


def _run_segment_dream(day: str, segment: str) -> None:
    """Run think-dream for a specific segment."""
    from think.runner import run_task

    logging.info(f"Starting segment dream: {day}/{segment}")
    success, exit_code = run_task(
        ["think-dream", "-v", "--day", day, "--segment", segment],
        callosum=_supervisor_callosum,
    )

    if success:
        logging.info(f"Segment dream completed: {day}/{segment}")
    else:
        logging.error(
            f"Segment dream failed with exit code {exit_code}: {day}/{segment}"
        )


def _handle_callosum_message(message: dict) -> None:
    """Dispatch incoming Callosum messages to appropriate handlers."""
    _handle_task_request(message)
    _handle_supervisor_request(message)
    _handle_segment_observed(message)


async def supervise(
    *,
    threshold: int = DEFAULT_THRESHOLD,
    interval: int = CHECK_INTERVAL,
    daily: bool = True,
    procs: list[ManagedProcess] | None = None,
) -> None:
    """Monitor heartbeat files and alert when they become stale.

    Main supervision loop runs at 1-second intervals for responsiveness.
    Subsystems manage their own timing (health checks every interval seconds,
    scheduled agents check continuously but only advance when ready).
    """
    alert_mgr = AlertManager()
    last_day = datetime.now().date()
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

            # Check for daily processing
            if daily:
                last_day = await handle_daily_tasks(last_day)

            # Advance scheduled agent execution (non-blocking)
            await check_scheduled_agents()

            # Sleep 1 second before next iteration (responsive to shutdown)
            await asyncio.sleep(1)
    finally:
        pass  # Callosum cleanup happens in main()


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor journaling health")
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
        help="Do not automatically start observe-gnome and observe-sense",
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
    args = setup_cli(parser)
    try:
        journal_path = _get_journal_path()
    except RuntimeError:
        parser.error("JOURNAL_PATH not set")
        return

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_path = journal_path / "health" / "supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=log_level,
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        format="%(asctime)s %(levelname)s %(message)s",
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

    logging.info("Supervisor starting...")

    global _managed_procs, _supervisor_callosum
    procs: list[ManagedProcess] = []

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

    # Now start other services (their startup events will be captured)
    if not args.no_observers:
        procs.extend(start_observers())
    if not args.no_cortex:
        procs.append(start_cortex_server())
    if not args.no_convey:
        procs.append(start_convey_server(verbose=args.verbose, debug=args.debug))

    # Make procs accessible to restart handler
    _managed_procs = procs

    logging.info(f"Started {len(procs)} processes, entering supervision loop")

    try:
        asyncio.run(
            supervise(
                threshold=args.threshold,
                interval=args.interval,
                daily=not args.no_daily,
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
