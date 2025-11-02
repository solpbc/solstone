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
from think.callosum import CallosumConnection
from think.domains import get_domains
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

    async def alert_if_ready(
        self, key: tuple, message: str, command: str = "notify-send"
    ) -> bool:
        """Send alert with exponential backoff. Returns True if sent."""
        now = time.time()

        if key in self._state:
            last_time, backoff = self._state[key]
            if now - last_time >= backoff:
                await send_notification(message, command, alert_key=key)
                new_backoff = min(backoff * 2, self._max_backoff)
                self._state[key] = (now, new_backoff)
                logging.info(f"Alert sent, next backoff: {new_backoff}s")
                return True
            else:
                remaining = int(backoff - (now - last_time))
                logging.info(f"Suppressing alert, next in {remaining}s")
                return False
        else:
            await send_notification(message, command, alert_key=key)
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
    "rescan_pending": False,  # Whether domain rescan needs to run
}

# State for task execution
_task_state = {
    "running_tasks": set(),  # Set of task_ids currently running
    "lock": threading.Lock(),  # Lock for thread-safe task state access
}

# Active task processes (task_id -> ManagedProcess)
_active_tasks: dict[str, RunnerManagedProcess] = {}


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
    process_id: str = ""

    def cleanup(self) -> None:
        for thread in self.threads:
            thread.join(timeout=1)
        self.logger.close()


def _launch_process(
    name: str,
    cmd: list[str],
    *,
    restart: bool = False,
    log_name: str | None = None,
) -> ManagedProcess:
    """Launch process with automatic output logging and restart policy tracking."""
    policy: RestartPolicy | None = None
    if restart:
        policy = _get_restart_policy(name)

    # Use unified runner to spawn process
    try:
        managed = RunnerManagedProcess.spawn(cmd, name=log_name or name)
    except RuntimeError as exc:
        logging.error(str(exc))
        raise

    if policy:
        policy.record_start()

    # Wrap in ManagedProcess for restart tracking
    return ManagedProcess(
        process=managed.process,
        name=name,
        logger=managed.log_writer,
        cmd=list(cmd),
        restart=restart,
        threads=managed._threads,
        process_id=managed.process_id,
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


async def send_notification(
    message: str, command: str = "notify-send", alert_key: tuple | None = None
) -> None:
    """Send a desktop notification with ``message``.

    Args:
        message: The notification message to display
        command: Legacy parameter for backwards compatibility (ignored)
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


def run_subprocess_task(name: str, cmd: list[str], log_name: str | None = None) -> bool:
    """Run a subprocess task while mirroring output to a dedicated log.

    Args:
        name: Display name for the task
        cmd: Command and arguments to execute
        log_name: Optional log filename (defaults to name)

    Returns:
        True when the subprocess exits successfully.
    """
    start = time.time()
    try:
        managed = RunnerManagedProcess.spawn(cmd, name=log_name or name)
        return_code = managed.wait()
    finally:
        managed.cleanup()

    duration = int(time.time() - start)
    logging.info(f"{name} finished in {duration} seconds")
    return return_code == 0


def run_dream() -> bool:
    """Run ``think.dream`` while mirroring output to a dedicated log."""
    return run_subprocess_task("dream", ["think-dream", "-v"])


def run_domain_rescan() -> bool:
    """Run ``think-indexer --rescan-domains`` while mirroring output to a dedicated log."""
    return run_subprocess_task(
        "domain_rescan", ["think-indexer", "--rescan-domains"], log_name="domain_rescan"
    )


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
        _scheduled_state["rescan_pending"] = True

        total_agents = sum(
            len(agents_list) for _, agents_list in _scheduled_state["pending_groups"]
        )
        logging.info(
            f"Prepared {len(_scheduled_state['pending_groups'])} priority groups "
            f"with {total_agents} total agents"
        )
    except Exception as e:
        logging.error(f"Failed to prepare scheduled agents: {e}")


def check_scheduled_agents() -> None:
    """Check and advance scheduled agent execution (non-blocking).

    Called from the main supervise loop to incrementally process priority groups.
    Each priority group completes before the next begins.
    """
    state = _scheduled_state

    # Nothing to do if no pending groups and no active agents
    if not state["pending_groups"] and not state["active_files"]:
        # Check if domain rescan is pending
        if state["rescan_pending"]:
            logging.info("All agent groups completed, running domain rescan...")
            state["rescan_pending"] = False
            if run_domain_rescan():
                logging.info("Domain rescan completed successfully")
            else:
                logging.warning("Domain rescan failed or exited with error")
        return

    # Check if current priority group is done
    if state["active_files"]:
        all_done = not any(f.exists() for f in state["active_files"])
        timed_out = (
            time.time() - state["start_time"]
        ) > 600  # 10 minute timeout per group

        if all_done:
            logging.info("Priority group completed")
            state["active_files"] = []
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
            state["active_files"] = []
        else:
            return  # Still running, check again next iteration

    # Check for shutdown before starting next group
    if shutdown_requested:
        state["pending_groups"] = []
        state["active_files"] = []
        state["rescan_pending"] = False
        return

    # Spawn next priority group
    if state["pending_groups"]:
        priority, agents_list = state["pending_groups"].pop(0)
        logging.info(f"Starting priority {priority} agents ({len(agents_list)} agents)")

        active_files = []
        for persona_id, config, yesterday in agents_list:
            try:
                # Check if this is a multi-domain agent
                if config.get("multi_domain"):
                    domains = get_domains()
                    # Filter out disabled domains for automated runs
                    enabled_domains = {
                        k: v for k, v in domains.items() if not v.get("disabled", False)
                    }
                    disabled_count = len(domains) - len(enabled_domains)
                    if disabled_count > 0:
                        disabled_names = [
                            k for k, v in domains.items() if v.get("disabled", False)
                        ]
                        logging.info(
                            f"Skipping {disabled_count} disabled domain(s) for {persona_id}: "
                            f"{', '.join(disabled_names)}"
                        )
                    for domain_name in enabled_domains.keys():
                        logging.info(f"Spawning {persona_id} for domain: {domain_name}")
                        request_file = cortex_request(
                            prompt=f"You are processing domain '{domain_name}' for yesterday ({yesterday}), use get_domain('{domain_name}') to load the correct context before starting.",
                            persona=persona_id,
                        )
                        active_files.append(Path(request_file))
                        agent_id = Path(request_file).stem.replace("_active", "")
                        logging.info(
                            f"Started {persona_id} for {domain_name} (ID: {agent_id})"
                        )
                else:
                    # Regular single-instance agent
                    request_file = cortex_request(
                        prompt=f"Running daily scheduled task for {persona_id}, yesterday was {yesterday}.",
                        persona=persona_id,
                    )
                    active_files.append(Path(request_file))
                    agent_id = Path(request_file).stem.replace("_active", "")
                    logging.info(f"Started {persona_id} agent (ID: {agent_id})")
            except Exception as e:
                logging.error(f"Failed to spawn {persona_id}: {e}")

        state["active_files"] = active_files
        state["start_time"] = time.time()


def _handle_task_request(message: dict) -> None:
    """Handle incoming task request from Callosum."""
    # Filter for task tract and request event
    if message.get("tract") != "task" or message.get("event") != "request":
        return

    task_id = message.get("task_id")
    cmd = message.get("cmd")

    if not task_id or not cmd:
        logging.error(f"Invalid task request: missing task_id or cmd: {message}")
        return

    # Check if task is already running
    with _task_state["lock"]:
        if task_id in _task_state["running_tasks"]:
            logging.debug(f"Task {task_id} already running, skipping duplicate")
            return
        _task_state["running_tasks"].add(task_id)

    # Spawn task in background thread
    threading.Thread(
        target=_run_task,
        args=(task_id, cmd),
        daemon=True,
    ).start()


def _run_task(task_id: str, cmd: list[str]) -> None:
    """Execute a task and broadcast events to Callosum."""
    callosum = CallosumConnection()
    managed = None

    try:
        # Start Callosum connection (auto-connects in background)
        callosum.start()

        # Emit start event
        callosum.emit("task", "start", task_id=task_id, cmd=cmd)
        logging.info(f"Starting task {task_id}: {' '.join(cmd)}")

        # Spawn process and track it
        managed = RunnerManagedProcess.spawn(cmd, log_name=task_id)
        _active_tasks[task_id] = managed

        # Wait for completion (blocks)
        exit_code = managed.wait()
        success = exit_code == 0

        # Emit finish or error event
        if success:
            callosum.emit("task", "finish", task_id=task_id, exit_code=exit_code)
            logging.info(f"Task {task_id} finished successfully")
        else:
            callosum.emit(
                "task",
                "error",
                task_id=task_id,
                error=f"Task exited with code {exit_code}",
                exit_code=exit_code,
            )
            logging.warning(f"Task {task_id} failed with exit code {exit_code}")

    except Exception as e:
        logging.exception(f"Task {task_id} encountered exception: {e}")
        callosum.emit(
            "task",
            "error",
            task_id=task_id,
            error=str(e),
            exit_code=-1,
        )
    finally:
        # Cleanup managed process
        if managed:
            managed.cleanup()
        _active_tasks.pop(task_id, None)

        # Remove from running tasks
        with _task_state["lock"]:
            _task_state["running_tasks"].discard(task_id)

        callosum.stop()


def cancel_task(task_id: str) -> bool:
    """Cancel a running task.

    Args:
        task_id: Task identifier

    Returns:
        True if task was found and terminated, False otherwise
    """
    if task_id not in _active_tasks:
        logging.warning(f"Cannot cancel task {task_id}: not found")
        return False

    managed = _active_tasks[task_id]
    if not managed.is_running():
        logging.debug(f"Task {task_id} already finished")
        return False

    logging.info(f"Cancelling task {task_id}...")
    managed.terminate()
    return True


def get_task_status(task_id: str) -> dict:
    """Get status of a task.

    Args:
        task_id: Task identifier

    Returns:
        Dict with status info, or {"status": "not_found"} if task doesn't exist
    """
    if task_id not in _active_tasks:
        return {"status": "not_found"}

    managed = _active_tasks[task_id]
    return {
        "status": "running" if managed.is_running() else "finished",
        "pid": managed.pid,
        "returncode": managed.returncode,
        "log_path": str(managed.log_writer.path),
        "cmd": managed.cmd,
    }


def list_running_tasks() -> list[str]:
    """List all currently running task IDs.

    Returns:
        List of task_ids for tasks that are still running
    """
    return [
        task_id for task_id, managed in _active_tasks.items() if managed.is_running()
    ]


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
                    "process": proc.process_id,
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
    for task_id, managed in _active_tasks.items():
        if managed.is_running():
            duration = int(now - managed._start_time)
            # Extract command name (first element of cmd)
            cmd_name = managed.cmd[0] if managed.cmd else "unknown"
            tasks.append(
                {
                    "task_id": task_id,
                    "name": cmd_name,
                    "process": managed.process_id,
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


async def emit_periodic_status(procs: list[ManagedProcess]) -> None:
    """Emit task/status events every 5 seconds."""
    callosum = CallosumConnection()
    callosum.start()

    while not shutdown_requested:
        try:
            # Collect and emit status
            status = collect_status(procs)
            callosum.emit("task", "status", **status)

        except Exception as e:
            logging.debug(f"Status emission failed: {e}")

        await asyncio.sleep(5)

    callosum.stop()


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


def start_callosum_server() -> ManagedProcess:
    """Launch the Callosum message bus server."""
    cmd = ["think-callosum", "-v"]
    return _launch_process("callosum", cmd, restart=True)


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
    command: str,
) -> None:
    """Check for and handle exited processes with restart policy."""
    exited = check_runner_exits(procs)
    if not exited:
        return

    exited_names = [managed.name for managed in exited]
    msg = f"Runner process exited: {', '.join(sorted(exited_names))}"
    logging.error(msg)
    exit_key = ("runner_exit", tuple(sorted(exited_names)))

    await alert_mgr.alert_if_ready(exit_key, msg, command)

    for managed in exited:
        returncode = managed.process.returncode
        logging.info("%s exited with code %s", managed.name, returncode)

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
                    time.sleep(1)
            if shutdown_requested:
                continue
            logging.info("Restarting %s...", managed.name)
            try:
                new_proc = _launch_process(
                    managed.name,
                    managed.cmd,
                    restart=True,
                    log_name=managed.logger.path.stem,
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
    command: str,
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
        await alert_mgr.alert_if_ready(stale_key, msg, command)

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
        if run_dream():
            spawn_scheduled_agents()
        return today
    return last_day


async def supervise(
    *,
    threshold: int = DEFAULT_THRESHOLD,
    interval: int = CHECK_INTERVAL,
    command: str = "notify-send",
    daily: bool = True,
    procs: list[ManagedProcess] | None = None,
) -> None:
    """Monitor heartbeat files and alert when they become stale.

    Main supervision loop runs at 1-second intervals for responsiveness.
    Subsystems manage their own timing (health checks every interval seconds,
    scheduled agents check continuously but only advance when ready).
    """
    global shutdown_requested
    alert_mgr = AlertManager()
    last_day = datetime.now().date()
    last_health_check = 0.0
    prev_stale: set[str] = set()

    # Connect to Callosum to receive task requests
    callosum = None
    try:
        callosum = CallosumConnection()
        callosum.start(callback=_handle_task_request)
        logging.info("Supervisor connected to Callosum for task requests")
    except Exception as e:
        logging.warning(f"Failed to start Callosum connection: {e}")

    try:
        while (
            not shutdown_requested
        ):  # pragma: no cover - loop checked via unit tests by patching
            # Check for runner exits first (immediate alert)
            if procs:
                await handle_runner_exits(procs, alert_mgr, command)

            # Check health periodically (interval-based timing)
            last_health_check, prev_stale = await handle_health_checks(
                last_health_check, interval, threshold, alert_mgr, command, prev_stale
            )

            # Check for daily processing
            if daily:
                last_day = await handle_daily_tasks(last_day)

            # Advance scheduled agent execution (non-blocking)
            check_scheduled_agents()

            # Sleep 1 second before next iteration (responsive to shutdown)
            await asyncio.sleep(1)
    finally:
        # Clean up Callosum connection
        if callosum:
            callosum.stop()
            logging.info("Supervisor disconnected from Callosum")


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
        "--notify-cmd",
        default="notify-send",
        help="Command used to send desktop notification",
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

    procs: list[ManagedProcess] = []
    # Start Callosum first - it's the message bus that other services depend on
    procs.append(start_callosum_server())
    if not args.no_observers:
        procs.extend(start_observers())
    if not args.no_cortex:
        procs.append(start_cortex_server())
    if not args.no_convey:
        procs.append(start_convey_server(verbose=args.verbose, debug=args.debug))

    logging.info(f"Started {len(procs)} processes, entering supervision loop")
    try:

        async def run_supervisor():
            """Run supervision loop."""
            tasks = [
                supervise(
                    threshold=args.threshold,
                    interval=args.interval,
                    command=args.notify_cmd,
                    daily=not args.no_daily,
                    procs=procs if procs else None,
                ),
                emit_periodic_status(procs),
            ]

            await asyncio.gather(*tasks)

        asyncio.run(run_supervisor())
    except KeyboardInterrupt:
        logging.info("Caught KeyboardInterrupt, shutting down...")
    finally:
        logging.info("Stopping all processes...")
        print(
            "\nShutting down gracefully (this may take up to 15 seconds)...", flush=True
        )
        for managed in procs:
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
                print(f" timeout, forcing kill...", flush=True)
                try:
                    proc.kill()
                    proc.wait(timeout=1)
                except Exception:
                    pass
            managed.cleanup()
        logging.info("Supervisor shutdown complete.")
        print("Shutdown complete.", flush=True)


if __name__ == "__main__":
    main()
