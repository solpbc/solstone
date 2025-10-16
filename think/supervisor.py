import argparse
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO

from muse.cortex_client import cortex_request
from think.domains import get_domains
from think.utils import get_agents, setup_cli

DEFAULT_THRESHOLD = 30
CHECK_INTERVAL = 30

# Global shutdown flag
shutdown_requested = False

# State for scheduled agent execution
_scheduled_state = {
    "pending_groups": [],  # List of (priority, [(persona_id, config, yesterday)])
    "active_files": [],  # List of Path objects for current priority group
    "start_time": 0,  # When current group started
    "rescan_pending": False,  # Whether domain rescan needs to run
}


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


class ProcessLogWriter:
    """Thread-safe writer that appends process output to a log file."""

    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = log_path
        self._lock = threading.Lock()
        self._fh = log_path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        with self._lock:
            self._fh.write(message)
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    @property
    def path(self) -> Path:
        return self._log_path


@dataclass
class ManagedProcess:
    process: subprocess.Popen
    name: str
    logger: ProcessLogWriter
    cmd: list[str]
    restart: bool = False
    threads: list[threading.Thread] = field(default_factory=list)

    def cleanup(self) -> None:
        for thread in self.threads:
            thread.join(timeout=1)
        self.logger.close()


def _format_log_line(process_name: str, stream_label: str, line: str) -> str:
    timestamp = datetime.now().isoformat(timespec="seconds")
    clean_line = line.rstrip("\n")
    return f"{timestamp} [{process_name}:{stream_label}] {clean_line}\n"


def _stream_output(
    pipe: TextIO | None,
    process_name: str,
    stream_label: str,
    logger: ProcessLogWriter,
) -> None:
    if pipe is None:
        return
    with pipe:
        for line in pipe:
            logger.write(_format_log_line(process_name, stream_label, line))


def _launch_process(
    name: str,
    cmd: list[str],
    *,
    restart: bool = False,
    log_name: str | None = None,
) -> ManagedProcess:
    journal_path = _get_journal_path()
    log_writer = ProcessLogWriter(journal_path / "health" / f"{(log_name or name)}.log")
    policy: RestartPolicy | None = None
    if restart:
        policy = _get_restart_policy(name)

    logging.info(f"Starting {name}: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except Exception as exc:
        logging.error(f"Failed to start {name}: {exc}")
        log_writer.close()
        raise
    logging.info(f"Started {name} with PID {proc.pid}")
    threads = [
        threading.Thread(
            target=_stream_output,
            args=(proc.stdout, name, "stdout", log_writer),
            daemon=True,
        ),
        threading.Thread(
            target=_stream_output,
            args=(proc.stderr, name, "stderr", log_writer),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    if policy:
        policy.record_start()
    return ManagedProcess(
        process=proc,
        name=name,
        logger=log_writer,
        cmd=list(cmd),
        restart=restart,
        threads=threads,
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


def send_notification(message: str, command: str = "notify-send") -> None:
    """Send a desktop notification with ``message`` using ``command``."""
    try:
        subprocess.run([command, message], check=False)
    except Exception as exc:  # pragma: no cover - system issues
        logging.error("Failed to send notification: %s", exc)


def run_dream() -> bool:
    """Run ``think.dream`` while mirroring output to a dedicated log.

    Returns ``True`` when the subprocess exits successfully.
    """

    start = time.time()
    managed = _launch_process("dream", ["think-dream", "-v"])
    try:
        return_code = managed.process.wait()
    finally:
        managed.cleanup()

    duration = int(time.time() - start)
    logging.info("think.dream finished in %s seconds", duration)
    return return_code == 0


def run_domain_rescan() -> bool:
    """Run ``think-indexer --rescan-domains`` while mirroring output to a dedicated log.

    Returns ``True`` when the subprocess exits successfully.
    """

    start = time.time()
    managed = _launch_process(
        "domain_rescan", ["think-indexer", "--rescan-domains"], log_name="domain_rescan"
    )
    try:
        return_code = managed.process.wait()
    finally:
        managed.cleanup()

    duration = int(time.time() - start)
    logging.info("think-indexer --rescan-domains finished in %s seconds", duration)
    return return_code == 0


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
                    for domain_name in domains.keys():
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


def start_cortex_server() -> ManagedProcess:
    """Launch the Cortex WebSocket API server."""
    cmd = ["muse-cortex", "-v"]
    return _launch_process("cortex", cmd, restart=True)


def start_convey_server(verbose: bool) -> ManagedProcess:
    """Launch the Convey web application with optional verbose logging."""

    cmd = ["convey"]
    if verbose:
        cmd.append("-v")
    return _launch_process("convey", cmd, restart=True)


def check_runner_exits(procs: list[ManagedProcess]) -> list[ManagedProcess]:
    """Return managed processes that have exited."""

    exited: list[ManagedProcess] = []
    for managed in procs:
        if managed.process.poll() is not None:
            exited.append(managed)
    return exited


def supervise(
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
    last_day = datetime.now().date()
    last_health_check = 0.0  # Track last health check time
    alert_state = {}  # Track {issue_key: (last_alert_time, backoff_seconds)}
    prev_stale: set[str] = set()
    initial_backoff = 60  # Start with 1 minute
    max_backoff = 3600  # Max 1 hour between alerts

    while (
        not shutdown_requested
    ):  # pragma: no cover - loop checked via unit tests by patching
        # Check for runner exits first (immediate alert)
        if procs:
            exited = check_runner_exits(procs)
            if exited:
                exited_names = [managed.name for managed in exited]
                msg = f"Runner process exited: {', '.join(sorted(exited_names))}"
                logging.error(msg)
                exit_key = ("runner_exit", tuple(sorted(exited_names)))
                now = time.time()

                if exit_key in alert_state:
                    last_time, backoff = alert_state[exit_key]
                    if now - last_time >= backoff:
                        send_notification(msg, command)
                        alert_state[exit_key] = (now, min(backoff * 2, max_backoff))
                        logging.info(
                            f"Alert sent, next backoff: {min(backoff * 2, max_backoff)}s"
                        )
                    else:
                        remaining = int(backoff - (now - last_time))
                        logging.info(f"Suppressing alert, next in {remaining}s")
                else:
                    send_notification(msg, command)
                    alert_state[exit_key] = (now, initial_backoff)

                for managed in exited:
                    returncode = managed.process.returncode
                    logging.info("%s exited with code %s", managed.name, returncode)
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

                    if managed.restart and not shutdown_requested:
                        policy = _get_restart_policy(managed.name)
                        uptime = (
                            time.time() - policy.last_start if policy.last_start else 0
                        )
                        if uptime >= 60:
                            policy.reset_attempts()
                        delay = policy.next_delay()
                        if delay:
                            logging.info(
                                "Waiting %ss before restarting %s", delay, managed.name
                            )
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
                        except Exception as exc:  # pragma: no cover - defensive
                            logging.exception(
                                "Failed to restart %s: %s", managed.name, exc
                            )
                            continue

                        insert_at = index if index is not None else len(procs)
                        procs.insert(insert_at, new_proc)
                        logging.info(
                            "Restarted %s after exit code %s", managed.name, returncode
                        )
                    else:
                        logging.info("Not restarting %s", managed.name)

        # Check health periodically (interval-based timing)
        now = time.time()
        if now - last_health_check >= interval:
            stale = check_health(threshold)
            stale_set = set(stale)

            recovered = sorted(prev_stale - stale_set)
            for name in recovered:
                logging.info("%s heartbeat recovered", name)

            if stale_set:
                msg = f"Journaling offline: {', '.join(sorted(stale_set))}"
                logging.warning(msg)

                # Apply exponential backoff
                stale_key = ("stale", tuple(sorted(stale_set)))

                if stale_key in alert_state:
                    last_time, backoff = alert_state[stale_key]
                    if now - last_time >= backoff:
                        send_notification(msg, command)
                        # Double the backoff for next time, up to max
                        alert_state[stale_key] = (now, min(backoff * 2, max_backoff))
                        logging.info(
                            f"Alert sent, next backoff: {min(backoff * 2, max_backoff)}s"
                        )
                    else:
                        remaining = int(backoff - (now - last_time))
                        logging.info(f"Suppressing alert, next in {remaining}s")
                else:
                    send_notification(msg, command)
                    alert_state[stale_key] = (now, initial_backoff)
                # Retain only alert state entries still relevant
                alert_state = {
                    k: v
                    for k, v in alert_state.items()
                    if k[0] != "stale" or set(k[1]).issubset(stale_set)
                }
            else:
                if prev_stale:
                    logging.info("Heartbeat OK")
                # Clear alert state for stale services when they recover
                alert_state = {k: v for k, v in alert_state.items() if k[0] != "stale"}

            prev_stale = stale_set
            last_health_check = now

        # Check for daily processing (fast date comparison)
        if daily and datetime.now().date() != last_day:
            if run_dream():
                spawn_scheduled_agents()
            last_day = datetime.now().date()

        # Advance scheduled agent execution (non-blocking)
        check_scheduled_agents()

        # Sleep 1 second before next iteration (responsive to shutdown)
        time.sleep(1)


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
    shutdown_requested = True
    logging.info("Shutdown requested, cleaning up...")


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

    if args.verbose:
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
    if not args.no_observers:
        procs.extend(start_observers())
    if not args.no_cortex:
        procs.append(start_cortex_server())
    if not args.no_convey:
        procs.append(start_convey_server(verbose=args.verbose))

    logging.info(f"Started {len(procs)} processes, entering supervision loop")
    try:
        supervise(
            threshold=args.threshold,
            interval=args.interval,
            command=args.notify_cmd,
            daily=not args.no_daily,
            procs=procs if procs else None,
        )
    except KeyboardInterrupt:
        logging.info("Caught KeyboardInterrupt, shutting down...")
    finally:
        logging.info("Stopping all processes...")
        print("\nShutting down gracefully (this may take up to 15 seconds)...")
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
                print(" done")
            except subprocess.TimeoutExpired:
                logging.warning(f"{name} did not terminate gracefully, killing...")
                print(f" timeout, forcing kill...")
                try:
                    proc.kill()
                    proc.wait(timeout=1)
                except Exception:
                    pass
            managed.cleanup()
        logging.info("Supervisor shutdown complete.")
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
