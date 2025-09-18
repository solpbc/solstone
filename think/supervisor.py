import argparse
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

from think.cortex_client import cortex_request
from think.utils import get_agents, setup_cli

DEFAULT_THRESHOLD = 90
CHECK_INTERVAL = 30

# Global shutdown flag
shutdown_requested = False


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


@dataclass
class ManagedProcess:
    process: subprocess.Popen
    name: str
    logger: ProcessLogWriter
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
    journal: str,
    *,
    env: dict[str, str] | None = None,
) -> ManagedProcess:
    log_writer = ProcessLogWriter(Path(journal) / "health" / f"{name}.log")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=env,
        )
    except Exception:
        log_writer.close()
        raise
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
    return ManagedProcess(process=proc, name=name, logger=log_writer, threads=threads)


def check_health(journal: str, threshold: int = DEFAULT_THRESHOLD) -> list[str]:
    """Return a list of stale heartbeat names."""
    now = time.time()
    stale: list[str] = []
    health_dir = Path(journal) / "health"
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


def run_process_day(journal: str) -> bool:
    """Run ``think.process_day`` while mirroring output to a dedicated log.

    Returns ``True`` when the subprocess exits successfully.
    """

    start = time.time()
    managed = _launch_process("process_day", ["think-process-day", "-v"], journal)
    try:
        return_code = managed.process.wait()
    finally:
        managed.cleanup()

    duration = int(time.time() - start)
    logging.info("think.process_day finished in %s seconds", duration)
    return return_code == 0


def spawn_scheduled_agents(journal: str) -> None:
    """Spawn agents that have schedule:daily in their metadata."""
    try:
        agents = get_agents()
        for persona_id, config in agents.items():
            if config.get("schedule") == "daily":
                logging.info(f"Spawning scheduled agent: {persona_id}")

                # Spawn via Cortex - it will load and merge the persona config
                request_file = cortex_request(
                    prompt=f"Running daily scheduled task for {persona_id}",
                    persona=persona_id,
                )

                # Extract agent_id from the filename
                agent_id = Path(request_file).stem.replace("_active", "")
                logging.info(f"Started {persona_id} agent (ID: {agent_id})")
    except Exception as e:
        logging.error(f"Failed to spawn scheduled agents: {e}")


def start_runners(journal: str) -> list[ManagedProcess]:
    """Launch hear and see runners with output logging."""
    procs: list[ManagedProcess] = []
    commands = {
        "hear": ["hear-runner", "-v"],
        "see": ["see-runner", "-v"],
    }
    for name, cmd in commands.items():
        procs.append(_launch_process(name, cmd, journal))
    return procs


def start_cortex_server(journal: str) -> ManagedProcess:
    """Launch the Cortex WebSocket API server."""
    cmd = ["think-cortex", "-v"]
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal
    return _launch_process("cortex", cmd, journal, env=env)


def check_runner_exits(procs: list[ManagedProcess]) -> list[str]:
    """Check if any runner processes have exited and return their names."""
    exited = []
    for managed in procs:
        if managed.process.poll() is not None:  # Process has exited
            exited.append(managed.name)
    return exited


def supervise(
    journal: str,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    interval: int = CHECK_INTERVAL,
    command: str = "notify-send",
    daily: bool = True,
    procs: list[ManagedProcess] | None = None,
) -> None:
    """Monitor heartbeat files and alert when they become stale."""
    global shutdown_requested
    last_day = datetime.now().date()
    alert_state = {}  # Track {issue_key: (last_alert_time, backoff_seconds)}
    initial_backoff = 60  # Start with 1 minute
    max_backoff = 3600  # Max 1 hour between alerts

    while (
        not shutdown_requested
    ):  # pragma: no cover - loop checked via unit tests by patching
        # Check for runner exits first (immediate alert)
        if procs:
            exited = check_runner_exits(procs)
            if exited:
                msg = f"Runner process exited: {', '.join(sorted(exited))}"
                logging.error(msg)
                exit_key = ("runner_exit", tuple(sorted(exited)))
                now = time.time()

                if exit_key in alert_state:
                    last_time, backoff = alert_state[exit_key]
                    if now - last_time >= backoff:
                        send_notification(msg, command)
                        # Double the backoff for next time, up to max
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

        stale = check_health(journal, threshold)
        if stale:
            msg = f"Journaling offline: {', '.join(sorted(stale))}"
            logging.warning(msg)

            # Apply exponential backoff
            stale_key = ("stale", tuple(sorted(stale)))
            now = time.time()

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
        else:
            logging.info("Heartbeat OK")
            # Clear alert state for stale services when they recover
            alert_state = {k: v for k, v in alert_state.items() if k[0] != "stale"}

        if daily and datetime.now().date() != last_day:
            if run_process_day(journal):
                spawn_scheduled_agents(journal)
            last_day = datetime.now().date()

        # Use shorter sleep intervals to check for shutdown
        for _ in range(interval):
            if shutdown_requested:
                break
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
        "--no-runners",
        action="store_true",
        help="Do not automatically start hear and see runners",
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
    return parser


def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logging.info("Shutdown requested, cleaning up...")


def main() -> None:
    parser = parse_args()
    args = setup_cli(parser)
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        parser.error("JOURNAL_PATH not set")

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_path = Path(journal) / "health" / "supervisor.log"
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

    procs: list[ManagedProcess] = []
    if not args.no_runners:
        procs.extend(start_runners(journal))
    if not args.no_cortex:
        procs.append(start_cortex_server(journal))
    try:
        supervise(
            journal,
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
        for managed in procs:
            name = managed.name
            proc = managed.process
            logging.info(f"Stopping {name}...")
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logging.warning(f"{name} did not terminate gracefully, killing...")
                try:
                    proc.kill()
                    proc.wait(timeout=1)
                except Exception:
                    pass
            managed.cleanup()
        logging.info("Supervisor shutdown complete.")


if __name__ == "__main__":
    main()
