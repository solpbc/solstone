import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from think.cortex_client import cortex_request
from think.utils import get_agents, setup_cli

DEFAULT_THRESHOLD = 90
CHECK_INTERVAL = 30

# Global shutdown flag
shutdown_requested = False


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


def run_process_day() -> bool:
    """Run ``think.process_day`` and log duration.

    Returns:
        True if process_day completed successfully, False otherwise.
    """
    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "think.process_day"],
        stdout=None,  # Inherit stdout
        stderr=None,  # Inherit stderr
    )
    duration = int(time.time() - start)
    logging.info("think.process_day finished in %s seconds", duration)
    return result.returncode == 0


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


def start_runners(
    journal: str, verbose: bool = False
) -> list[tuple[subprocess.Popen, str]]:
    """Launch hear and see runners with output to console."""
    procs = []
    for module in ("hear.runner", "see.runner"):
        cmd = [sys.executable, "-m", module]
        if verbose:
            cmd.append("-v")
        proc = subprocess.Popen(
            cmd,
            stdout=None,  # Inherit stdout
            stderr=None,  # Inherit stderr
            start_new_session=True,
        )
        runner_name = module.split(".")[0]  # "hear" or "see"
        procs.append((proc, runner_name))
    return procs


def start_mcp_server(journal: str, verbose: bool = False) -> subprocess.Popen:
    """Launch the MCP tools HTTP server."""
    cmd = [sys.executable, "-m", "think.mcp_tools", "--transport", "http"]
    if verbose:
        cmd.append("-v")
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal
    proc = subprocess.Popen(
        cmd,
        stdout=None,  # Inherit stdout
        stderr=None,  # Inherit stderr
        start_new_session=True,
        env=env,
    )
    return proc


def start_cortex_server(journal: str, verbose: bool = False) -> subprocess.Popen:
    """Launch the Cortex WebSocket API server."""
    cmd = [sys.executable, "-m", "think.cortex"]
    if verbose:
        cmd.append("-v")
    env = os.environ.copy()
    env["JOURNAL_PATH"] = journal
    proc = subprocess.Popen(
        cmd,
        stdout=None,  # Inherit stdout
        stderr=None,  # Inherit stderr
        start_new_session=True,
        env=env,
    )
    return proc


def check_runner_exits(procs: list[tuple[subprocess.Popen, str]]) -> list[str]:
    """Check if any runner processes have exited and return their names."""
    exited = []
    for proc, name in procs:
        if proc.poll() is not None:  # Process has exited
            exited.append(name)
    return exited


def supervise(
    journal: str,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    interval: int = CHECK_INTERVAL,
    command: str = "notify-send",
    daily: bool = True,
    procs: list[tuple[subprocess.Popen, str]] | None = None,
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
            if run_process_day():
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

    # Only log to console
    level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=level,
        handlers=[logging.StreamHandler()],
        format="%(asctime)s %(levelname)s %(message)s",
    )

    os.environ.setdefault("SUNSTONE_MCP_URL", "http://127.0.0.1:6270/mcp")

    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    procs: list[tuple[subprocess.Popen, str]] = []
    if not args.no_runners:
        procs = start_runners(journal, verbose=args.verbose)
    mcp_proc = start_mcp_server(journal, verbose=args.verbose)
    procs.append((mcp_proc, "mcp"))
    cortex_proc = start_cortex_server(journal, verbose=args.verbose)
    procs.append((cortex_proc, "cortex"))
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
        for proc, name in procs:
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
        logging.info("Supervisor shutdown complete.")


if __name__ == "__main__":
    main()
