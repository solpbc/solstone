import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from think.utils import setup_cli

DEFAULT_THRESHOLD = 90
CHECK_INTERVAL = 30


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


def run_process_day(log_path: Path) -> None:
    """Run ``think.process_day`` and log duration to ``log_path``."""
    start = time.time()
    with open(log_path, "ab") as log_file:
        subprocess.run(
            [sys.executable, "-m", "think.process_day"],
            stdout=log_file,
            stderr=log_file,
        )
    duration = int(time.time() - start)
    logging.info("think.process_day finished in %s seconds", duration)


def start_runners(journal: str) -> list[tuple[subprocess.Popen, str]]:
    """Launch hear and see runners logging output to supervisor.log."""
    log_path = Path(journal) / "health" / "supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    procs = []
    for module in ("hear.runner", "see.runner"):
        with open(log_path, "ab") as log_file:
            proc = subprocess.Popen(
                [sys.executable, "-m", module],
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
        runner_name = module.split(".")[0]  # "hear" or "see"
        procs.append((proc, runner_name))
    return procs


def start_mcp_server(journal: str) -> subprocess.Popen:
    """Launch the MCP tools HTTP server."""
    log_path = Path(journal) / "health" / "supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log_file:
        env = os.environ.copy()
        env["JOURNAL_PATH"] = journal
        proc = subprocess.Popen(
            [sys.executable, "-m", "think.mcp_tools", "--transport", "http"],
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
            env=env,
        )
    return proc


def start_cortex_server(journal: str) -> subprocess.Popen:
    """Launch the Cortex WebSocket API server."""
    log_path = Path(journal) / "health" / "supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log_file:
        env = os.environ.copy()
        env["JOURNAL_PATH"] = journal
        proc = subprocess.Popen(
            [sys.executable, "-m", "think.cortex"],
            stdout=log_file,
            stderr=log_file,
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
    log_path = Path(journal) / "health" / "supervisor.log"
    last_day = datetime.now().date()
    while True:  # pragma: no cover - loop checked via unit tests by patching
        # Check for runner exits first (immediate alert)
        if procs:
            exited = check_runner_exits(procs)
            if exited:
                msg = f"Runner process exited: {', '.join(sorted(exited))}"
                logging.error(msg)
                send_notification(msg, command)

        stale = check_health(journal, threshold)
        if stale:
            msg = f"Journaling offline: {', '.join(sorted(stale))}"
            logging.warning(msg)
            send_notification(msg, command)
        else:
            logging.info("Heartbeat OK")
        if daily and datetime.now().date() != last_day:
            run_process_day(log_path)
            last_day = datetime.now().date()
        time.sleep(interval)


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


def main() -> None:
    parser = parse_args()
    args = setup_cli(parser)
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        parser.error("JOURNAL_PATH not set")

    log_path = Path(journal) / "health" / "supervisor.log"
    handlers: list[logging.Handler] = [logging.FileHandler(log_path, encoding="utf-8")]
    if args.verbose:
        handlers.append(logging.StreamHandler())

    level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=level, handlers=handlers, format="%(asctime)s %(levelname)s %(message)s"
    )

    os.environ.setdefault("SUNSTONE_MCP_URL", "http://127.0.0.1:6270/mcp")

    procs: list[tuple[subprocess.Popen, str]] = []
    if not args.no_runners:
        procs = start_runners(journal)
    mcp_proc = start_mcp_server(journal)
    procs.append((mcp_proc, "mcp"))
    cortex_proc = start_cortex_server(journal)
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
    finally:
        for proc, name in procs:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass


if __name__ == "__main__":
    main()
