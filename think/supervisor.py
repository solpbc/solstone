import argparse
import logging
import os
import subprocess
import time
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


def supervise(
    journal: str,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    interval: int = CHECK_INTERVAL,
    command: str = "notify-send",
) -> None:
    """Monitor heartbeat files and alert when they become stale."""
    while True:  # pragma: no cover - loop checked via unit tests by patching
        stale = check_health(journal, threshold)
        if stale:
            msg = f"Journaling offline: {', '.join(sorted(stale))}"
            logging.warning(msg)
            send_notification(msg, command)
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
    return parser


def main() -> None:
    parser = parse_args()
    args = setup_cli(parser)
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        parser.error("JOURNAL_PATH not set")

    supervise(
        journal,
        threshold=args.threshold,
        interval=args.interval,
        command=args.notify_cmd,
    )


if __name__ == "__main__":
    main()
