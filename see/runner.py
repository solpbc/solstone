"""Run scan.py and describe.py concurrently with automatic restarts."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Thread

from think.reduce_screen import reduce_day

STOP_EVENT = Event()


def _signal_handler(signum: int, frame) -> None:  # type: ignore[unused-argument]
    STOP_EVENT.set()


def _run_scan(journal: str, interval: int, extra_args: list[str]) -> None:
    script_path = Path(__file__).with_name("scan.py")
    cmd_base = [sys.executable, str(script_path), journal, "--min", "250", *extra_args]
    while not STOP_EVENT.is_set():
        start_ts = time.strftime("%Y%m%d_%H%M%S")
        print(f"Running scan.py at {start_ts}", flush=True)
        proc = subprocess.Popen(cmd_base, start_new_session=True)
        while not STOP_EVENT.is_set():
            try:
                proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
        if STOP_EVENT.is_set():
            print("Stopping scan.py...", flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            return

        # Opportunistically reduce the previous 5 minute window
        now = datetime.now()
        prev_minute = now - timedelta(minutes=1)
        block_end = prev_minute.replace(
            minute=(prev_minute.minute // 5) * 5,
            second=0,
            microsecond=0,
        )
        block_start = block_end - timedelta(minutes=5)
        day_dir = os.path.join(journal, prev_minute.strftime("%Y%m%d"))
        try:
            reduce_day(day_dir, start=block_start, end=block_end)
        except Exception as exc:
            print(f"reduce_day failed: {exc}", flush=True)

        time.sleep(interval)


def _run_describe(journal: str, extra_args: list[str]) -> None:
    script_path = Path(__file__).with_name("describe.py")
    cmd_base = [sys.executable, str(script_path), journal, *extra_args]
    while not STOP_EVENT.is_set():
        start_ts = time.strftime("%Y%m%d_%H%M%S")
        print(f"Starting describe.py at {start_ts}", flush=True)
        proc = subprocess.Popen(cmd_base, start_new_session=True)
        while not STOP_EVENT.is_set():
            try:
                proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
        if STOP_EVENT.is_set():
            print("Stopping describe.py...", flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            return
        print("describe.py exited, restarting in 1 second...", flush=True)
        time.sleep(1)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: gemini-see <interval_seconds> <journal_dir> [args...]", file=sys.stderr)
        sys.exit(1)
    interval = int(sys.argv[1])
    journal = sys.argv[2]
    extra_args = sys.argv[3:]

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    scan_thread = Thread(target=_run_scan, args=(journal, interval, extra_args))
    describe_thread = Thread(target=_run_describe, args=(journal, extra_args))

    scan_thread.start()
    describe_thread.start()

    scan_thread.join()
    describe_thread.join()


if __name__ == "__main__":
    main()
