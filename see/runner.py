"""Run scan.py and describe.py concurrently with automatic restarts."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

from dotenv import load_dotenv

STOP_EVENT = Event()


def _signal_handler(signum: int, frame) -> None:  # type: ignore[unused-argument]
    STOP_EVENT.set()


def _run_scan(interval: int, extra_args: list[str]) -> None:
    script_path = Path(__file__).with_name("scan.py")
    cmd_base = [sys.executable, str(script_path), "--min", "250", *extra_args]
    while not STOP_EVENT.is_set():
        start_ts = time.strftime("%Y%m%d_%H%M%S")
        print(f"Running scan.py at {start_ts}", flush=True)
        try:
            proc = subprocess.Popen(cmd_base, start_new_session=True)
        except Exception as exc:  # catch startup errors
            print(f"Failed to start scan.py: {exc}", flush=True)
            time.sleep(interval)
            continue
        start_time = time.time()
        while not STOP_EVENT.is_set():
            try:
                proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                if time.time() - start_time > 5:
                    print("scan.py timed out after 5 seconds", flush=True)
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                continue
            except Exception as exc:
                print(f"Error waiting for scan.py: {exc}", flush=True)
                break
        if STOP_EVENT.is_set():
            print("Stopping scan.py...", flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            return

        time.sleep(interval)


def _run_describe(extra_args: list[str]) -> None:
    script_path = Path(__file__).with_name("describe.py")
    cmd_base = [sys.executable, str(script_path), *extra_args]
    while not STOP_EVENT.is_set():
        start_ts = time.strftime("%Y%m%d_%H%M%S")
        print(f"Starting describe.py at {start_ts}", flush=True)
        try:
            proc = subprocess.Popen(cmd_base, start_new_session=True)
        except Exception as exc:
            print(f"Failed to start describe.py: {exc}", flush=True)
            time.sleep(1)
            continue
        while not STOP_EVENT.is_set():
            try:
                proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
            except Exception as exc:
                print(f"Error waiting for describe.py: {exc}", flush=True)
                break
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
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: gemini-see <interval_seconds> [args...]", file=sys.stderr)
        sys.exit(1)
    interval = int(sys.argv[1])
    extra_args = sys.argv[2:]

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    scan_thread = Thread(target=_run_scan, args=(interval, extra_args))
    describe_thread = Thread(target=_run_describe, args=(extra_args,))

    scan_thread.start()
    describe_thread.start()

    scan_thread.join()
    describe_thread.join()


if __name__ == "__main__":
    main()
