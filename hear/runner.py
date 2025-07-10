"""Run capture.py and transcribe.py in parallel with automatic restarts."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

STOP_EVENT = Event()


def _signal_handler(signum: int, frame) -> None:  # type: ignore[unused-argument]
    STOP_EVENT.set()


def _run_loop(script: str, args: list[str]) -> None:
    script_path = Path(__file__).with_name(script)
    cmd = [sys.executable, str(script_path), *args]
    while not STOP_EVENT.is_set():
        start_ts = time.strftime("%Y%m%d_%H%M%S")
        print(f"Starting {script} at {start_ts}", flush=True)
        try:
            proc = subprocess.Popen(cmd, start_new_session=True)
        except Exception as exc:  # Catch anything that prevents start
            print(f"Failed to start {script}: {exc}", flush=True)
            time.sleep(1)
            continue
        while not STOP_EVENT.is_set():
            try:
                proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
            except Exception as exc:  # Unexpected errors while waiting
                print(f"Error waiting for {script}: {exc}", flush=True)
                break
        if STOP_EVENT.is_set():
            print(f"Stopping {script}...", flush=True)
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            return
        print(
            f"{script} exited with code {proc.returncode}, restarting in 1 second...",
            flush=True,
        )
        time.sleep(1)


def main() -> None:
    args = sys.argv[1:]

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    capture_args = args + ["--ws-port", "9987"]
    capture_thread = Thread(target=_run_loop, args=("capture.py", capture_args))
    transcribe_thread = Thread(target=_run_loop, args=("transcribe.py", args))

    capture_thread.start()
    transcribe_thread.start()

    capture_thread.join()
    transcribe_thread.join()


if __name__ == "__main__":
    main()
