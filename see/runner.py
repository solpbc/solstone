"""Run scan.py and describe.py concurrently with automatic restarts."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event, Thread

from think.utils import setup_cli

STOP_EVENT = Event()


def _signal_handler(signum: int, frame) -> None:  # type: ignore[unused-argument]
    STOP_EVENT.set()


def _run_scan(interval: int, extra_args: list[str]) -> None:
    script_path = Path(__file__).with_name("scan.py")
    cmd_base = [sys.executable, str(script_path), "--min", "250", *extra_args]
    while not STOP_EVENT.is_set():
        start_ts = time.strftime("%Y%m%d_%H%M%S")
        logging.info("Running scan.py at %s", start_ts)
        try:
            proc = subprocess.Popen(cmd_base, start_new_session=True)
        except Exception as exc:  # catch startup errors
            logging.error("Failed to start scan.py: %s", exc)
            time.sleep(interval)
            continue
        start_time = time.time()
        while not STOP_EVENT.is_set():
            try:
                proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                if time.time() - start_time > 5:
                    logging.error("scan.py timed out after 5 seconds")
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    break
                continue
            except Exception as exc:
                logging.error("Error waiting for scan.py: %s", exc)
                break
        if STOP_EVENT.is_set():
            logging.info("Stopping scan.py...")
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
        logging.info("Starting describe.py at %s", start_ts)
        try:
            proc = subprocess.Popen(cmd_base, start_new_session=True)
        except Exception as exc:
            logging.error("Failed to start describe.py: %s", exc)
            time.sleep(1)
            continue
        while not STOP_EVENT.is_set():
            try:
                proc.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
            except Exception as exc:
                logging.error("Error waiting for describe.py: %s", exc)
                break
        if STOP_EVENT.is_set():
            logging.info("Stopping describe.py...")
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            return
        logging.info("describe.py exited, restarting in 1 second...")
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scan.py and describe.py concurrently with automatic restarts."
    )
    parser.add_argument("interval", type=int, help="Seconds between scan runs")
    args, extra_args = setup_cli(parser, parse_known=True)

    if args.verbose and "-v" not in extra_args and "--verbose" not in extra_args:
        extra_args = ["-v", *extra_args]

    interval = args.interval

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
