#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
Standalone tmux terminal capture observer.

Continuously polls all active tmux sessions and captures terminal content,
creating 5-minute segments with draft-to-final atomic rename.

Always-on: no idle detection, no screen activity checks. Just captures
whatever tmux sessions exist on the configurable interval.
"""

import argparse
import asyncio
import logging
import os
import platform
import signal
import socket
import sys
import time
from pathlib import Path

from observe.tmux.capture import TmuxCapture, write_captures_jsonl
from observe.utils import create_draft_folder, get_timestamp_parts
from think.callosum import CallosumConnection
from think.streams import stream_name, update_stream, write_segment_stream
from think.utils import day_path, get_config, get_rev, setup_cli

logger = logging.getLogger(__name__)

HOST = socket.gethostname()
PLATFORM = platform.system().lower()


class TmuxObserver:
    def __init__(self, interval: int = 300):
        self.interval = interval
        self.tmux_capture = TmuxCapture()
        self.running = True
        self.stream = stream_name(host=HOST, qualifier="tmux")
        self._callosum: CallosumConnection | None = None
        self.start_at = time.time()
        self.start_at_mono = time.monotonic()
        self.draft_dir: str | None = None
        self.captures: list[dict] = []
        self.capture_id = 0
        self.sessions_seen: set[str] = set()
        self.last_capture_time: float = 0
        self.capture_interval = 5

    def _load_config(self) -> tuple[bool, int]:
        """Load tmux settings from journal config."""
        enabled = True
        capture_interval = 5
        try:
            config = get_config()
            observe_config = config.get("observe", {})
            tmux_config = observe_config.get("tmux", {})
            enabled = tmux_config.get("enabled", True)
            capture_interval = tmux_config.get("capture_interval", 5)
        except Exception as e:
            logger.warning(f"Failed to load tmux config, using defaults: {e}")
        return enabled, capture_interval

    def setup(self) -> bool:
        """Initialize config, tmux availability, and Callosum."""
        enabled, self.capture_interval = self._load_config()
        if not enabled:
            logger.info("Tmux capture disabled in config")
            return False

        if not self.tmux_capture.is_available():
            logger.error("Tmux not available")
            return False

        self._callosum = CallosumConnection(defaults={"rev": get_rev()})
        self._callosum.start()
        logger.info("Callosum connection started")
        return True

    def capture(self):
        """Poll tmux and accumulate captures based on capture interval."""
        now = time.time()
        time_since_capture = now - self.last_capture_time

        if time_since_capture < self.capture_interval:
            return

        active_sessions = self.tmux_capture.get_active_sessions(time_since_capture)
        if not active_sessions:
            return

        self.last_capture_time = now

        for session_info in active_sessions:
            session = session_info["session"]
            self.sessions_seen.add(session)

            result = self.tmux_capture.capture_changed(session)
            if not result:
                continue

            self.capture_id += 1
            relative_ts = now - self.start_at
            capture_dict = self.tmux_capture.result_to_dict(
                result, self.capture_id, relative_ts
            )
            self.captures.append(capture_dict)
            logger.debug(f"Captured tmux session {session}: {len(result.panes)} panes")

    def _reset_capture_state(self):
        """Reset per-segment capture tracking."""
        self.captures = []
        self.capture_id = 0
        self.sessions_seen = set()
        self.tmux_capture.reset_hashes()
        self.last_capture_time = 0

    def _remove_empty_draft(self):
        """Remove an empty draft folder, ignoring errors."""
        if self.draft_dir:
            try:
                os.rmdir(self.draft_dir)
            except OSError:
                pass

    def finalize_segment(self) -> list[str]:
        """Write captures to disk, finalize the segment, and emit observing."""
        if not self.captures or not self.draft_dir:
            self._remove_empty_draft()
            self._reset_capture_state()
            return []

        tmux_files = write_captures_jsonl(self.captures, Path(self.draft_dir))
        if not tmux_files:
            self._remove_empty_draft()
            self._reset_capture_state()
            return []

        date_part, time_part = get_timestamp_parts(self.start_at)
        duration = int(time.time() - self.start_at)
        day_dir = day_path(date_part)
        segment_key = f"{time_part}_{duration}"
        final_segment_dir = str(day_dir / self.stream / segment_key)

        try:
            os.rename(self.draft_dir, final_segment_dir)
            logger.info(f"Segment finalized: {self.draft_dir} -> {final_segment_dir}")
        except OSError as e:
            logger.error(f"Failed to rename draft folder: {e}")
            tmux_files = []

        if tmux_files:
            try:
                result = update_stream(
                    self.stream,
                    date_part,
                    segment_key,
                    type="observer",
                    host=HOST,
                    platform=PLATFORM,
                )
                write_segment_stream(
                    final_segment_dir,
                    self.stream,
                    result["prev_day"],
                    result["prev_segment"],
                    result["seq"],
                )
            except Exception as e:
                logger.warning(f"Failed to write stream identity: {e}")

            if self._callosum:
                self._callosum.emit(
                    "observe",
                    "observing",
                    day=date_part,
                    segment=segment_key,
                    files=tmux_files,
                    host=HOST,
                    platform=PLATFORM,
                    stream=self.stream,
                )

        self._reset_capture_state()
        return tmux_files

    def _start_segment(self):
        """Start a new draft segment."""
        self.start_at = time.time()
        self.start_at_mono = time.monotonic()
        self.draft_dir = create_draft_folder(self.start_at, self.stream)

    def emit_status(self):
        """Emit observe.status with current tmux capture state."""
        if not self._callosum:
            return

        elapsed = int(time.monotonic() - self.start_at_mono)
        tmux_info = {
            "capturing": True,
            "captures": len(self.captures),
            "sessions": sorted(self.sessions_seen),
            "window_elapsed_seconds": elapsed,
        }
        self._callosum.emit(
            "observe",
            "status",
            mode="tmux",
            tmux=tmux_info,
            host=HOST,
            platform=PLATFORM,
            stream=self.stream,
        )

    async def main_loop(self):
        """Run the polling loop until shutdown."""
        self._start_segment()

        while self.running:
            await asyncio.sleep(1)  # Short sleep for responsive shutdown
            self.capture()

            elapsed = time.monotonic() - self.start_at_mono
            if elapsed >= self.interval:
                self.finalize_segment()
                self._start_segment()

            self.emit_status()

        await self.shutdown()

    async def shutdown(self):
        """Finalize the current segment and stop Callosum."""
        self.finalize_segment()
        self.draft_dir = None
        if self._callosum:
            self._callosum.stop()
            self._callosum = None


async def async_main(args):
    """Async entry point."""
    observer = TmuxObserver(interval=args.interval)

    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        observer.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    if not observer.setup():
        logger.error("Tmux observer setup failed")
        return 1

    try:
        await observer.main_loop()
    except RuntimeError as e:
        logger.error(f"Tmux observer runtime error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Tmux observer error: {e}", exc_info=True)
        return 1

    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Standalone tmux terminal capture observer."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Duration per tmux segment in seconds (default: 300 = 5 minutes).",
    )
    args = setup_cli(parser)

    try:
        rc = asyncio.run(async_main(args))
        sys.exit(rc)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
