#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
Standalone tmux terminal capture observer.

Continuously polls all active tmux sessions and captures terminal content,
creating 5-minute segments uploaded via HTTP to the ingest server.

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

from observe.remote_client import ObserverClient, cleanup_draft
from observe.tmux.capture import TmuxCapture, write_captures_jsonl
from observe.utils import create_draft_folder, get_timestamp_parts
from think.streams import stream_name
from think.utils import get_config, setup_cli

logger = logging.getLogger(__name__)

HOST = socket.gethostname()
PLATFORM = platform.system().lower()


class TmuxObserver:
    def __init__(self, interval: int = 300):
        self.interval = interval
        self.tmux_capture = TmuxCapture()
        self.running = True
        self.stream = stream_name(host=HOST, qualifier="tmux")
        self._client: ObserverClient | None = None
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
        """Initialize config, tmux availability, and remote client."""
        enabled, self.capture_interval = self._load_config()
        if not enabled:
            logger.info("Tmux capture disabled in config")
            return False

        if not self.tmux_capture.is_available():
            logger.error("Tmux not available")
            return False

        self._client = ObserverClient(self.stream)
        logger.info("Remote client initialized")
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
        """Write captures to disk, upload the segment, and clean up draft state."""
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
        segment_key = f"{time_part}_{duration}"

        # Upload from draft directory
        draft_path = Path(self.draft_dir)
        draft_files = [
            draft_path / f
            for f in os.listdir(self.draft_dir)
            if (draft_path / f).is_file()
        ]
        if draft_files and self._client:
            meta = {"host": HOST, "platform": PLATFORM, "stream": self.stream}
            result = self._client.upload_segment(
                date_part, segment_key, draft_files, meta
            )
            if result.success:
                logger.info(
                    f"Segment uploaded: {segment_key} ({len(draft_files)} files)"
                )
            else:
                logger.error(f"Segment upload failed: {segment_key}")
        cleanup_draft(self.draft_dir)

        self._reset_capture_state()
        return tmux_files

    def _start_segment(self):
        """Start a new draft segment."""
        self.start_at = time.time()
        self.start_at_mono = time.monotonic()
        self.draft_dir = create_draft_folder(self.start_at, self.stream)

    def emit_status(self):
        """Emit observe.status with current tmux capture state."""
        if not self._client:
            return

        elapsed = int(time.monotonic() - self.start_at_mono)
        tmux_info = {
            "capturing": True,
            "captures": len(self.captures),
            "sessions": sorted(self.sessions_seen),
            "window_elapsed_seconds": elapsed,
        }
        self._client.relay_event(
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
        """Finalize the current segment and stop the remote client."""
        self.finalize_segment()
        self.draft_dir = None
        if self._client:
            self._client.stop()
            self._client = None


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
