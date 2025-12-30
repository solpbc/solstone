#!/usr/bin/env python3
"""
macOS observer for audio and screencast capture using ScreenCaptureKit.

Continuously captures audio and video using sck-cli based on activity detection.
Creates 5-minute windows, saving both video and audio when voice activity is detected.
"""

import argparse
import asyncio
import datetime
import logging
import os
import signal
import sys
import time
from pathlib import Path

from observe.macos.activity import (
    get_idle_time_ms,
    get_monitor_metadata_string,
    is_screen_locked,
)
from observe.macos.screencapture import ScreenCaptureKitManager
from think.callosum import CallosumConnection
from think.utils import day_path, setup_cli

logger = logging.getLogger(__name__)

# Constants
IDLE_THRESHOLD_MS = 5 * 60 * 1000  # 5 minutes
CHUNK_DURATION = 5  # seconds


class MacOSObserver:
    """macOS audio and screencast observer using ScreenCaptureKit."""

    def __init__(self, interval: int = 300):
        """
        Initialize the macOS observer.

        Args:
            interval: Window duration in seconds (default: 300 = 5 minutes)
        """
        self.interval = interval
        self.screencapture = ScreenCaptureKitManager()
        self.running = True
        self.callosum: CallosumConnection | None = None

        # State tracking
        self.start_at = time.time()  # Wall-clock for filenames
        self.start_at_mono = time.monotonic()  # Monotonic for elapsed calculations
        self.capture_running = False
        self.current_output_base = None  # Base path for current capture
        self.pending_finalization = (
            None  # Tuple of (temp_base, final_video, final_audio)
        )
        self.last_video_size = 0  # Track file size for health checks

        # Activity status cache (updated each loop)
        self.cached_is_active = False
        self.cached_idle_time_ms = 0
        self.cached_screen_locked = False

    async def setup(self):
        """Initialize ScreenCaptureKit and Callosum connection."""
        # TODO: Implement setup
        # 1. Verify sck-cli is available in PATH
        # 2. Start Callosum connection for status events
        # 3. Log initialization success
        logger.warning("setup() not yet implemented")
        return False

    async def check_activity_status(self) -> bool:
        """
        Check system activity status and cache values.

        Returns:
            True if user is active (not idle and screen unlocked)
        """
        # TODO: Implement activity checking
        # 1. Call get_idle_time_ms()
        # 2. Call is_screen_locked()
        # 3. Cache values for status events
        # 4. Determine if active (not idle and not locked)
        # 5. Return activity status
        logger.warning("check_activity_status() not yet implemented")
        return False

    def get_timestamp_parts(self, timestamp: float = None) -> tuple[str, str]:
        """
        Get date and time parts from timestamp.

        Args:
            timestamp: Unix timestamp (default: current time)

        Returns:
            Tuple of (date_part, time_part) like ("20250101", "143022")
        """
        if timestamp is None:
            timestamp = time.time()
        dt = datetime.datetime.fromtimestamp(timestamp)
        date_part = dt.strftime("%Y%m%d")
        time_part = dt.strftime("%H%M%S")
        return date_part, time_part

    async def handle_boundary(self, is_active: bool):
        """
        Handle window boundary rollover.

        Args:
            is_active: Whether system is currently active
        """
        # TODO: Implement boundary handling
        # 1. Get timestamp parts and calculate duration
        # 2. Stop current capture if running
        # 3. Queue files for finalization (temp paths -> final paths with duration)
        # 4. Reset timing for new window
        # 5. Start new capture if active and screen not locked
        # 6. Emit Callosum observing event with saved files
        logger.warning("handle_boundary() not yet implemented")

    async def initialize_capture(self) -> bool:
        """
        Start a new screencast and audio recording.

        Returns:
            True if capture started successfully, False otherwise
        """
        # TODO: Implement capture initialization
        # 1. Get timestamp for filename
        # 2. Build temp output base path (e.g., .HHMMSS)
        # 3. Start sck-cli via ScreenCaptureKitManager
        # 4. Update state tracking variables
        # 5. Log capture start
        logger.warning("initialize_capture() not yet implemented")
        return False

    def emit_status(self):
        """Emit observe.status event with current state."""
        # TODO: Implement status emission
        # 1. Build capture info dict (recording status, file path, elapsed time)
        # 2. Build activity info dict (active, idle_time_ms, screen_locked)
        # 3. Emit via Callosum
        logger.warning("emit_status() not yet implemented")

    async def finalize_capture(
        self, temp_base: Path, final_video: Path, final_audio: Path
    ):
        """
        Add monitor metadata to video and rename files from temp to final paths.

        Args:
            temp_base: Temporary base path (e.g., /path/.HHMMSS)
            final_video: Final video path (e.g., /path/HHMMSS_300_screen.mov)
            final_audio: Final audio path (e.g., /path/HHMMSS_300_audio.m4a)
        """
        # TODO: Implement finalization
        # 1. Check if temp files exist
        # 2. Get monitor metadata string
        # 3. Call screencapture.finalize() to add metadata and rename
        # 4. Log success/failure
        logger.warning("finalize_capture() not yet implemented")

    async def main_loop(self):
        """Run the main observer loop."""
        # TODO: Implement main loop
        # 1. Check activity and start capture if active
        # 2. Loop with CHUNK_DURATION sleep intervals:
        #    a. Process pending finalization if queued
        #    b. Check activity status
        #    c. Detect activation edge (idle -> active transition)
        #    d. Check for window boundary or activation edge
        #    e. Handle boundary if needed
        #    f. Touch health files (see, hear)
        #    g. Emit status event
        # 3. Call shutdown on exit
        logger.warning("main_loop() not yet implemented")

    async def shutdown(self):
        """Clean shutdown of observer."""
        # TODO: Implement shutdown
        # 1. Stop current capture if running
        # 2. Wait for files to be written
        # 3. Finalize pending captures
        # 4. Stop Callosum connection
        # 5. Log shutdown complete
        logger.warning("shutdown() not yet implemented")


async def async_main(args):
    """Async entry point."""
    observer = MacOSObserver(interval=args.interval)

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        observer.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Initialize
    if not await observer.setup():
        logger.error("Observer setup failed")
        return 1

    # Run main loop
    try:
        await observer.main_loop()
    except Exception as e:
        logger.error(f"Observer error: {e}", exc_info=True)
        return 1

    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="macOS audio and screencast observer for journaling."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Duration per capture window in seconds (default: 300 = 5 minutes).",
    )
    parser.add_argument(
        "--sck-cli-path",
        type=str,
        default="sck-cli",
        help="Path to sck-cli executable (default: sck-cli from PATH).",
    )
    args = setup_cli(parser)

    # Verify journal path exists
    journal = os.getenv("JOURNAL_PATH")
    if not journal or not os.path.exists(journal):
        logger.error(f"JOURNAL_PATH not set or does not exist: {journal}")
        sys.exit(1)

    # Run async main
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
