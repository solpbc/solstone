#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

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
import shutil
import signal
import sys
import time
from pathlib import Path

import av
import numpy as np

from observe.macos.activity import (
    get_idle_time_ms,
    is_output_muted,
    is_power_save_active,
    is_screen_locked,
)
from observe.macos.screencapture import AudioInfo, DisplayInfo, ScreenCaptureKitManager
from observe.remote import HOST, PLATFORM
from think.callosum import CallosumConnection
from think.utils import day_path, setup_cli

logger = logging.getLogger(__name__)

# Constants
IDLE_THRESHOLD_MS = 5 * 60 * 1000  # 5 minutes
CHUNK_DURATION = 5  # seconds
RMS_THRESHOLD = 0.01
MIN_HITS_FOR_SAVE = 3
SAMPLE_RATE = 48000  # Standard audio sample rate


class MacOSObserver:
    """macOS audio and screencast observer using ScreenCaptureKit."""

    def __init__(self, interval: int = 300, sck_cli_path: str = "sck-cli"):
        """
        Initialize the macOS observer.

        Args:
            interval: Window duration in seconds (default: 300 = 5 minutes)
            sck_cli_path: Path to sck-cli executable
        """
        self.interval = interval
        self.screencapture = ScreenCaptureKitManager(sck_cli_path=sck_cli_path)
        self.running = True
        self.callosum: CallosumConnection | None = None

        # State tracking
        self.start_at = time.time()  # Wall-clock for filenames
        self.start_at_mono = time.monotonic()  # Monotonic for elapsed calculations
        self.capture_running = False

        # Multi-display tracking (similar to Linux observer)
        self.current_displays: list[DisplayInfo] = []
        self.current_audio: AudioInfo | None = None

        # Draft folder for current segment (HHMMSS_draft/)
        self.draft_dir: str | None = None

        # Activity status cache (updated each loop)
        self.cached_is_active = False
        self.cached_idle_time_ms = 0
        self.cached_screen_locked = False
        self.cached_is_muted = False
        self.cached_power_save = False

        # Mute state at segment start
        self.segment_is_muted = False

    async def setup(self):
        """Initialize ScreenCaptureKit and Callosum connection."""
        # Verify sck-cli is available
        sck_path = shutil.which(self.screencapture.sck_cli_path)
        if not sck_path:
            logger.error(f"sck-cli not found: {self.screencapture.sck_cli_path}")
            return False
        logger.info(f"Found sck-cli at: {sck_path}")

        # Start Callosum connection for status events
        self.callosum = CallosumConnection()
        self.callosum.start()
        logger.info("Callosum connection started")

        return True

    def check_activity_status(self) -> bool:
        """
        Check system activity status and cache values.

        Returns:
            True if user is active (not idle and screen unlocked)
        """
        idle_time = get_idle_time_ms()
        screen_locked = is_screen_locked()
        power_save = is_power_save_active()
        output_muted = is_output_muted()

        # Cache values for status events
        self.cached_idle_time_ms = idle_time
        self.cached_screen_locked = screen_locked
        self.cached_power_save = power_save
        self.cached_is_muted = output_muted

        is_idle = (idle_time > IDLE_THRESHOLD_MS) or screen_locked or power_save
        is_active = not is_idle

        # Cache result
        self.cached_is_active = is_active

        return is_active

    def get_timestamp_parts(self, timestamp: float | None = None) -> tuple[str, str]:
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

    def _check_audio_threshold(self, audio_path: str) -> bool:
        """
        Check if audio file has enough voice activity to save.

        Decodes the m4a file and applies the same 3-chunk RMS threshold
        logic as Linux observer uses for real-time audio.

        Args:
            audio_path: Path to the m4a audio file

        Returns:
            True if audio should be saved (enough voice activity), False otherwise
        """
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found for threshold check: {audio_path}")
            return False

        container = None
        try:
            container = av.open(audio_path)
            audio_streams = list(container.streams.audio)

            if not audio_streams:
                logger.warning(f"No audio streams in {audio_path}")
                return False

            stream = audio_streams[0]
            sample_rate = stream.rate or SAMPLE_RATE

            # Decode audio and collect samples
            samples = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                # Convert to mono if stereo (average channels)
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                samples.append(arr.flatten())

            if not samples:
                logger.warning(f"No audio samples decoded from {audio_path}")
                return False

            # Concatenate all samples
            all_samples = np.concatenate(samples)

            # Split into CHUNK_DURATION (5 second) chunks and count threshold hits
            chunk_samples = int(sample_rate * CHUNK_DURATION)
            threshold_hits = 0

            for i in range(0, len(all_samples), chunk_samples):
                chunk = all_samples[i : i + chunk_samples]
                if len(chunk) == 0:
                    continue

                # Compute RMS for this chunk
                rms = float(np.sqrt(np.mean(chunk**2)))
                if rms > RMS_THRESHOLD:
                    threshold_hits += 1

            logger.debug(
                f"Audio threshold check: {threshold_hits}/{MIN_HITS_FOR_SAVE} hits"
            )
            return threshold_hits >= MIN_HITS_FOR_SAVE

        except Exception as e:
            logger.warning(f"Error checking audio threshold for {audio_path}: {e}")
            # On error, keep the file (safer default)
            return True
        finally:
            if container is not None:
                container.close()

    def _finalize_segment(self) -> tuple[str, str, list[str]]:
        """
        Finalize current segment by renaming files and folder.

        Renames video files to simple names, checks audio threshold and
        renames/deletes accordingly, then renames draft folder to final name.

        Returns:
            Tuple of (date_part, segment_key, saved_files)
        """
        # Get timestamp parts for this window and calculate duration
        date_part, time_part = self.get_timestamp_parts(self.start_at)
        duration = int(time.time() - self.start_at)
        day_dir = day_path(date_part)
        segment_key = f"{time_part}_{duration}"

        saved_files: list[str] = []

        # Rename video files to simple names in draft folder
        for display in self.current_displays:
            if os.path.exists(display.file_path):
                # Simple name: position_displayID_screen.mov
                simple_name = f"{display.position}_{display.display_id}_screen.mov"
                simple_path = Path(self.draft_dir) / simple_name
                try:
                    os.rename(display.file_path, simple_path)
                    saved_files.append(simple_name)
                except OSError as e:
                    logger.error(f"Failed to rename {display.file_path}: {e}")

        # Check audio threshold and rename if passing
        if self.current_audio and os.path.exists(self.current_audio.file_path):
            if self._check_audio_threshold(self.current_audio.file_path):
                simple_name = "audio.m4a"
                simple_path = Path(self.draft_dir) / simple_name
                try:
                    os.rename(self.current_audio.file_path, simple_path)
                    saved_files.append(simple_name)
                    logger.info(f"Audio passed threshold check, saving: {simple_name}")
                except OSError as e:
                    logger.error(f"Failed to rename audio: {e}")
            else:
                # Delete the audio file
                try:
                    os.remove(self.current_audio.file_path)
                    logger.info("Audio below threshold, discarded")
                except OSError as e:
                    logger.warning(f"Failed to remove audio file: {e}")

        # Clear capture state
        self.current_displays = []
        self.current_audio = None

        # Rename draft folder to final segment name (atomic handoff)
        if self.draft_dir and saved_files:
            final_segment_dir = str(day_dir / segment_key)
            try:
                os.rename(self.draft_dir, final_segment_dir)
                logger.info(f"Segment finalized: {segment_key}")
            except OSError as e:
                logger.error(f"Failed to rename draft folder: {e}")
                saved_files = []  # Don't emit event if rename failed
        elif self.draft_dir:
            # No files to save, remove empty draft folder
            try:
                os.rmdir(self.draft_dir)
                logger.debug(f"Removed empty draft folder: {self.draft_dir}")
            except OSError:
                pass  # May have other files, ignore

        self.draft_dir = None

        return date_part, segment_key, saved_files

    def handle_boundary(self, is_active: bool):
        """
        Handle window boundary rollover.

        Closes the current draft folder, renames files to simple names,
        renames folder to final segment name, and emits the observing event.

        Args:
            is_active: Whether system is currently active
        """
        date_part = ""
        segment_key = ""
        saved_files: list[str] = []

        if self.capture_running:
            logger.info("Stopping previous capture")
            self.screencapture.stop()
            self.capture_running = False

            # Finalize segment (rename files and folder)
            date_part, segment_key, saved_files = self._finalize_segment()

        # Reset timing for new window
        self.start_at = time.time()
        self.start_at_mono = time.monotonic()

        # Update segment mute state
        self.segment_is_muted = self.cached_is_muted

        # Start new capture if active and screen not locked (creates new draft folder)
        if is_active and not self.cached_screen_locked:
            self.initialize_capture()

        # Emit observing event with saved files
        if saved_files and self.callosum:
            self.callosum.emit(
                "observe",
                "observing",
                day=date_part,
                segment=segment_key,
                files=saved_files,
                host=HOST,
                platform=PLATFORM,
            )
            logger.info(f"Segment observing: {segment_key} ({len(saved_files)} files)")

    def _create_draft_folder(self) -> str:
        """
        Create a draft folder for the current segment.

        Returns:
            Path to the draft folder (YYYYMMDD/HHMMSS_draft/)
        """
        date_part, time_part = self.get_timestamp_parts(self.start_at)
        day_dir = day_path(date_part)

        # Create draft folder: YYYYMMDD/HHMMSS_draft/
        draft_name = f"{time_part}_draft"
        draft_path = str(day_dir / draft_name)
        os.makedirs(draft_path, exist_ok=True)

        self.draft_dir = draft_path
        logger.debug(f"Created draft folder: {draft_path}")
        return draft_path

    def initialize_capture(self) -> bool:
        """
        Start a new screencast and audio recording.

        Creates a draft folder and starts sck-cli recording.

        Returns:
            True if capture started successfully, False otherwise
        """

        # Create draft folder for this segment
        draft_path = self._create_draft_folder()

        # Build output base for sck-cli (inside draft folder)
        # sck-cli will create files like: draft/capture_1.mov, draft/capture.m4a
        output_base = Path(draft_path) / "capture"

        try:
            displays, audio = self.screencapture.start(
                output_base, self.interval, frame_rate=1.0
            )
        except RuntimeError as e:
            logger.error(f"Failed to start capture: {e}")
            return False

        self.current_displays = displays
        self.current_audio = audio
        self.capture_running = True

        logger.info(f"Started capture with {len(displays)} display(s)")
        for display in displays:
            logger.info(
                f"  Display {display.display_id}: {display.position} -> {display.file_path}"
            )
        if audio:
            logger.info(f"  Audio: {audio.file_path}")

        return True

    def emit_status(self):
        """Emit observe.status event with current state.

        Event structure matches Linux observer for compatibility:
        - mode: "screencast" or "idle" (macOS doesn't have tmux mode)
        - screencast: recording status and display info
        - tmux: always empty (not supported on macOS)
        - audio: always empty (macOS checks threshold at boundary, not real-time)
        - activity: system activity status
        """
        if not self.callosum:
            return

        journal_path = os.getenv("JOURNAL_PATH", "")

        # Determine mode (macOS is binary: screencast or idle)
        mode = "screencast" if self.capture_running else "idle"

        # Build screencast info (matches Linux observer structure)
        if self.capture_running and self.current_displays:
            elapsed = int(time.monotonic() - self.start_at_mono)
            streams_info = []
            for display in self.current_displays:
                try:
                    rel_file = (
                        os.path.relpath(display.file_path, journal_path)
                        if journal_path
                        else display.file_path
                    )
                except ValueError:
                    rel_file = display.file_path

                streams_info.append(
                    {
                        "position": display.position,
                        "connector": str(display.display_id),
                        "file": rel_file,
                    }
                )

            screencast_info = {
                "recording": True,
                "streams": streams_info,
                "window_elapsed_seconds": elapsed,
            }
        else:
            screencast_info = {"recording": False}

        # Tmux info (not supported on macOS)
        tmux_info = {"capturing": False}

        # Audio info (macOS checks threshold at boundary, not real-time)
        audio_info = {
            "threshold_hits": 0,
            "will_save": False,
        }

        # Activity info
        activity_info = {
            "active": self.cached_is_active,
            "idle_time_ms": self.cached_idle_time_ms,
            "screen_locked": self.cached_screen_locked,
            "power_save": self.cached_power_save,
            "sink_muted": self.cached_is_muted,
        }

        self.callosum.emit(
            "observe",
            "status",
            mode=mode,
            screencast=screencast_info,
            tmux=tmux_info,
            audio=audio_info,
            activity=activity_info,
            host=HOST,
            platform=PLATFORM,
        )

    async def main_loop(self):
        """Run the main observer loop."""
        logger.info(f"Starting observer loop (interval={self.interval}s)")

        # Check initial activity and start capture if active
        is_active = self.check_activity_status()
        self.segment_is_muted = self.cached_is_muted

        if is_active and not self.cached_screen_locked:
            if not self.initialize_capture():
                logger.error("Failed to start initial capture")
                self.running = False
                return

        while self.running:
            # Sleep for chunk duration
            await asyncio.sleep(CHUNK_DURATION)

            # Check activity status
            is_active = self.check_activity_status()

            # Check if sck-cli process died unexpectedly
            if self.capture_running and not self.screencapture.is_running():
                logger.warning("Capture process died, handling boundary")
                self.handle_boundary(is_active)
                continue

            # Detect activation edge (idle -> active transition)
            activation_edge = is_active and not self.capture_running

            # Detect mute state transition
            mute_transition = self.cached_is_muted != self.segment_is_muted
            if mute_transition:
                logger.info(
                    f"Mute state changed: "
                    f"{'muted' if self.segment_is_muted else 'unmuted'} -> "
                    f"{'muted' if self.cached_is_muted else 'unmuted'}"
                )

            # Check for window boundary (use monotonic to avoid DST/clock jumps)
            now_mono = time.monotonic()
            elapsed = now_mono - self.start_at_mono
            is_boundary = (
                (elapsed >= self.interval) or activation_edge or mute_transition
            )

            if is_boundary:
                logger.info(
                    f"Boundary: elapsed={elapsed:.1f}s edge={activation_edge} "
                    f"mute_change={mute_transition}"
                )
                self.handle_boundary(is_active)

            # Emit status event
            self.emit_status()

        # Cleanup on exit
        logger.info("Observer loop stopped, cleaning up...")
        await self.shutdown()

    async def shutdown(self):
        """Clean shutdown of observer."""

        # Stop capture if running
        if self.capture_running:
            logger.info("Stopping capture for shutdown")
            self.screencapture.stop()
            self.capture_running = False

            # Brief delay for files to be written
            await asyncio.sleep(0.5)

            # Finalize segment (rename files and folder)
            date_part, segment_key, saved_files = self._finalize_segment()

            # Emit observing event for final segment
            if saved_files and self.callosum:
                self.callosum.emit(
                    "observe",
                    "observing",
                    day=date_part,
                    segment=segment_key,
                    files=saved_files,
                    host=HOST,
                    platform=PLATFORM,
                )

        # Stop Callosum connection
        if self.callosum:
            self.callosum.stop()
            logger.info("Callosum connection stopped")

        logger.info("Shutdown complete")


async def async_main(args):
    """Async entry point."""
    observer = MacOSObserver(
        interval=args.interval,
        sck_cli_path=args.sck_cli_path,
    )

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
