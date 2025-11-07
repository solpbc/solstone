#!/usr/bin/env python3
"""
Unified observer for audio and screencast capture.

Continuously captures audio and manages screencast recording based on activity.
Creates 5-minute windows, saving audio if voice activity detected and recording
screencasts during active periods.
"""

import argparse
import asyncio
import datetime
import logging
import os
import signal
import subprocess
import sys
import time

import numpy as np
from dbus_next.aio import MessageBus
from dbus_next.constants import BusType

from observe.gnome.dbus import (
    get_idle_time_ms,
    get_monitor_geometries,
    is_screen_locked,
)
from observe.gnome.screencast import Screencaster
from observe.hear import AudioRecorder
from think.callosum import CallosumConnection
from think.utils import day_path, setup_cli, touch_health

logger = logging.getLogger(__name__)

# Constants
IDLE_THRESHOLD_MS = 5 * 60 * 1000  # 5 minutes
RMS_THRESHOLD = 0.01
MIN_HITS_FOR_SAVE = 3
CHUNK_DURATION = 5  # seconds


class Observer:
    """Unified audio and screencast observer."""

    def __init__(self, interval: int = 300):
        self.interval = interval
        self.audio_recorder = AudioRecorder()
        self.screencaster = Screencaster()
        self.bus: MessageBus | None = None
        self.running = True
        self.callosum: CallosumConnection | None = None

        # State tracking
        self.start_at = time.time()  # Wall-clock for filenames
        self.start_at_mono = time.monotonic()  # Monotonic for elapsed calculations
        self.threshold_hits = 0
        self.accumulated_audio_buffer = np.array([], dtype=np.float32).reshape(0, 2)
        self.screencast_running = False
        self.current_screencast_path = None
        self.pending_finalization = None  # Path to screencast awaiting finalization
        self.last_screencast_size = 0  # Track file size for health checks

        # Activity status cache (updated each loop)
        self.cached_is_active = False
        self.cached_idle_time_ms = 0
        self.cached_screen_locked = False

    async def setup(self):
        """Initialize audio devices and DBus connection."""
        # Detect and start audio recorder
        if not self.audio_recorder.detect():
            logger.error("Failed to detect audio devices")
            return False

        self.audio_recorder.start_recording()
        logger.info("Audio recording started")

        # Connect to DBus for screencast
        self.bus = await MessageBus(bus_type=BusType.SESSION).connect()
        await self.screencaster.connect()
        logger.info("DBus connection established")

        # Start Callosum connection for status events
        self.callosum = CallosumConnection()
        self.callosum.start()
        logger.info("Callosum connection started")

        return True

    async def check_activity_status(self) -> bool:
        """
        Check system activity status and cache values.

        Returns:
            True if user is active (not idle and screen unlocked, OR has audio activity)
        """
        idle_time = await get_idle_time_ms(self.bus)
        screen_locked = await is_screen_locked(self.bus)

        # Cache values for status events
        self.cached_idle_time_ms = idle_time
        self.cached_screen_locked = screen_locked

        is_idle = (idle_time > IDLE_THRESHOLD_MS) or screen_locked
        has_audio_activity = self.threshold_hits >= MIN_HITS_FOR_SAVE
        is_active = (not is_idle) or has_audio_activity

        # Cache result
        self.cached_is_active = is_active

        return is_active

    def compute_rms(self, audio_buffer: np.ndarray) -> float:
        """Compute per-channel RMS and return maximum (stereo: mic=left, sys=right)."""
        if audio_buffer.size == 0:
            return 0.0
        # Compute RMS for each channel separately
        rms_left = float(np.sqrt(np.mean(audio_buffer[:, 0] ** 2)))
        rms_right = float(np.sqrt(np.mean(audio_buffer[:, 1] ** 2)))
        return max(rms_left, rms_right)

    def get_timestamp_parts(self, timestamp: float = None) -> tuple[str, str]:
        """
        Get date and time parts from timestamp.

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
        # Save audio if we have enough threshold hits
        if self.threshold_hits >= MIN_HITS_FOR_SAVE:
            if self.accumulated_audio_buffer.size > 0:
                date_part, time_part = self.get_timestamp_parts(self.start_at)
                day_dir = day_path(date_part)

                flac_bytes = self.audio_recorder.create_flac_bytes(
                    self.accumulated_audio_buffer
                )
                flac_path = day_dir / f"{time_part}_raw.flac"
                with open(flac_path, "wb") as f:
                    f.write(flac_bytes)
                logger.info(f"Saved audio to {flac_path} ({self.threshold_hits} hits)")
            else:
                logger.warning("Threshold hits met but no audio buffer")
        else:
            logger.debug(
                f"Skipping audio save (only {self.threshold_hits}/{MIN_HITS_FOR_SAVE} hits)"
            )

        # Reset audio state
        self.accumulated_audio_buffer = np.array([], dtype=np.float32).reshape(0, 2)
        self.threshold_hits = 0

        # Handle screencast rollover
        did_stop = False
        previous_screencast_path = None

        if self.screencast_running:
            logger.info("Stopping previous screencast")
            await self.screencaster.stop()
            previous_screencast_path = self.current_screencast_path
            self.screencast_running = False
            self.current_screencast_path = None
            self.last_screencast_size = 0
            did_stop = True

        # Reset timing for new window
        self.start_at = time.time()  # Wall-clock for filenames
        self.start_at_mono = time.monotonic()  # Monotonic for elapsed

        # Start new screencast if active
        if is_active:
            await self.initialize_screencast()

        # Queue previous screencast for finalization (file may not exist yet)
        if did_stop and previous_screencast_path:
            self.pending_finalization = previous_screencast_path

        # Emit window_complete with what we saved this boundary
        files = []
        if self.threshold_hits >= MIN_HITS_FOR_SAVE:  # We saved audio
            files.append(f"{time_part}_raw.flac")
        if did_stop:  # We stopped a screencast (will be finalized soon)
            files.append(f"{time_part}_screen.webm")

        if files:
            self.callosum.emit(
                "observe",
                "period",
                day=date_part,
                timestamp=time_part,
                files=files,
            )
            logger.info(f"Period complete: {date_part}/{time_part} ({len(files)} files)")

    async def initialize_screencast(self) -> bool:
        """
        Start a new screencast recording.

        Returns:
            True if screencast started successfully, False otherwise
        """
        date_part, time_part = self.get_timestamp_parts(self.start_at)
        day_dir = day_path(date_part)

        # Use _live.webm name (GNOME won't append .webm), rename to _screen.webm later
        live_path = str(day_dir / f"{time_part}_live.webm")
        screencast_path = str(day_dir / f"{time_part}_screen.webm")

        ok, _ = await self.screencaster.start(live_path, framerate=1, draw_cursor=True)
        if ok:
            self.screencast_running = True
            self.current_screencast_path = screencast_path
            self.last_screencast_size = 0
            logger.info(f"Started new screencast: {screencast_path}")
            return True
        else:
            logger.warning("Failed to start screencast")
            return False

    def emit_status(self):
        """Emit observe.status event with current state."""
        # Calculate screencast info
        screencast_info = None
        if self.screencast_running and self.current_screencast_path:
            journal_path = os.getenv("JOURNAL_PATH", "")
            try:
                rel_file = (
                    os.path.relpath(self.current_screencast_path, journal_path)
                    if journal_path
                    else self.current_screencast_path
                )
            except ValueError:
                rel_file = self.current_screencast_path

            elapsed = int(time.monotonic() - self.start_at_mono)
            screencast_info = {
                "recording": True,
                "file": rel_file,
                "window_elapsed_seconds": elapsed,
            }
        else:
            screencast_info = {"recording": False}

        # Audio info
        audio_info = {
            "threshold_hits": self.threshold_hits,
            "will_save": self.threshold_hits >= MIN_HITS_FOR_SAVE,
        }

        # Activity info
        activity_info = {
            "active": self.cached_is_active,
            "idle_time_ms": self.cached_idle_time_ms,
            "screen_locked": self.cached_screen_locked,
        }

        self.callosum.emit(
            "observe",
            "status",
            screencast=screencast_info,
            audio=audio_info,
            activity=activity_info,
        )

    async def finalize_screencast(self, screencast_path: str):
        """
        Add monitor metadata to screencast and rename from _live.webm to _screen.webm.

        Args:
            screencast_path: Final destination path (_screen.webm)
        """
        live_path = screencast_path.replace("_screen.webm", "_live.webm")

        if not os.path.exists(live_path):
            logger.warning(f"Screencast file not found: {live_path}")
            return

        # Build monitor geometry metadata
        try:
            geometries = get_monitor_geometries()
            title_parts = []
            for geom_info in geometries:
                x1, y1, x2, y2 = geom_info["box"]
                title_parts.append(
                    f"{geom_info['id']}:{geom_info['position']},{x1},{y1},{x2},{y2}"
                )
            title = " ".join(title_parts)

            # Update video title with monitor dimensions
            subprocess.run(
                [
                    "mkvpropedit",
                    live_path,
                    "--edit",
                    "info",
                    "--set",
                    f"title={title}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.debug(f"Added monitor metadata to {live_path}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to update video title: {e.stderr}")
        except FileNotFoundError:
            logger.warning("mkvpropedit not found, skipping title update")
        except Exception as e:
            logger.warning(f"Error adding monitor metadata: {e}")

        # Atomically rename to final destination
        try:
            os.replace(live_path, screencast_path)
            logger.info(f"Finalized screencast: {screencast_path}")
        except OSError as e:
            logger.error(f"Failed to rename {live_path} to {screencast_path}: {e}")

    async def main_loop(self):
        """Run the main observer loop."""
        logger.info(f"Starting observer loop (interval={self.interval}s)")

        # Start screencast immediately if active
        is_active = await self.check_activity_status()
        if is_active:
            await self.initialize_screencast()

        while self.running:
            # Sleep for chunk duration
            await asyncio.sleep(CHUNK_DURATION)

            # Process pending screencast finalization
            if self.pending_finalization:
                live_path = self.pending_finalization.replace(
                    "_screen.webm", "_live.webm"
                )
                if os.path.exists(live_path):
                    await self.finalize_screencast(self.pending_finalization)
                    self.pending_finalization = None
                else:
                    logger.warning(f"Pending screencast not found: {live_path}")
                    self.pending_finalization = None

            # Check activity status
            is_active = await self.check_activity_status()

            # Transition from idle to active
            activation_edge = is_active and not self.screencast_running

            # Capture audio buffer for this chunk
            audio_chunk = self.audio_recorder.get_buffers()

            if audio_chunk.size > 0:
                # Append to accumulated buffer
                self.accumulated_audio_buffer = np.vstack(
                    (self.accumulated_audio_buffer, audio_chunk)
                )

                # Compute RMS and check threshold
                rms = self.compute_rms(audio_chunk)
                if rms > RMS_THRESHOLD:
                    self.threshold_hits += 1
                    logger.debug(
                        f"RMS {rms:.4f} > threshold (hit {self.threshold_hits})"
                    )
                else:
                    logger.debug(f"RMS {rms:.4f} below threshold")
            else:
                logger.debug("No audio data in chunk")

            # Check for window boundary (use monotonic to avoid DST/clock jumps)
            now_mono = time.monotonic()
            elapsed = now_mono - self.start_at_mono
            is_boundary = (elapsed >= self.interval) or activation_edge

            if is_boundary:
                logger.info(
                    f"Boundary: elapsed={elapsed:.1f}s edge={activation_edge} "
                    f"hits={self.threshold_hits}/{MIN_HITS_FOR_SAVE}"
                )
                await self.handle_boundary(is_active)

            # Touch health for audio processing (always)
            touch_health("hear")

            # Touch health for screencast based on file existence and growth
            if not is_active:
                # Healthy not to record when idle/locked
                touch_health("see")
            elif self.screencast_running and self.current_screencast_path:
                # Check if recording file exists and is growing
                live_path = self.current_screencast_path.replace(
                    "_screen.webm", "_live.webm"
                )
                if os.path.exists(live_path):
                    current_size = os.path.getsize(live_path)
                    if current_size > self.last_screencast_size:
                        touch_health("see")
                        self.last_screencast_size = current_size

            # Emit status event
            self.emit_status()

        # Cleanup on exit
        logger.info("Observer loop stopped, cleaning up...")
        await self.shutdown()

    async def shutdown(self):
        """Clean shutdown of observer."""
        # Perform final boundary logic without restarting screencast
        if (
            self.threshold_hits >= MIN_HITS_FOR_SAVE
            and self.accumulated_audio_buffer.size > 0
        ):
            date_part, time_part = self.get_timestamp_parts(self.start_at)
            day_dir = day_path(date_part)

            flac_bytes = self.audio_recorder.create_flac_bytes(
                self.accumulated_audio_buffer
            )
            flac_path = day_dir / f"{time_part}_raw.flac"
            with open(flac_path, "wb") as f:
                f.write(flac_bytes)
            logger.info(f"Saved final audio to {flac_path}")

        # Stop screencast if running
        if self.screencast_running:
            logger.info("Stopping screencast for shutdown")
            await self.screencaster.stop()
            if self.current_screencast_path:
                # Wait 1s for GNOME Shell to create the file
                await asyncio.sleep(1.0)
                live_path = self.current_screencast_path.replace(
                    "_screen.webm", "_live.webm"
                )
                if os.path.exists(live_path):
                    await self.finalize_screencast(self.current_screencast_path)
                else:
                    logger.warning(
                        f"Screencast file not found after shutdown: {live_path}"
                    )
            self.screencast_running = False

        # Process any remaining pending finalization
        if self.pending_finalization:
            await asyncio.sleep(1.0)
            live_path = self.pending_finalization.replace("_screen.webm", "_live.webm")
            if os.path.exists(live_path):
                await self.finalize_screencast(self.pending_finalization)
            else:
                logger.warning(
                    f"Pending screencast not found after shutdown: {live_path}"
                )

        # Stop audio recorder
        self.audio_recorder.stop_recording()
        logger.info("Audio recording stopped")

        # Stop Callosum connection
        self.callosum.stop()
        logger.info("Callosum connection stopped")


async def async_main(args):
    """Async entry point."""
    observer = Observer(interval=args.interval)

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
        description="Unified audio and screencast observer for journaling."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Duration per screencast window in seconds (default: 300 = 5 minutes).",
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
