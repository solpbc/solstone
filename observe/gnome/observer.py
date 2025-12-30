#!/usr/bin/env python3
"""
Unified observer for audio and screencast capture.

Continuously captures audio and manages screencast recording based on activity.
Creates 5-minute windows, saving audio if voice activity detected and recording
screencasts during active segments. Each monitor is recorded as a separate file.
"""

import argparse
import asyncio
import datetime
import logging
import os
import signal
import sys
import time

import numpy as np
from dbus_next.aio import MessageBus
from dbus_next.constants import BusType

from observe.gnome.dbus import (
    get_idle_time_ms,
    is_power_save_active,
    is_screen_locked,
    is_sink_muted,
)
from observe.gnome.screencast import Screencaster, StreamInfo
from observe.hear import AudioRecorder
from think.callosum import CallosumConnection
from think.utils import day_path, setup_cli

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

        # Multi-file screencast tracking
        self.current_streams: list[StreamInfo] = []
        self.pending_finalizations: list[tuple[str, str]] | None = None
        self.last_screencast_sizes: dict[str, int] = {}

        # Activity status cache (updated each loop)
        self.cached_is_active = False
        self.cached_idle_time_ms = 0
        self.cached_screen_locked = False
        self.cached_is_muted = False
        self.cached_power_save = False

        # Mute state at segment start (determines save format)
        self.segment_is_muted = False

        # Health tracking - whether screencast files are actively growing
        self.files_growing = False

    async def setup(self):
        """Initialize audio devices and DBus connection."""
        # Detect and start audio recorder
        if not self.audio_recorder.detect():
            logger.error("Failed to detect audio devices")
            return False

        self.audio_recorder.start_recording()
        logger.info("Audio recording started")

        # Connect to DBus for idle/lock detection
        self.bus = await MessageBus(bus_type=BusType.SESSION).connect()
        logger.info("DBus connection established")

        # Verify portal is available (exit if not)
        if not await self.screencaster.connect():
            logger.error("Screencast portal not available")
            return False
        logger.info("Screencast portal connected")

        # Start Callosum connection for status events
        self.callosum = CallosumConnection()
        self.callosum.start()
        logger.info("Callosum connection started")

        return True

    async def check_activity_status(self) -> bool:
        """
        Check system activity status and cache values.

        Returns:
            True if user is active (not idle/locked/power-save, OR has audio activity)
        """
        idle_time = await get_idle_time_ms(self.bus)
        screen_locked = await is_screen_locked(self.bus)
        power_save = await is_power_save_active(self.bus)
        sink_muted = await is_sink_muted()

        # Cache values for status events
        self.cached_idle_time_ms = idle_time
        self.cached_screen_locked = screen_locked
        self.cached_is_muted = sink_muted
        self.cached_power_save = power_save

        is_idle = (idle_time > IDLE_THRESHOLD_MS) or screen_locked or power_save
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

    def _save_audio_segment(
        self, day_dir, time_part: str, duration: int, is_muted: bool
    ) -> list[str]:
        """
        Save accumulated audio buffer to disk.

        Args:
            day_dir: Path to the day directory
            time_part: Timestamp string (HHMMSS)
            duration: Segment duration in seconds
            is_muted: Whether to save as split mono files (muted) or stereo (unmuted)

        Returns:
            List of saved filenames (empty if nothing saved)
        """
        if self.accumulated_audio_buffer.size == 0:
            logger.warning("No audio buffer to save")
            return []

        if is_muted:
            # Split mode: save mic and sys as separate mono files
            mic_data = self.accumulated_audio_buffer[:, 0]
            sys_data = self.accumulated_audio_buffer[:, 1]

            mic_bytes = self.audio_recorder.create_mono_flac_bytes(mic_data)
            sys_bytes = self.audio_recorder.create_mono_flac_bytes(sys_data)

            mic_name = f"{time_part}_{duration}_mic_audio.flac"
            sys_name = f"{time_part}_{duration}_sys_audio.flac"

            mic_path = day_dir / mic_name
            sys_path = day_dir / sys_name

            with open(mic_path, "wb") as f:
                f.write(mic_bytes)
            with open(sys_path, "wb") as f:
                f.write(sys_bytes)

            logger.info(f"Saved split audio (muted): {mic_path}, {sys_path}")
            return [mic_name, sys_name]
        else:
            # Normal mode: save combined stereo file
            flac_bytes = self.audio_recorder.create_flac_bytes(
                self.accumulated_audio_buffer
            )
            audio_name = f"{time_part}_{duration}_audio.flac"
            flac_path = day_dir / audio_name

            with open(flac_path, "wb") as f:
                f.write(flac_bytes)

            logger.info(f"Saved audio to {flac_path}")
            return [audio_name]

    async def handle_boundary(self, is_active: bool):
        """
        Handle window boundary rollover.

        Args:
            is_active: Whether system is currently active
        """
        # Get timestamp parts for this window and calculate duration
        date_part, time_part = self.get_timestamp_parts(self.start_at)
        duration = int(time.time() - self.start_at)
        day_dir = day_path(date_part)

        # Save audio if we have enough threshold hits
        did_save_audio = self.threshold_hits >= MIN_HITS_FOR_SAVE
        audio_files: list[str] = []
        if did_save_audio:
            audio_files = self._save_audio_segment(
                day_dir, time_part, duration, self.segment_is_muted
            )
            if audio_files:
                logger.info(
                    f"Saved {len(audio_files)} audio file(s) ({self.threshold_hits} hits)"
                )
        else:
            logger.debug(
                f"Skipping audio save (only {self.threshold_hits}/{MIN_HITS_FOR_SAVE} hits)"
            )

        # Reset audio state
        self.accumulated_audio_buffer = np.array([], dtype=np.float32).reshape(0, 2)
        self.threshold_hits = 0

        # Handle screencast rollover
        stopped_streams: list[StreamInfo] = []
        screen_files: list[str] = []

        if self.screencast_running:
            logger.info("Stopping previous screencast")
            stopped_streams = await self.screencaster.stop()
            self.screencast_running = False
            self.current_streams = []
            self.last_screencast_sizes = {}

            # Build finalization list and file names
            finalizations = []
            for stream in stopped_streams:
                final_name = stream.final_name(time_part, duration)
                final_path = str(day_dir / final_name)
                finalizations.append((stream.temp_path, final_path))
                screen_files.append(final_name)

            if finalizations:
                self.pending_finalizations = finalizations

        # Reset timing for new window
        self.start_at = time.time()  # Wall-clock for filenames
        self.start_at_mono = time.monotonic()  # Monotonic for elapsed

        # Update segment mute state for new segment
        self.segment_is_muted = self.cached_is_muted

        # Start new screencast if active AND screen not locked
        # (is_active can be True due to audio activity even when locked)
        if is_active and not self.cached_screen_locked:
            await self.initialize_screencast()

        # Emit observing event with what we saved this boundary
        files = audio_files + screen_files

        if files:
            segment = f"{time_part}_{duration}"
            self.callosum.emit(
                "observe",
                "observing",
                segment=segment,
                files=files,
            )
            logger.info(f"Segment observing: {segment} ({len(files)} files)")

    async def initialize_screencast(self) -> bool:
        """
        Start a new screencast recording.

        Returns:
            True if screencast started successfully, False otherwise.

        Raises:
            RuntimeError: If recording fails to start (caller should exit).
        """
        date_part, time_part = self.get_timestamp_parts(self.start_at)
        day_dir = day_path(date_part)

        try:
            streams = await self.screencaster.start(
                str(day_dir), time_part, framerate=1, draw_cursor=True
            )
        except RuntimeError as e:
            logger.error(f"Failed to start screencast: {e}")
            raise

        if not streams:
            logger.error("No streams returned from screencast start")
            raise RuntimeError("No streams available")

        self.screencast_running = True
        self.current_streams = streams
        self.last_screencast_sizes = {s.temp_path: 0 for s in streams}

        logger.info(f"Started screencast with {len(streams)} stream(s)")
        for stream in streams:
            logger.info(f"  {stream.position} ({stream.connector}): {stream.temp_path}")

        return True

    def emit_status(self):
        """Emit observe.status event with current state."""
        journal_path = os.getenv("JOURNAL_PATH", "")

        # Calculate screencast info
        if self.screencast_running and self.current_streams:
            elapsed = int(time.monotonic() - self.start_at_mono)
            streams_info = []
            for stream in self.current_streams:
                try:
                    rel_file = (
                        os.path.relpath(stream.temp_path, journal_path)
                        if journal_path
                        else stream.temp_path
                    )
                except ValueError:
                    rel_file = stream.temp_path

                streams_info.append(
                    {
                        "position": stream.position,
                        "connector": stream.connector,
                        "file": rel_file,
                    }
                )

            screencast_info = {
                "recording": True,
                "streams": streams_info,
                "window_elapsed_seconds": elapsed,
                "files_growing": self.files_growing,
            }
        else:
            screencast_info = {"recording": False, "files_growing": False}

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
            "sink_muted": self.cached_is_muted,
            "power_save": self.cached_power_save,
        }

        self.callosum.emit(
            "observe",
            "status",
            screencast=screencast_info,
            audio=audio_info,
            activity=activity_info,
        )

    def finalize_screencast(self, temp_path: str, final_path: str):
        """
        Rename screencast from temp to final path.

        Args:
            temp_path: Temporary hidden path (.HHMMSS_position_connector.webm)
            final_path: Final destination path (HHMMSS_LEN_position_connector_screen.webm)
        """
        if not os.path.exists(temp_path):
            logger.warning(f"Screencast file not found: {temp_path}")
            return

        try:
            os.replace(temp_path, final_path)
            logger.info(f"Finalized screencast: {final_path}")
        except OSError as e:
            logger.error(f"Failed to rename {temp_path} to {final_path}: {e}")

    async def main_loop(self):
        """Run the main observer loop."""
        logger.info(f"Starting observer loop (interval={self.interval}s)")

        # Start screencast immediately if active and screen not locked
        is_active = await self.check_activity_status()
        self.segment_is_muted = self.cached_is_muted  # Sync initial mute state
        if is_active and not self.cached_screen_locked:
            try:
                await self.initialize_screencast()
            except RuntimeError:
                # Failed to start screencast, exit
                self.running = False
                return

        while self.running:
            # Sleep for chunk duration
            await asyncio.sleep(CHUNK_DURATION)

            # Process pending screencast finalizations
            if self.pending_finalizations:
                for temp_path, final_path in self.pending_finalizations:
                    if os.path.exists(temp_path):
                        self.finalize_screencast(temp_path, final_path)
                    else:
                        logger.warning(f"Pending screencast not found: {temp_path}")
                self.pending_finalizations = None

            # Check activity status
            is_active = await self.check_activity_status()

            # Check for GStreamer failure mid-recording
            if self.screencast_running and not self.screencaster.is_healthy():
                logger.warning("Screencast recording failed, stopping gracefully")
                stopped_streams = await self.screencaster.stop()
                self.screencast_running = False

                # Finalize whatever we have
                if stopped_streams:
                    date_part, time_part = self.get_timestamp_parts(self.start_at)
                    duration = int(time.time() - self.start_at)
                    day_dir = day_path(date_part)

                    for stream in stopped_streams:
                        if os.path.exists(stream.temp_path):
                            final_path = str(
                                day_dir / stream.final_name(time_part, duration)
                            )
                            self.finalize_screencast(stream.temp_path, final_path)

                self.current_streams = []
                self.last_screencast_sizes = {}

            # Transition from idle to active
            activation_edge = is_active and not self.screencast_running

            # Detect mute state transition
            mute_transition = self.cached_is_muted != self.segment_is_muted
            if mute_transition:
                logger.info(
                    f"Mute state changed: "
                    f"{'muted' if self.segment_is_muted else 'unmuted'} -> "
                    f"{'muted' if self.cached_is_muted else 'unmuted'}"
                )

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
            is_boundary = (
                (elapsed >= self.interval) or activation_edge or mute_transition
            )

            if is_boundary:
                logger.info(
                    f"Boundary: elapsed={elapsed:.1f}s edge={activation_edge} "
                    f"mute_change={mute_transition} "
                    f"hits={self.threshold_hits}/{MIN_HITS_FOR_SAVE}"
                )
                await self.handle_boundary(is_active)

            # Check if screencast files are actively growing (for health reporting)
            if self.screencast_running and self.current_streams:
                any_growing = False
                for stream in self.current_streams:
                    if os.path.exists(stream.temp_path):
                        current_size = os.path.getsize(stream.temp_path)
                        last_size = self.last_screencast_sizes.get(stream.temp_path, 0)
                        if current_size > last_size:
                            any_growing = True
                            self.last_screencast_sizes[stream.temp_path] = current_size
                self.files_growing = any_growing
            else:
                self.files_growing = False

            # Emit status event (supervisor derives health from this)
            self.emit_status()

        # Cleanup on exit
        logger.info("Observer loop stopped, cleaning up...")
        await self.shutdown()

    async def shutdown(self):
        """Clean shutdown of observer."""
        # Perform final boundary logic without restarting screencast
        if self.threshold_hits >= MIN_HITS_FOR_SAVE:
            date_part, time_part = self.get_timestamp_parts(self.start_at)
            duration = int(time.time() - self.start_at)
            day_dir = day_path(date_part)

            audio_files = self._save_audio_segment(
                day_dir, time_part, duration, self.segment_is_muted
            )
            if audio_files:
                logger.info(f"Saved final audio: {len(audio_files)} file(s)")

        # Stop screencast if running
        if self.screencast_running:
            logger.info("Stopping screencast for shutdown")
            stopped_streams = await self.screencaster.stop()

            if stopped_streams:
                # Brief delay for files to be written
                await asyncio.sleep(0.5)

                duration = int(time.time() - self.start_at)
                date_part, time_part = self.get_timestamp_parts(self.start_at)
                day_dir = day_path(date_part)

                for stream in stopped_streams:
                    if os.path.exists(stream.temp_path):
                        final_path = str(
                            day_dir / stream.final_name(time_part, duration)
                        )
                        self.finalize_screencast(stream.temp_path, final_path)
                    else:
                        logger.warning(
                            f"Screencast file not found after shutdown: {stream.temp_path}"
                        )

            self.screencast_running = False

        # Process any remaining pending finalizations
        if self.pending_finalizations:
            await asyncio.sleep(0.5)
            for temp_path, final_path in self.pending_finalizations:
                if os.path.exists(temp_path):
                    self.finalize_screencast(temp_path, final_path)
                else:
                    logger.warning(
                        f"Pending screencast not found after shutdown: {temp_path}"
                    )
            self.pending_finalizations = None

        # Stop audio recorder
        self.audio_recorder.stop_recording()
        logger.info("Audio recording stopped")

        # Stop Callosum connection
        if self.callosum:
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
    except RuntimeError as e:
        logger.error(f"Observer runtime error: {e}")
        return 1
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
