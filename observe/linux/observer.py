#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""
Unified observer for audio and screencast capture.

Continuously captures audio and manages screencast/tmux recording based on activity.
Creates 5-minute windows, saving audio if voice activity detected and recording
screencasts during active segments. When screen is idle but tmux sessions are active,
captures tmux terminal content instead.

State machine:
    SCREENCAST: Screen is active, recording video
    TMUX: Screen is idle but tmux has recent activity
    IDLE: Both screen and tmux are inactive
"""

import argparse
import asyncio
import datetime
import logging
import os
import signal
import socket
import sys
import time

import numpy as np
from dbus_next.aio import MessageBus
from dbus_next.constants import BusType

from observe.gnome.activity import (
    get_idle_time_ms,
    is_power_save_active,
    is_screen_locked,
)
from observe.hear import AudioRecorder
from observe.linux.audio import is_sink_muted
from observe.linux.screencast import Screencaster, StreamInfo
from observe.remote import RemoteClient
from observe.tmux.capture import TmuxCapture, write_captures_jsonl
from think.callosum import CallosumConnection
from think.utils import day_path, setup_cli

logger = logging.getLogger(__name__)

# Constants
IDLE_THRESHOLD_MS = 5 * 60 * 1000  # 5 minutes
RMS_THRESHOLD = 0.01
MIN_HITS_FOR_SAVE = 3
CHUNK_DURATION = 5  # seconds

# Host identification for multi-host scenarios
_HOST = socket.gethostname()
_PLATFORM = "linux"

# Capture modes
MODE_IDLE = "idle"
MODE_SCREENCAST = "screencast"
MODE_TMUX = "tmux"


class Observer:
    """Unified audio and screencast/tmux observer."""

    def __init__(self, interval: int = 300, remote_url: str | None = None):
        self.interval = interval
        self.remote_url = remote_url
        self.audio_recorder = AudioRecorder()
        self.screencaster = Screencaster()
        self.tmux_capture = TmuxCapture()
        self.bus: MessageBus | None = None
        self.running = True
        self.callosum: CallosumConnection | None = None
        self.remote_client: RemoteClient | None = None

        # State tracking
        self.start_at = time.time()  # Wall-clock for filenames
        self.start_at_mono = time.monotonic()  # Monotonic for elapsed calculations
        self.threshold_hits = 0
        self.accumulated_audio_buffer = np.array([], dtype=np.float32).reshape(0, 2)

        # Mode tracking (replaces screencast_running boolean)
        self.current_mode = MODE_IDLE

        # Multi-file screencast tracking
        self.current_streams: list[StreamInfo] = []
        self.pending_finalizations: list[tuple[str, str]] | None = None
        self.last_screencast_sizes: dict[str, int] = {}

        # Tmux capture tracking
        self.tmux_captures: list[dict] = []
        self.tmux_capture_id = 0
        self.tmux_sessions_seen: set[str] = set()

        # Activity status cache (updated each loop)
        self.cached_is_active = False
        self.cached_idle_time_ms = 0
        self.cached_screen_locked = False
        self.cached_is_muted = False
        self.cached_power_save = False
        self.cached_tmux_active = False

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

        # Check tmux availability
        if self.tmux_capture.is_available():
            logger.info("Tmux available for fallback capture")
        else:
            logger.info("Tmux not available (will only use screencast)")

        # Start Callosum connection for status events (or remote client)
        if self.remote_url:
            self.remote_client = RemoteClient(self.remote_url)
            self.remote_client.start()
            logger.info(f"Remote client started: {self.remote_url[:50]}...")
        else:
            self.callosum = CallosumConnection()
            self.callosum.start()
            logger.info("Callosum connection started")

        return True

    async def check_activity_status(self) -> str:
        """
        Check system activity status and determine capture mode.

        Returns:
            Capture mode: MODE_SCREENCAST, MODE_TMUX, or MODE_IDLE
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

        # Determine screen activity
        screen_idle = (idle_time > IDLE_THRESHOLD_MS) or screen_locked or power_save
        screen_active = not screen_idle

        # Check tmux activity (only if screen is idle)
        if screen_active:
            tmux_active = False
        else:
            tmux_active = self.tmux_capture.is_active(poll_interval=CHUNK_DURATION)
        self.cached_tmux_active = tmux_active

        # Determine mode with priority: screen > tmux > idle
        if screen_active:
            mode = MODE_SCREENCAST
        elif tmux_active:
            mode = MODE_TMUX
        else:
            mode = MODE_IDLE

        # Cache legacy is_active for audio threshold logic
        has_audio_activity = self.threshold_hits >= MIN_HITS_FOR_SAVE
        self.cached_is_active = screen_active or tmux_active or has_audio_activity

        return mode

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

    async def handle_boundary(self, new_mode: str):
        """
        Handle window boundary rollover.

        Args:
            new_mode: The mode for the new segment
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

        # Handle screencast rollover (if we were in screencast mode)
        stopped_streams: list[StreamInfo] = []
        screen_files: list[str] = []

        if self.current_mode == MODE_SCREENCAST:
            logger.info("Stopping previous screencast")
            stopped_streams = await self.screencaster.stop()
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

        # Handle tmux capture save (if we were in tmux mode)
        tmux_files: list[str] = []
        if self.current_mode == MODE_TMUX and self.tmux_captures:
            segment_key = f"{time_part}_{duration}"
            segment_dir = day_dir / segment_key
            tmux_files = write_captures_jsonl(self.tmux_captures, segment_dir)

        # Reset tmux state
        self.tmux_captures = []
        self.tmux_capture_id = 0
        self.tmux_sessions_seen = set()
        self.tmux_capture.reset_hashes()

        # Reset timing for new window
        self.start_at = time.time()  # Wall-clock for filenames
        self.start_at_mono = time.monotonic()  # Monotonic for elapsed

        # Update segment mute state for new segment
        self.segment_is_muted = self.cached_is_muted

        # Update mode
        old_mode = self.current_mode
        self.current_mode = new_mode

        # Start new capture based on mode
        if new_mode == MODE_SCREENCAST and not self.cached_screen_locked:
            await self.initialize_screencast()
        # MODE_TMUX doesn't need initialization, captures happen in main loop

        logger.info(f"Mode transition: {old_mode} -> {new_mode}")

        # Emit observing event with what we saved this boundary
        files = audio_files + screen_files + tmux_files

        if files:
            segment = f"{time_part}_{duration}"

            if self.remote_client:
                # Remote mode: upload files to remote server
                file_paths = [day_dir / f for f in files]
                if self.remote_client.upload_and_cleanup(
                    date_part, segment, file_paths
                ):
                    logger.info(f"Segment uploaded: {segment} ({len(files)} files)")
                else:
                    logger.error(
                        f"Segment upload failed: {segment} - files kept locally"
                    )
            elif self.callosum:
                # Local mode: emit to local Callosum
                self.callosum.emit(
                    "observe",
                    "observing",
                    day=date_part,
                    segment=segment,
                    files=files,
                    host=_HOST,
                    platform=_PLATFORM,
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

        self.current_streams = streams
        self.last_screencast_sizes = {s.temp_path: 0 for s in streams}

        logger.info(f"Started screencast with {len(streams)} stream(s)")
        for stream in streams:
            logger.info(f"  {stream.position} ({stream.connector}): {stream.temp_path}")

        return True

    def capture_tmux(self):
        """Poll tmux and accumulate captures for this chunk."""
        active_sessions = self.tmux_capture.get_active_sessions(CHUNK_DURATION)
        if not active_sessions:
            return

        ts = time.time()

        for session_info in active_sessions:
            session = session_info["session"]
            self.tmux_sessions_seen.add(session)

            result = self.tmux_capture.capture_changed(session)
            if not result:
                continue

            self.tmux_capture_id += 1
            relative_ts = ts - self.start_at
            capture_dict = self.tmux_capture.result_to_dict(
                result, self.tmux_capture_id, relative_ts
            )
            self.tmux_captures.append(capture_dict)
            logger.debug(f"Captured tmux session {session}: {len(result.panes)} panes")

    def emit_status(self):
        """Emit observe.status event with current state."""
        journal_path = os.getenv("JOURNAL_PATH", "")
        elapsed = int(time.monotonic() - self.start_at_mono)

        # Calculate screencast info
        if self.current_mode == MODE_SCREENCAST and self.current_streams:
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

        # Calculate tmux info
        if self.current_mode == MODE_TMUX:
            tmux_info = {
                "capturing": True,
                "captures": len(self.tmux_captures),
                "sessions": sorted(self.tmux_sessions_seen),
                "window_elapsed_seconds": elapsed,
            }
        else:
            tmux_info = {"capturing": False}

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
            "tmux_active": self.cached_tmux_active,
        }

        # Emit to remote or local Callosum
        if self.remote_client:
            self.remote_client.emit(
                "observe",
                "status",
                mode=self.current_mode,
                screencast=screencast_info,
                tmux=tmux_info,
                audio=audio_info,
                activity=activity_info,
                host=_HOST,
                platform=_PLATFORM,
            )
        elif self.callosum:
            self.callosum.emit(
                "observe",
                "status",
                mode=self.current_mode,
                screencast=screencast_info,
                tmux=tmux_info,
                audio=audio_info,
                activity=activity_info,
                host=_HOST,
                platform=_PLATFORM,
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

        # Determine initial mode
        new_mode = await self.check_activity_status()
        self.segment_is_muted = self.cached_is_muted  # Sync initial mute state
        self.current_mode = new_mode

        # Start initial capture based on mode
        if new_mode == MODE_SCREENCAST and not self.cached_screen_locked:
            try:
                await self.initialize_screencast()
            except RuntimeError:
                # Failed to start screencast, exit
                self.running = False
                return

        logger.info(f"Initial mode: {self.current_mode}")

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

            # Check activity status and determine new mode
            new_mode = await self.check_activity_status()

            # Check for GStreamer failure mid-recording
            if (
                self.current_mode == MODE_SCREENCAST
                and not self.screencaster.is_healthy()
            ):
                logger.warning("Screencast recording failed, stopping gracefully")
                stopped_streams = await self.screencaster.stop()

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
                # Force recalculate mode without screencast
                self.current_mode = MODE_IDLE

            # Detect mode transition
            mode_transition = new_mode != self.current_mode
            if mode_transition:
                logger.info(f"Mode changing: {self.current_mode} -> {new_mode}")

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

            # Capture tmux if in tmux mode
            if self.current_mode == MODE_TMUX:
                self.capture_tmux()

            # Check for window boundary (use monotonic to avoid DST/clock jumps)
            now_mono = time.monotonic()
            elapsed = now_mono - self.start_at_mono
            is_boundary = (
                (elapsed >= self.interval) or mode_transition or mute_transition
            )

            if is_boundary:
                logger.info(
                    f"Boundary: elapsed={elapsed:.1f}s mode_change={mode_transition} "
                    f"mute_change={mute_transition} "
                    f"hits={self.threshold_hits}/{MIN_HITS_FOR_SAVE}"
                )
                await self.handle_boundary(new_mode)

            # Check if screencast files are actively growing (for health reporting)
            if self.current_mode == MODE_SCREENCAST and self.current_streams:
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
        # Get timestamp parts for final save
        date_part, time_part = self.get_timestamp_parts(self.start_at)
        duration = int(time.time() - self.start_at)
        day_dir = day_path(date_part)

        # Save final audio if threshold met
        if self.threshold_hits >= MIN_HITS_FOR_SAVE:
            audio_files = self._save_audio_segment(
                day_dir, time_part, duration, self.segment_is_muted
            )
            if audio_files:
                logger.info(f"Saved final audio: {len(audio_files)} file(s)")

        # Stop screencast if running
        if self.current_mode == MODE_SCREENCAST:
            logger.info("Stopping screencast for shutdown")
            stopped_streams = await self.screencaster.stop()

            if stopped_streams:
                # Brief delay for files to be written
                await asyncio.sleep(0.5)

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

        # Save tmux captures if in tmux mode
        if self.current_mode == MODE_TMUX and self.tmux_captures:
            segment_key = f"{time_part}_{duration}"
            segment_dir = day_dir / segment_key
            tmux_files = write_captures_jsonl(self.tmux_captures, segment_dir)
            if tmux_files:
                logger.info(f"Saved final tmux captures: {len(tmux_files)} file(s)")

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

        # Stop Callosum or remote client
        if self.remote_client:
            self.remote_client.stop()
            logger.info("Remote client stopped")
        elif self.callosum:
            self.callosum.stop()
            logger.info("Callosum connection stopped")


async def async_main(args):
    """Async entry point."""
    observer = Observer(
        interval=args.interval,
        remote_url=getattr(args, "remote", None),
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
        description="Unified audio, screencast, and tmux observer for journaling."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Duration per screencast window in seconds (default: 300 = 5 minutes).",
    )
    parser.add_argument(
        "--remote",
        type=str,
        help="Remote server URL for uploading segments (e.g., https://server:5000/app/remote/ingest/KEY)",
    )
    args = setup_cli(parser)

    # Verify journal path exists
    journal = os.getenv("JOURNAL_PATH")
    if not journal or not os.path.exists(journal):
        logger.error(f"JOURNAL_PATH not set or does not exist: {journal}")
        sys.exit(1)

    # Log remote mode if enabled
    if args.remote:
        logger.info(f"Remote mode enabled: {args.remote[:50]}...")

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
