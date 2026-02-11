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
import logging
import os
import platform
import signal
import socket
import sys
import time
from pathlib import Path

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
from observe.tmux.capture import TmuxCapture, write_captures_jsonl
from observe.utils import create_draft_folder, get_timestamp_parts
from think.callosum import CallosumConnection
from think.streams import stream_name, update_stream, write_segment_stream
from think.utils import day_path, get_config, get_journal, setup_cli

logger = logging.getLogger(__name__)

# Host identification
HOST = socket.gethostname()
PLATFORM = platform.system().lower()

# Constants
IDLE_THRESHOLD_MS = 5 * 60 * 1000  # 5 minutes
RMS_THRESHOLD = 0.01
MIN_HITS_FOR_SAVE = 3
CHUNK_DURATION = 5  # seconds
TMUX_ACTIVITY_THRESHOLD = 5  # seconds - fixed window for activity detection


# Capture modes
MODE_IDLE = "idle"
MODE_SCREENCAST = "screencast"
MODE_TMUX = "tmux"


class Observer:
    """Unified audio and screencast/tmux observer."""

    def __init__(self, interval: int = 300):
        self.interval = interval
        self.audio_recorder = AudioRecorder()
        self.screencaster = Screencaster()
        self.tmux_capture = TmuxCapture()
        self.bus: MessageBus | None = None
        self.running = True
        self.stream = stream_name(host=HOST)

        # Callosum connection for events
        self._callosum: CallosumConnection | None = None
        self._journal_path: Path | None = None

        # State tracking
        self.start_at = time.time()  # Wall-clock for filenames
        self.start_at_mono = time.monotonic()  # Monotonic for elapsed calculations
        self.threshold_hits = 0
        self.accumulated_audio_buffer = np.array([], dtype=np.float32).reshape(0, 2)

        # Mode tracking (replaces screencast_running boolean)
        self.current_mode = MODE_IDLE

        # Draft folder for current segment (HHMMSS_draft/)
        self.draft_dir: str | None = None

        # Multi-file screencast tracking
        self.current_streams: list[StreamInfo] = []

        # Tmux capture tracking
        self.tmux_captures: list[dict] = []
        self.tmux_capture_id = 0
        self.tmux_sessions_seen: set[str] = set()
        self.last_tmux_capture_time: float = 0  # For capture interval timing
        self.in_tmux_segment: bool = False  # True while tmux segment is open

        # Activity status cache (updated each loop)
        self.cached_is_active = False
        self.cached_idle_time_ms = 0
        self.cached_screen_locked = False
        self.cached_is_muted = False
        self.cached_power_save = False

        # Mute state at segment start (determines save format)
        self.segment_is_muted = False

        # Tmux configuration (loaded from journal config)
        self.tmux_enabled = True
        self.tmux_capture_interval = (
            CHUNK_DURATION  # User-configurable capture frequency
        )

    def _load_tmux_config(self):
        """Load tmux settings from journal config."""
        try:
            config = get_config()
            observe_config = config.get("observe", {})
            tmux_config = observe_config.get("tmux", {})

            self.tmux_enabled = tmux_config.get("enabled", True)
            self.tmux_capture_interval = tmux_config.get(
                "capture_interval", CHUNK_DURATION
            )

            if not self.tmux_enabled:
                logger.info("Tmux capture disabled in config")
            elif self.tmux_capture_interval != CHUNK_DURATION:
                logger.info(f"Tmux capture interval: {self.tmux_capture_interval}s")
        except Exception as e:
            logger.warning(f"Failed to load tmux config, using defaults: {e}")

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

        # Load tmux configuration from journal
        self._load_tmux_config()

        # Check tmux availability (only if enabled)
        if self.tmux_enabled:
            if self.tmux_capture.is_available():
                logger.info("Tmux available for fallback capture")
            else:
                logger.info("Tmux not available (will only use screencast)")

        # Start Callosum connection for events
        self._callosum = CallosumConnection()
        self._callosum.start()
        self._journal_path = Path(get_journal())
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

        # Check tmux activity (only if screen is idle and tmux is enabled)
        # Uses fixed threshold for mode detection, not capture interval
        if screen_active or not self.tmux_enabled:
            tmux_active = False
        else:
            tmux_active = self.tmux_capture.is_active(
                poll_interval=TMUX_ACTIVITY_THRESHOLD
            )

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

    def _save_audio_segment(self, segment_dir: str, is_muted: bool) -> list[str]:
        """
        Save accumulated audio buffer to segment directory.

        Args:
            segment_dir: Path to the segment directory (YYYYMMDD/HHMMSS_LEN/)
            is_muted: Whether to save as split mono files (muted) or stereo (unmuted)

        Returns:
            List of saved filenames (empty if nothing saved)
        """
        if self.accumulated_audio_buffer.size == 0:
            logger.warning("No audio buffer to save")
            return []

        segment_path = Path(segment_dir)

        if is_muted:
            # Split mode: save mic and sys as separate mono files
            mic_data = self.accumulated_audio_buffer[:, 0]
            sys_data = self.accumulated_audio_buffer[:, 1]

            mic_bytes = self.audio_recorder.create_mono_flac_bytes(mic_data)
            sys_bytes = self.audio_recorder.create_mono_flac_bytes(sys_data)

            mic_name = "mic_audio.flac"
            sys_name = "sys_audio.flac"

            mic_path = segment_path / mic_name
            sys_path = segment_path / sys_name

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
            audio_name = "audio.flac"
            flac_path = segment_path / audio_name

            with open(flac_path, "wb") as f:
                f.write(flac_bytes)

            logger.info(f"Saved audio to {flac_path}")
            return [audio_name]

    async def handle_boundary(self, new_mode: str):
        """
        Handle window boundary rollover.

        Closes the current draft folder, renames it to final segment name,
        and emits the observing event.

        Args:
            new_mode: The mode for the new segment
        """
        # Get timestamp parts for this window and calculate duration
        date_part, time_part = get_timestamp_parts(self.start_at)
        duration = int(time.time() - self.start_at)
        day_dir = day_path(date_part)

        # Stop screencast first (closes file handles)
        stopped_streams: list[StreamInfo] = []
        screen_files: list[str] = []

        if self.current_mode == MODE_SCREENCAST:
            logger.info("Stopping previous screencast")
            stopped_streams = await self.screencaster.stop()
            self.current_streams = []

            # Collect screen filenames (files are already in draft dir with final names)
            screen_files = [stream.filename for stream in stopped_streams]

        # Save audio if we have enough threshold hits (to draft dir)
        did_save_audio = self.threshold_hits >= MIN_HITS_FOR_SAVE
        audio_files: list[str] = []
        if did_save_audio and self.draft_dir:
            audio_files = self._save_audio_segment(
                self.draft_dir, self.segment_is_muted
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

        # Handle tmux capture save (to draft dir)
        # Save if we have captures, regardless of current mode (mode may have
        # changed to IDLE within the segment without triggering a boundary)
        tmux_files: list[str] = []
        if self.tmux_captures and self.draft_dir:
            # write_captures_jsonl expects a Path and creates it if needed
            # Draft dir already exists
            tmux_files = write_captures_jsonl(self.tmux_captures, Path(self.draft_dir))

        # Reset tmux state
        self.tmux_captures = []
        self.tmux_capture_id = 0
        self.tmux_sessions_seen = set()
        self.tmux_capture.reset_hashes()
        self.last_tmux_capture_time = 0  # Allow immediate capture in new segment
        self.in_tmux_segment = new_mode == MODE_TMUX  # Track if new segment is tmux

        # Collect all files saved in this segment
        files = audio_files + screen_files + tmux_files
        segment_key = f"{time_part}_{duration}"

        # Rename draft folder to final segment name (atomic handoff)
        if self.draft_dir and files:
            final_segment_dir = str(day_dir / self.stream / segment_key)
            try:
                os.rename(self.draft_dir, final_segment_dir)
                logger.info(
                    f"Segment finalized: {self.draft_dir} -> {final_segment_dir}"
                )
            except OSError as e:
                logger.error(f"Failed to rename draft folder: {e}")
                # Files stay in draft folder, won't be processed
                files = []

            # Write stream identity for this segment
            if files:
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
        elif self.draft_dir and not files:
            # No files to save, remove empty draft folder
            try:
                os.rmdir(self.draft_dir)
                logger.debug(f"Removed empty draft folder: {self.draft_dir}")
            except OSError:
                pass  # May have other files, ignore

        self.draft_dir = None

        # Reset timing for new window
        self.start_at = time.time()  # Wall-clock for filenames
        self.start_at_mono = time.monotonic()  # Monotonic for elapsed

        # Update segment mute state for new segment
        self.segment_is_muted = self.cached_is_muted

        # Update mode
        old_mode = self.current_mode
        self.current_mode = new_mode

        # Start new capture based on mode (creates new draft folder)
        if new_mode == MODE_SCREENCAST and not self.cached_screen_locked:
            await self.initialize_screencast()
        elif new_mode == MODE_TMUX or new_mode == MODE_IDLE:
            # Create draft folder for audio/tmux even without screencast
            self._create_draft_folder()
        # MODE_TMUX doesn't need initialization, captures happen in main loop

        logger.info(f"Mode transition: {old_mode} -> {new_mode}")

        # Emit observing event with what we saved this boundary
        if files and self._callosum:
            self._callosum.emit(
                "observe",
                "observing",
                day=date_part,
                segment=segment_key,
                files=files,
                host=HOST,
                platform=PLATFORM,
                stream=self.stream,
            )
            logger.info(f"Segment observing: {segment_key} ({len(files)} files)")

    def _create_draft_folder(self) -> str:
        """Create a draft folder for the current segment."""
        self.draft_dir = create_draft_folder(self.start_at, self.stream)
        logger.debug(f"Created draft folder: {self.draft_dir}")
        return self.draft_dir

    async def initialize_screencast(self) -> bool:
        """
        Start a new screencast recording.

        Creates a draft folder and starts GStreamer recording to it.

        Returns:
            True if screencast started successfully, False otherwise.

        Raises:
            RuntimeError: If recording fails to start (caller should exit).
        """
        # Create draft folder for this segment
        draft_path = self._create_draft_folder()

        try:
            streams = await self.screencaster.start(
                draft_path, framerate=1, draw_cursor=True
            )
        except RuntimeError as e:
            logger.error(f"Failed to start screencast: {e}")
            raise

        if not streams:
            logger.error("No streams returned from screencast start")
            raise RuntimeError("No streams available")

        self.current_streams = streams

        logger.info(f"Started screencast with {len(streams)} stream(s)")
        for stream in streams:
            logger.info(f"  {stream.position} ({stream.connector}): {stream.file_path}")

        return True

    def capture_tmux(self):
        """Poll tmux and accumulate captures based on capture interval.

        Only captures if:
        1. Enough time has passed since last capture (capture_interval)
        2. There was activity since the last capture
        """
        now = time.time()
        time_since_capture = now - self.last_tmux_capture_time

        # Check if capture interval has elapsed
        if time_since_capture < self.tmux_capture_interval:
            return

        # Get sessions with activity since last capture
        active_sessions = self.tmux_capture.get_active_sessions(time_since_capture)
        if not active_sessions:
            return

        # Update capture time before capturing
        self.last_tmux_capture_time = now

        for session_info in active_sessions:
            session = session_info["session"]
            self.tmux_sessions_seen.add(session)

            result = self.tmux_capture.capture_changed(session)
            if not result:
                continue

            self.tmux_capture_id += 1
            relative_ts = now - self.start_at
            capture_dict = self.tmux_capture.result_to_dict(
                result, self.tmux_capture_id, relative_ts
            )
            self.tmux_captures.append(capture_dict)
            logger.debug(f"Captured tmux session {session}: {len(result.panes)} panes")

    def emit_status(self):
        """Emit observe.status event with current state."""
        if not self._callosum:
            return

        journal_path = str(self._journal_path) if self._journal_path else ""
        elapsed = int(time.monotonic() - self.start_at_mono)

        # Calculate screencast info
        if self.current_mode == MODE_SCREENCAST and self.current_streams:
            streams_info = []
            for stream in self.current_streams:
                try:
                    rel_file = os.path.relpath(stream.file_path, journal_path)
                except ValueError:
                    rel_file = stream.file_path

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
            }
        else:
            screencast_info = {"recording": False}

        # Calculate tmux info (show capturing=true while tmux segment is open)
        if self.in_tmux_segment:
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
        }

        # Determine reported mode (segment type, not instantaneous state)
        # Screencast takes priority, then tmux segment, then idle
        if self.current_mode == MODE_SCREENCAST:
            reported_mode = MODE_SCREENCAST
        elif self.in_tmux_segment:
            reported_mode = MODE_TMUX
        else:
            reported_mode = MODE_IDLE

        # Emit status
        self._callosum.emit(
            "observe",
            "status",
            mode=reported_mode,
            screencast=screencast_info,
            tmux=tmux_info,
            audio=audio_info,
            activity=activity_info,
            host=HOST,
            platform=PLATFORM,
            stream=self.stream,
        )

    async def main_loop(self):
        """Run the main observer loop."""
        logger.info(f"Starting observer loop (interval={self.interval}s)")

        # Determine initial mode
        new_mode = await self.check_activity_status()
        self.segment_is_muted = self.cached_is_muted  # Sync initial mute state
        self.current_mode = new_mode
        self.in_tmux_segment = new_mode == MODE_TMUX

        # Start initial capture based on mode (creates draft folder)
        if new_mode == MODE_SCREENCAST and not self.cached_screen_locked:
            try:
                await self.initialize_screencast()
            except RuntimeError:
                # Failed to start screencast, exit
                self.running = False
                return
        else:
            # Create draft folder for audio/tmux even without screencast
            self._create_draft_folder()

        logger.info(f"Initial mode: {self.current_mode}")

        while self.running:
            # Sleep for chunk duration
            await asyncio.sleep(CHUNK_DURATION)

            # Check activity status and determine new mode
            new_mode = await self.check_activity_status()

            # Check for GStreamer failure mid-recording
            if (
                self.current_mode == MODE_SCREENCAST
                and not self.screencaster.is_healthy()
            ):
                logger.warning("Screencast recording failed, stopping gracefully")
                await self.screencaster.stop()

                # Files are already in draft folder, will be finalized at next boundary
                self.current_streams = []
                # Force recalculate mode without screencast
                self.current_mode = MODE_IDLE

            # Detect mode change
            mode_changed = new_mode != self.current_mode
            if mode_changed:
                logger.info(f"Mode changing: {self.current_mode} -> {new_mode}")

            # Only trigger segment boundary on screencast transitions (not tmux<->idle)
            # This allows tmux segments to run full 5min windows like screencast
            screencast_transition = mode_changed and (
                self.current_mode == MODE_SCREENCAST or new_mode == MODE_SCREENCAST
            )

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
                (elapsed >= self.interval) or screencast_transition or mute_transition
            )

            if is_boundary:
                logger.info(
                    f"Boundary: elapsed={elapsed:.1f}s screencast_change={screencast_transition} "
                    f"mute_change={mute_transition} "
                    f"hits={self.threshold_hits}/{MIN_HITS_FOR_SAVE}"
                )
                await self.handle_boundary(new_mode)
            elif mode_changed:
                # Update mode without boundary (tmux<->idle transitions)
                self.current_mode = new_mode
                # Mark tmux segment active when entering TMUX mode
                if new_mode == MODE_TMUX:
                    self.in_tmux_segment = True

            # Emit status event
            self.emit_status()

        # Cleanup on exit
        logger.info("Observer loop stopped, cleaning up...")
        await self.shutdown()

    async def shutdown(self):
        """Clean shutdown of observer."""
        # Get timestamp parts for final save
        date_part, time_part = get_timestamp_parts(self.start_at)
        duration = int(time.time() - self.start_at)
        day_dir = day_path(date_part)

        # Stop screencast first (closes file handles)
        stopped_streams: list[StreamInfo] = []
        if self.current_mode == MODE_SCREENCAST:
            logger.info("Stopping screencast for shutdown")
            stopped_streams = await self.screencaster.stop()
            # Brief delay for files to be flushed
            await asyncio.sleep(0.5)

        # Save final audio if threshold met (to draft dir)
        audio_files: list[str] = []
        if self.threshold_hits >= MIN_HITS_FOR_SAVE and self.draft_dir:
            audio_files = self._save_audio_segment(
                self.draft_dir, self.segment_is_muted
            )
            if audio_files:
                logger.info(f"Saved final audio: {len(audio_files)} file(s)")

        # Save tmux captures (to draft dir)
        # Save if we have captures, regardless of current mode (mode may have
        # changed to IDLE within the segment without triggering a boundary)
        tmux_files: list[str] = []
        if self.tmux_captures and self.draft_dir:
            tmux_files = write_captures_jsonl(self.tmux_captures, Path(self.draft_dir))
            if tmux_files:
                logger.info(f"Saved final tmux captures: {len(tmux_files)} file(s)")

        # Collect all files and finalize segment
        screen_files = [stream.filename for stream in stopped_streams]
        files = audio_files + screen_files + tmux_files
        segment_key = f"{time_part}_{duration}"

        if self.draft_dir and files:
            final_segment_dir = str(day_dir / self.stream / segment_key)
            try:
                os.rename(self.draft_dir, final_segment_dir)
                logger.info(f"Final segment: {self.draft_dir} -> {final_segment_dir}")

                # Write stream identity for this segment
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

                # Emit final observing event
                if self._callosum:
                    self._callosum.emit(
                        "observe",
                        "observing",
                        day=date_part,
                        segment=segment_key,
                        files=files,
                        host=HOST,
                        platform=PLATFORM,
                        stream=self.stream,
                    )
                    logger.info(
                        f"Segment observing: {segment_key} ({len(files)} files)"
                    )
            except OSError as e:
                logger.error(f"Failed to rename final draft folder: {e}")
        elif self.draft_dir:
            # No files, remove empty draft folder
            try:
                os.rmdir(self.draft_dir)
            except OSError:
                pass

        self.draft_dir = None

        # Stop audio recorder
        self.audio_recorder.stop_recording()
        logger.info("Audio recording stopped")

        # Stop Callosum connection
        if self._callosum:
            self._callosum.stop()
            self._callosum = None
        logger.info("Callosum connection stopped")


async def async_main(args):
    """Async entry point."""
    observer = Observer(
        interval=args.interval,
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
    args = setup_cli(parser)

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
