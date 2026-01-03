# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""ScreenCaptureKit integration via sck-cli subprocess.

This module manages the sck-cli subprocess lifecycle for video and audio capture
on macOS using ScreenCaptureKit. sck-cli captures all displays simultaneously
and outputs JSONL metadata to stdout with display geometry information.
"""

import json
import logging
import select
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from observe.utils import assign_monitor_positions

# Timeout for reading metadata from sck-cli (seconds)
METADATA_TIMEOUT = 5.0

logger = logging.getLogger(__name__)


@dataclass
class DisplayInfo:
    """Information about a single display's recording."""

    display_id: int
    position: str
    x: int
    y: int
    width: int
    height: int
    temp_path: str

    def final_name(self, time_part: str, duration: int) -> str:
        """Generate the final filename for this display's video."""
        return f"{time_part}_{duration}_{self.position}_{self.display_id}_screen.mov"


@dataclass
class AudioInfo:
    """Information about the audio recording."""

    temp_path: str
    tracks: list[str]

    def final_name(self, time_part: str, duration: int) -> str:
        """Generate the final filename for audio."""
        return f"{time_part}_{duration}_audio.m4a"


class ScreenCaptureKitManager:
    """
    Manages sck-cli subprocess for synchronized video and audio capture.

    Wraps the sck-cli tool to provide lifecycle management, handles process
    monitoring, parses JSONL output for display geometry, and manages output
    file finalization.
    """

    def __init__(self, sck_cli_path: str = "sck-cli"):
        """
        Initialize the ScreenCaptureKit manager.

        Args:
            sck_cli_path: Path to sck-cli executable (default: "sck-cli" from PATH)
        """
        self.sck_cli_path = sck_cli_path
        self.process: Optional[subprocess.Popen] = None
        self.displays: list[DisplayInfo] = []
        self.audio: Optional[AudioInfo] = None
        self._output_threads: list[threading.Thread] = []
        self._exit_logged: bool = False

    def start(
        self,
        output_base: Path,
        duration: int,
        frame_rate: float = 1.0,
    ) -> tuple[list[DisplayInfo], Optional[AudioInfo]]:
        """
        Start video and audio capture.

        Launches sck-cli as a subprocess with the specified parameters.
        Parses JSONL output from stdout to get display geometry information.
        Files are written to output_base_<displayID>.mov and output_base.m4a.

        Args:
            output_base: Base path for output files (without extension)
            duration: Capture duration in seconds
            frame_rate: Frame rate in Hz (default: 1.0)

        Returns:
            Tuple of (list of DisplayInfo, AudioInfo or None)

        Raises:
            RuntimeError: If sck-cli fails to start or returns no displays

        Example:
            >>> manager = ScreenCaptureKitManager()
            >>> day_dir = Path("journal/20250101")
            >>> output_base = day_dir / ".120000"  # Hidden temp file
            >>> displays, audio = manager.start(output_base, duration=300)
        """
        # Build command
        cmd = [
            self.sck_cli_path,
            str(output_base),
            "-r",
            str(frame_rate),
            "-l",
            str(duration),
        ]

        logger.info(f"Starting sck-cli: {' '.join(cmd)}")
        self._exit_logged = False

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffering for real-time output
            )
        except FileNotFoundError:
            raise RuntimeError(f"sck-cli not found at: {self.sck_cli_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to start sck-cli: {e}")

        # Read JSONL metadata from stdout (sck-cli outputs this immediately)
        # Each line is either a display info or audio info
        displays_raw = []
        audio_info = None

        # Read lines until we get both display and audio metadata.
        # Use select() with timeout to avoid blocking forever - the process
        # keeps running for the capture duration but outputs metadata upfront.
        # Note: "for line in file:" uses block buffering which can hang.
        deadline = time.monotonic() + METADATA_TIMEOUT
        stdout_fd = self.process.stdout.fileno()

        try:
            while time.monotonic() < deadline:
                # Wait for data with remaining timeout
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break

                readable, _, _ = select.select([stdout_fd], [], [], min(remaining, 1.0))
                if not readable:
                    # No data yet, keep polling until deadline
                    continue

                line = self.process.stdout.readline()
                if not line:
                    # EOF - process closed stdout
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if data.get("type") == "display":
                        displays_raw.append(data)
                    elif data.get("type") == "audio":
                        audio_info = data
                except json.JSONDecodeError:
                    # Not JSON, just a log message (already logged above)
                    pass

                # Break once we have both display and audio info
                if displays_raw and audio_info is not None:
                    break
        except Exception as e:
            logger.warning(f"Error reading sck-cli stdout: {e}")

        if not displays_raw:
            self.stop()
            raise RuntimeError("sck-cli returned no display information")

        # Convert raw display data to monitor format for position assignment
        monitors = []
        for d in displays_raw:
            x = int(d.get("x", 0))
            y = int(d.get("y", 0))
            width = int(d.get("width", 0))
            height = int(d.get("height", 0))
            monitors.append(
                {
                    "id": str(d["displayID"]),
                    "box": [x, y, x + width, y + height],
                    "_raw": d,
                }
            )

        # Assign position labels based on geometry
        monitors = assign_monitor_positions(monitors)

        # Build DisplayInfo objects
        self.displays = []
        for mon in monitors:
            raw = mon["_raw"]
            self.displays.append(
                DisplayInfo(
                    display_id=raw["displayID"],
                    position=mon["position"],
                    x=mon["box"][0],
                    y=mon["box"][1],
                    width=mon["box"][2] - mon["box"][0],
                    height=mon["box"][3] - mon["box"][1],
                    temp_path=raw["filename"],
                )
            )

        # Build AudioInfo if present
        if audio_info:
            tracks = [t["name"] for t in audio_info.get("tracks", [])]
            self.audio = AudioInfo(
                temp_path=audio_info["filename"],
                tracks=tracks,
            )
        else:
            self.audio = None

        logger.info(f"sck-cli started with {len(self.displays)} display(s)")
        for display in self.displays:
            logger.info(
                f"  Display {display.display_id}: {display.position} "
                f"({display.width}x{display.height}) -> {display.temp_path}"
            )
        if self.audio:
            logger.info(f"  Audio: {self.audio.temp_path} ({self.audio.tracks})")

        # Start background threads to log remaining stdout/stderr in real-time
        self._output_threads = [
            threading.Thread(
                target=self._stream_stdout,
                daemon=True,
                name="sck-cli-stdout",
            ),
            threading.Thread(
                target=self._stream_stderr,
                daemon=True,
                name="sck-cli-stderr",
            ),
        ]
        for thread in self._output_threads:
            thread.start()

        return self.displays, self.audio

    def stop(self) -> None:
        """
        Stop the running capture gracefully.

        Sends SIGTERM to the sck-cli process and waits for it to finish writing
        files properly. This ensures video and audio files are finalized correctly.
        """
        if self.process is None:
            return

        if self.process.poll() is None:
            logger.info("Stopping sck-cli...")
            try:
                self.process.send_signal(signal.SIGTERM)
                try:
                    exit_code = self.process.wait(timeout=5)
                    logger.info(f"sck-cli stopped with exit code {exit_code}")
                except subprocess.TimeoutExpired:
                    logger.warning("sck-cli did not exit cleanly, killing")
                    self.process.kill()
                    exit_code = self.process.wait()
                    logger.info(f"sck-cli killed with exit code {exit_code}")
            except Exception as e:
                logger.warning(f"Error stopping sck-cli: {e}")

        # Wait for output threads to finish (they exit when pipes close)
        for thread in self._output_threads:
            thread.join(timeout=1)
        self._output_threads = []

        self.process = None

    def _stream_stdout(self) -> None:
        """Background thread: stream remaining stdout lines to logger."""
        if self.process is None or self.process.stdout is None:
            return

        try:
            for line in self.process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"sck-cli: {line}")
        except Exception as e:
            logger.debug(f"Error reading sck-cli stdout: {e}")

    def _stream_stderr(self) -> None:
        """Background thread: stream stderr lines to logger."""
        if self.process is None or self.process.stderr is None:
            return

        try:
            for line in self.process.stderr:
                line = line.strip()
                if line:
                    logger.info(f"sck-cli stderr: {line}")
        except Exception as e:
            logger.debug(f"Error reading sck-cli stderr: {e}")

    def is_running(self) -> bool:
        """
        Check if the capture subprocess is currently running.

        Returns:
            True if subprocess is active, False otherwise
        """
        if self.process is None:
            return False
        exit_code = self.process.poll()
        if exit_code is not None:
            if not self._exit_logged:
                logger.info(f"sck-cli exited with code {exit_code}")
                self._exit_logged = True
            return False
        return True
