"""ScreenCaptureKit integration via sck-cli subprocess.

This module manages the sck-cli subprocess lifecycle for video and audio capture
on macOS using ScreenCaptureKit. sck-cli captures all displays simultaneously
and outputs JSONL metadata to stdout with display geometry information.
"""

import json
import logging
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from observe.utils import assign_monitor_positions

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
        self.output_base: Optional[Path] = None

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
        self.output_base = output_base

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

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            raise RuntimeError(f"sck-cli not found at: {self.sck_cli_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to start sck-cli: {e}")

        # Read JSONL metadata from stdout (sck-cli outputs this immediately)
        # Each line is either a display info or audio info
        displays_raw = []
        audio_info = None

        # Read lines until we get all metadata (sck-cli outputs then starts capture)
        # We need to read non-blocking since the process keeps running
        try:
            for line in self.process.stdout:
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
                    # Not JSON, might be a log message - ignore
                    pass

                # sck-cli outputs all metadata before starting capture
                # Once we have both displays and audio (or displays only if no audio)
                # we can stop reading. But we also need to not block forever.
                # Actually, sck-cli flushes stdout after metadata, so readline
                # will return empty when no more data. But process is still running.
                # We break after getting audio info or when stdout blocks.
                if audio_info is not None:
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
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("sck-cli did not exit cleanly, killing")
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                logger.warning(f"Error stopping sck-cli: {e}")

        self.process = None

    def is_running(self) -> bool:
        """
        Check if the capture subprocess is currently running.

        Returns:
            True if subprocess is active, False otherwise
        """
        if self.process is None:
            return False
        return self.process.poll() is None
