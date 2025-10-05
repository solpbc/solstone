#!/usr/bin/env python3
"""
gnome_screencast.py — minimal GNOME Shell screencast via D-Bus (no portals)

Requirements:
  pip install dbus-next

Examples:
  # 5 minute recording (default), saved to journal
  gnome-screencast

  # 10 second recording with custom path
  gnome-screencast --screencast 10 --out /tmp/sample.webm

  # 5 minute recording at 15 fps without cursor
  gnome-screencast --fps 15 --no-cursor
"""

import argparse
import asyncio
import datetime
import logging
import os
import signal
import subprocess
import sys
import tempfile

from dbus_next.aio import MessageBus

from observe.gnome.dbus import (
    get_monitor_geometries,
    start_screencast,
    stop_screencast,
)
from think.utils import setup_cli, touch_health


class Screencaster:
    """Higher-level screencast manager with state tracking."""

    def __init__(self):
        self.bus: MessageBus | None = None
        self._started = False

    async def connect(self):
        """Establish DBus session connection."""
        if self.bus is None:
            self.bus = await MessageBus(bus_type=BusType.SESSION).connect()

    async def start(
        self, out_path: str, framerate: int = 30, draw_cursor: bool = True
    ) -> tuple[bool, str]:
        """
        Start screencast recording.

        Returns:
            Tuple of (ok: bool, resolved_output_path: str)
        """
        await self.connect()
        ok, resolved = await start_screencast(self.bus, out_path, framerate, draw_cursor)
        self._started = bool(ok)
        return bool(ok), resolved

    async def stop(self):
        """Stop screencast recording."""
        if self.bus is None:
            return
        try:
            await stop_screencast(self.bus)
        finally:
            self._started = False

    @property
    def started(self) -> bool:
        return self._started


async def run_screencast(
    duration_s: int, out_path: str, fps: int, draw_cursor: bool
) -> int:
    """Record screencast with monitor geometry metadata."""
    # Capture monitor geometries before starting recording
    geometries = get_monitor_geometries()

    # Create temp file in same directory for atomic move
    out_dir = os.path.dirname(out_path)
    temp_fd, temp_path = tempfile.mkstemp(dir=out_dir, suffix=".webm.tmp")
    os.close(temp_fd)  # Close fd, we just need the path

    try:
        sc = Screencaster()
        ok, resolved_path = await sc.start(temp_path, fps, draw_cursor)
        if not ok:
            print("ERROR: Failed to start screencast.", file=sys.stderr)
            if os.path.exists(temp_path):
                error_path = out_path.replace(".webm", ".webm.error")
                os.replace(temp_path, error_path)
                print(f"Moved failed recording to {error_path}", file=sys.stderr)
            return 1

        print(f"Recording… ({duration_s}s) -> {out_path}")

        # Graceful Ctrl-C handling (stop and exit)
        stop_event = asyncio.Event()

        def _signal_handler():
            stop_event.set()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                # Windows / restricted environments
                pass

        try:
            # Wait for either duration elapsed or a signal
            done = asyncio.create_task(asyncio.sleep(duration_s))
            interrupted = asyncio.create_task(stop_event.wait())
            await asyncio.wait({done, interrupted}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            await sc.stop()
            print("Stopped.")

        # Update video title with monitor dimensions
        # Format: "connector-id:position,x1,y1,x2,y2 connector-id:position,x1,y1,x2,y2 ..."
        title_parts = []
        for geom_info in geometries:
            x1, y1, x2, y2 = geom_info["box"]
            title_parts.append(
                f"{geom_info['id']}:{geom_info['position']},{x1},{y1},{x2},{y2}"
            )
        title = " ".join(title_parts)

        try:
            subprocess.run(
                ["mkvpropedit", resolved_path, "--edit", "info", "--set", f"title={title}"],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"Updated video title with monitor dimensions: {title}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to update video title: {e.stderr}", file=sys.stderr)
        except FileNotFoundError:
            print("Warning: mkvpropedit not found, skipping title update", file=sys.stderr)

        # Atomically move temp file to final location
        os.replace(resolved_path, out_path)
        print(f"Saved to {out_path}")

        touch_health("screencast")
        return 0

    except Exception as e:
        # Move temp file to error location for debugging
        if os.path.exists(temp_path):
            try:
                error_path = out_path.replace(".webm", ".webm.error")
                os.replace(temp_path, error_path)
                print(f"Moved partial recording to {error_path}", file=sys.stderr)
            except OSError:
                pass
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="GNOME Shell screencast via D-Bus with journal integration."
    )
    parser.add_argument(
        "--screencast",
        type=int,
        metavar="SECONDS",
        default=300,
        help="Record a screencast for the given number of seconds (default: 300 = 5 minutes).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output file path for the screencast (default: <journal>/YYYYMMDD/HHMMSS_screencast.webm).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Framerate for the screencast (default: 30).",
    )
    parser.add_argument(
        "--no-cursor", action="store_true", help="Do not draw the mouse cursor."
    )

    args = setup_cli(parser)

    # Compute output path if not specified
    out_path = args.out
    if out_path is None:
        journal_path = os.getenv("JOURNAL_PATH")
        if not journal_path:
            print("ERROR: JOURNAL_PATH not set and --out not specified.", file=sys.stderr)
            sys.exit(1)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        date_part, time_part = timestamp.split("_", 1)
        day_dir = os.path.join(journal_path, date_part)
        os.makedirs(day_dir, exist_ok=True)
        out_path = os.path.join(day_dir, f"{time_part}_screencast.webm")

    # Basic sanity on FPS
    fps = max(1, int(args.fps))

    try:
        rc = asyncio.run(
            run_screencast(
                duration_s=int(args.screencast),
                out_path=out_path,
                fps=fps,
                draw_cursor=not args.no_cursor,
            )
        )
        sys.exit(rc)
    except Exception as e:
        # Common issues: not running GNOME Shell; calling from a non-GNOME compositor session.
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
