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

  # 5 minute recording at 1 fps without cursor
  gnome-screencast --fps 1 --no-cursor
"""

import argparse
import asyncio
import datetime
import logging
import os
import signal
import subprocess
import sys

from dbus_next.aio import MessageBus
from dbus_next.constants import BusType

from observe.gnome.dbus import (
    get_monitor_geometries,
    start_screencast,
    stop_screencast,
)
from think.utils import day_path, setup_cli, touch_health

logger = logging.getLogger(__name__)


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
        self, out_path: str, framerate: int = 1, draw_cursor: bool = True
    ) -> tuple[bool, str]:
        """
        Start screencast recording.

        Returns:
            Tuple of (ok: bool, resolved_output_path: str)
        """
        await self.connect()
        ok, resolved = await start_screencast(
            self.bus, out_path, framerate, draw_cursor
        )
        self._started = bool(ok)
        if not ok:
            logger.error("Failed to start screencast via DBus")
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
    logger.info(f"Detected {len(geometries)} monitor(s)")

    # Record to .live file, then atomically rename when complete
    live_path = out_path + ".live"

    try:
        sc = Screencaster()
        ok, _ = await sc.start(live_path, fps, draw_cursor)
        if not ok:
            print("ERROR: Failed to start screencast.", file=sys.stderr)
            if os.path.exists(live_path):
                error_path = out_path + ".error"
                os.replace(live_path, error_path)
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

        # Build monitor geometry metadata for video title
        # Format: "connector-id:position,x1,y1,x2,y2 connector-id:position,x1,y1,x2,y2 ..."
        title_parts = []
        for geom_info in geometries:
            x1, y1, x2, y2 = geom_info["box"]
            title_parts.append(
                f"{geom_info['id']}:{geom_info['position']},{x1},{y1},{x2},{y2}"
            )
        title = " ".join(title_parts)

        # Update video title with monitor dimensions
        try:
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
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to update video title: {e.stderr}")
        except FileNotFoundError:
            logger.warning("mkvpropedit not found, skipping title update")

        # Atomically rename to final destination
        # This ensures filesystem watchers only see the complete file
        os.replace(live_path, out_path)
        print(f"Saved to {out_path}")

        touch_health("screencast")
        return 0

    except Exception as e:
        # Clean up partial recording on failure
        logger.error(f"Screencast failed: {e}", exc_info=True)
        if os.path.exists(live_path):
            try:
                error_path = out_path + ".error"
                os.replace(live_path, error_path)
                print(f"Moved partial recording to {error_path}", file=sys.stderr)
            except OSError:
                pass
        raise


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
        default=1,
        help="Framerate for the screencast (default: 1).",
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
            print(
                "ERROR: JOURNAL_PATH not set and --out not specified.", file=sys.stderr
            )
            sys.exit(1)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        date_part, time_part = timestamp.split("_", 1)
        out_path = str(day_path(date_part) / f"{time_part}_screencast.webm")

    # Basic sanity on FPS
    fps = max(1, int(args.fps))
    if fps != args.fps:
        logger.warning(f"FPS adjusted from {args.fps} to {fps}")

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
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
