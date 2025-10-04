#!/usr/bin/env python3
"""
gnome_screencast.py — minimal GNOME Shell screencast via D-Bus (no portals)

Requirements:
  pip install dbus-next

Examples:
  # 10s sample at 30 fps with cursor, saved to ./screencast.webm
  python gnome_screencast.py --screencast 10

  # 5s sample at 15 fps without cursor, custom path
  python gnome_screencast.py --screencast 5 --fps 15 --no-cursor --out /tmp/sample.webm
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
from typing import Tuple

from dbus_next import Variant
from dbus_next.aio import MessageBus
from dbus_next.constants import BusType

from see.screen_dbus import get_monitor_geometries

SCREencast_BUS = "org.gnome.Shell.Screencast"
SCREencast_PATH = "/org/gnome/Shell/Screencast"
SCREencast_IFACE = "org.gnome.Shell.Screencast"


class Screencaster:
    def __init__(self):
        self.bus: MessageBus | None = None
        self.iface = None
        self._started = False

    async def connect(self):
        self.bus = await MessageBus(bus_type=BusType.SESSION).connect()
        introspection = await self.bus.introspect(SCREencast_BUS, SCREencast_PATH)
        obj = self.bus.get_proxy_object(SCREencast_BUS, SCREencast_PATH, introspection)
        self.iface = obj.get_interface(SCREencast_IFACE)

    async def start(
        self, out_path: str, framerate: int = 30, draw_cursor: bool = True
    ) -> Tuple[bool, str]:
        """
        Call org.gnome.Shell.Screencast.Screencast("file://...", { 'framerate': u, 'draw-cursor': b })
        Returns (ok: bool, resolved_output_path: str)
        """
        if self.iface is None:
            await self.connect()

        # GNOME expects a file:// URI (not a plain filesystem path)
        uri = f"{out_path}"
        options = {
            "framerate": Variant("u", int(framerate)),
            "draw-cursor": Variant("b", bool(draw_cursor)),
            # Additional options that GNOME understands exist, but these two keep it simple and robust.
        }

        ok, resolved = await self.iface.call_screencast(uri, options)
        self._started = bool(ok)
        return bool(ok), resolved

    async def stop(self):
        if self.iface is None:
            return
        try:
            await self.iface.call_stop_screencast()
        finally:
            self._started = False

    @property
    def started(self) -> bool:
        return self._started


async def run_screencast(
    duration_s: int, out_path: str, fps: int, draw_cursor: bool
) -> int:
    # Capture monitor geometries before starting recording
    geometries = get_monitor_geometries()

    sc = Screencaster()
    ok, resolved_path = await sc.start(out_path, fps, draw_cursor)
    if not ok:
        print("ERROR: Failed to start screencast.", file=sys.stderr)
        return 1

    print(f"Recording… ({duration_s}s) -> {resolved_path}")

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

    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal GNOME Shell screencast via D-Bus.")
    p.add_argument(
        "--screencast",
        type=int,
        metavar="SECONDS",
        help="Record a screencast for the given number of seconds, then stop.",
    )
    p.add_argument(
        "--out",
        default="./screencast.webm",
        help="Output file path for the screencast (default: ./screencast.webm).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Framerate for the screencast (default: 30).",
    )
    p.add_argument(
        "--no-cursor", action="store_true", help="Do not draw the mouse cursor."
    )
    return p.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])

    if args.screencast is None:
        print(
            "Nothing to do. Example:\n  python gnome_screencast.py --screencast 10\n",
            file=sys.stderr,
        )
        sys.exit(2)

    # Basic sanity on FPS
    fps = max(1, int(args.fps))

    try:
        rc = asyncio.run(
            run_screencast(
                duration_s=int(args.screencast),
                out_path=args.out,
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
