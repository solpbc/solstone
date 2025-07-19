import asyncio
import io
import os
import tempfile
import time

import gi
from dbus_next.aio import MessageBus
from dbus_next.constants import BusType

gi.require_version("Gdk", "4.0")  # noqa: E402
gi.require_version("Gtk", "4.0")  # noqa: E402
from gi.repository import Gdk, Gtk  # noqa: E402
from PIL import Image  # noqa: E402

# Global timestamp for the last screenshot (in seconds)
last_screenshot_timestamp = 0


async def take_screenshot(bus):
    introspection = await bus.introspect(
        "org.gnome.Shell.Screenshot", "/org/gnome/Shell/Screenshot"
    )
    proxy_obj = bus.get_proxy_object(
        "org.gnome.Shell.Screenshot", "/org/gnome/Shell/Screenshot", introspection
    )
    interface = proxy_obj.get_interface("org.gnome.Shell.Screenshot")
    temp_path = os.path.join(tempfile.gettempdir(), "screenshot.png")
    # Call the Screenshot method (non-interactive, no flash)
    await interface.call_screenshot(False, False, temp_path)
    with open(temp_path, "rb") as f:
        screenshot_bytes = f.read()
    os.remove(temp_path)
    return screenshot_bytes


async def get_idle_time_ms(bus):
    introspection = await bus.introspect(
        "org.gnome.Mutter.IdleMonitor", "/org/gnome/Mutter/IdleMonitor/Core"
    )
    proxy_obj = bus.get_proxy_object(
        "org.gnome.Mutter.IdleMonitor",
        "/org/gnome/Mutter/IdleMonitor/Core",
        introspection,
    )
    idle_monitor = proxy_obj.get_interface("org.gnome.Mutter.IdleMonitor")
    idle_time = await idle_monitor.call_get_idletime()
    return idle_time


async def _idle_time_ms_async():
    """Return the current idle time of the desktop in milliseconds."""
    bus = await MessageBus(bus_type=BusType.SESSION).connect()
    return await get_idle_time_ms(bus)


def idle_time_ms():
    """Synchronous wrapper around ``_idle_time_ms_async``."""
    return asyncio.run(_idle_time_ms_async())


def get_monitor_geometries():
    # Initialize GTK before using GDK functions
    Gtk.init()

    # Get the default display. If it is None, try opening one from the environment.
    display = Gdk.Display.get_default()
    if display is None:
        env_display = os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")
        if env_display is not None:
            display = Gdk.Display.open(env_display)
        if display is None:
            raise RuntimeError("No display available")
    # In GTK 4, get_monitors() returns a list of Gdk.Monitor objects.
    monitors = display.get_monitors()
    geometries = []
    for monitor in monitors:
        geom = (
            monitor.get_geometry()
        )  # geom is a Gdk.Rectangle with attributes: x, y, width, height
        geometries.append(geom)
    return geometries


# asynchronous snapshot helper that returns an array of monitor images only if user is active
async def screen_snap_async():
    global last_screenshot_timestamp
    now = time.time()
    bus = await MessageBus(bus_type=BusType.SESSION).connect()
    if last_screenshot_timestamp:
        idle_time_ms = await get_idle_time_ms(bus)
        elapsed = now - last_screenshot_timestamp
        if (idle_time_ms / 1000) > elapsed:
            # User has been idle since before the last screenshot.
            return []
    # Take the screenshot for all monitors.
    screenshot_data = await take_screenshot(bus)
    im = Image.open(io.BytesIO(screenshot_data))
    geometries = get_monitor_geometries()
    monitor_images = []
    for geom in geometries:
        box = (geom.x, geom.y, geom.x + geom.width, geom.y + geom.height)
        monitor_img = im.crop(box)
        monitor_images.append(monitor_img)
    last_screenshot_timestamp = now
    return monitor_images


def screen_snap():
    return asyncio.run(screen_snap_async())


def main():
    monitor_images = screen_snap()
    for idx, img in enumerate(monitor_images, start=1):
        filename = f"monitor_{idx}.png"
        img.save(filename)
        print(f"Saved {filename}")
    print(f"Processed {len(monitor_images)} monitor images.")


if __name__ == "__main__":
    main()
