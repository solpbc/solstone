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


async def is_screen_locked(bus) -> bool:
    """Check if the screen is currently locked using GNOME ScreenSaver."""
    try:
        intro = await bus.introspect("org.gnome.ScreenSaver", "/org/gnome/ScreenSaver")
        obj = bus.get_proxy_object(
            "org.gnome.ScreenSaver", "/org/gnome/ScreenSaver", intro
        )
        iface = obj.get_interface("org.gnome.ScreenSaver")
        return bool(await iface.call_get_active())
    except Exception:
        # If the interface isn't present on this session, treat as unlocked
        return False


async def is_power_save_active(bus) -> bool:
    """Check if display power save mode is active (screen blanked)."""
    try:
        intro = await bus.introspect(
            "org.gnome.Mutter.DisplayConfig", "/org/gnome/Mutter/DisplayConfig"
        )
        obj = bus.get_proxy_object(
            "org.gnome.Mutter.DisplayConfig", "/org/gnome/Mutter/DisplayConfig", intro
        )
        iface = obj.get_interface("org.freedesktop.DBus.Properties")
        # Get("org.gnome.Mutter.DisplayConfig", "PowerSaveMode") -> int32
        # 0 = on, nonzero = blanked
        mode_variant = await iface.call_get(
            "org.gnome.Mutter.DisplayConfig", "PowerSaveMode"
        )
        mode = int(mode_variant.value)
        return mode != 0
    except Exception:
        # Property or service not available -> assume not blanked
        return False


async def _idle_time_ms_async():
    """Return the current idle time of the desktop in milliseconds."""
    bus = await MessageBus(bus_type=BusType.SESSION).connect()
    return await get_idle_time_ms(bus)


def idle_time_ms():
    """Synchronous wrapper around ``_idle_time_ms_async``."""
    return asyncio.run(_idle_time_ms_async())


async def _check_screen_state_async():
    """Check screen lock and power save state."""
    bus = await MessageBus(bus_type=BusType.SESSION).connect()
    locked = await is_screen_locked(bus)
    power_save = await is_power_save_active(bus)
    return {"locked": locked, "power_save": power_save}


def check_screen_state():
    """Synchronous wrapper to check screen lock and power save state."""
    return asyncio.run(_check_screen_state_async())


def get_monitor_geometries():
    """
    Get structured monitor information.

    Returns:
        List of dicts with format:
        [{"id": "connector-id", "box": [x1, y1, x2, y2], "position": "center|left|right|top|bottom|left-top|..."}, ...]
        where box contains [left, top, right, bottom] coordinates
    """
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

    # First pass: collect all geometries and compute union bounding box
    monitor_data = []
    for monitor in monitors:
        geom = monitor.get_geometry()
        connector = monitor.get_connector() or f"monitor-{len(monitor_data)}"
        monitor_data.append(
            {
                "monitor": monitor,
                "connector": connector,
                "x": geom.x,
                "y": geom.y,
                "width": geom.width,
                "height": geom.height,
            }
        )

    # Compute union bounding box
    min_x = min(m["x"] for m in monitor_data)
    min_y = min(m["y"] for m in monitor_data)
    max_x = max(m["x"] + m["width"] for m in monitor_data)
    max_y = max(m["y"] + m["height"] for m in monitor_data)

    # Compute midlines
    union_mid_x = (min_x + max_x) / 2
    union_mid_y = (min_y + max_y) / 2

    # Epsilon for intersection detection (1 pixel tolerance)
    epsilon = 1

    # Second pass: assign positions based on midline intersections
    geometries = []
    for m in monitor_data:
        x_left = m["x"]
        x_right = m["x"] + m["width"]
        y_top = m["y"]
        y_bottom = m["y"] + m["height"]

        # Horizontal position
        if x_left <= union_mid_x + epsilon and x_right >= union_mid_x - epsilon:
            h_pos = "center"
        elif x_right < union_mid_x - epsilon:
            h_pos = "left"
        else:
            h_pos = "right"

        # Vertical position
        if y_top <= union_mid_y + epsilon and y_bottom >= union_mid_y - epsilon:
            v_pos = "center"
        elif y_bottom < union_mid_y - epsilon:
            v_pos = "top"
        else:
            v_pos = "bottom"

        # Combine positions
        if h_pos == "center" and v_pos == "center":
            position = "center"
        elif h_pos == "center":
            position = v_pos
        elif v_pos == "center":
            position = h_pos
        else:
            position = f"{h_pos}-{v_pos}"

        geometries.append(
            {
                "id": m["connector"],
                "box": [m["x"], m["y"], m["x"] + m["width"], m["y"] + m["height"]],
                "position": position,
            }
        )

    return geometries


# asynchronous snapshot helper that returns an array of monitor images only if user is active
async def screen_snap_async(skip_if_locked=True):
    global last_screenshot_timestamp
    now = time.time()
    bus = await MessageBus(bus_type=BusType.SESSION).connect()

    # Check if screen is locked or in power save mode
    if skip_if_locked:
        locked = await is_screen_locked(bus)
        power_save = await is_power_save_active(bus)
        if locked or power_save:
            # Screen is locked or blanked, skip screenshot but update heartbeat
            return []

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
    for geom_info in geometries:
        box = tuple(geom_info["box"])  # [x, y, x+w, y+h]
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
