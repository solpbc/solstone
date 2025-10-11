"""GNOME Shell and Mutter DBus interface primitives."""

import asyncio
import io
import os
import tempfile
import time

import gi
from dbus_next import Variant
from dbus_next.aio import MessageBus
from dbus_next.constants import BusType

gi.require_version("Gdk", "4.0")  # noqa: E402
gi.require_version("Gtk", "4.0")  # noqa: E402
from gi.repository import Gdk, Gtk  # noqa: E402
from PIL import Image  # noqa: E402


# DBus service constants
SCREENSHOT_BUS = "org.gnome.Shell.Screenshot"
SCREENSHOT_PATH = "/org/gnome/Shell/Screenshot"
SCREENSHOT_IFACE = "org.gnome.Shell.Screenshot"

IDLE_MONITOR_BUS = "org.gnome.Mutter.IdleMonitor"
IDLE_MONITOR_PATH = "/org/gnome/Mutter/IdleMonitor/Core"
IDLE_MONITOR_IFACE = "org.gnome.Mutter.IdleMonitor"

SCREENSAVER_BUS = "org.gnome.ScreenSaver"
SCREENSAVER_PATH = "/org/gnome/ScreenSaver"
SCREENSAVER_IFACE = "org.gnome.ScreenSaver"

DISPLAY_CONFIG_BUS = "org.gnome.Mutter.DisplayConfig"
DISPLAY_CONFIG_PATH = "/org/gnome/Mutter/DisplayConfig"
DISPLAY_CONFIG_IFACE = "org.gnome.Mutter.DisplayConfig"

SCREENCAST_BUS = "org.gnome.Shell.Screencast"
SCREENCAST_PATH = "/org/gnome/Shell/Screencast"
SCREENCAST_IFACE = "org.gnome.Shell.Screencast"


# Global timestamp for the last screenshot (in seconds)
last_screenshot_timestamp = 0


async def take_screenshot(bus: MessageBus) -> bytes:
    """
    Take a screenshot via GNOME Shell DBus interface.

    Args:
        bus: Connected DBus session bus

    Returns:
        Screenshot image data as bytes (PNG format)
    """
    introspection = await bus.introspect(SCREENSHOT_BUS, SCREENSHOT_PATH)
    proxy_obj = bus.get_proxy_object(SCREENSHOT_BUS, SCREENSHOT_PATH, introspection)
    interface = proxy_obj.get_interface(SCREENSHOT_IFACE)
    temp_path = os.path.join(tempfile.gettempdir(), "screenshot.png")
    # Call the Screenshot method (non-interactive, no flash)
    await interface.call_screenshot(False, False, temp_path)
    with open(temp_path, "rb") as f:
        screenshot_bytes = f.read()
    os.remove(temp_path)
    return screenshot_bytes


async def get_idle_time_ms(bus: MessageBus) -> int:
    """
    Get the current idle time in milliseconds.

    Args:
        bus: Connected DBus session bus

    Returns:
        Idle time in milliseconds
    """
    introspection = await bus.introspect(IDLE_MONITOR_BUS, IDLE_MONITOR_PATH)
    proxy_obj = bus.get_proxy_object(
        IDLE_MONITOR_BUS, IDLE_MONITOR_PATH, introspection
    )
    idle_monitor = proxy_obj.get_interface(IDLE_MONITOR_IFACE)
    idle_time = await idle_monitor.call_get_idletime()
    return idle_time


async def is_screen_locked(bus: MessageBus) -> bool:
    """
    Check if the screen is currently locked using GNOME ScreenSaver.

    Args:
        bus: Connected DBus session bus

    Returns:
        True if screen is locked, False otherwise
    """
    try:
        intro = await bus.introspect(SCREENSAVER_BUS, SCREENSAVER_PATH)
        obj = bus.get_proxy_object(SCREENSAVER_BUS, SCREENSAVER_PATH, intro)
        iface = obj.get_interface(SCREENSAVER_IFACE)
        return bool(await iface.call_get_active())
    except Exception:
        # If the interface isn't present on this session, treat as unlocked
        return False


async def is_power_save_active(bus: MessageBus) -> bool:
    """
    Check if display power save mode is active (screen blanked).

    Args:
        bus: Connected DBus session bus

    Returns:
        True if power save is active, False otherwise
    """
    try:
        intro = await bus.introspect(DISPLAY_CONFIG_BUS, DISPLAY_CONFIG_PATH)
        obj = bus.get_proxy_object(DISPLAY_CONFIG_BUS, DISPLAY_CONFIG_PATH, intro)
        iface = obj.get_interface("org.freedesktop.DBus.Properties")
        # Get("org.gnome.Mutter.DisplayConfig", "PowerSaveMode") -> int32
        # 0 = on, nonzero = blanked
        mode_variant = await iface.call_get(DISPLAY_CONFIG_IFACE, "PowerSaveMode")
        mode = int(mode_variant.value)
        return mode != 0
    except Exception:
        # Property or service not available -> assume not blanked
        return False


async def start_screencast(
    bus: MessageBus, out_path: str, framerate: int = 1, draw_cursor: bool = True
) -> tuple[bool, str]:
    """
    Start GNOME Shell screencast recording.

    Args:
        bus: Connected DBus session bus
        out_path: Output file path
        framerate: Frames per second (default: 1)
        draw_cursor: Whether to draw mouse cursor (default: True)

    Returns:
        Tuple of (success: bool, resolved_path: str)
    """
    introspection = await bus.introspect(SCREENCAST_BUS, SCREENCAST_PATH)
    obj = bus.get_proxy_object(SCREENCAST_BUS, SCREENCAST_PATH, introspection)
    iface = obj.get_interface(SCREENCAST_IFACE)

    options = {
        "framerate": Variant("u", int(framerate)),
        "draw-cursor": Variant("b", bool(draw_cursor)),
    }

    ok, resolved = await iface.call_screencast(out_path, options)
    return bool(ok), resolved


async def stop_screencast(bus: MessageBus) -> None:
    """
    Stop active GNOME Shell screencast recording.

    Args:
        bus: Connected DBus session bus
    """
    introspection = await bus.introspect(SCREENCAST_BUS, SCREENCAST_PATH)
    obj = bus.get_proxy_object(SCREENCAST_BUS, SCREENCAST_PATH, introspection)
    iface = obj.get_interface(SCREENCAST_IFACE)
    await iface.call_stop_screencast()


def get_monitor_geometries() -> list[dict]:
    """
    Get structured monitor information.

    Returns:
        List of dicts with format:
        [{"id": "connector-id", "box": [x1, y1, x2, y2], "position": "center|left|right|..."}, ...]
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


async def screen_snap_async(skip_if_locked: bool = True) -> list[Image.Image]:
    """
    Take a screenshot and split it into per-monitor images.

    Only captures if user is active (not idle since last screenshot).
    Skips if screen is locked or in power save mode.

    Args:
        skip_if_locked: Skip screenshot if screen locked/blanked (default: True)

    Returns:
        List of PIL Images, one per monitor (empty list if skipped)
    """
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
