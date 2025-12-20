"""GNOME Shell and Mutter DBus interface primitives."""

import os

import gi
from dbus_next.aio import MessageBus

gi.require_version("Gdk", "4.0")  # noqa: E402
gi.require_version("Gtk", "4.0")  # noqa: E402
from gi.repository import Gdk, Gtk  # noqa: E402

# DBus service constants
IDLE_MONITOR_BUS = "org.gnome.Mutter.IdleMonitor"
IDLE_MONITOR_PATH = "/org/gnome/Mutter/IdleMonitor/Core"
IDLE_MONITOR_IFACE = "org.gnome.Mutter.IdleMonitor"

SCREENSAVER_BUS = "org.gnome.ScreenSaver"
SCREENSAVER_PATH = "/org/gnome/ScreenSaver"
SCREENSAVER_IFACE = "org.gnome.ScreenSaver"

DISPLAY_CONFIG_BUS = "org.gnome.Mutter.DisplayConfig"
DISPLAY_CONFIG_PATH = "/org/gnome/Mutter/DisplayConfig"
DISPLAY_CONFIG_IFACE = "org.gnome.Mutter.DisplayConfig"


async def get_idle_time_ms(bus: MessageBus) -> int:
    """
    Get the current idle time in milliseconds.

    Args:
        bus: Connected DBus session bus

    Returns:
        Idle time in milliseconds
    """
    introspection = await bus.introspect(IDLE_MONITOR_BUS, IDLE_MONITOR_PATH)
    proxy_obj = bus.get_proxy_object(IDLE_MONITOR_BUS, IDLE_MONITOR_PATH, introspection)
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


def get_monitor_geometries() -> list[dict]:
    """
    Get structured monitor information.

    Returns:
        List of dicts with format:
        [{"id": "connector-id", "box": [x1, y1, x2, y2], "position": "center|left|right|..."}, ...]
        where box contains [left, top, right, bottom] coordinates
    """
    from observe.utils import assign_monitor_positions

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

    # Collect monitor geometries
    geometries = []
    for monitor in monitors:
        geom = monitor.get_geometry()
        connector = monitor.get_connector() or f"monitor-{len(geometries)}"
        geometries.append(
            {
                "id": connector,
                "box": [geom.x, geom.y, geom.x + geom.width, geom.y + geom.height],
            }
        )

    # Assign position labels using shared algorithm
    return assign_monitor_positions(geometries)
