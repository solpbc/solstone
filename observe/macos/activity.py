"""macOS system activity detection using PyObjC.

This module mirrors the GNOME dbus.py structure, providing activity detection
primitives using native macOS APIs via PyObjC.
"""

import logging
import subprocess

from Quartz import (
    CGDisplayIsAsleep,
    CGEventSourceSecondsSinceLastEventType,
    CGMainDisplayID,
    CGSessionCopyCurrentDictionary,
    kCGAnyInputEventType,
)

logger = logging.getLogger(__name__)


def get_idle_time_ms() -> int:
    """
    Get the current system idle time in milliseconds.

    Uses Quartz CGEventSourceSecondsSinceLastEventType to detect time since last
    user input event (keyboard, mouse, etc.).

    Returns:
        Idle time in milliseconds

    Example:
        >>> idle_ms = get_idle_time_ms()
        >>> print(f"User idle for {idle_ms / 1000:.1f} seconds")
    """
    try:
        # kCGEventSourceStateHIDSystemState = 1 (hardware input events)
        seconds = CGEventSourceSecondsSinceLastEventType(1, kCGAnyInputEventType)
        return int(seconds * 1000)
    except Exception as e:
        logger.warning(f"Failed to get idle time: {e}")
        return 0


def is_screen_locked() -> bool:
    """
    Check if the screen is currently locked.

    Queries the macOS session state via CGSessionCopyCurrentDictionary.
    When the screen is locked, kCGSSessionOnConsoleKey becomes False.

    Returns:
        True if screen is locked, False otherwise

    Example:
        >>> if is_screen_locked():
        ...     print("Screen is locked, skipping capture")
    """
    try:
        session_dict = CGSessionCopyCurrentDictionary()
        if session_dict is None:
            logger.warning("CGSessionCopyCurrentDictionary returned None")
            return False

        # kCGSSessionOnConsoleKey is True when user is on console (not locked)
        # When screen is locked, this becomes False
        on_console = session_dict.get("kCGSSessionOnConsoleKey", True)
        return not on_console
    except Exception as e:
        logger.warning(f"Failed to check screen lock status: {e}")
        return False


def is_power_save_active() -> bool:
    """
    Check if display power save mode is active (screen blanked/sleep).

    Uses CGDisplayIsAsleep to detect if the main display is sleeping,
    similar to GNOME's DisplayConfig PowerSaveMode check.

    Returns:
        True if power save is active (displays off), False otherwise

    Example:
        >>> if is_power_save_active():
        ...     print("Displays are sleeping")
    """
    try:
        main_display = CGMainDisplayID()
        is_asleep = CGDisplayIsAsleep(main_display)
        return bool(is_asleep)
    except Exception as e:
        logger.warning(f"Failed to check display sleep status: {e}")
        return False


def is_output_muted() -> bool:
    """
    Check if the system audio output is muted.

    Uses osascript to query macOS volume settings, similar to how GNOME
    uses pactl for PulseAudio mute status.

    Returns:
        True if muted, False otherwise (including on error).

    Example:
        >>> if is_output_muted():
        ...     print("Audio is muted")
    """
    try:
        result = subprocess.run(
            ["osascript", "-e", "output muted of (get volume settings)"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            logger.warning(
                f"osascript failed (rc={result.returncode}): {result.stderr}"
            )
            return False

        return result.stdout.strip().lower() == "true"
    except subprocess.TimeoutExpired:
        logger.warning("osascript timed out checking mute status")
        return False
    except FileNotFoundError:
        logger.warning("osascript not found")
        return False
    except Exception as e:
        logger.warning(f"Error checking output mute status: {e}")
        return False
