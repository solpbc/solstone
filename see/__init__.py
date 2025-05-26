"""Utilities for screen capture and Gemini-based vision processing."""

from .screen_dbus import screen_snap
from .screen_compare import compare_images
from .gemini_look import initialize as gemini_initialize, gemini_describe_region

__all__ = [
    "screen_snap",
    "compare_images",
    "gemini_initialize",
    "gemini_describe_region",
]
