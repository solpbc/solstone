# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""MCP resource handlers for media."""

from pathlib import Path

from fastmcp.resources import FileResource

from think.mcp import mcp
from think.utils import get_journal, get_raw_file


@mcp.resource("journal://media/{day}/{name}")
def get_media(day: str, name: str) -> FileResource:
    """Return a raw FLAC or PNG file referenced by a transcript.

    Parameters
    ----------
    day:
        Day folder in ``YYYYMMDD`` format.
    name:
        Transcript JSON filename such as ``HHMMSS_audio.json`` or
        ``HHMMSS_monitor_1_diff.json``.

    Returns
    -------
    FileResource
        Resource pointing to the raw media file referenced by ``name``.
    """

    rel_path, mime, _ = get_raw_file(day, name)
    abs_path = Path(get_journal()) / day / rel_path
    return FileResource(
        uri=f"journal://media/{day}/{name}",
        name=f"Media: {name}",
        description=f"Raw media file from {day}",
        mime_type=mime,
        path=abs_path,
    )
