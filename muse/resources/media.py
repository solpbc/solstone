"""MCP resource handlers for media."""

import os
from pathlib import Path

from fastmcp.resources import FileResource

from muse.mcp import mcp
from think.utils import get_raw_file


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
    journal = os.getenv("JOURNAL_PATH", "journal")
    abs_path = Path(journal) / day / rel_path
    return FileResource(
        uri=f"journal://media/{day}/{name}",
        name=f"Media: {name}",
        description=f"Raw media file from {day}",
        mime_type=mime,
        path=abs_path,
    )
