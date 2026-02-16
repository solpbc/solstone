# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""General utilities for solstone.

This module provides core utilities for journal access, date/segment handling,
configuration loading, and CLI setup. Muse-related utilities (prompt loading,
agent configs, etc.) have been moved to think/muse.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import platformdirs
from dotenv import load_dotenv
from timefhuman import timefhuman

DATE_RE = re.compile(r"\d{8}")
_journal_path_cache: str | None = None


def now_ms() -> int:
    """Return current time as Unix epoch milliseconds."""
    return int(time.time() * 1000)


_rev_cache: str | None = "__unset__"


def get_rev() -> str | None:
    """Return short git commit hash, cached after first call. None if unavailable."""
    global _rev_cache
    if _rev_cache != "__unset__":
        return _rev_cache
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        _rev_cache = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        _rev_cache = None
    return _rev_cache


def truncated_echo(text: str, max_bytes: int = 16384) -> None:
    """Print text to stdout, truncating if it exceeds *max_bytes* UTF-8 bytes.

    When the encoded output exceeds the limit it is cut at a clean UTF-8
    character boundary and a warning is written to stderr reporting the
    original size.  Pass ``max_bytes=0`` to disable the limit.
    """
    encoded = text.encode("utf-8")
    if max_bytes > 0 and len(encoded) > max_bytes:
        truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
        sys.stdout.write(truncated)
        sys.stdout.write("\n")
        sys.stderr.write(
            f"[truncated: {len(encoded):,} bytes total, --max {max_bytes:,}]\n"
        )
    else:
        sys.stdout.write(text)
        sys.stdout.write("\n")


def get_journal_info() -> tuple[str, str]:
    """Return the journal path and its source.

    Determines where JOURNAL_PATH came from:
    - "shell": Set in shell environment before process started
    - "dotenv": Loaded from .env file
    - "default": Platform-specific default

    This function does NOT auto-create directories or modify the environment.
    Use get_journal() for normal operations that need the path created.

    Returns
    -------
    tuple[str, str]
        (path, source) where path is the journal directory and source is
        one of "shell", "dotenv", or "default".
    """
    # Check if already set in shell environment (before loading .env)
    shell_value = os.environ.get("JOURNAL_PATH")

    if shell_value:
        return shell_value, "shell"

    # Load .env and check if it provides JOURNAL_PATH
    load_dotenv()
    dotenv_value = os.environ.get("JOURNAL_PATH")

    if dotenv_value:
        return dotenv_value, "dotenv"

    # Fall back to platform default
    data_dir = platformdirs.user_data_dir("solstone")
    default_journal = os.path.join(data_dir, "journal")
    return default_journal, "default"


def get_journal() -> str:
    """Return the journal path, auto-creating it if it doesn't exist.

    Resolution order:
    1. JOURNAL_PATH environment variable (from .env or shell) - created if missing
    2. Cached platform default from previous call
    3. Platform-specific default: <user_data_dir>/solstone/journal

    When using the platform default, the path is cached and set in os.environ.
    Environment variable changes are always respected (no caching for explicit config).
    An INFO log message is emitted when auto-creating the default path.

    Returns
    -------
    str
        Absolute path to the journal directory.
    """
    global _journal_path_cache

    # Always check environment first (allows tests to override)
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")

    if journal:
        # User explicitly configured a path - create it if needed and use it
        os.makedirs(journal, exist_ok=True)
        return journal

    # Use cached platform default if available
    if _journal_path_cache:
        return _journal_path_cache

    # Create platform-specific default
    data_dir = platformdirs.user_data_dir("solstone")
    default_journal = os.path.join(data_dir, "journal")

    # Create directory if needed
    os.makedirs(default_journal, exist_ok=True)

    # Set environment for this process and children
    os.environ["JOURNAL_PATH"] = default_journal
    _journal_path_cache = default_journal

    logging.info("Using default journal path: %s", default_journal)
    return default_journal


def day_path(day: Optional[str] = None) -> Path:
    """Return absolute path for a day directory within the journal.

    Parameters
    ----------
    day : str, optional
        Day in YYYYMMDD format. If None, uses today's date.

    Returns
    -------
    Path
        Absolute path to the day directory. Directory is created if it doesn't exist.

    Raises
    ------
    ValueError
        If day format is invalid.
    """
    journal = get_journal()

    # Handle "today" case
    if day is None:
        day = datetime.now().strftime("%Y%m%d")
    elif not DATE_RE.fullmatch(day):
        raise ValueError("day must be in YYYYMMDD format")

    path = Path(journal) / day
    path.mkdir(parents=True, exist_ok=True)
    return path


def day_dirs() -> dict[str, str]:
    """Return mapping of YYYYMMDD day names to absolute paths.

    Returns
    -------
    dict[str, str]
        Mapping of day folder names to their full paths.
        Example: {"20250101": "/path/to/journal/20250101", ...}
    """
    journal = get_journal()

    days: dict[str, str] = {}
    for name in os.listdir(journal):
        if DATE_RE.fullmatch(name):
            path = os.path.join(journal, name)
            if os.path.isdir(path):
                days[name] = path
    return days


def dirty_days(exclude: set[str] | None = None) -> list[str]:
    """Return journal days with pending stream data not yet processed daily.

    A day is "dirty" when it has a ``health/stream.updated`` marker that is
    newer than its ``health/daily.updated`` marker (or daily.updated is missing).
    Days without ``stream.updated`` are skipped entirely.

    Parameters
    ----------
    exclude : set of str, optional
        Day strings (YYYYMMDD) to skip.

    Returns
    -------
    list of str
        Sorted list of dirty day strings.
    """
    days = day_dirs()
    dirty: list[str] = []
    for name, path in days.items():
        if exclude and name in exclude:
            continue
        stream = os.path.join(path, "health", "stream.updated")
        if not os.path.isfile(stream):
            continue
        daily = os.path.join(path, "health", "daily.updated")
        if not os.path.isfile(daily):
            dirty.append(name)
            continue
        if os.path.getmtime(stream) > os.path.getmtime(daily):
            dirty.append(name)
    dirty.sort()
    return dirty


def segment_path(day: str, segment: str, stream: str) -> Path:
    """Return absolute path for a segment directory within a stream.

    Parameters
    ----------
    day : str
        Day in YYYYMMDD format.
    segment : str
        Segment key in HHMMSS_LEN format.
    stream : str
        Stream name (e.g., "archon", "import.apple").

    Returns
    -------
    Path
        Absolute path to the segment directory (created if it doesn't exist).
    """
    path = day_path(day) / stream / segment
    path.mkdir(parents=True, exist_ok=True)
    return path


def day_from_path(path: str | Path) -> str | None:
    """Extract the YYYYMMDD day from a journal path.

    Walks up the path's parents and returns the first directory name
    that matches the YYYYMMDD date format.

    Parameters
    ----------
    path : str or Path
        Any path within the journal directory structure.

    Returns
    -------
    str or None
        The YYYYMMDD day string, or None if no date directory is found.
    """
    path = Path(path)
    for parent in (path, *path.parents):
        if DATE_RE.fullmatch(parent.name):
            return parent.name
    return None


def iter_segments(day: str | Path) -> list[tuple[str, str, Path]]:
    """Return all segments in a day, sorted chronologically.

    Traverses the stream directory structure under a day directory and
    returns segment information for all streams.

    Parameters
    ----------
    day : str or Path
        Day in YYYYMMDD format (str) or path to day directory (Path).

    Returns
    -------
    list of (stream_name, segment_key, segment_path) tuples
        Sorted by segment_key across all streams for chronological order.
    """
    if isinstance(day, Path):
        day_dir = day
    else:
        day_dir = day_path(day)

    if not day_dir.exists():
        return []

    results = []
    for entry in day_dir.iterdir():
        if not entry.is_dir():
            continue
        stream_name = entry.name
        for seg_entry in entry.iterdir():
            if seg_entry.is_dir() and segment_key(seg_entry.name):
                results.append((stream_name, seg_entry.name, seg_entry))

    results.sort(key=lambda x: x[1])
    return results


def segment_key(name_or_path: str) -> str | None:
    """Extract segment key (HHMMSS_LEN) from any path/filename.

    Parameters
    ----------
    name_or_path : str
        Segment name, filename, or full path containing segment.

    Returns
    -------
    str or None
        Segment key in HHMMSS_LEN format if valid, None otherwise.

    Examples
    --------
    >>> segment_key("143022_300")
    "143022_300"
    >>> segment_key("143022_300_summary.txt")
    "143022_300"
    >>> segment_key("/journal/20250109/143022_300/audio.jsonl")
    "143022_300"
    >>> segment_key("invalid")
    None
    """
    # Match HHMMSS_LEN format: 6 digits, underscore, 1+ digits
    pattern = r"\b(\d{6})_(\d+)(?:_|\b)"
    match = re.search(pattern, name_or_path)
    if match:
        time_part = match.group(1)
        len_part = match.group(2)
        return f"{time_part}_{len_part}"
    return None


def segment_parse(
    name_or_path: str,
) -> tuple[datetime.time, datetime.time] | tuple[None, None]:
    """Parse segment to extract start and end times as datetime objects.

    Parameters
    ----------
    name_or_path : str
        Segment name (e.g., "143022_300") or full path containing segment.

    Returns
    -------
    tuple of (datetime.time, datetime.time) or (None, None)
        Tuple of (start_time, end_time) where:
        - start_time: datetime.time for HHMMSS
        - end_time: datetime.time computed from start + LEN seconds
        Returns (None, None) if not a valid HHMMSS_LEN segment format.

    Examples
    --------
    >>> segment_parse("143022_300")  # 14:30:22 + 300 seconds = 14:35:22
    (datetime.time(14, 30, 22), datetime.time(14, 35, 22))
    >>> segment_parse("/journal/20250109/143022_300/audio.jsonl")
    (datetime.time(14, 30, 22), datetime.time(14, 35, 22))
    >>> segment_parse("invalid")
    (None, None)
    """
    from datetime import time, timedelta

    # Extract just the segment name if it's a path
    if "/" in name_or_path or "\\" in name_or_path:
        path_parts = Path(name_or_path).parts
        # Look for segment key in path parts after a YYYYMMDD day directory.
        # Layout is YYYYMMDD/stream/HHMMSS_LEN/...
        name = None
        for i, part in enumerate(path_parts):
            if part.isdigit() and len(part) == 8:
                # Scan subsequent parts for a segment key
                for j in range(i + 1, len(path_parts)):
                    if segment_key(path_parts[j]):
                        name = path_parts[j]
                        break
                if name:
                    break
        if name is None:
            return (None, None)
    else:
        name = name_or_path

    # Validate and extract HHMMSS_LEN from segment name
    if "_" not in name:
        return (None, None)

    parts = name.split("_", 1)  # Split on first underscore only
    if (
        len(parts) != 2
        or not parts[0].isdigit()
        or len(parts[0]) != 6
        or not parts[1].isdigit()
    ):
        return (None, None)

    time_str = parts[0]
    length_str = parts[1]

    # Parse HHMMSS to datetime.time
    try:
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])

        # Validate ranges
        if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
            return (None, None)

        start_time = time(hour, minute, second)
    except (ValueError, IndexError):
        return (None, None)

    # Parse LEN and compute end time
    try:
        length_seconds = int(length_str)
        # Compute end time by adding duration
        start_dt = datetime.combine(datetime.today(), start_time)
        end_dt = start_dt + timedelta(seconds=length_seconds)
        end_time = end_dt.time()
        return (start_time, end_time)
    except ValueError:
        return (None, None)


def format_day(day: str) -> str:
    """Format a day string (YYYYMMDD) as a human-readable date.

    Parameters
    ----------
    day:
        Day in YYYYMMDD format.

    Returns
    -------
    str
        Formatted date like "Friday, January 24, 2026".
        Returns the original string if parsing fails.

    Examples
    --------
    >>> format_day("20260124")
    "Friday, January 24, 2026"
    """
    try:
        dt = datetime.strptime(day, "%Y%m%d")
        return dt.strftime("%A, %B %d, %Y")
    except ValueError:
        return day


def iso_date(day: str) -> str:
    """Convert a day string (YYYYMMDD) to ISO format (YYYY-MM-DD).

    Parameters
    ----------
    day:
        Day in YYYYMMDD format.

    Returns
    -------
    str
        ISO formatted date like "2026-01-24".
    """
    return f"{day[:4]}-{day[4:6]}-{day[6:8]}"


def format_segment_times(segment: str) -> tuple[str, str] | tuple[None, None]:
    """Format segment start and end times as human-readable strings.

    Parameters
    ----------
    segment:
        Segment key in HHMMSS_LEN format (e.g., "143022_300").

    Returns
    -------
    tuple[str, str] | tuple[None, None]
        Tuple of (start_time, end_time) as formatted strings like "2:30 PM".
        Returns (None, None) if segment format is invalid.

    Examples
    --------
    >>> format_segment_times("143022_300")
    ("2:30 PM", "2:35 PM")
    >>> format_segment_times("090000_3600")
    ("9:00 AM", "10:00 AM")
    """
    start_time, end_time = segment_parse(segment)
    if start_time is None or end_time is None:
        return (None, None)

    return (_format_time(start_time), _format_time(end_time))


def _format_time(t: datetime.time) -> str:
    """Format a time as 12-hour with AM/PM, no leading zero on hour.

    Uses lstrip('0') for cross-platform compatibility (%-I is Unix-only).
    """
    return datetime.combine(datetime.today(), t).strftime("%I:%M %p").lstrip("0")


def _load_default_config() -> dict[str, Any]:
    """Load the default journal configuration from journal_default.json.

    Returns
    -------
    dict
        Default configuration structure.
    """
    default_path = Path(__file__).parent / "journal_default.json"
    with open(default_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Cached default config (loaded once at first use)
_default_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Return the journal configuration from config/journal.json.

    When no journal.json exists, returns a deep copy of the defaults from
    think/journal_default.json. Once journal.json exists it is the master
    and is returned as-is with no merging of defaults.

    Returns
    -------
    dict
        Journal configuration.
    """
    global _default_config
    if _default_config is None:
        _default_config = _load_default_config()

    journal = get_journal()
    config_path = Path(journal) / "config" / "journal.json"

    # Return defaults when no config file exists yet
    if not config_path.exists():
        return copy.deepcopy(_default_config)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        # Log error but return defaults to avoid breaking callers
        logging.getLogger(__name__).warning(
            "Failed to load config from %s: %s", config_path, exc
        )
        return copy.deepcopy(_default_config)


def _append_task_log(dir_path: str | Path, message: str) -> None:
    """Append ``message`` to ``task_log.txt`` inside ``dir_path``."""
    path = Path(dir_path) / "task_log.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{int(time.time())}\t{message}\n")
    except Exception:
        pass


def day_log(day: str, message: str) -> None:
    """Convenience wrapper to log message for ``day``."""
    _append_task_log(str(day_path(day)), message)


def journal_log(message: str) -> None:
    """Append ``message`` to the journal's ``task_log.txt``."""
    _append_task_log(get_journal(), message)


def day_input_summary(day: str) -> str:
    """Return a human-readable summary of recording data available for a day.

    Uses cluster_segments() to detect recording segments and computes
    total duration from segment keys (HHMMSS_LEN format).

    Parameters
    ----------
    day:
        Day in YYYYMMDD format.

    Returns
    -------
    str
        Human-readable summary like "No recordings", "Light activity: 2 segments,
        ~3 minutes", or "18 segments, ~7.5 hours".
    """
    from think.cluster import cluster_segments

    segments = cluster_segments(day)

    if not segments:
        return "No recordings"

    # Compute total duration from segment keys (HHMMSS_LEN format)
    total_seconds = 0
    for seg in segments:
        key = seg.get("key", "")
        if "_" in key:
            parts = key.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                total_seconds += int(parts[1])

    # Format duration
    if total_seconds < 60:
        duration_str = f"~{total_seconds} seconds"
    elif total_seconds < 3600:
        minutes = total_seconds / 60
        duration_str = f"~{minutes:.0f} minutes"
    else:
        hours = total_seconds / 3600
        duration_str = f"~{hours:.1f} hours"

    segment_count = len(segments)

    # Categorize activity level
    if segment_count < 5 or total_seconds < 1800:  # < 5 segments or < 30 min
        return f"Light activity: {segment_count} segment{'s' if segment_count != 1 else ''}, {duration_str}"
    else:
        return f"{segment_count} segments, {duration_str}"


def setup_cli(parser: argparse.ArgumentParser, *, parse_known: bool = False):
    """Parse command line arguments and configure logging.

    The parser will be extended with ``-v``/``--verbose`` and ``-d``/``--debug`` flags.
    The journal path is resolved via get_journal() which loads .env and auto-creates
    a default path if needed. Environment variables from the journal config's ``env``
    section are loaded as fallbacks for any keys not already set in the shell or .env.
    The parsed arguments are returned. If ``parse_known`` is ``True`` a tuple of
    ``(args, extra)`` is returned using :func:`argparse.ArgumentParser.parse_known_args`.
    """
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )
    if parse_known:
        args, extra = parser.parse_known_args()
    else:
        args = parser.parse_args()
        extra = None

    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(level=log_level)

    # Initialize journal path (may auto-create default)
    get_journal()

    # Load config env as fallback for missing environment variables
    # Precedence: shell env > .env file > journal config env
    config = get_config()
    for key, value in config.get("env", {}).items():
        if not os.environ.get(key):  # Only set if missing or empty
            os.environ[key] = str(value)

    return (args, extra) if parse_known else args


def parse_time_range(text: str) -> Optional[tuple[str, str, str]]:
    """Return ``(day, start, end)`` from a natural language time range.

    Parameters
    ----------
    text:
        Natural language description of a time range.

    Returns
    -------
    tuple[str, str, str] | None
        ``(day, start, end)`` if a single range within one day was detected.
        ``day`` is ``YYYYMMDD`` and ``start``/``end`` are ``HHMMSS``. ``None``
        if parsing fails.
    """

    try:
        result = timefhuman(text)
    except Exception as exc:  # pragma: no cover - unexpected library failure
        logging.info("timefhuman failed for %s: %s", text, exc)
        return None

    logging.debug("timefhuman(%s) -> %r", text, result)

    if len(result) != 1:
        logging.info("timefhuman did not return a single expression for %s", text)
        return None

    range_item = result[0]
    if not isinstance(range_item, tuple) or len(range_item) != 2:
        logging.info("Expected a range from %s but got %r", text, range_item)
        return None

    start_dt, end_dt = range_item
    if start_dt.date() != end_dt.date():
        logging.info("Range must be within a single day: %s -> %s", start_dt, end_dt)
        return None

    day = start_dt.strftime("%Y%m%d")
    start = start_dt.strftime("%H%M%S")
    end = end_dt.strftime("%H%M%S")
    return day, start, end


def get_raw_file(day: str, name: str) -> tuple[str, str, Any]:
    """Return raw file path, mime type and metadata for a transcript.

    Parameters
    ----------
    day:
        Day folder in ``YYYYMMDD`` format.
    name:
        Transcript filename such as ``HHMMSS/audio.jsonl``,
        ``HHMMSS/monitor_1_diff.json``, or ``HHMMSS/screen.jsonl``.

    Returns
    -------
    tuple[str, str, Any]
        ``(path, mime_type, metadata)`` where ``path`` is relative to the day
        directory (read from metadata header), ``mime_type`` is determined
        from the raw file extension, and ``metadata`` contains the parsed
        JSON data (empty on failure).
    """

    day_dir = day_path(day)
    transcript_path = day_dir / name

    rel = None
    meta: Any = {}

    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            if name.endswith(".jsonl"):
                # First line is metadata header with "raw" field
                first_line = f.readline().strip()
                if first_line:
                    header = json.loads(first_line)
                    rel = header.get("raw")

                # Read remaining lines as metadata
                meta = [json.loads(line) for line in f if line.strip()]
            else:
                # Non-JSONL format (e.g., _diff.json)
                meta = json.load(f)
                rel = meta.get("raw")
    except Exception:  # pragma: no cover - optional metadata
        logging.debug("Failed to read %s", transcript_path)

    if not rel:
        raise ValueError(f"No 'raw' field found in metadata for {name}")

    # Determine MIME type from raw file extension
    if rel.endswith(".flac"):
        mime = "audio/flac"
    elif rel.endswith(".ogg"):
        mime = "audio/ogg"
    elif rel.endswith(".m4a"):
        mime = "audio/mp4"
    elif rel.endswith(".png"):
        mime = "image/png"
    elif rel.endswith(".webm"):
        mime = "video/webm"
    elif rel.endswith(".mp4"):
        mime = "video/mp4"
    elif rel.endswith(".mov"):
        mime = "video/quicktime"
    else:
        # Default fallback for unknown types
        mime = "application/octet-stream"

    return rel, mime, meta


# =============================================================================
# SOL_* Environment Variable Helpers
# =============================================================================


def get_sol_day() -> str | None:
    """Read SOL_DAY from the environment."""
    return os.environ.get("SOL_DAY") or None


def get_sol_facet() -> str | None:
    """Read SOL_FACET from the environment."""
    return os.environ.get("SOL_FACET") or None


def get_sol_segment() -> str | None:
    """Read SOL_SEGMENT from the environment."""
    return os.environ.get("SOL_SEGMENT") or None


def get_sol_stream() -> str | None:
    """Read SOL_STREAM from the environment."""
    return os.environ.get("SOL_STREAM") or None


def get_sol_activity() -> str | None:
    """Read SOL_ACTIVITY from the environment."""
    return os.environ.get("SOL_ACTIVITY") or None


def resolve_sol_day(arg: str | None) -> str:
    """Return *arg* if provided, else SOL_DAY from env, else exit with error.

    Intended for CLI commands where ``day`` is required but can be supplied
    via the SOL_DAY environment variable as a convenience.
    """
    if arg:
        return arg
    env = get_sol_day()
    if env:
        return env
    import typer

    typer.echo("Error: day is required (pass as argument or set SOL_DAY).", err=True)
    raise typer.Exit(1)


def resolve_sol_facet(arg: str | None) -> str:
    """Return *arg* if provided, else SOL_FACET from env, else exit with error.

    Intended for CLI commands where ``facet`` is required but can be supplied
    via the SOL_FACET environment variable as a convenience.
    """
    if arg:
        return arg
    env = get_sol_facet()
    if env:
        return env
    import typer

    typer.echo(
        "Error: facet is required (pass as argument or set SOL_FACET).", err=True
    )
    raise typer.Exit(1)


def resolve_sol_segment(arg: str | None) -> str | None:
    """Return *arg* if provided, else SOL_SEGMENT from env, else None.

    Unlike :func:`resolve_sol_day` this does **not** error when missing
    because segment is typically optional.
    """
    if arg:
        return arg
    return get_sol_segment()


# =============================================================================
# Service Port Discovery
# =============================================================================


def find_available_port(host: str = "127.0.0.1") -> int:
    """Find an available port by binding to port 0.

    Uses the socket bind/getsockname/close pattern to let the OS assign
    an available port.

    Args:
        host: Host address to bind to (default: 127.0.0.1)

    Returns:
        Available port number
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    _, port = sock.getsockname()
    sock.close()
    return port


def write_service_port(service: str, port: int) -> None:
    """Write a service's port to the health directory.

    Creates $JOURNAL_PATH/health/{service}.port with the port number.

    Args:
        service: Service name (e.g., "convey", "cortex")
        port: Port number to write
    """
    health_dir = Path(get_journal()) / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    port_file = health_dir / f"{service}.port"
    port_file.write_text(str(port))


def read_service_port(service: str) -> int | None:
    """Read a service's port from the health directory.

    Args:
        service: Service name (e.g., "convey", "cortex")

    Returns:
        Port number if file exists and is valid, None otherwise
    """
    port_file = Path(get_journal()) / "health" / f"{service}.port"
    try:
        return int(port_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
