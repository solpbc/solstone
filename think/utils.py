# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, NamedTuple, Optional

import frontmatter
import platformdirs
from dotenv import load_dotenv
from timefhuman import timefhuman

DATE_RE = re.compile(r"\d{8}")
_journal_path_cache: str | None = None


def now_ms() -> int:
    """Return current time as Unix epoch milliseconds."""
    return int(time.time() * 1000)


MUSE_DIR = Path(__file__).parent.parent / "muse"
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Cached raw template content loaded from think/templates/*.md
_templates_cache: dict[str, str] | None = None


def _load_raw_templates() -> dict[str, str]:
    """Load raw template files from think/templates/ directory.

    Templates are cached on first load. Each .md file becomes a template
    variable named after its stem (e.g., daily_preamble.md -> $daily_preamble).

    Returns
    -------
    dict[str, str]
        Mapping of template variable names to their raw content (no substitution).
    """
    global _templates_cache
    if _templates_cache is not None:
        return _templates_cache

    _templates_cache = {}
    if TEMPLATES_DIR.is_dir():
        for md_path in TEMPLATES_DIR.glob("*.md"):
            var_name = md_path.stem
            try:
                post = frontmatter.load(
                    md_path,
                )
                _templates_cache[var_name] = post.content.strip()
            except Exception as exc:
                logging.debug("Failed to load template %s: %s", md_path, exc)

    return _templates_cache


def _load_templates(template_vars: dict[str, str] | None = None) -> dict[str, str]:
    """Load and substitute template files from think/templates/ directory.

    Raw templates are cached, but substitution is performed on each call
    to support context-dependent variables like $date and $segment_start.

    Parameters
    ----------
    template_vars:
        Optional variables to substitute into templates. Templates can use
        identity vars ($name, $preferred), context vars ($day, $date,
        $segment_start, $segment_end), and other template vars.

    Returns
    -------
    dict[str, str]
        Mapping of template variable names to their substituted content.
    """
    raw_templates = _load_raw_templates()

    if not template_vars:
        return dict(raw_templates)

    # Substitute variables into each template
    substituted = {}
    for var_name, content in raw_templates.items():
        try:
            template = Template(content)
            substituted[var_name] = template.safe_substitute(template_vars)
        except Exception as exc:
            logging.debug("Template substitution failed for %s: %s", var_name, exc)
            substituted[var_name] = content

    return substituted


class PromptContent(NamedTuple):
    """Container for prompt text, metadata, and its resolved path."""

    text: str
    path: Path
    metadata: dict[str, Any] = {}


class PromptNotFoundError(FileNotFoundError):
    """Raised when a prompt file cannot be located."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"Prompt file not found: {path}")


def _flatten_identity_to_template_vars(identity: dict[str, Any]) -> dict[str, str]:
    """Flatten identity config into template variables with uppercase-first versions.

    Parameters
    ----------
    identity:
        Identity configuration dictionary from get_config()['identity'].

    Returns
    -------
    dict[str, str]
        Template variables including flattened nested objects and uppercase-first versions.
        For example:
        - 'name' → identity['name']
        - 'pronouns_possessive' → identity['pronouns']['possessive']
        - 'Pronouns_possessive' → identity['pronouns']['possessive'].capitalize()
        - 'bio' → identity['bio']
    """
    template_vars: dict[str, str] = {}

    # Flatten top-level and nested values
    for key, value in identity.items():
        if isinstance(value, dict):
            # Flatten nested dictionaries with underscore separator
            for subkey, subvalue in value.items():
                var_name = f"{key}_{subkey}"
                template_vars[var_name] = str(subvalue)
                # Create uppercase-first version
                template_vars[var_name.capitalize()] = str(subvalue).capitalize()
        elif isinstance(value, (str, int, float, bool)):
            # Top-level scalar values
            template_vars[key] = str(value)
            # Create uppercase-first version
            template_vars[key.capitalize()] = str(value).capitalize()

    return template_vars


def load_prompt(
    name: str,
    base_dir: str | Path | None = None,
    *,
    include_journal: bool = False,
    context: dict[str, Any] | None = None,
) -> PromptContent:
    """Return the text contents, metadata, and path for a ``.md`` prompt file.

    Prompt files use JSON frontmatter for metadata. Supports Python
    string.Template variable substitution using:
    - Identity config from get_config()['identity']:
      - Top-level fields: $name, $preferred, $bio, $timezone
      - Nested fields with underscores: $pronouns_possessive, $pronouns_subject
      - Uppercase-first versions: $Pronouns_possessive, $Name, $Bio
    - Templates from think/templates/*.md:
      - Each file becomes a variable named after its stem
      - Example: daily_preamble.md -> $daily_preamble
      - Templates are pre-processed with identity and context vars, so templates
        can use $date, $preferred, etc. before being substituted into prompts

    Callers can provide additional context variables via the ``context`` parameter.
    Context variables override identity and template variables if there's a collision.
    Uppercase-first versions are automatically created for context variables.

    Parameters
    ----------
    name:
        Base filename of the prompt without the ``.md`` suffix. If the suffix is
        included, it will not be duplicated.
    base_dir:
        Optional directory containing the prompt file. Defaults to the directory
        of this module when not provided.
    include_journal:
        If True, prepends the content of ``think/journal.md`` to the requested
        prompt. Defaults to False. Context variables are passed through to the
        journal template as well.
    context:
        Optional dictionary of additional template variables. Values are converted
        to strings. For each key, an uppercase-first version is also created
        (e.g., ``{"day": "20250110"}`` adds both ``$day`` and ``$Day``).

    Returns
    -------
    PromptContent
        The prompt text (with surrounding whitespace removed and template variables
        substituted), the resolved path to the ``.md`` file, and metadata from
        the JSON frontmatter.
    """

    if not name:
        raise ValueError("Prompt name must be provided")

    if name.endswith(".md"):
        filename = name
    else:
        filename = f"{name}.md"

    prompt_dir = Path(base_dir) if base_dir is not None else Path(__file__).parent
    prompt_path = prompt_dir / filename
    try:
        post = frontmatter.load(
            prompt_path,
        )
        text = post.content.strip()
        metadata = dict(post.metadata)
    except FileNotFoundError as exc:  # pragma: no cover - caller handles missing prompt
        raise PromptNotFoundError(prompt_path) from exc

    # Perform template substitution
    try:
        config = get_config()
        identity = config.get("identity", {})
        template_vars = _flatten_identity_to_template_vars(identity)

        # Merge caller-provided context (overrides identity vars if collision)
        if context:
            for key, value in context.items():
                str_value = str(value)
                template_vars[key] = str_value
                # Add uppercase-first version
                template_vars[key.capitalize()] = str_value.capitalize()

        # Load templates with identity and context vars so templates can use them
        templates = _load_templates(template_vars)
        template_vars.update(templates)

        # Use safe_substitute to avoid errors for undefined variables
        template = Template(text)
        text = template.safe_substitute(template_vars)
    except Exception as exc:
        # Log but don't fail - return original text if substitution fails
        logging.debug("Template substitution failed for %s: %s", prompt_path, exc)

    # Prepend journal content if requested
    if include_journal and name != "journal":
        journal_content = load_prompt("journal", context=context)
        text = f"{journal_content.text}\n\n{text}"

    return PromptContent(text=text, path=prompt_path, metadata=metadata)


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
        # Look for YYYYMMDD/HHMMSS_LEN pattern
        for i, part in enumerate(path_parts):
            if part.isdigit() and len(part) == 8 and i + 1 < len(path_parts):
                name = path_parts[i + 1]
                break
        else:
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


def get_config() -> dict[str, Any]:
    """Return the journal configuration from config/journal.json.

    Returns
    -------
    dict
        Journal configuration with at least an 'identity' key containing
        name, preferred, bio, pronouns, aliases, email_addresses, and
        timezone fields. Returns default empty structure if config file doesn't exist.
    """
    # Default identity structure - defined once
    default_identity = {
        "name": "",
        "preferred": "",
        "bio": "",
        "pronouns": {
            "subject": "",
            "object": "",
            "possessive": "",
            "reflexive": "",
        },
        "aliases": [],
        "email_addresses": [],
        "timezone": "",
    }

    journal = get_journal()
    config_path = Path(journal) / "config" / "journal.json"

    # Return default structure if file doesn't exist
    if not config_path.exists():
        return {"identity": default_identity.copy()}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Ensure identity section exists with all required fields
        if "identity" not in config:
            config["identity"] = {}

        # Fill in any missing fields with defaults
        for key, default in default_identity.items():
            if key not in config["identity"]:
                config["identity"][key] = default

        return config
    except (json.JSONDecodeError, OSError) as exc:
        # Log error but return default structure to avoid breaking callers
        logging.getLogger(__name__).warning(
            "Failed to load config from %s: %s", config_path, exc
        )
        return {"identity": default_identity.copy()}


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


def get_output_topic(key: str) -> str:
    """Convert agent/generator key to filesystem-safe basename (no extension).

    Parameters
    ----------
    key:
        Generator key in format "topic" (system) or "app:topic" (app).

    Returns
    -------
    str
        Filesystem-safe name: "topic" or "_app_topic".

    Examples
    --------
    >>> get_output_topic("activity")
    'activity'
    >>> get_output_topic("chat:sentiment")
    '_chat_sentiment'
    """
    if ":" in key:
        app, topic = key.split(":", 1)
        return f"_{app}_{topic}"
    return key


def key_to_context(key: str) -> str:
    """Convert muse config key to context pattern.

    Parameters
    ----------
    key:
        Muse config key in format "name" (system) or "app:name" (app).

    Returns
    -------
    str
        Context pattern: "muse.system.{name}" or "muse.{app}.{name}".

    Examples
    --------
    >>> key_to_context("meetings")
    'muse.system.meetings'
    >>> key_to_context("entities:observer")
    'muse.entities.observer'
    """
    if ":" in key:
        app, name = key.split(":", 1)
        return f"muse.{app}.{name}"
    return f"muse.system.{key}"


def get_output_path(
    day_dir: os.PathLike[str],
    key: str,
    segment: str | None = None,
    output_format: str | None = None,
) -> Path:
    """Return output path for generator agent output.

    Shared utility for determining where to write generator results.
    Used by both think/generate.py and think/cortex.py.

    Parameters
    ----------
    day_dir:
        Day directory path (YYYYMMDD).
    key:
        Generator key or agent name (e.g., "activity", "chat:sentiment",
        "decisionalizer", "entities:observer").
    segment:
        Optional segment key (HHMMSS_LEN) for segment-level output.
    output_format:
        Output format - "json" for JSON, anything else for markdown.

    Returns
    -------
    Path
        Output file path:
        - With segment: YYYYMMDD/{segment}/{topic}.{ext}
        - Without segment: YYYYMMDD/agents/{topic}.{ext}
        Where topic is derived from key and ext is "json" or "md".
    """
    day = Path(day_dir)
    topic = get_output_topic(key)
    ext = "json" if output_format == "json" else "md"

    if segment:
        # Segment output goes directly in segment directory
        return day / segment / f"{topic}.{ext}"
    else:
        # Daily output goes in agents/ subdirectory
        return day / "agents" / f"{topic}.{ext}"


def _load_prompt_metadata(md_path: Path) -> dict[str, object]:
    """Load prompt metadata from .md file with JSON frontmatter.

    Parameters
    ----------
    md_path:
        Path to the .md prompt file with JSON frontmatter.

    Returns
    -------
    dict
        Metadata dict with path, mtime, color, and frontmatter fields.
    """
    mtime = int(md_path.stat().st_mtime)
    info: dict[str, object] = {
        "path": str(md_path),
        "mtime": mtime,
    }

    try:
        post = frontmatter.load(
            md_path,
        )
        if post.metadata:
            info.update(post.metadata)
    except Exception as exc:  # pragma: no cover - metadata optional
        logging.debug("Error reading frontmatter from %s: %s", md_path, exc)

    # Apply default color if not specified
    if "color" not in info:
        info["color"] = "#6c757d"

    return info


def get_muse_configs(
    *,
    has_tools: bool | None = None,
    has_output: bool | None = None,
    schedule: str | None = None,
    include_disabled: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load muse configs from system and app directories.

    Unified function for loading both tool-using agents and generators from
    muse/*.md files. Filters based on presence of tools/output fields.

    Args:
        has_tools: If True, only configs with "tools" field (agents).
            If False, only configs without "tools" field.
            If None, no filtering on tools presence.
        has_output: If True, only configs with "output" field (generators).
            If False, only configs without "output" field.
            If None, no filtering on output presence.
        schedule: If provided, only configs where schedule matches this value
            (e.g., "segment", "daily").
        include_disabled: If True, include configs with disabled=True.
            Default False (for processing pipelines).

    Returns:
        Dictionary mapping config keys to their metadata including:
        - path: Path to the .md file
        - source: "system" or "app"
        - app: App name (only for app configs)
        - All fields from frontmatter
    """
    configs: dict[str, dict[str, Any]] = {}

    def matches_filter(info: dict) -> bool:
        """Check if config matches the filter criteria."""
        # Check has_tools filter
        if has_tools is True and "tools" not in info:
            return False
        if has_tools is False and "tools" in info:
            return False

        # Check has_output filter
        if has_output is True and "output" not in info:
            return False
        if has_output is False and "output" in info:
            return False

        # Check specific schedule value
        if schedule is not None and info.get("schedule") != schedule:
            return False

        # Check disabled status
        if not include_disabled and info.get("disabled", False):
            return False

        return True

    # System configs from muse/
    if MUSE_DIR.is_dir():
        for md_path in sorted(MUSE_DIR.glob("*.md")):
            name = md_path.stem
            info = _load_prompt_metadata(md_path)

            if not matches_filter(info):
                continue

            info["source"] = "system"
            configs[name] = info

    # App configs from apps/*/muse/
    apps_dir = Path(__file__).parent.parent / "apps"
    if apps_dir.is_dir():
        for app_path in sorted(apps_dir.iterdir()):
            if not app_path.is_dir() or app_path.name.startswith("_"):
                continue
            app_muse_dir = app_path / "muse"
            if not app_muse_dir.is_dir():
                continue
            app_name = app_path.name
            for md_path in sorted(app_muse_dir.glob("*.md")):
                item_name = md_path.stem
                info = _load_prompt_metadata(md_path)

                if not matches_filter(info):
                    continue

                key = f"{app_name}:{item_name}"
                info["source"] = "app"
                info["app"] = app_name
                configs[key] = info

    # Merge journal config overrides from providers.contexts
    providers_config = get_config().get("providers", {})
    contexts = providers_config.get("contexts", {})

    for key, info in configs.items():
        context_key = key_to_context(key)

        # Check for exact match in contexts
        override = contexts.get(context_key)
        if override and isinstance(override, dict):
            # Merge supported override fields
            if "disabled" in override:
                info["disabled"] = override["disabled"]
            if "extract" in override:
                info["extract"] = override["extract"]
            if "tier" in override:
                info["tier"] = override["tier"]
            if "provider" in override:
                info["provider"] = override["provider"]

    return configs


def _resolve_agent_path(name: str) -> tuple[Path, str]:
    """Resolve agent name to directory path and agent filename.

    Parameters
    ----------
    name:
        Agent name - either system agent (e.g., "default") or
        app-namespaced agent (e.g., "chat:helper").

    Returns
    -------
    tuple[Path, str]
        (agent_directory, agent_name) tuple.
    """
    if ":" in name:
        # App agent: "chat:helper" -> apps/chat/muse/helper
        app, agent_name = name.split(":", 1)
        agent_dir = Path(__file__).parent.parent / "apps" / app / "muse"
    else:
        # System agent: "default" -> muse/default
        agent_dir = MUSE_DIR
        agent_name = name
    return agent_dir, agent_name


# Default instruction configuration
_DEFAULT_INSTRUCTIONS = {
    "system": "journal",
    "facets": "short",
    "sources": {
        "audio": True,
        "screen": True,
        "agents": False,
    },
}


def _merge_instructions_config(defaults: dict, overrides: dict | None) -> dict:
    """Merge instruction config overrides into defaults.

    Handles nested "sources" dict specially.

    Parameters
    ----------
    defaults:
        Default instruction configuration.
    overrides:
        Optional overrides from .json "instructions" key.

    Returns
    -------
    dict
        Merged configuration.
    """
    if not overrides:
        return defaults.copy()

    result = defaults.copy()

    # Merge top-level keys
    for key in ("system", "facets"):
        if key in overrides:
            result[key] = overrides[key]

    # Merge sources dict if present
    if "sources" in overrides and isinstance(overrides["sources"], dict):
        result["sources"] = {**defaults.get("sources", {}), **overrides["sources"]}

    return result


def compose_instructions(
    *,
    user_prompt: str | None = None,
    user_prompt_dir: Path | None = None,
    facet: str | None = None,
    include_datetime: bool = True,
    config_overrides: dict | None = None,
) -> dict:
    """Compose instruction components for agents or generators.

    This is the shared function for building system_instruction, user_instruction,
    extra_context, and sources configuration. Both agents and generators use this
    to ensure consistent prompt composition.

    Parameters
    ----------
    user_prompt:
        Name of the user instruction prompt to load (e.g., "default" for agents).
        If None, no user_instruction is included (typical for generators).
    user_prompt_dir:
        Directory to load user_prompt from. If None, uses think/ directory.
    facet:
        Optional facet name to focus on. When provided, extra_context includes
        detailed information for just this facet instead of all facets.
    include_datetime:
        Whether to include current date/time in extra_context. Default True
        for agents (real-time chat), typically False for generators (past analysis).
    config_overrides:
        Optional dict from .json "instructions" key. Supported keys:
        - "system": prompt name for system instruction (default: "journal")
        - "facets": "none" | "short" | "detailed" (default: "short")
        - "sources": {"audio": bool, "screen": bool, "agents": bool}

    Returns
    -------
    dict
        Composed instruction configuration:
        - system_instruction: str - loaded from "system" prompt
        - system_prompt_name: str - name of system prompt (for cache keys)
        - user_instruction: str | None - loaded from user_prompt if provided
        - extra_context: str | None - facets + datetime
        - sources: dict - {"audio": bool, "screen": bool, "agents": bool}
    """
    # Merge defaults with overrides
    cfg = _merge_instructions_config(_DEFAULT_INSTRUCTIONS, config_overrides)

    result: dict = {}

    # Load system instruction
    system_name = cfg.get("system", "journal")
    system_prompt = load_prompt(system_name)
    result["system_instruction"] = system_prompt.text
    result["system_prompt_name"] = system_name

    # Load user instruction if specified
    if user_prompt:
        base_dir = user_prompt_dir if user_prompt_dir else Path(__file__).parent
        user_prompt_obj = load_prompt(user_prompt, base_dir=base_dir)
        result["user_instruction"] = user_prompt_obj.text
    else:
        result["user_instruction"] = None

    # Build extra_context based on facets setting
    extra_parts = []
    facets_mode = cfg.get("facets", "short")

    # Focused facet always gets context (facets setting only affects plural)
    if facet:
        try:
            from think.facets import facet_summary

            detailed = facet_summary(facet)
            extra_parts.append(f"## Facet Focus\n{detailed}")
        except Exception:
            pass  # Ignore if facet can't be loaded
    elif facets_mode != "none":
        # General mode: all facets (controlled by facets setting)
        try:
            from think.facets import facet_summaries

            detailed = facets_mode == "detailed"
            facets_summary = facet_summaries(detailed=detailed)
            if facets_summary and facets_summary != "No facets found.":
                extra_parts.append(facets_summary)
        except Exception:
            pass  # Ignore if facets can't be loaded

    # Add current date/time if requested
    if include_datetime:
        now = datetime.now()
        try:
            import tzlocal

            local_tz = tzlocal.get_localzone()
            now_local = now.astimezone(local_tz)
            time_str = now_local.strftime("%A, %B %d, %Y at %I:%M %p %Z")
        except Exception:
            time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
        extra_parts.append(f"## Current Date and Time\nToday is {time_str}")

    result["extra_context"] = "\n\n".join(extra_parts).strip() if extra_parts else None

    # Include sources config
    result["sources"] = cfg.get("sources", _DEFAULT_INSTRUCTIONS["sources"])

    return result


def source_is_enabled(value: bool | str) -> bool:
    """Check if a source should be loaded based on its config value.

    Sources can be configured as:
    - False: don't load
    - True: load if available
    - "required": load (and generation will fail if none found)

    Both True and "required" mean the source should be loaded.

    Args:
        value: The source config value (bool or "required" string)

    Returns:
        True if the source should be loaded, False otherwise.
    """
    return value is True or value == "required"


def source_is_required(value: bool | str) -> bool:
    """Check if a source must have content for generation to proceed.

    Args:
        value: The source config value (bool or "required" string)

    Returns:
        True if the source is required (generation should skip if no content).
    """
    return value == "required"


def get_agent(name: str = "default", facet: str | None = None) -> dict:
    """Return complete agent configuration by name.

    Loads configuration from .md file with JSON frontmatter and instruction text,
    merges with runtime context.

    Parameters
    ----------
    name:
        Agent name to load. Can be a system agent (e.g., "default")
        or an app-namespaced agent (e.g., "chat:helper" for apps/chat/muse/helper).
    facet:
        Optional facet name to focus on. When provided, includes detailed
        information for just this facet (with full entity details) instead
        of summaries of all facets.

    Returns
    -------
    dict
        Complete agent configuration including system_instruction, user_instruction,
        extra_context, model, backend, etc.
    """
    # Resolve agent path based on namespace
    agent_dir, agent_name = _resolve_agent_path(name)

    # Verify agent prompt file exists
    md_path = agent_dir / f"{agent_name}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"Agent not found: {name}")

    # Load config from frontmatter
    post = frontmatter.load(
        md_path,
    )
    config = dict(post.metadata) if post.metadata else {}

    # Extract instructions config if present
    instructions_config = config.pop("instructions", None)

    # Use compose_instructions for consistent prompt composition
    instructions = compose_instructions(
        user_prompt=agent_name,
        user_prompt_dir=agent_dir,
        facet=facet,
        include_datetime=True,
        config_overrides=instructions_config,
    )

    # Merge instruction results into config
    config["system_instruction"] = instructions["system_instruction"]
    config["user_instruction"] = instructions["user_instruction"]
    if instructions["extra_context"]:
        config["extra_context"] = instructions["extra_context"]

    # Set agent name
    config["name"] = name

    return config


def create_mcp_client(http_uri: str) -> Any:
    """Return a FastMCP HTTP client for solstone tools."""

    http_uri = http_uri.strip()
    if not http_uri:
        raise RuntimeError("MCP server URL not provided")

    from fastmcp import Client

    return Client(http_uri, timeout=15.0)


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
