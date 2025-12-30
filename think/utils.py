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

from dotenv import load_dotenv
from timefhuman import timefhuman

DATE_RE = re.compile(r"\d{8}")

# Insight colors are now stored in each insight's JSON metadata file

AGENT_DIR = Path(__file__).parent.parent / "muse" / "agents"


class PromptContent(NamedTuple):
    """Container for prompt text and its resolved path."""

    text: str
    path: Path


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
    name: str, base_dir: str | Path | None = None, *, include_journal: bool = False
) -> PromptContent:
    """Return the text contents and path for a ``.txt`` prompt file.

    Supports Python string.Template variable substitution using identity config
    from get_config()['identity']. Template variables include:
    - Top-level fields: $name, $preferred, $bio, $timezone
    - Nested fields with underscores: $pronouns_possessive, $pronouns_subject
    - Uppercase-first versions: $Pronouns_possessive, $Name, $Bio

    Parameters
    ----------
    name:
        Base filename of the prompt without the ``.txt`` suffix. If the suffix is
        included, it will not be duplicated.
    base_dir:
        Optional directory containing the prompt file. Defaults to the directory
        of this module when not provided.
    include_journal:
        If True, prepends the content of ``think/journal.txt`` to the requested
        prompt. Defaults to False.

    Returns
    -------
    PromptContent
        The prompt text (with surrounding whitespace removed and template variables
        substituted) and the resolved path to the ``.txt`` file.
    """

    if not name:
        raise ValueError("Prompt name must be provided")

    filename = name if name.endswith(".txt") else f"{name}.txt"
    prompt_dir = Path(base_dir) if base_dir is not None else Path(__file__).parent
    prompt_path = prompt_dir / filename
    try:
        text = prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:  # pragma: no cover - caller handles missing prompt
        raise PromptNotFoundError(prompt_path) from exc

    # Perform template substitution
    try:
        config = get_config()
        identity = config.get("identity", {})
        template_vars = _flatten_identity_to_template_vars(identity)

        # Use safe_substitute to avoid errors for undefined variables
        template = Template(text)
        text = template.safe_substitute(template_vars)
    except Exception as exc:
        # Log but don't fail - return original text if substitution fails
        logging.debug("Template substitution failed for %s: %s", prompt_path, exc)

    # Prepend journal content if requested
    if include_journal and name != "journal":
        journal_content = load_prompt("journal")
        text = f"{journal_content.text}\n\n{text}"

    return PromptContent(text=text, path=prompt_path)


def day_path(day: Optional[str] = None) -> Path:
    """Return absolute path for a day from ``JOURNAL_PATH`` environment variable.

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
    RuntimeError
        If JOURNAL_PATH is not set.
    ValueError
        If day format is invalid.
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

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

    Uses JOURNAL_PATH from environment (must be set via load_dotenv() or setup_cli()).

    Returns
    -------
    dict[str, str]
        Mapping of day folder names to their full paths.
        Example: {"20250101": "/path/to/journal/20250101", ...}

    Raises
    ------
    RuntimeError
        If JOURNAL_PATH environment variable is not set.
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")
    if not os.path.isdir(journal):
        return {}

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


def get_config() -> dict[str, Any]:
    """Return the journal configuration from config/journal.json.

    Returns
    -------
    dict
        Journal configuration with at least an 'identity' key containing
        name, preferred, bio, pronouns, aliases, email_addresses, and
        timezone fields. Returns default empty structure if config file doesn't exist.

    Raises
    ------
    RuntimeError
        If JOURNAL_PATH is not set.
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

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

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
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if journal:
        _append_task_log(journal, message)


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

    The parser will be extended with ``-v``/``--verbose`` and ``-d``/``--debug`` flags. Environment
    variables from ``.env`` are loaded and ``JOURNAL_PATH`` is validated. The
    parsed arguments are returned. If ``parse_known`` is ``True`` a tuple of
    ``(args, extra)`` is returned using :func:`argparse.ArgumentParser.parse_known_args`.
    """

    load_dotenv()
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

    journal = os.getenv("JOURNAL_PATH")
    if not journal or not os.path.isdir(journal):
        parser.error("JOURNAL_PATH not set or invalid")

    return (args, extra) if parse_known else args


def get_insight_topic(key: str) -> str:
    """Convert insight key to filesystem-safe basename (no extension).

    Parameters
    ----------
    key:
        Insight key in format "topic" (system) or "app:topic" (app).

    Returns
    -------
    str
        Filesystem-safe name: "topic" or "_app_topic".

    Examples
    --------
    >>> get_insight_topic("activity")
    'activity'
    >>> get_insight_topic("chat:sentiment")
    '_chat_sentiment'
    """
    if ":" in key:
        app, topic = key.split(":", 1)
        return f"_{app}_{topic}"
    return key


def _load_insight_metadata(txt_path: Path) -> dict[str, object]:
    """Load insight metadata from .txt and optional .json file.

    Parameters
    ----------
    txt_path:
        Path to the .txt prompt file.

    Returns
    -------
    dict
        Metadata dict with path, mtime, color, and any JSON fields.
    """
    mtime = int(txt_path.stat().st_mtime)
    info: dict[str, object] = {
        "path": str(txt_path),
        "mtime": mtime,
    }
    json_path = txt_path.with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                info.update(data)
                if "color" not in info:
                    info["color"] = "#6c757d"
        except Exception as exc:  # pragma: no cover - metadata optional
            logging.debug("Error reading %s: %s", json_path, exc)
            info["color"] = "#6c757d"
    else:
        info["color"] = "#6c757d"
    return info


def get_insights() -> dict[str, dict[str, object]]:
    """Return available insights with metadata.

    Scans both system insights (think/insights/) and app insights
    (apps/*/insights/). Each key is the insight name:
    - System: "activity", "meetings"
    - App: "app:topic" (e.g., "chat:sentiment")

    The value contains the ``path`` to the ``.txt`` file, the ``color``
    from the metadata JSON, the file ``mtime``, a ``source`` field
    ("system" or "app"), and any keys loaded from a matching ``.json``
    metadata file.
    """
    insights: dict[str, dict[str, object]] = {}

    # System insights from think/insights/
    system_dir = Path(__file__).parent / "insights"
    for txt_path in sorted(system_dir.glob("*.txt")):
        name = txt_path.stem
        info = _load_insight_metadata(txt_path)
        info["source"] = "system"
        insights[name] = info

    # App insights from apps/*/insights/
    apps_dir = Path(__file__).parent.parent / "apps"
    if apps_dir.is_dir():
        for app_path in sorted(apps_dir.iterdir()):
            if not app_path.is_dir() or app_path.name.startswith("_"):
                continue
            app_insights_dir = app_path / "insights"
            if not app_insights_dir.is_dir():
                continue
            app_name = app_path.name
            for txt_path in sorted(app_insights_dir.glob("*.txt")):
                topic = txt_path.stem
                key = f"{app_name}:{topic}"
                info = _load_insight_metadata(txt_path)
                info["source"] = "app"
                info["app"] = app_name
                insights[key] = info

    return insights


def _resolve_agent_path(persona: str) -> tuple[Path, str]:
    """Resolve agent persona to directory path and agent name.

    Parameters
    ----------
    persona:
        Agent key - either system agent name (e.g., "default") or
        app-namespaced agent (e.g., "chat:helper").

    Returns
    -------
    tuple[Path, str]
        (agent_directory, agent_name) tuple.
    """
    if ":" in persona:
        # App agent: "chat:helper" -> apps/chat/agents/helper
        app, agent_name = persona.split(":", 1)
        agent_dir = Path(__file__).parent.parent / "apps" / app / "agents"
    else:
        # System agent: "default" -> muse/agents/default
        agent_dir = AGENT_DIR
        agent_name = persona
    return agent_dir, agent_name


def get_agent(persona: str = "default", facet: str | None = None) -> dict:
    """Return complete agent configuration for a persona.

    Loads JSON configuration and instruction text, merges with runtime context.

    Parameters
    ----------
    persona:
        Name of the persona to load. Can be a system agent name (e.g., "default")
        or an app-namespaced agent (e.g., "chat:helper" for apps/chat/agents/helper).
    facet:
        Optional facet name to focus on. When provided, includes detailed
        information for just this facet (with full entity details) instead
        of summaries of all facets.

    Returns
    -------
    dict
        Complete agent configuration including instruction, model, backend, etc.
    """
    config = {}

    # Resolve agent path based on namespace
    agent_dir, agent_name = _resolve_agent_path(persona)

    # Load JSON config if exists
    json_path = agent_dir / f"{agent_name}.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Load instruction text
    txt_path = agent_dir / f"{agent_name}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Agent persona not found: {persona}")
    prompt_data = load_prompt(agent_name, base_dir=agent_dir, include_journal=True)
    config["instruction"] = prompt_data.text

    # Add runtime context (facets with entities)
    extra_parts = []

    # Add facet context - either focused single facet or all facets summary
    journal = os.getenv("JOURNAL_PATH")
    if journal:
        try:
            if facet:
                # Focused mode: detailed view of single facet with full entities
                from think.facets import facet_summary

                detailed = facet_summary(facet)
                extra_parts.append(f"## Facet Focus\n{detailed}")
            else:
                # General mode: summary of all facets
                from think.facets import facet_summaries

                facets_summary = facet_summaries()
                if facets_summary and facets_summary != "No facets found.":
                    extra_parts.append(facets_summary)
        except Exception:
            pass  # Ignore if facets can't be loaded

    # Add insights to agent instructions
    insights = get_insights()
    if insights:
        insights_list = []
        for insight_name, info in sorted(insights.items()):
            desc = str(info.get("contains", "")).replace("\n", " ").strip()
            if desc:
                insights_list.append(f"- `{insight_name}`: {desc}")
            else:
                insights_list.append(f"- `{insight_name}`")
        extra_parts.append("## Available Insights\n" + "\n".join(insights_list))

    # Add current date/time
    now = datetime.now()
    try:
        import tzlocal

        local_tz = tzlocal.get_localzone()
        now_local = now.astimezone(local_tz)
        time_str = now_local.strftime("%A, %B %d, %Y at %I:%M %p %Z")
    except Exception:
        time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")
    extra_parts.append(f"## Current Date and Time\nToday is {time_str}")

    if extra_parts:
        config["extra_context"] = "\n\n".join(extra_parts).strip()

    # Set persona name
    config["persona"] = persona

    return config


def create_mcp_client(http_uri: str) -> Any:
    """Return a FastMCP HTTP client for Sunstone tools."""

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
    elif rel.endswith(".png"):
        mime = "image/png"
    elif rel.endswith(".webm"):
        mime = "video/webm"
    else:
        # Default fallback for unknown types
        mime = "application/octet-stream"

    return rel, mime, meta


def get_agents() -> dict[str, dict[str, Any]]:
    """Load agent metadata from system and app directories.

    Scans both system agents (muse/agents/) and app agents (apps/*/agents/).
    System agents use simple keys like "default", while app agents are
    namespaced as "app:agent" (e.g., "chat:helper").

    Returns:
        Dictionary mapping agent keys to their metadata including:
        - title: Display title for the agent
        - source: "system" or "app"
        - app: App name (only for app agents)
        - All configuration fields from get_agent()
    """
    agents = {}

    # System agents from muse/agents/
    if AGENT_DIR.exists():
        for txt_path in sorted(AGENT_DIR.glob("*.txt")):
            agent_id = txt_path.stem
            try:
                config = get_agent(agent_id)
                config["title"] = config.get("title", agent_id)
                config["source"] = "system"
                agents[agent_id] = config
            except Exception:
                pass  # Skip agents that can't be loaded

    # App agents from apps/*/agents/
    apps_dir = Path(__file__).parent.parent / "apps"
    if apps_dir.is_dir():
        for app_path in sorted(apps_dir.iterdir()):
            if not app_path.is_dir() or app_path.name.startswith("_"):
                continue
            agents_dir = app_path / "agents"
            if not agents_dir.is_dir():
                continue
            app_name = app_path.name
            for txt_path in sorted(agents_dir.glob("*.txt")):
                agent_name = txt_path.stem
                key = f"{app_name}:{agent_name}"
                try:
                    config = get_agent(key)
                    config["title"] = config.get("title", agent_name)
                    config["source"] = "app"
                    config["app"] = app_name
                    agents[key] = config
                except Exception:
                    pass  # Skip agents that can't be loaded

    return agents
