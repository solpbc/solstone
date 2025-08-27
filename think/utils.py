import argparse
import json
import logging
import os
import re
import time
import zoneinfo
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

from dotenv import load_dotenv
from timefhuman import timefhuman

DATE_RE = re.compile(r"\d{8}")

# Topic colors are now stored in each topic's JSON metadata file

AGENT_DIR = Path(__file__).with_name("agents")


def day_path(day: str) -> str:
    """Return absolute path for *day* from ``JOURNAL_PATH`` environment variable."""
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")
    if not DATE_RE.fullmatch(day):
        raise ValueError("day must be in YYYYMMDD format")
    return os.path.join(journal, day)


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
    _append_task_log(day_path(day), message)


def journal_log(message: str) -> None:
    """Append ``message`` to the journal's ``task_log.txt``."""
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if journal:
        _append_task_log(journal, message)


def touch_health(name: str) -> None:
    """Update the journal's ``name`` heartbeat file.

    The journal path is read from ``JOURNAL_PATH`` in the environment.
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return
    path = Path(journal) / "health" / f"{name}.up"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    except Exception:
        pass


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


def get_topics() -> dict[str, dict[str, object]]:
    """Return available topics with metadata.

    Each key is the topic name. The value contains the ``path`` to the
    ``.txt`` file, the ``color`` from the metadata JSON, the file
    ``mtime`` and any keys loaded from a matching ``.json`` metadata file.
    """

    topics_dir = Path(__file__).parent / "topics"
    topics: dict[str, dict[str, object]] = {}
    for txt_path in sorted(topics_dir.glob("*.txt")):
        name = txt_path.stem
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
                    # Ensure color exists, fallback to a default if missing
                    if "color" not in info:
                        info["color"] = "#6c757d"  # Default gray color
            except Exception as exc:  # pragma: no cover - metadata optional
                logging.debug("Error reading %s: %s", json_path, exc)
                info["color"] = "#6c757d"  # Default gray color
        else:
            info["color"] = "#6c757d"  # Default gray color
        topics[name] = info
    return topics


def agent_instructions(persona: str = "default") -> Tuple[str, str, dict[str, object]]:
    """Return system instruction, initial user context and metadata for ``persona``."""

    txt_path = AGENT_DIR / f"{persona}.txt"
    system_instruction = txt_path.read_text(encoding="utf-8")

    meta: dict[str, object] = {}
    json_path = txt_path.with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                meta.update(data)
        except Exception as exc:  # pragma: no cover - optional metadata
            logging.debug("Error reading %s: %s", json_path, exc)

    extra_parts: list[str] = []
    journal = os.getenv("JOURNAL_PATH")
    if journal:
        ent_path = Path(journal) / "entities.md"
        if ent_path.is_file():
            entities = ent_path.read_text(encoding="utf-8").strip()
            if entities:
                extra_parts.append("## Well-Known Entities\n" + entities)

        # Add domains to agent instructions
        try:
            from think.domains import get_domains

            domains = get_domains()
            if domains:
                lines = ["## Domains"]
                for name, info in sorted(domains.items()):
                    title = str(info.get("title", name))
                    desc = str(info.get("description", ""))
                    emoji = str(info.get("emoji", ""))

                    # Format with emoji if available
                    if emoji:
                        title_with_emoji = f"{emoji} {title}"
                    else:
                        title_with_emoji = title

                    # Build domain line with hashtag format
                    if desc:
                        lines.append(f"* Domain: {title_with_emoji} (#{name}) - {desc}")
                    else:
                        lines.append(f"* Domain: {title_with_emoji} (#{name})")
                extra_parts.append("\n".join(lines))
        except Exception as exc:
            logging.debug("Error loading domains: %s", exc)

    topics = get_topics()
    if topics:
        lines = [
            "## Topics",
            "These are the topics available for use in tool and resource requests:",
        ]
        for name, info in sorted(topics.items()):
            desc = str(info.get("contains", ""))
            lines.append(f"* Topic: `{name}`: {desc}")
        extra_parts.append("\n".join(lines))

    now = datetime.now()
    try:
        local_tz = zoneinfo.ZoneInfo(str(now.astimezone().tzinfo))
        now_local = now.astimezone(local_tz)
        time_str = now_local.strftime("%A, %B %d, %Y at %I:%M %p %Z")
    except Exception:
        time_str = now.strftime("%A, %B %d, %Y at %I:%M %p")

    extra_parts.append(f"## Current Date and Time\n{time_str}")

    extra_context = "\n\n".join(extra_parts).strip()
    return system_instruction, extra_context, meta


def create_mcp_client() -> Any:
    """Return a fastMCP HTTP client for Sunstone tools."""

    # Auto-discover HTTP server URI from journal
    journal_path = os.getenv("JOURNAL_PATH")
    if not journal_path:
        raise RuntimeError("JOURNAL_PATH not set")

    uri_file = Path(journal_path) / "agents" / "mcp.uri"
    if not uri_file.exists():
        raise RuntimeError(f"MCP server URI file not found: {uri_file}")

    try:
        http_uri = uri_file.read_text().strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to read MCP server URI: {exc}")

    if not http_uri:
        raise RuntimeError("MCP server URI file is empty")

    from fastmcp import Client

    return Client(http_uri)


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
        Transcript filename such as ``HHMMSS_audio.json`` or
        ``HHMMSS_monitor_1_diff.json``.

    Returns
    -------
    tuple[str, str, Any]
        ``(path, mime_type, metadata)`` where ``path`` is relative to the day
        directory, ``mime_type`` is either ``image/png`` or ``audio/flac`` and
        ``metadata`` contains the parsed JSON data (empty on failure).
    """

    day_dir = Path(day_path(day))
    json_path = day_dir / name

    if name.endswith("_audio.json"):
        # Audio files are stored as _raw.flac in the heard directory
        raw_name = name.replace("_audio.json", "_raw.flac")
        rel = f"heard/{raw_name}"
        mime = "audio/flac"
    elif name.endswith("_diff.json"):
        raw_name = name[:-5] + ".png"
        rel = f"seen/{raw_name}"
        mime = "image/png"
    else:
        raise ValueError(f"unsupported transcript name: {name}")

    meta: Any = {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:  # pragma: no cover - optional metadata
        logging.debug("Failed to read %s", json_path)

    return rel, mime, meta


def load_entity_names(
    journal_path: str | Path | None = None, required: bool = False
) -> str | None:
    """Load entity names from journal/entities.md for AI transcription context.

    This function extracts just the entity names (no types or descriptions) from
    the top-level entities.md file and returns them as a comma-delimited string.
    This is specifically optimized for transcription accuracy in hear/ and see/
    modules where the AI needs to recognize entity names but doesn't need the
    full context.

    Args:
        journal_path: Path to journal directory. If None, uses JOURNAL_PATH env var.
        required: If True, raises FileNotFoundError when entities.md is missing.
                 If False, returns None when missing.

    Returns:
        Comma-delimited string of entity names (e.g., "John Smith, Acme Corp, Project X"),
        or None if the file is not found and required=False.

    Raises:
        FileNotFoundError: If required=True and entities.md doesn't exist.
        ValueError: If journal_path is not provided and JOURNAL_PATH env var is not set.
    """
    if journal_path is None:
        load_dotenv()
        journal_path = os.getenv("JOURNAL_PATH")
        if not journal_path:
            raise ValueError("JOURNAL_PATH not set and no journal_path provided")

    journal_path = Path(journal_path)
    entities_path = journal_path / "entities.md"

    if not entities_path.is_file():
        if required:
            raise FileNotFoundError(
                f"Required entities file not found: {entities_path}"
            )
        return None

    # Import here to avoid circular dependency
    from think.indexer import parse_entity_line

    # Parse entity names from the file
    entity_names = []
    with open(entities_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_entity_line(line)
            if parsed:
                _, name, _ = parsed  # Ignore type and description
                if name and name not in entity_names:  # Avoid duplicates
                    entity_names.append(name)

    if not entity_names:
        return None

    return ", ".join(entity_names)


def get_todos(day: str) -> dict[str, list[dict[str, Any]]] | None:
    """Parse TODO.md file for a given day and return structured data.

    Parameters
    ----------
    day:
        Day folder in YYYYMMDD format.

    Returns
    -------
    dict[str, list[dict[str, Any]]] | None
        Dictionary with 'today' and 'future' keys containing parsed todos,
        or None if TODO.md doesn't exist.

    Example
    -------
    >>> todos = get_todos("20250124")
    >>> if todos:
    ...     for item in todos['today']:
    ...         print(f"{item['type']}: {item['description']}")
    """
    day_dir = Path(day_path(day))
    todo_path = day_dir / "TODO.md"

    if not todo_path.exists():
        return None

    todos = {"today": [], "future": []}

    current_section = None
    line_number = 0  # Track actual line numbers in file (1-based for editors)

    try:
        with open(todo_path, "r", encoding="utf-8") as f:
            for line in f:
                line_number += 1  # Increment before processing (1-based)
                line = line.strip()

                # Check for section headers
                if line == "# Today":
                    current_section = "today"
                    continue
                elif line == "# Future":
                    current_section = "future"
                    continue

                # Skip empty lines or non-todo lines
                if not line.startswith("- ["):
                    continue

                # Parse checkbox state
                completed = False
                cancelled = False

                if line.startswith("- [x]"):
                    completed = True
                    line = line[6:].strip()
                elif line.startswith("- [ ]"):
                    line = line[5:].strip()
                else:
                    continue  # Not a valid todo line

                # Check for strikethrough (cancelled) - may have content after ~~
                if line.startswith("~~"):
                    # Find the closing ~~
                    close_idx = line.find("~~", 2)
                    if close_idx > 0:
                        cancelled = True
                        # Extract the content between ~~ and reconstruct line
                        cancelled_content = line[2:close_idx].strip()
                        after_content = line[close_idx + 2 :].strip()
                        line = (
                            cancelled_content + " " + after_content
                            if after_content
                            else cancelled_content
                        )

                # Parse the type (bold text) if present, otherwise use the whole line
                type_match = re.match(r"\*\*([^*]+)\*\*:\s*(.*)", line)
                if type_match:
                    todo_type = type_match.group(1)
                    remainder = type_match.group(2)
                else:
                    # No type prefix, use default type and entire line as description
                    todo_type = None
                    remainder = line

                # Parse based on section
                if current_section == "today":
                    # Today format: description #optional-domain-tag (HH:MM)
                    # Match time only if it's a valid hour:minute format at the very end
                    # Hours can be 00-23 or 01-12 (with optional leading zero)
                    time_match = re.search(r"\(([0-2]?\d:[0-5]\d)\)\s*$", remainder)
                    if time_match:
                        # Validate hour is in valid range (00-23)
                        time_str = time_match.group(1)
                        hour = int(time_str.split(":")[0])
                        if hour <= 23:
                            remainder = remainder[: time_match.start()].strip()
                        else:
                            time_str = None
                    else:
                        time_str = None

                    # Extract domain tag if present
                    # Domain tags should be #word with alphanumeric/hyphen/underscore only
                    # and should be preceded by whitespace to avoid matching issue numbers
                    domain_match = re.search(
                        r"\s#([a-zA-Z][a-zA-Z0-9_-]*)\b", remainder
                    )
                    if domain_match:
                        domain = domain_match.group(1)
                        description = remainder[: domain_match.start()].strip()
                    else:
                        domain = None
                        description = remainder

                    todo_item = {
                        "type": todo_type,
                        "description": description,
                        "completed": completed,
                        "cancelled": cancelled,
                        "time": time_str,
                        "line_number": line_number,  # Add line number for tracking
                    }
                    if domain:
                        todo_item["domain"] = domain

                    todos["today"].append(todo_item)

                elif current_section == "future":
                    # Future format: description #optional-domain-tag MM/DD/YYYY
                    date_match = re.search(r"(\d{2}/\d{2}/\d{4})$", remainder)
                    if date_match:
                        date_str = date_match.group(1)
                        remainder = remainder[: date_match.start()].strip()
                    else:
                        date_str = None

                    # Extract domain tag if present
                    # Domain tags should be #word with alphanumeric/hyphen/underscore only
                    # and should be preceded by whitespace to avoid matching issue numbers
                    domain_match = re.search(
                        r"\s#([a-zA-Z][a-zA-Z0-9_-]*)\b", remainder
                    )
                    if domain_match:
                        domain = domain_match.group(1)
                        description = remainder[: domain_match.start()].strip()
                    else:
                        domain = None
                        description = remainder

                    todo_item = {
                        "type": todo_type,
                        "description": description,
                        "completed": completed,
                        "cancelled": cancelled,
                        "date": date_str,
                        "line_number": line_number,  # Add line number for tracking
                    }
                    if domain:
                        todo_item["domain"] = domain

                    todos["future"].append(todo_item)

    except Exception as exc:
        logging.debug("Error parsing TODO.md for day %s: %s", day, exc)
        return None

    # Sort future todos by date
    if todos["future"]:
        from datetime import datetime

        def parse_date(date_str):
            """Parse MM/DD/YYYY format to datetime for sorting."""
            if not date_str:
                return datetime(9999, 12, 31)
            try:
                parts = date_str.split("/")
                if len(parts) == 3:
                    month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
                    return datetime(year, month, day)
            except (ValueError, IndexError):
                pass
            return datetime(9999, 12, 31)

        todos["future"].sort(key=lambda x: parse_date(x.get("date")))

    return todos


def get_personas() -> dict[str, dict[str, Any]]:
    """Load persona metadata from think/agents directory.

    Returns:
        Dictionary mapping persona IDs to their metadata including:
        - title: Display title for the persona
        - txt_path: Absolute path to the .txt file
        - json_path: Absolute path to the .json file (if exists)
        - config: Contents of the JSON configuration (if exists)
    """
    personas = {}
    agents_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "think", "agents"
    )

    if not os.path.isdir(agents_path):
        return personas

    txt_files = [f for f in os.listdir(agents_path) if f.endswith(".txt")]
    for txt_file in txt_files:
        base_name = txt_file[:-4]
        txt_path = os.path.join(agents_path, txt_file)
        json_file = base_name + ".json"
        json_path = os.path.join(agents_path, json_file)

        # Build metadata for this persona
        metadata = {
            "title": base_name,  # Default to ID
            "txt_path": txt_path,
        }

        # Load JSON config if exists
        if os.path.isfile(json_path):
            metadata["json_path"] = json_path
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    agent_config = json.load(f)
                    metadata["config"] = agent_config
                    metadata["title"] = agent_config.get("title", base_name)
            except Exception:
                pass

        personas[base_name] = metadata

    return personas
