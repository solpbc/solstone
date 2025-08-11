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

# Colors used for topic visualization in the dream app
CATEGORY_COLORS = [
    "#007bff",
    "#28a745",
    "#17a2b8",
    "#ffc107",
    "#6f42c1",
    "#fd7e14",
    "#e83e8c",
    "#6c757d",
    "#20c997",
    "#ff5722",
    "#9c27b0",
    "#795548",
]

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


def importer_log(message: str) -> None:
    """Append ``message`` to the journal's ``importer/task_log.txt``."""
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if journal:
        _append_task_log(Path(journal) / "importer", message)


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
    ``.txt`` file, the assigned ``color`` from :data:`CATEGORY_COLORS`, the file
    ``mtime`` and any keys loaded from a matching ``.json`` metadata file.
    """

    topics_dir = Path(__file__).parent / "topics"
    topics: dict[str, dict[str, object]] = {}
    for idx, txt_path in enumerate(sorted(topics_dir.glob("*.txt"))):
        name = txt_path.stem
        color = CATEGORY_COLORS[idx % len(CATEGORY_COLORS)]
        mtime = int(txt_path.stat().st_mtime)
        info: dict[str, object] = {
            "path": str(txt_path),
            "color": color,
            "mtime": mtime,
        }
        json_path = txt_path.with_suffix(".json")
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    info.update(data)
            except Exception as exc:  # pragma: no cover - metadata optional
                logging.debug("Error reading %s: %s", json_path, exc)
        topics[name] = info
    return topics


def get_domains() -> dict[str, dict[str, object]]:
    """Return available domains with metadata.

    Each key is the domain name. The value contains the domain metadata
    from domain.json including title, description, and the domain path.
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    domains_dir = Path(journal) / "domains"
    domains: dict[str, dict[str, object]] = {}

    if not domains_dir.exists():
        return domains

    for domain_path in sorted(domains_dir.iterdir()):
        if not domain_path.is_dir():
            continue

        domain_name = domain_path.name
        domain_json = domain_path / "domain.json"

        if not domain_json.exists():
            continue

        try:
            with open(domain_json, "r", encoding="utf-8") as f:
                domain_data = json.load(f)

            if isinstance(domain_data, dict):
                domain_info = {
                    "path": str(domain_path),
                    "title": domain_data.get("title", domain_name),
                    "description": domain_data.get("description", ""),
                    "color": domain_data.get("color", ""),
                    "emoji": domain_data.get("emoji", ""),
                }
                domains[domain_name] = domain_info
        except Exception as exc:  # pragma: no cover - metadata optional
            logging.debug("Error reading %s: %s", domain_json, exc)

    return domains


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


def get_matters(
    domain: str, *, limit: Optional[int] = None, offset: int = 0
) -> dict[str, dict[str, object]]:
    """Return matters for the specified domain with pagination support.

    Parameters
    ----------
    domain:
        Domain name to get matters for.
    limit:
        Maximum number of matters to return. If None, returns all matters.
    offset:
        Number of matters to skip from the beginning (for pagination).

    Returns
    -------
    dict[str, dict[str, object]]
        Dictionary where keys are matter IDs and values contain
        matter metadata from the matter.json file plus 'activity_log_path' field.
        Results are sorted by matter ID (newest first).
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    domain_path = Path(journal) / "domains" / domain
    matters: dict[str, dict[str, object]] = {}

    if not domain_path.exists():
        return matters

    # Find all matter_X directories
    matter_dirs = [
        d
        for d in domain_path.iterdir()
        if d.is_dir() and d.name.startswith("matter_") and d.name[7:].isdigit()
    ]
    # Sort by matter number in descending order (newest first)
    matter_dirs.sort(key=lambda d: int(d.name[7:]), reverse=True)

    # Apply offset and limit
    start_idx = offset
    end_idx = start_idx + limit if limit is not None else len(matter_dirs)
    matter_dirs = matter_dirs[start_idx:end_idx]

    for matter_dir in matter_dirs:
        matter_id = matter_dir.name
        json_path = matter_dir / "matter.json"
        jsonl_path = matter_dir / "activity_log.jsonl"

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                matter_data = json.load(f)

            if isinstance(matter_data, dict):
                matter_info = {
                    "matter_id": matter_id,
                    "metadata_path": str(json_path),
                    "activity_log_path": str(jsonl_path),
                    "activity_log_exists": jsonl_path.exists(),
                    **matter_data,  # Include all fields from the JSON metadata
                }
                matters[matter_id] = matter_info
        except Exception as exc:  # pragma: no cover - metadata optional
            logging.debug("Error reading %s: %s", json_path, exc)

    return matters


def get_matter(domain: str, matter_id: str) -> dict[str, Any]:
    """Return complete matter data including metadata, logs, objectives, and attachments.

    Parameters
    ----------
    domain:
        Domain name containing the matter.
    matter_id:
        Matter ID in matter_X format.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - metadata: matter metadata from matter.json
        - activity_log: parsed matter activity log from activity_log.jsonl
        - objectives: dict of objectives keyed by name, each containing name, objective, outcome (if completed), created, and modified timestamps
        - attachments: dict of attachment metadata from .json files

    Raises
    ------
    FileNotFoundError
        If the matter directory doesn't exist.
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    matter_path = Path(journal) / "domains" / domain / matter_id
    if not matter_path.exists():
        raise FileNotFoundError(f"Matter {matter_id} not found in domain {domain}")

    result: dict[str, Any] = {
        "metadata": {},
        "activity_log": [],
        "objectives": {},
        "attachments": {},
    }

    # Load matter metadata
    matter_json = matter_path / "matter.json"
    if matter_json.exists():
        try:
            with open(matter_json, "r", encoding="utf-8") as f:
                result["metadata"] = json.load(f)
        except Exception as exc:
            logging.debug("Error reading %s: %s", matter_json, exc)

    # Load matter activity log
    matter_jsonl = matter_path / "activity_log.jsonl"
    if matter_jsonl.exists():
        try:
            with open(matter_jsonl, "r", encoding="utf-8") as f:
                result["activity_log"] = [
                    json.loads(line.strip()) for line in f if line.strip()
                ]
        except Exception as exc:
            logging.debug("Error reading %s: %s", matter_jsonl, exc)

    # Load objectives (new format: objective_<name> directories with OBJECTIVE.md and OUTCOME.md)
    for obj_dir in matter_path.iterdir():
        if obj_dir.is_dir() and obj_dir.name.startswith("objective_"):
            obj_name = obj_dir.name[len("objective_") :]  # Remove "objective_" prefix
            obj_data = {
                "name": obj_name,
                "objective": "",
                "outcome": None,
                "created": None,
                "modified": None,
            }

            # Load timestamps from directory metadata
            try:
                stat = obj_dir.stat()
                obj_data["created"] = stat.st_ctime
                obj_data["modified"] = stat.st_mtime
            except Exception as exc:
                logging.debug("Error reading directory stats for %s: %s", obj_dir, exc)

            # Load OBJECTIVE.md
            objective_file = obj_dir / "OBJECTIVE.md"
            if objective_file.exists():
                try:
                    with open(objective_file, "r", encoding="utf-8") as f:
                        obj_data["objective"] = f.read().strip()
                except Exception as exc:
                    logging.debug("Error reading %s: %s", objective_file, exc)

            # Load OUTCOME.md if it exists
            outcome_file = obj_dir / "OUTCOME.md"
            if outcome_file.exists():
                try:
                    with open(outcome_file, "r", encoding="utf-8") as f:
                        obj_data["outcome"] = f.read().strip()
                except Exception as exc:
                    logging.debug("Error reading %s: %s", outcome_file, exc)

            result["objectives"][obj_name] = obj_data

    # Load attachment metadata
    attachments_dir = matter_path / "attachments"
    if attachments_dir.exists():
        for meta_file in attachments_dir.glob("*.json"):
            # Skip if this is a metadata file for a non-existent attachment
            attachment_name = meta_file.stem
            # Check if the actual attachment file exists (any extension)
            attachment_exists = any(
                f.stem == attachment_name and f.suffix != ".json"
                for f in attachments_dir.iterdir()
            )

            if attachment_exists:
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        result["attachments"][attachment_name] = json.load(f)
                except Exception as exc:
                    logging.debug("Error reading %s: %s", meta_file, exc)

    return result


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
