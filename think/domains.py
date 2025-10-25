"""Domain-specific utilities and tooling for the think module."""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


def _normalize_matter_timestamp(value: Any) -> tuple[Optional[str], Optional[Any]]:
    """Return an ISO 8601 timestamp and the original value for matter logs.

    Handles epoch timestamps stored in seconds or milliseconds, numeric
    strings, and already formatted timestamp strings. Returns (None, None)
    when the value cannot be normalized.
    """

    if value is None:
        return None, None

    original_value = value

    # Normalize whitespace for strings
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None, original_value

        if stripped.isdigit():
            value = float(stripped)
        else:
            try:
                numeric_value = float(stripped)
            except ValueError:
                # Assume it's already a formatted timestamp string
                return stripped, original_value
            else:
                value = numeric_value

    elif isinstance(value, (int, float)):
        value = float(value)
    else:
        return None, original_value

    # Detect millisecond epoch timestamps
    if value > 1e12:
        value /= 1000.0
    elif value > 1e10:
        value /= 1000.0

    try:
        timestamp = datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None, original_value

    return timestamp, original_value


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


def domain_summary(domain: str) -> str:
    """Generate a nicely formatted markdown summary of a domain.

    Args:
        domain: The domain name to summarize

    Returns:
        Formatted markdown string with domain title, description, and entities

    Raises:
        FileNotFoundError: If the domain doesn't exist
        RuntimeError: If JOURNAL_PATH is not set
    """
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal or journal == "":
        raise RuntimeError("JOURNAL_PATH not set")

    domain_path = Path(journal) / "domains" / domain
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain '{domain}' not found at {domain_path}")

    # Load domain metadata
    domain_json_path = domain_path / "domain.json"
    if not domain_json_path.exists():
        raise FileNotFoundError(f"domain.json not found for domain '{domain}'")

    with open(domain_json_path, "r", encoding="utf-8") as f:
        domain_data = json.load(f)

    # Extract metadata
    title = domain_data.get("title", domain)
    description = domain_data.get("description", "")
    color = domain_data.get("color", "")

    # Build markdown summary
    lines = []

    # Title without emoji
    lines.append(f"# {title}")

    # Add color as a badge if available
    if color:
        lines.append(f"![Color]({color})")
        lines.append("")

    # Description
    if description:
        lines.append(f"**Description:** {description}")
        lines.append("")

    # Load entities if available
    entities_path = domain_path / "entities.md"
    if entities_path.exists():
        with open(entities_path, "r", encoding="utf-8") as f:
            entities_content = f.read().strip()

        if entities_content:
            lines.append("## Entities")
            lines.append("")
            lines.append(entities_content)
            lines.append("")

    # Check for matters
    matters = []
    for item in domain_path.iterdir():
        if item.is_dir() and item.name.startswith("matter_"):
            matter_json = item / "matter.json"
            if matter_json.exists():
                with open(matter_json, "r", encoding="utf-8") as f:
                    matter_data = json.load(f)
                    matters.append(
                        {
                            "id": item.name,
                            "title": matter_data.get("title", item.name),
                            "status": matter_data.get("status", "unknown"),
                            "priority": matter_data.get("priority", "normal"),
                        }
                    )

    if matters:
        lines.append("## Matters")
        lines.append("")
        lines.append(f"**Total:** {len(matters)} matter(s)")
        lines.append("")

        # Group by status
        by_status = {}
        for matter in matters:
            status = matter["status"]
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(matter)

        for status in ["active", "pending", "completed", "archived"]:
            if status in by_status:
                lines.append(f"### {status.capitalize()} ({len(by_status[status])})")
                lines.append("")
                for matter in by_status[status]:
                    priority_marker = (
                        "🔴"
                        if matter["priority"] == "high"
                        else "🟡" if matter["priority"] == "medium" else ""
                    )
                    lines.append(
                        f"- {priority_marker} **{matter['id']}**: {matter['title']}"
                    )
                lines.append("")

    return "\n".join(lines)


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


def get_domain_news(
    domain: str,
    *,
    cursor: Optional[str] = None,
    limit: int = 1,
    day: Optional[str] = None,
) -> dict[str, Any]:
    """Return domain news entries grouped by day, newest first.

    Parameters
    ----------
    domain:
        Domain name containing the news directory.
    cursor:
        Optional date string (``YYYYMMDD``). When provided, only news files with
        a date strictly earlier than the cursor are returned. This supports
        pagination in the UI where older entries are fetched on demand.
    limit:
        Maximum number of news days to return. Defaults to one day per request.
    day:
        Optional specific day (``YYYYMMDD``) to return. When provided, returns
        only news for that specific day if it exists. Overrides cursor and limit.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``days`` (list of news day payloads), ``next_cursor``
        (date string for subsequent requests) and ``has_more`` boolean flag.
    """

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise RuntimeError("JOURNAL_PATH not set")

    news_dir = Path(journal) / "domains" / domain / "news"
    if not news_dir.exists():
        return {"days": [], "next_cursor": None, "has_more": False}

    # If specific day requested, check for that file directly
    if day:
        news_path = news_dir / f"{day}.md"
        if news_path.exists() and news_path.is_file():
            selected = [news_path]
        else:
            return {"days": [], "next_cursor": None, "has_more": False}
    else:
        news_files = [
            path
            for path in news_dir.iterdir()
            if path.is_file() and re.fullmatch(r"\d{8}\.md", path.name)
        ]

        # Sort newest first by file name (YYYYMMDD.md)
        news_files.sort(key=lambda p: p.stem, reverse=True)

        if cursor:
            news_files = [path for path in news_files if path.stem < cursor]

        if limit is not None and limit > 0:
            selected = news_files[:limit]
        else:
            selected = news_files

    days: list[dict[str, Any]] = []

    for news_path in selected:
        date_key = news_path.stem

        # Read the raw markdown content
        raw_content = ""
        try:
            raw_content = news_path.read_text(encoding="utf-8")
        except Exception:
            pass

        days.append(
            {
                "date": date_key,
                "raw_content": raw_content,
            }
        )

    # When specific day requested, no pagination
    if day:
        has_more = False
        next_cursor = None
    else:
        has_more = len(news_files) > len(selected)
        next_cursor = selected[-1].stem if has_more and selected else None

    return {"days": days, "next_cursor": next_cursor, "has_more": has_more}


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
            activity_entries: list[dict[str, Any]] = []
            with open(matter_jsonl, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logging.debug(
                            "Error parsing activity log line in %s: %s",
                            matter_jsonl,
                            exc,
                        )
                        continue

                    normalized_ts, original_ts = _normalize_matter_timestamp(
                        entry.get("timestamp")
                    )
                    if original_ts is not None:
                        entry["timestamp_raw"] = original_ts
                    if normalized_ts is not None:
                        entry["timestamp"] = normalized_ts

                    activity_entries.append(entry)

            result["activity_log"] = activity_entries
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


def domain_summaries() -> str:
    """Generate a formatted list summary of all domains for use in agent prompts.

    Returns a markdown-formatted string with each domain as a list item including:
    - Domain name and hashtag ID
    - Description
    - Entity names (if available)

    Returns:
        Formatted markdown string with all domains and their entities

    Raises:
        RuntimeError: If JOURNAL_PATH is not set
    """
    from think.entities import load_entity_names

    domains = get_domains()
    if not domains:
        return "No domains found."

    lines = []
    lines.append("## Available Domains\n")

    for domain_name, domain_info in sorted(domains.items()):
        # Build domain header with name in parentheses
        title = domain_info.get("title", domain_name)
        description = domain_info.get("description", "")

        # Main list item for domain
        lines.append(f"- **{title}** (`{domain_name}`)")

        if description:
            lines.append(f"  {description}")

        # Load entities for this domain
        try:
            entity_names = load_entity_names(domain=domain_name)
            if entity_names:
                lines.append(f"  - **{title} Entities**: {entity_names}")
        except Exception:
            # No entities file or error loading - that's fine, skip it
            pass

        lines.append("")  # Empty line between domains

    return "\n".join(lines).strip()
