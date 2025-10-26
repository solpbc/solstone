"""Domain-specific utilities and tooling for the think module."""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv


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

    # Load entities if available using load_entities
    from think.entities import load_entities

    entities = load_entities(domain)
    if entities:
        lines.append("## Entities")
        lines.append("")
        for entity in entities:
            entity_type = entity.get("type", "")
            name = entity.get("name", "")
            desc = entity.get("description", "")
            # Include aka values in parentheses after name
            aka_list = entity.get("aka", [])
            if isinstance(aka_list, list) and aka_list:
                aka_str = ", ".join(aka_list)
                formatted_name = f"{name} ({aka_str})"
            else:
                formatted_name = name

            if desc:
                lines.append(f"- **{entity_type}**: {formatted_name} - {desc}")
            else:
                lines.append(f"- **{entity_type}**: {formatted_name}")
        lines.append("")

    return "\n".join(lines)


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


def domain_summaries(*, detailed_entities: bool = False) -> str:
    """Generate a formatted list summary of all domains for use in agent prompts.

    Returns a markdown-formatted string with each domain as a list item including:
    - Domain name and hashtag ID
    - Description
    - Entity names (if available)

    Parameters
    ----------
    detailed_entities:
        If True, includes full entity details (name: description).
        If False (default), includes only entity names as a comma-separated list.

    Returns
    -------
    str
        Formatted markdown string with all domains and their entities

    Raises
    ------
    RuntimeError
        If JOURNAL_PATH is not set
    """
    from think.entities import load_entity_names, load_entities

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
            if detailed_entities:
                entities = load_entities(domain_name)
                if entities:
                    lines.append(f"  - **{title} Entities**:")
                    for entity in entities:
                        name = entity.get("name", "")
                        desc = entity.get("description", "")
                        # Include aka values in parentheses after name
                        aka_list = entity.get("aka", [])
                        if isinstance(aka_list, list) and aka_list:
                            aka_str = ", ".join(aka_list)
                            formatted_name = f"{name} ({aka_str})"
                        else:
                            formatted_name = name

                        if desc:
                            lines.append(f"    - {formatted_name}: {desc}")
                        else:
                            lines.append(f"    - {formatted_name}")
            else:
                entity_names = load_entity_names(domain=domain_name)
                if entity_names:
                    lines.append(f"  - **{title} Entities**: {entity_names}")
        except Exception:
            # No entities file or error loading - that's fine, skip it
            pass

        lines.append("")  # Empty line between domains

    return "\n".join(lines).strip()
