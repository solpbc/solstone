"""Domain-specific utilities and tooling for the think module."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from think.indexer.entities import parse_entities


def get_domains() -> dict[str, dict[str, object]]:
    """Return available domains with metadata and parsed entities.

    Each key is the domain name. The value contains the domain metadata
    from domain.json including title, description, the domain path, and
    parsed entities grouped by type.
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
                    "entities": {},  # Will be populated below
                }

                # Parse entities for this domain
                try:
                    entity_tuples = parse_entities(str(domain_path))
                    entities_by_type: dict[str, list[str]] = {}

                    for etype, name, desc in entity_tuples:
                        if etype not in entities_by_type:
                            entities_by_type[etype] = []
                        if name not in entities_by_type[etype]:
                            entities_by_type[etype].append(name)

                    domain_info["entities"] = entities_by_type
                except Exception as exc:
                    logging.debug("Error parsing entities for %s: %s", domain_name, exc)

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
    emoji = domain_data.get("emoji", "")
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


def domain_summaries() -> str:
    """Generate a formatted list summary of all domains for use in agent prompts.

    Returns a markdown-formatted string with each domain as a list item including:
    - Domain name and hashtag ID
    - Description
    - Sub-list of entities grouped by type

    Returns:
        Formatted markdown string with all domains and their entities

    Raises:
        RuntimeError: If JOURNAL_PATH is not set
    """
    domains = get_domains()  # This now includes parsed entities
    if not domains:
        return "No domains found."

    lines = []
    lines.append("## Available Domains\n")

    for domain_name, domain_info in sorted(domains.items()):
        # Build domain header with name and hashtag
        emoji = domain_info.get("emoji", "")
        title = domain_info.get("title", domain_name)
        description = domain_info.get("description", "")

        # Main list item for domain
        lines.append(f"- **{title}** (#{domain_name})")

        if description:
            lines.append(f"  {description}")

        # Add entities as sub-list grouped by type (now from get_domains())
        entities_by_type = domain_info.get("entities", {})
        if entities_by_type:
            for entity_type in ["Person", "Company", "Project", "Tool"]:
                if entity_type in entities_by_type:
                    entity_list = entities_by_type[entity_type]
                    if entity_list:
                        # Format as comma-separated list
                        entities_str = ", ".join(entity_list)
                        lines.append(f"  - **{entity_type}**: {entities_str}")

        lines.append("")  # Empty line between domains

    return "\n".join(lines).strip()
