import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from think.entities import parse_entity_line
from think.models import GEMINI_FLASH

DATE_RE = re.compile(r"\d{8}")


def adjacent_days(journal: str, day: str) -> tuple[Optional[str], Optional[str]]:
    """Return previous and next day folder names if they exist."""
    if not journal or not os.path.isdir(journal):
        return None, None
    days = sorted(d for d in os.listdir(journal) if DATE_RE.fullmatch(d))
    if day not in days:
        return None, None
    idx = days.index(day)
    prev_day = days[idx - 1] if idx > 0 else None
    next_day = days[idx + 1] if idx < len(days) - 1 else None
    return prev_day, next_day


def format_date(date_str: str) -> str:
    """Convert YYYYMMDD to 'Wednesday April 2nd' format."""
    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        day = date_obj.day
        if 10 <= day % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return date_obj.strftime(f"%A %B {day}{suffix}")
    except ValueError:
        return date_str


def log_entity_operation(
    log_dir: str, operation: str, day: str, etype: str, name: str, new_name: Optional[str] = None
) -> None:
    """Log entity operations to entity_review.log."""
    log_path = os.path.join(log_dir, "entity_review.log")
    timestamp = datetime.now().isoformat()
    if new_name:
        log_entry = f"{timestamp} {operation} {day} {etype}: {name} -> {new_name}\n"
    else:
        log_entry = f"{timestamp} {operation} {day} {etype}: {name}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)


def modify_entity_in_file(
    file_path: str,
    etype: str,
    name: str,
    new_name: Optional[str] = None,
    operation: str = "remove",
    require_match: bool = True,
) -> bool:
    """Remove or rename an entity entry in an entities.md file."""
    if not os.path.isfile(file_path):
        if require_match:
            raise ValueError(f"entities.md not found at {file_path}")
        return False
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    matches: List[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        parsed = parse_entity_line(line)
        if not parsed:
            continue
        t, n, desc = parsed
        if t == etype and n == name:
            matches.append((idx, desc))
    if len(matches) == 0:
        if require_match:
            raise ValueError(f"No match found for '{etype}: {name}' in {file_path}")
        return False
    if len(matches) > 1:
        raise ValueError(f"Multiple matches found for '{etype}: {name}' in {file_path}")
    idx, desc = matches[0]
    newline = "\n" if lines[idx].endswith("\n") else ""
    if operation == "remove":
        del lines[idx]
    elif operation == "rename" and new_name:
        new_line = f"* {etype}: {new_name}"
        if desc:
            new_line += f" - {desc}"
        lines[idx] = new_line + newline
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return True


def modify_entity_file(
    journal: str,
    day: str,
    etype: str,
    name: str,
    new_name: Optional[str] = None,
    operation: str = "remove",
) -> None:
    """Remove or rename an entity entry in a day's ``entities.md`` file."""
    file_path = os.path.join(journal, day, "entities.md")
    modify_entity_in_file(file_path, etype, name, new_name, operation, require_match=True)
    log_entity_operation(journal, operation, day, etype, name, new_name)


def update_top_entry(journal: str, etype: str, name: str, desc: str) -> None:
    """Add or update an entry in the top entities.md file."""
    desc = desc.replace("\n", " ").replace("\r", " ").strip()
    file_path = os.path.join(journal, "entities.md")
    lines: List[str] = []
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    found = False
    for idx, line in enumerate(lines):
        parsed = parse_entity_line(line)
        if not parsed:
            continue
        t, n, _ = parsed
        if t == etype and n == name:
            newline = "\n" if line.endswith("\n") else ""
            lines[idx] = f"* {etype}: {name} - {desc}" + newline
            found = True
            break
    if not found:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"* {etype}: {name} - {desc}\n")
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def generate_top_summary(info: Dict[str, Any], api_key: str) -> str:
    """Merge entity descriptions into a single summary via Gemini."""
    descs = list(info.get("descriptions", {}).values())
    if not descs and info.get("primary"):
        descs.append(info["primary"])
    joined = "\n".join(f"- {d}" for d in descs if d)
    prompt = (
        "Merge the following entity descriptions into one concise summary about"
        " the same length as any individual line. Only return the final merged summary text."
    )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_FLASH,
        contents=[joined],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192 * 2,
            system_instruction=prompt,
        ),
    )
    return response.text


def _combine(day: str, time_str: str) -> str:
    """Return ISO timestamp string for ``day`` + ``time_str``."""

    return f"{day[:4]}-{day[4:6]}-{day[6:]}T{time_str}"


def build_occurrence_index(journal: str) -> Dict[str, List[Dict[str, Any]]]:
    """Aggregate occurrences from all ponder_*.json files for each day."""

    index: Dict[str, List[Dict[str, Any]]] = {}
    for name in os.listdir(journal):
        if not DATE_RE.fullmatch(name):
            continue
        path = os.path.join(journal, name)
        if not os.path.isdir(path):
            continue

        occs: List[Dict[str, Any]] = []
        for fname in os.listdir(path):
            if not (fname.startswith("ponder_") and fname.endswith(".json")):
                continue
            file_path = os.path.join(path, fname)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data.get("occurrences", []) if isinstance(data, dict) else data
            except Exception:
                continue
            if not isinstance(items, list):
                continue
            for occ in items:
                o: Dict[str, Any] = {
                    "title": occ.get("title", ""),
                    "summary": occ.get("summary", ""),
                    "type": occ.get("type", ""),
                }
                source = occ.get("source")
                if (
                    isinstance(source, str)
                    and source.startswith("ponder_")
                    and source.endswith(".md")
                ):
                    o["slug"] = source[7:-3]
                if occ.get("start"):
                    o["startTime"] = _combine(name, occ["start"])
                if occ.get("end"):
                    o["endTime"] = _combine(name, occ["end"])
                occs.append(o)
        if occs:
            index[name] = occs

    return index


# Backwards compatibility
build_index = build_occurrence_index
