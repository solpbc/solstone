"""Entity indexing utilities."""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

DATE_RE = re.compile(r"\d{8}")
ITEM_RE = re.compile(r"^\s*[-*]\s*(.*)")
TOP_KEY = "__top__"


def find_day_dirs(journal: str) -> Dict[str, str]:
    """Return mapping of YYYYMMDD strings to absolute paths."""
    days: Dict[str, str] = {}
    for name in os.listdir(journal):
        if DATE_RE.fullmatch(name):
            path = os.path.join(journal, name)
            if os.path.isdir(path):
                days[name] = path
    return days


def parse_entity_line(line: str) -> Optional[Tuple[str, str, str]]:
    """Parse a single entity line from an ``entities.md`` file."""
    cleaned = line.replace("**", "")
    match = ITEM_RE.match(cleaned)
    if not match:
        return None

    text = match.group(1).strip()
    if ":" not in text:
        return None

    etype, rest = text.split(":", 1)
    rest = rest.strip()
    if " - " in rest:
        name, desc = rest.split(" - ", 1)
    else:
        name, desc = rest, ""

    return etype.strip(), name.strip(), desc.strip()


def parse_entities(path: str) -> List[Tuple[str, str, str]]:
    """Return parsed entity tuples from ``entities.md`` inside ``path``."""
    items: List[Tuple[str, str, str]] = []
    valid_types = {"Person", "Company", "Project", "Tool"}

    file_path = os.path.join(path, "entities.md")
    if not os.path.isfile(file_path):
        return items

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not ITEM_RE.match(line.replace("**", "")):
                continue
            parsed = parse_entity_line(line)
            if not parsed:
                continue
            etype, name, desc = parsed
            if etype not in valid_types:
                continue
            items.append((etype, name, desc))

    return items


def load_cache(journal: str) -> Dict[str, dict]:
    """Load the indexer cache from ``journal``."""
    cache_path = os.path.join(journal, "indexer.json")
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(journal: str, cache: Dict[str, dict]) -> None:
    """Save the indexer cache to ``journal``."""
    cache_path = os.path.join(journal, "indexer.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def build_entities(cache: Dict[str, dict]) -> Dict[str, Dict[str, dict]]:
    """Transform cached entries into a structured entity index."""
    index: Dict[str, Dict[str, dict]] = {}
    for key, info in cache.items():
        is_top = info.get("top", False)
        for etype, name, desc in info.get("entries", []):
            type_map = index.setdefault(etype, {})
            entry = type_map.setdefault(name, {"dates": [], "descriptions": {}, "top": False})
            if is_top:
                entry["top"] = True
                if desc:
                    entry["top_desc"] = desc
            else:
                if key not in entry["dates"]:
                    entry["dates"].append(key)
                if desc:
                    entry["descriptions"][key] = desc

    for type_map in index.values():
        for info in type_map.values():
            if info.get("top"):
                info["primary"] = info.get("top_desc", "")
            elif info["descriptions"]:
                latest = max(info["descriptions"].keys())
                info["primary"] = info["descriptions"].get(latest, "")
            else:
                info["primary"] = ""
            info.pop("top_desc", None)

    return index


def scan_entities(journal: str, cache: Dict[str, dict]) -> bool:
    """Scan ``journal`` for entities and update ``cache`` in-place."""
    days = find_day_dirs(journal)
    changed = False

    # handle top entities file in parent directory
    top_path = os.path.join(journal, "entities.md")
    if os.path.isfile(top_path):
        mtime = int(os.path.getmtime(top_path))
        info = cache.get(TOP_KEY)
        if info is None or info.get("mtime") != mtime:
            cache[TOP_KEY] = {
                "file": os.path.relpath(top_path, journal),
                "mtime": mtime,
                "entries": parse_entities(journal),
                "top": True,
            }
            changed = True
    elif TOP_KEY in cache:
        del cache[TOP_KEY]
        changed = True

    # remove days no longer present (ignore top key)
    for day in list(cache.keys()):
        if day == TOP_KEY:
            continue
        if day not in days:
            del cache[day]
            changed = True

    for day, path in days.items():
        md_path = os.path.join(path, "entities.md")
        if not os.path.isfile(md_path):
            if day in cache:
                del cache[day]
                changed = True
            continue

        mtime = int(os.path.getmtime(md_path))
        day_info = cache.get(day)
        if day_info is None or day_info.get("mtime") != mtime:
            entries = parse_entities(path)
            cache[day] = {
                "file": os.path.relpath(md_path, journal),
                "mtime": mtime,
                "entries": entries,
            }
            changed = True

    return changed


def get_entities(journal: str) -> Dict[str, Dict[str, dict]]:
    """Return the entity index for ``journal``."""
    cache = load_cache(journal)
    if scan_entities(journal, cache):
        save_cache(journal, cache)

    return build_entities(cache)
