import json
import os
import re
from typing import Dict, List, Optional, Tuple

MASTER_KEY = "__master__"

DATE_RE = re.compile(r"\d{8}")
ITEM_RE = re.compile(r"^\s*[-*]\s*(.*)")


def find_day_dirs(journal: str) -> Dict[str, str]:
    days = {}
    for name in os.listdir(journal):
        if DATE_RE.fullmatch(name):
            path = os.path.join(journal, name)
            if os.path.isdir(path):
                days[name] = path
    return days


def parse_entity_line(line: str) -> Optional[Tuple[str, str, str]]:
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
    cache_path = os.path.join(journal, "indexer.json")
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(journal: str, cache: Dict[str, dict]) -> None:
    cache_path = os.path.join(journal, "indexer.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def build_index(cache: Dict[str, dict]) -> Dict[str, Dict[str, dict]]:
    index: Dict[str, Dict[str, dict]] = {}
    for key, info in cache.items():
        is_master = info.get("master", False)
        for etype, name, desc in info.get("entries", []):
            type_map = index.setdefault(etype, {})
            entry = type_map.setdefault(name, {"dates": [], "descriptions": {}, "master": False})
            if is_master:
                entry["master"] = True
                if desc:
                    entry["master_desc"] = desc
            else:
                if key not in entry["dates"]:
                    entry["dates"].append(key)
                if desc:
                    entry["descriptions"][key] = desc

    for type_map in index.values():
        for info in type_map.values():
            if info.get("master"):
                info["primary"] = info.get("master_desc", "")
            elif info["descriptions"]:
                latest = max(info["descriptions"].keys())
                info["primary"] = info["descriptions"].get(latest, "")
            else:
                info["primary"] = ""
            info.pop("master_desc", None)

    return index


def get_entities(journal: str) -> Dict[str, Dict[str, dict]]:
    cache = load_cache(journal)
    days = find_day_dirs(journal)
    changed = False

    # handle master file in parent directory
    master_path = os.path.join(journal, "entities.md")
    if os.path.isfile(master_path):
        mtime = int(os.path.getmtime(master_path))
        info = cache.get(MASTER_KEY)
        if info is None or info.get("mtime") != mtime:
            cache[MASTER_KEY] = {
                "file": os.path.relpath(master_path, journal),
                "mtime": mtime,
                "entries": parse_entities(journal),
                "master": True,
            }
            changed = True
    elif MASTER_KEY in cache:
        del cache[MASTER_KEY]
        changed = True

    # remove days no longer present (ignore master key)
    for day in list(cache.keys()):
        if day == MASTER_KEY:
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

    if changed:
        save_cache(journal, cache)

    return build_index(cache)
