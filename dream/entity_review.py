import argparse
import json
import os
import re
from datetime import datetime
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional

DATE_RE = re.compile(r"\d{8}")
ITEM_RE = re.compile(r"^\s*[-*]\s*(.*)")


def format_date(date_str: str) -> str:
    """Convert YYYYMMDD to 'Wednesday April 2nd' format"""
    try:
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        day = date_obj.day

        # Add ordinal suffix
        if 10 <= day % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

        return date_obj.strftime(f"%A %B {day}{suffix}")
    except ValueError:
        return date_str  # Return original if parsing fails


def find_day_dirs(parent: str) -> Dict[str, str]:
    days = {}
    for name in os.listdir(parent):
        if DATE_RE.fullmatch(name):
            path = os.path.join(parent, name)
            if os.path.isdir(path):
                days[name] = path
    return days


def parse_entities(path: str) -> tuple[List[tuple[str, str, str]], int, int]:
    items: List[tuple[str, str, str]] = []
    bullet_count = 0
    skipped_count = 0
    valid_types = {"Person", "Company", "Project", "Tool"}

    file_path = os.path.join(path, "entities.md")
    if not os.path.isfile(file_path):
        return items, bullet_count, skipped_count

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # Remove all bold formatting first
            line = line.replace("**", "").strip()
            if not line:
                continue

            # Only process lines that start with "*" bullet
            if not line.startswith("*"):
                continue

            bullet_count += 1

            # Remove the bullet and get the text
            text = line[1:].strip()

            if ":" not in text:
                skipped_count += 1
                continue
            etype, rest = text.split(":", 1)
            etype = etype.strip()

            # Skip if entity type is not in valid types
            if etype not in valid_types:
                skipped_count += 1
                continue

            rest = rest.strip()
            if " - " in rest:
                name, desc = rest.split(" - ", 1)
            else:
                name, desc = rest, ""
            items.append((etype, name.strip(), desc.strip()))

    return items, bullet_count, skipped_count


def build_index(parent: str) -> Dict[str, Dict[str, dict]]:
    days = find_day_dirs(parent)
    index: Dict[str, Dict[str, dict]] = {}
    for day, path in days.items():
        entities, bullet_count, skipped_count = parse_entities(path)
        print(f"Day {day}: {bullet_count} bullet lines found, {skipped_count} skipped")
        for etype, name, desc in entities:
            type_map = index.setdefault(etype, {})
            entry = type_map.setdefault(name, {"dates": [], "descriptions": {}})
            if day not in entry["dates"]:
                entry["dates"].append(day)
            if desc:
                entry["descriptions"][day] = desc
    return index


class EntityHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        index: Optional[Dict[str, Dict[str, dict]]] = None,
        directory: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.index: Dict[str, Dict[str, dict]] = index or {}
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/data":
            data: Dict[str, List[Dict[str, object]]] = {}
            for etype, names in self.index.items():
                data[etype] = []
                for name, info in names.items():
                    desc = info["descriptions"].get(info["dates"][0], "")
                    # Format dates and descriptions with friendly dates
                    formatted_descriptions = {}
                    for date, desc_text in info["descriptions"].items():
                        formatted_descriptions[format_date(date)] = desc_text

                    data[etype].append(
                        {
                            "name": name,
                            "dates": [format_date(date) for date in sorted(info["dates"])],
                            "desc": desc,
                            "descriptions": formatted_descriptions,
                        }
                    )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            if self.path in ["/", ""]:
                self.path = "index.html"
            super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="Review entities from daily folders")
    parser.add_argument("parent", help="Directory containing YYYYMMDD folders")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    index = build_index(args.parent)

    directory = os.path.join(os.path.dirname(__file__), "entity_review")
    handler = partial(EntityHandler, index=index, directory=directory)
    httpd = HTTPServer(("", args.port), handler)
    print(f"Serving on http://localhost:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
