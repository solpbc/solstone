import argparse
import json
import os
from datetime import datetime
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional

from think.indexer import get_entities, parse_entity_line


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


def log_entity_operation(
    log_dir: str, operation: str, day: str, etype: str, name: str, new_name: Optional[str] = None
) -> None:
    """Log entity operations to entity_review.log"""
    log_path = os.path.join(log_dir, "entity_review.log")
    timestamp = datetime.now().isoformat()

    log_entry = f"{timestamp} {operation} {day} {etype}: {name}\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_entry)


def modify_entity_file(
    parent: str,
    day: str,
    etype: str,
    name: str,
    new_name: Optional[str] = None,
    operation: str = "remove",
) -> None:
    """Remove or rename an entity entry in a day's ``entities.md`` file."""

    file_path = os.path.join(parent, day, "entities.md")
    if not os.path.isfile(file_path):
        raise ValueError(f"entities.md not found for {day}")

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

    if len(matches) != 1:
        raise ValueError(
            f"Expected 1 match for '{etype}: {name}' in {file_path}, found {len(matches)}"
        )

    idx, desc = matches[0]
    newline = "\n" if lines[idx].endswith("\n") else ""
    if new_name is None:
        del lines[idx]
    else:
        new_line = f"* {etype}: {new_name}"
        if desc:
            new_line += f" - {desc}"
        lines[idx] = new_line + newline

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Log the operation
    log_entity_operation(parent, operation, day, etype, name, new_name)


class EntityHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        index: Optional[Dict[str, Dict[str, dict]]] = None,
        directory: Optional[str] = None,
        root: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.index: Dict[str, Dict[str, dict]] = index or {}
        self.root = root or os.getcwd()
        super().__init__(*args, directory=directory, **kwargs)

    def reload_index(self) -> None:
        print(f"Reloading index from {self.root}")
        old_count = sum(len(names) for names in self.index.values())
        new_index = get_entities(self.root)
        self.index.clear()
        self.index.update(new_index)
        new_count = sum(len(names) for names in self.index.values())
        print(f"Index reloaded: {old_count} -> {new_count} total entities")

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
                            "raw_dates": sorted(info["dates"]),
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

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Invalid JSON")
            return

        action = None
        if self.path == "/api/remove":
            action = "remove"
        elif self.path == "/api/rename":
            action = "rename"
        else:
            self.send_response(404)
            self.end_headers()
            return

        days = payload.get("days", [])
        etype = payload.get("type")
        name = payload.get("name")
        new_name = payload.get("new_name") if action == "rename" else None

        try:
            for day in days:
                modify_entity_file(self.root, day, etype, name, new_name, action)
            self.reload_index()
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Review entities from daily folders")
    parser.add_argument("parent", help="Directory containing YYYYMMDD folders")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    index = get_entities(args.parent)

    directory = os.path.join(os.path.dirname(__file__), "entity_review")
    handler = partial(EntityHandler, index=index, directory=directory, root=args.parent)
    httpd = HTTPServer(("", args.port), handler)
    print(f"Serving on http://localhost:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
