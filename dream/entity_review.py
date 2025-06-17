import argparse
import json
import os
import re
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional

DATE_RE = re.compile(r"\d{8}")
ITEM_RE = re.compile(r"^\s*[-*]\s*(.*)")
BOLD_RE = re.compile(r"\*\*(.*?)\*\*")


def find_day_dirs(parent: str) -> Dict[str, str]:
    days = {}
    for name in os.listdir(parent):
        if DATE_RE.fullmatch(name):
            path = os.path.join(parent, name)
            if os.path.isdir(path):
                days[name] = path
    return days


def parse_entities(path: str) -> List[tuple[str, str, str]]:
    items: List[tuple[str, str, str]] = []
    file_path = os.path.join(path, "entities.md")
    if not os.path.isfile(file_path):
        return items
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            m = ITEM_RE.match(line)
            if not m:
                continue
            text = BOLD_RE.sub(r"\1", m.group(1)).strip()
            if ":" not in text:
                continue
            etype, rest = text.split(":", 1)
            etype = etype.strip()
            rest = rest.strip()
            if " - " in rest:
                name, desc = rest.split(" - ", 1)
            else:
                name, desc = rest, ""
            items.append((etype, name.strip(), desc.strip()))
    return items


def build_index(parent: str) -> Dict[str, Dict[str, dict]]:
    days = find_day_dirs(parent)
    index: Dict[str, Dict[str, dict]] = {}
    for day, path in days.items():
        for etype, name, desc in parse_entities(path):
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
                    data[etype].append(
                        {
                            "name": name,
                            "dates": sorted(info["dates"]),
                            "desc": desc,
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
