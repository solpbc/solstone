import argparse
import json
import os
import re
from datetime import datetime
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List

DATE_RE = re.compile(r"\d{8}")


def find_day_dirs(journal: str) -> Dict[str, str]:
    days = {}
    for name in os.listdir(journal):
        if DATE_RE.fullmatch(name):
            path = os.path.join(journal, name)
            if os.path.isdir(path):
                days[name] = path
    return days


def load_meetings(path: str) -> List[Dict[str, Any]]:
    file_path = os.path.join(path, "meetings.json")
    if not os.path.isfile(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def build_index(journal: str) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for day, path in find_day_dirs(journal).items():
        meetings = load_meetings(path)
        if meetings:
            index[day] = meetings
    return index


class MeetingHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        index: Dict[str, List[Dict[str, Any]]] | None = None,
        directory: str | None = None,
        root: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.index = index or {}
        self.root = root or os.getcwd()
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/meetings":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.index).encode("utf-8"))
        else:
            if self.path in ["/", ""]:
                self.path = "index.html"
            super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="Review meetings from daily folders")
    parser.add_argument("journal", help="Journal directory containing YYYYMMDD folders")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    index = build_index(args.journal)

    directory = os.path.join(os.path.dirname(__file__), "meeting_calendar")
    handler = partial(MeetingHandler, index=index, directory=directory, root=args.journal)
    httpd = HTTPServer(("", args.port), handler)
    print(f"Serving on http://localhost:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
