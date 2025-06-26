import argparse
import json
import os
from datetime import datetime
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from think.cluster_glob import FLASH_MODEL, PRO_MODEL
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
    """Remove or rename an entity entry in an entities.md file.

    Returns True if a match was found and modified, False otherwise.
    """
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

    # Log the operation
    log_entity_operation(journal, operation, day, etype, name, new_name)


def update_top_entry(journal: str, etype: str, name: str, desc: str) -> None:
    """Add or update an entry in the top entities.md file."""
    # Sanitize description to prevent newlines that would break formatting
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
        "Merge the following entity descriptions into one concise summary about "
        "the same length as any individual line. Only return the final merged summary text."
    )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=[joined],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192 * 2,
            system_instruction=prompt,
        ),
    )
    return response.text


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
                    primary = info.get("primary", "")
                    formatted_descriptions = {}
                    for date, desc_text in info.get("descriptions", {}).items():
                        formatted_descriptions[format_date(date)] = desc_text

                    data[etype].append(
                        {
                            "name": name,
                            "dates": [format_date(date) for date in sorted(info.get("dates", []))],
                            "raw_dates": sorted(info.get("dates", [])),
                            "desc": primary,
                            "top": info.get("top", False),
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

        if self.path == "/api/top_generate":
            etype = payload.get("type")
            name = payload.get("name")
            info = self.index.get(etype, {}).get(name)
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key or info is None:
                self.send_response(400)
                self.end_headers()
                return

            try:
                desc = generate_top_summary(info, api_key)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"desc": desc}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": f"Failed to generate summary: {str(e)}"}).encode("utf-8")
                )
                print(f"Error generating summary for {etype}: {name} - {e}")
            return
        elif self.path == "/api/top_update":
            etype = payload.get("type")
            name = payload.get("name")
            desc = payload.get("desc", "")
            # Sanitize description to prevent newlines that would break formatting
            desc = desc.replace("\n", " ").replace("\r", " ").strip()
            update_top_entry(self.root, etype, name, desc)
            self.reload_index()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
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

            # If renaming, also update the top entities.md file
            if action == "rename" and new_name:
                top_file_path = os.path.join(self.root, "entities.md")
                modify_entity_in_file(
                    top_file_path, etype, name, new_name, "rename", require_match=False
                )

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
    parser.add_argument("journal", help="Journal directory containing YYYYMMDD folders")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    index = get_entities(args.journal)

    directory = os.path.join(os.path.dirname(__file__), "entity_review")
    handler = partial(EntityHandler, index=index, directory=directory, root=args.journal)
    httpd = HTTPServer(("", args.port), handler)
    print(f"Serving on http://localhost:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
