import argparse
import glob
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from think.crumbs import CrumbBuilder
from think.models import GEMINI_PRO
from think.utils import day_log, day_path, setup_cli


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """Extract date from filename containing YYYYMMDD pattern."""
    date_match = re.search(r"(\d{8})", filename)
    if not date_match:
        return None

    date_str = date_match.group(1)
    try:
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        return datetime(year, month, day)
    except ValueError:
        return None


def format_friendly_date(dt: datetime) -> str:
    """Convert datetime to friendly format like 'Monday May 1st, 2025'."""
    day_name = dt.strftime("%A")
    month_name = dt.strftime("%B")
    day_num = dt.day
    year = dt.year

    if 10 <= day_num % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day_num % 10, "th")

    return f"{day_name} {month_name} {day_num}{suffix}, {year}"


def cluster_glob(filepaths: List[str]) -> str:
    """Generate markdown from files with friendly date headers."""
    if not filepaths:
        return "No files provided"

    file_data: List[Tuple[datetime, str, str]] = []

    for filepath in filepaths:
        if not os.path.isfile(filepath):
            print(f"Warning: File not found {filepath}. Skipping.", file=sys.stderr)
            continue

        date = extract_date_from_filename(filepath)
        if date is None:
            print(
                f"Warning: Could not extract date from filename {filepath}. Skipping.",
                file=sys.stderr,
            )
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            file_data.append((date, filepath, content))
        except Exception as e:
            print(f"Warning: Could not read file {filepath}: {e}", file=sys.stderr)
            continue

    if not file_data:
        return "No valid files with extractable dates found"

    file_data.sort(key=lambda x: x[0])

    lines = []
    for date, filepath, content in file_data:
        friendly_date = format_friendly_date(date)
        lines.append(f"# {friendly_date}")
        lines.append("")
        lines.append(content.strip())
        lines.append("")

    return "\n".join(lines)


def send_to_gemini(
    markdown_content: str,
    prompt_text: str,
    api_key: str,
    model_name: str,
    is_json_mode: bool,
) -> tuple[Optional[str], Optional[object]]:
    """Send markdown content and a prompt to Gemini API."""
    client = genai.Client(api_key=api_key)

    done = threading.Event()

    def progress():
        elapsed = 0
        while not done.is_set():
            time.sleep(5)
            elapsed += 5
            if not done.is_set():
                print(f"... {elapsed}s elapsed", file=sys.stderr)

    t = threading.Thread(target=progress, daemon=True)
    t.start()
    try:
        generation_config_args = {
            "temperature": 0.3,
            "max_output_tokens": 8192 * 2,
            "thinking_config": types.ThinkingConfig(
                thinking_budget=8192 * 2,
            ),
            "system_instruction": prompt_text,
        }
        if is_json_mode:
            generation_config_args["response_mime_type"] = "application/json"

        response = client.models.generate_content(
            model=model_name,
            contents=[markdown_content],
            config=types.GenerateContentConfig(**generation_config_args),
        )
        return response.text, response.usage_metadata

    except Exception as e:
        print(f"Error during Gemini API call: {e}", file=sys.stderr)
        return None, None
    finally:
        done.set()
        t.join()


DATE_RE = re.compile(r"\d{8}")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "entity_roll.txt")


def find_day_dirs(journal: str) -> Dict[str, str]:
    """Return mapping of YYYYMMDD string to full path."""
    days = {}
    for name in os.listdir(journal):
        if DATE_RE.fullmatch(name):
            path = os.path.join(journal, name)
            if os.path.isdir(path):
                days[name] = path
    return days


def gather_files(day: datetime, day_dirs: Dict[str, str]) -> List[str]:
    files = []
    for i in range(8):
        d = day - timedelta(days=i)
        key = d.strftime("%Y%m%d")
        dir_path = day_dirs.get(key)
        if not dir_path:
            continue
        graph_pattern = os.path.join(dir_path, "topics", "knowledge_graph*.md")
        graph_files = glob.glob(graph_pattern)
        files.extend(graph_files)
    return files


def scan_day(day: str) -> Dict[str, List[str]]:
    """Return lists of processed and missing entity markdown files."""
    day_dir = Path(day_path(day))
    processed: List[str] = []
    repairable: List[str] = []
    if (day_dir / "entities.md").exists():
        processed.append("entities.md")
    elif list((day_dir / "topics").glob("knowledge_graph*.md")):
        repairable.append("entities.md")
    return {"processed": processed, "repairable": repairable}


def process_day(
    day_str: str, day_dirs: Dict[str, str], force: bool, verbose: bool = False
) -> None:
    out_path = os.path.join(day_dirs[day_str], "entities.md")
    if os.path.exists(out_path) and not force:
        print(f"Skipping {day_str}: entities.md exists")
        return

    success = False

    day = datetime.strptime(day_str, "%Y%m%d")
    files = gather_files(day, day_dirs)
    if not files:
        print(f"No topics/knowledge_graph.md files for {day_str}")
        return

    print(f"Processing {day_str}:")
    print(f"  Found {len(files)} knowledge_graph files from 8-day window")
    for file in files:
        print(f"    {os.path.basename(file)}")

    print("  Clustering and merging content...")
    markdown = cluster_glob(files)

    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("GOOGLE_API_KEY not found in environment")
            return

        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        print("  Sending to Gemini for entity extraction...")
        result, _ = send_to_gemini(markdown, prompt, api_key, GEMINI_PRO, verbose)
        if not result:
            print(f"Gemini returned no result for {day_str}")
            return

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Wrote {out_path}")

        crumb_builder = (
            CrumbBuilder().add_file(PROMPT_PATH).add_files(files).add_model(GEMINI_PRO)
        )
        crumb_path = crumb_builder.commit(out_path)
        print(f"Crumb saved to: {crumb_path}")
        success = True
    finally:
        msg = f"entity-roll {'ok' if success else 'failed'}"
        if force:
            msg += " --force"
        day_log(day_str, msg)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge topics/knowledge_graph files from a rolling 8-day window and generate entities.md"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--day", help="Process a single day (YYYYMMDD)")
    args = setup_cli(parser)
    journal = os.getenv("JOURNAL_PATH")

    day_dirs = find_day_dirs(journal)
    if not day_dirs:
        parser.error("No YYYYMMDD directories found")

    if args.day:
        if args.day not in day_dirs:
            parser.error(f"Day not found: {args.day}")
        process_day(args.day, day_dirs, args.force, verbose=args.verbose)
    else:
        for day_str in sorted(day_dirs.keys()):
            process_day(day_str, day_dirs, args.force, verbose=args.verbose)


if __name__ == "__main__":
    main()
