import argparse
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta

from dotenv import load_dotenv
from google import genai
from google.genai import types

from think.crumbs import CrumbBuilder

DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "reduce_screen.txt")
FLASH_MODEL = "gemini-2.5-flash"


class TokenTracker:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_candidates_tokens = 0

    def add_usage(self, usage_metadata):
        prompt_tokens = getattr(usage_metadata, "prompt_token_count", None) or 0
        candidates_tokens = getattr(usage_metadata, "candidates_token_count", None) or 0
        self.total_prompt_tokens += prompt_tokens
        self.total_candidates_tokens += candidates_tokens

    def get_compression_percent(self):
        if self.total_prompt_tokens == 0:
            return 0.0
        return (1 - self.total_candidates_tokens / self.total_prompt_tokens) * 100

    def print_summary(self):
        print("\n=== Summary ===")
        print(f"Total tokens in: {self.total_prompt_tokens}")
        print(f"Total tokens out: {self.total_candidates_tokens}")
        print(f"Total compression: {self.get_compression_percent():.2f}%")


def parse_monitor_files(day_dir):
    pattern = re.compile(r"^(\d{6})_monitor_(\d+)_diff\.json$")
    entries = []
    for name in os.listdir(day_dir):
        m = pattern.match(name)
        if not m:
            continue
        time_part, mon = m.groups()
        try:
            ts = datetime.strptime(time_part, "%H%M%S")
        except ValueError:
            continue
        entries.append(
            {
                "timestamp": ts,
                "monitor": int(mon),
                "path": os.path.join(day_dir, name),
            }
        )
    entries.sort(key=lambda e: e["timestamp"])
    return entries


def group_entries(entries):
    groups = {}
    for e in entries:
        interval_minute = e["timestamp"].minute - (e["timestamp"].minute % 5)
        start = e["timestamp"].replace(minute=interval_minute, second=0, microsecond=0)
        key = start  # Remove monitor from key to group all monitors together
        groups.setdefault(key, []).append(e)
    return groups


def load_prompt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Prompt file not found: {path}", file=sys.stderr)
        sys.exit(1)


def call_gemini(markdown, prompt, api_key, debug=False):
    client = genai.Client(api_key=api_key)
    done = threading.Event()

    def progress():
        elapsed = 0
        while not done.is_set():
            time.sleep(5)
            elapsed += 5
            if not done.is_set():
                print(f"... {elapsed}s elapsed")

    if debug:
        print("\n=== DEBUG: Prompt to Gemini ===")
        print(f"System instruction: {prompt}")
        print("\n=== DEBUG: Content to Gemini ===")
        print(markdown)
        print("\n=== DEBUG: End of input ===\n")

    t = threading.Thread(target=progress, daemon=True)
    t.start()
    try:
        response = client.models.generate_content(
            model=FLASH_MODEL,
            contents=[markdown],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8192 * 2,
                system_instruction=prompt,
            ),
        )

        if debug:
            print("\n=== DEBUG: Response from Gemini ===")
            print(response.text)
            print("\n=== DEBUG: End of response ===\n")

        return response.text, response.usage_metadata
    finally:
        done.set()
        t.join()


def process_group(
    start, files, prompt, prompt_path, api_key, force, day_dir, token_tracker, debug=False
):
    out_name = f"{start.strftime('%H%M%S')}_screen.md"
    out_path = os.path.join(day_dir, out_name)
    if os.path.exists(out_path) and not force:
        print(f"Skipping existing {out_name}")
        return

    lines = []
    end = start + timedelta(minutes=5)
    lines.append(f"## Screen Activity {start.strftime('%H:%M')} - {end.strftime('%H:%M')}")
    lines.append("")

    # Group files by monitor for organized display
    monitor_files = {}
    for e in files:
        monitor_files.setdefault(e["monitor"], []).append(e)

    # Process each monitor in order
    for monitor in sorted(monitor_files.keys()):
        monitor_entries = sorted(monitor_files[monitor], key=lambda x: x["timestamp"])
        lines.append(f"### Monitor {monitor}")
        lines.append("")

        for e in monitor_entries:
            with open(e["path"], "r", encoding="utf-8") as f:
                content = f.read().strip()
            lines.append(f"#### {e['timestamp'].strftime('%H:%M:%S')}")
            lines.append("```json")
            lines.append(content)
            lines.append("```")
            lines.append("")

    markdown = "\n".join(lines)
    monitor_count = len(monitor_files)
    file_count = len(files)
    print(
        f"Processing screen activity starting {start.strftime('%H:%M')} ({monitor_count} monitors, {file_count} files)"
    )
    try:
        result, usage_metadata = call_gemini(markdown, prompt, api_key, debug)
        token_tracker.add_usage(usage_metadata)

        # Calculate compression for this file
        prompt_tokens = usage_metadata.prompt_token_count
        candidates_tokens = usage_metadata.candidates_token_count
        compression = (1 - candidates_tokens / prompt_tokens) * 100 if prompt_tokens > 0 else 0

        print(f"Tokens: ({prompt_tokens}) -> ({candidates_tokens}) compression: {compression:.2f}%")
    except Exception as e:
        print(f"Gemini call failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Writing {out_name} ({len(result)})")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Saved {out_path}")

    crumb_builder = (
        CrumbBuilder()
        .add_file(prompt_path)
        .add_files([e["path"] for e in files])
        .add_model(FLASH_MODEL)
    )
    crumb_path = crumb_builder.commit(out_path)
    print(f"Crumb saved to: {crumb_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize all monitor JSON files in 5 minute chunks using Gemini"
    )
    parser.add_argument("day_dir", help="Day directory containing *_monitor_*_diff.json files")
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT_PATH, help="Prompt file")
    parser.add_argument("--force", action="store_true", help="Overwrite existing markdown files")
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Print prompt and response from Gemini"
    )
    args = parser.parse_args()

    day_dir = args.day_dir
    if not os.path.isdir(day_dir):
        parser.error(f"Folder not found: {day_dir}")
    print(f"Processing folder: {day_dir}")

    entries = parse_monitor_files(day_dir)
    if not entries:
        parser.error("No monitor diff JSON files found")

    groups = group_entries(entries)

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        parser.error("GOOGLE_API_KEY not set in environment")

    prompt = load_prompt(args.prompt)

    token_tracker = TokenTracker()

    for start, files in sorted(groups.items()):
        process_group(
            start,
            files,
            prompt,
            args.prompt,
            api_key,
            args.force,
            day_dir,
            token_tracker,
            args.debug,
        )

    token_tracker.print_summary()


if __name__ == "__main__":
    main()
