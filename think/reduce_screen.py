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

DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "reduce_screen.txt")
FLASH_MODEL = "gemini-2.5-flash-preview-05-20"


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
        entries.append({
            "timestamp": ts,
            "monitor": int(mon),
            "path": os.path.join(day_dir, name),
        })
    entries.sort(key=lambda e: e["timestamp"])
    return entries


def group_entries(entries):
    groups = {}
    for e in entries:
        interval_minute = e["timestamp"].minute - (e["timestamp"].minute % 5)
        start = e["timestamp"].replace(minute=interval_minute, second=0, microsecond=0)
        key = (e["monitor"], start)
        groups.setdefault(key, []).append(e)
    return groups


def load_prompt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Prompt file not found: {path}", file=sys.stderr)
        sys.exit(1)


def call_gemini(markdown, prompt, api_key):
    client = genai.Client(api_key=api_key)
    done = threading.Event()

    def progress():
        elapsed = 0
        while not done.is_set():
            time.sleep(5)
            elapsed += 5
            if not done.is_set():
                print(f"... {elapsed}s elapsed")

    t = threading.Thread(target=progress, daemon=True)
    t.start()
    try:
        response = client.models.generate_content(
            model=FLASH_MODEL,
            contents=[markdown],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
                system_instruction=prompt,
            ),
        )
        return response.text
    finally:
        done.set()
        t.join()


def process_group(monitor, start, files, prompt, api_key, force, day_dir):
    out_name = f"{start.strftime('%H%M%S')}_monitor_{monitor}.md"
    out_path = os.path.join(day_dir, out_name)
    if os.path.exists(out_path) and not force:
        print(f"Skipping existing {out_name}")
        return

    lines = []
    end = start + timedelta(minutes=5)
    lines.append(f"## Monitor {monitor} {start.strftime('%H:%M')} - {end.strftime('%H:%M')}")
    lines.append("")
    for e in files:
        with open(e["path"], "r", encoding="utf-8") as f:
            content = f.read().strip()
        lines.append(f"### {e['timestamp'].strftime('%H:%M:%S')}")
        lines.append("```json")
        lines.append(content)
        lines.append("```")
        lines.append("")
    markdown = "\n".join(lines)
    print(f"Processing monitor {monitor} starting {start.strftime('%H:%M')} ({len(files)} files)")
    try:
        result = call_gemini(markdown, prompt, api_key)
    except Exception as e:
        print(f"Gemini call failed: {e}", file=sys.stderr)
        sys.exit(1)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Saved {out_name}")


def main():
    parser = argparse.ArgumentParser(description="Summarize per-monitor JSON files in 5 minute chunks using Gemini")
    parser.add_argument("folder", help="Day directory containing *_monitor_*_diff.json files")
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT_PATH, help="Prompt file")
    parser.add_argument("--force", action="store_true", help="Overwrite existing markdown files")
    args = parser.parse_args()

    day_dir = args.folder
    if not os.path.isdir(day_dir):
        parser.error(f"Folder not found: {day_dir}")

    entries = parse_monitor_files(day_dir)
    if not entries:
        parser.error("No monitor diff JSON files found")

    groups = group_entries(entries)

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        parser.error("GOOGLE_API_KEY not set in environment")

    prompt = load_prompt(args.prompt)

    for (monitor, start), files in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        process_group(monitor, start, files, prompt, api_key, args.force, day_dir)


if __name__ == "__main__":
    main()
