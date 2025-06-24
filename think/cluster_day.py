import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, Tuple


def cluster_day(day_dir: str) -> Tuple[str, int]:
    """Return Markdown summary for one day's JSON files and the number processed.

    ``day_dir`` must point directly at the ``YYYYMMDD`` folder.
    """

    # Determine which directory actually holds the day's files.
    base = os.path.basename(os.path.normpath(day_dir))
    if re.fullmatch(r"\d{8}", base):
        date_str = base
    else:
        raise ValueError("day_dir must end with YYYYMMDD")

    # Patterns for the two file types we care about
    audio_pattern = re.compile(r"^(\d{6})_audio\.json$")
    screen_pattern = re.compile(r"^(\d{6})_screen\.md$")

    all_files_data = []

    # Process all files in the directory
    for filename in os.listdir(day_dir):
        audio_match = audio_pattern.match(filename)
        screen_match = screen_pattern.match(filename)

        if audio_match:
            time_part = audio_match.group(1)
            prefix = "audio"
            is_json = True
        elif screen_match:
            time_part = screen_match.group(1)
            prefix = "screen"
            is_json = False
        else:
            continue  # Skip files that don't match our patterns

        try:
            year = int(date_str[0:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_part[0:2])
            minute = int(time_part[2:4])
            second = int(time_part[4:6])
            timestamp = datetime(year, month, day, hour, minute, second)
            full_path = os.path.join(day_dir, filename)

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"Warning: Could not read file {filename}: {e}", file=sys.stderr)
                continue

            all_files_data.append(
                {
                    "filepath": full_path,
                    "basename": filename,
                    "timestamp": timestamp,
                    "prefix": prefix,
                    "content": content,
                    "is_json": is_json,
                }
            )
        except ValueError:
            print(
                f"Warning: Could not parse time from filename {filename}. Skipping.",
                file=sys.stderr,
            )

    # Sort all files by timestamp
    all_files_data.sort(key=lambda x: x["timestamp"])

    # Group files into 5-minute intervals
    grouped_files = defaultdict(list)
    for file_data in all_files_data:
        ts = file_data["timestamp"]
        interval_minute = ts.minute - (ts.minute % 5)
        interval_start_time = ts.replace(minute=interval_minute, second=0, microsecond=0)
        grouped_files[interval_start_time].append(file_data)

    lines = []
    sorted_interval_keys = sorted(grouped_files.keys())

    if not sorted_interval_keys:
        return f"No audio or screen files found for date {date_str} in {day_dir}.", 0

    for interval_start in sorted_interval_keys:
        interval_end = interval_start + timedelta(minutes=5)
        lines.append(
            f"## {interval_start.strftime('%Y-%m-%d %H:%M')} - {interval_end.strftime('%H:%M')}"
        )
        lines.append("")

        files_in_group = grouped_files[interval_start]
        for file_data in files_in_group:
            if file_data["prefix"] == "screen":
                lines.append(f"### Screen Activity Summary")
                lines.append('"""')
                lines.append(file_data["content"].strip())
                lines.append('"""')
                lines.append("")
            else:
                lines.append(f"### Audio Transcript")
                lines.append("```json")
                lines.append(file_data["content"].strip())
                lines.append("```")
                lines.append("")

    return "\n".join(lines), len(all_files_data)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report for a day's JSON files grouped by 5-minute intervals."
    )
    parser.add_argument(
        "day_dir",
        help="Path to the journal day folder (YYYYMMDD)",
    )

    args = parser.parse_args()

    # Validate day_dir argument
    if not os.path.isdir(args.day_dir):
        print(f"Error: Folder not found at specified path: {args.day_dir}", file=sys.stderr)
        sys.exit(1)

    base = os.path.basename(os.path.normpath(args.day_dir))
    if not re.fullmatch(r"\d{8}", base):
        print("Error: Folder name must be in YYYYMMDD format (e.g., 20250524).", file=sys.stderr)
        sys.exit(1)

    markdown, _ = cluster_day(args.day_dir)
    print(markdown)


if __name__ == "__main__":
    main()
