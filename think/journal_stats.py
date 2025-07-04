import argparse
import glob
import os
import re
import sys
from collections import Counter
from datetime import timedelta
from typing import Dict

import soundfile as sf
from dotenv import load_dotenv

DATE_RE = re.compile(r"\d{8}")
FLAC_RE = re.compile(r"^(\d{6})_audio\.flac$")
AUDIO_JSON_RE = re.compile(r"^(\d{6})_audio\.json$")
DIFF_PNG_RE = re.compile(r"^(\d{6})_monitor_\d+_diff\.png$")
BOX_JSON_RE = re.compile(r"^(\d{6})_monitor_\d+_diff_box\.json$")
DESC_JSON_RE = re.compile(r"^(\d{6})_monitor_\d+_diff\.json$")
SCREEN_MD_RE = re.compile(r"^(\d{6})_screen\.md$")

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "ponder")
PROMPT_BASENAMES = [
    os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(PROMPT_DIR, "*.txt"))
]
PONDER_BASENAMES = [f"ponder_{b}" for b in PROMPT_BASENAMES]


class JournalStats:
    def __init__(self) -> None:
        self.days: Dict[str, Dict[str, float | int]] = {}
        self.totals: Counter[str] = Counter()
        self.total_audio_sec = 0.0
        self.total_audio_bytes = 0
        self.total_image_bytes = 0

    def scan_day(self, day: str, path: str) -> None:
        stats: Counter[str] = Counter()
        stats_bool: Dict[str, bool] = {
            "entities": False,
            "ponder": False,
        }
        audio_sec = 0.0
        audio_bytes = 0
        image_bytes = 0
        for name in os.listdir(path):
            file_path = os.path.join(path, name)
            if FLAC_RE.match(name):
                stats["audio_flac"] += 1
                try:
                    info = sf.info(file_path)
                    audio_sec += float(info.frames) / float(info.samplerate)
                except Exception:
                    pass
                try:
                    audio_bytes += os.path.getsize(file_path)
                except OSError:
                    pass
            elif AUDIO_JSON_RE.match(name):
                stats["audio_json"] += 1
            elif DIFF_PNG_RE.match(name):
                stats["diff_png"] += 1
                try:
                    image_bytes += os.path.getsize(file_path)
                except OSError:
                    pass
            elif BOX_JSON_RE.match(name):
                stats["box_json"] += 1
            elif DESC_JSON_RE.match(name):
                stats["desc_json"] += 1
            elif SCREEN_MD_RE.match(name):
                stats["screen_md"] += 1
            elif name == "entities.md":
                stats_bool["entities"] = True
            else:
                base, ext = os.path.splitext(name)
                if ext in {".md", ".json"} and (base in PONDER_BASENAMES):
                    stats_bool["ponder"] = True
                elif name.startswith("ponder_"):
                    stats_bool["ponder"] = True

        stats["entities"] = int(stats_bool["entities"])
        stats["ponder"] = int(stats_bool["ponder"])
        counts_for_totals = dict(stats)
        self.totals.update(counts_for_totals)
        stats["audio_seconds"] = audio_sec
        stats["audio_bytes"] = audio_bytes
        stats["image_bytes"] = image_bytes
        stats["activity"] = sum(counts_for_totals.values())
        self.days[day] = dict(stats)
        self.total_audio_sec += audio_sec
        self.total_audio_bytes += audio_bytes
        self.total_image_bytes += image_bytes

    def scan(self, journal: str) -> None:
        day_dirs = [d for d in os.listdir(journal) if DATE_RE.fullmatch(d)]
        day_dirs.sort()
        for idx, day in enumerate(day_dirs, 1):
            path = os.path.join(journal, day)
            if not os.path.isdir(path):
                continue
            print(f"[{idx}/{len(day_dirs)}] Scanning {day}...", end="\r", flush=True)
            self.scan_day(day, path)
        print()

    def report(self) -> None:
        day_count = len(self.days)
        print(f"Days scanned: {day_count}")
        print(
            f"Total audio files: {self.totals.get('audio_flac', 0)} | Transcripts: {self.totals.get('audio_json', 0)}"
        )
        missing = self.totals.get("audio_flac", 0) - self.totals.get("audio_json", 0)
        if missing > 0:
            print(f"  Missing transcripts: {missing}")
        print(f"Total audio duration: {timedelta(seconds=int(self.total_audio_sec))}")
        print(
            f"Total audio size: {self.total_audio_bytes / (1024 * 1024):.2f} MB | Image size: {self.total_image_bytes / (1024 * 1024):.2f} MB"
        )
        print(
            f"Screenshot diffs: {self.totals.get('diff_png', 0)} | Descriptions: {self.totals.get('desc_json', 0)}"
        )
        missing_desc = self.totals.get("diff_png", 0) - self.totals.get("desc_json", 0)
        if missing_desc > 0:
            print(f"  Missing descriptions: {missing_desc}")
        print(
            f"Screen summaries: {self.totals.get('screen_md', 0)} | Days with entities.md: {self.totals.get('entities', 0)}"
        )
        print(f"Days with ponder results: {self.totals.get('ponder', 0)}")

        # per-day audio hours
        if day_count:
            print("\nHours of audio per day:")
            for day, data in sorted(self.days.items()):
                sec = data.get("audio_seconds", 0.0)
                if sec:
                    print(f"  {day}: {sec / 3600:.2f}h")

        # simple activity graph for last 30 days
        if self.days:
            print("\nRecent activity (last 30 days):")
            days_sorted = sorted(self.days.keys())[-30:]
            max_val = max(self.days[d]["activity"] for d in days_sorted)
            scale = 40 / max_val if max_val else 1
            for d in days_sorted:
                val = self.days[d]["activity"]
                bar = "#" * int(val * scale)
                print(f"{d} | {bar} {val}")

    def to_markdown(self) -> str:
        """Return a markdown summary of the collected statistics."""
        lines = ["# Journal Summary", ""]
        day_count = len(self.days)
        lines.append(f"Days scanned: {day_count}")
        lines.append("")
        lines.append("## Totals")
        lines.append("")
        lines.append(
            f"- Total audio files: {self.totals.get('audio_flac', 0)}"
            f" | Transcripts: {self.totals.get('audio_json', 0)}"
        )
        missing = self.totals.get("audio_flac", 0) - self.totals.get("audio_json", 0)
        if missing > 0:
            lines.append(f"  - Missing transcripts: {missing}")
        total_dur = timedelta(seconds=int(self.total_audio_sec))
        lines.append(f"- Total audio duration: {total_dur}")
        lines.append(
            f"- Total audio size: {self.total_audio_bytes / (1024 * 1024):.2f} MB"
            f" | Image size: {self.total_image_bytes / (1024 * 1024):.2f} MB"
        )
        lines.append(
            f"- Screenshot diffs: {self.totals.get('diff_png', 0)}"
            f" | Descriptions: {self.totals.get('desc_json', 0)}"
        )
        missing_desc = self.totals.get("diff_png", 0) - self.totals.get("desc_json", 0)
        if missing_desc > 0:
            lines.append(f"  - Missing descriptions: {missing_desc}")
        lines.append(
            f"- Screen summaries: {self.totals.get('screen_md', 0)}"
            f" | Days with entities.md: {self.totals.get('entities', 0)}"
        )
        lines.append(f"- Days with ponder results: {self.totals.get('ponder', 0)}")

        if day_count:
            lines.append("")
            lines.append("## Hours of audio per day")
            for day, data in sorted(self.days.items()):
                sec = data.get("audio_seconds", 0.0)
                if sec:
                    lines.append(f"- {day}: {sec / 3600:.2f}h")

        if self.days:
            lines.append("")
            lines.append("## Recent activity (last 30 days)")
            days_sorted = sorted(self.days.keys())[-30:]
            max_val = max(self.days[d]["activity"] for d in days_sorted)
            scale = 40 / max_val if max_val else 1
            for d in days_sorted:
                val = self.days[d]["activity"]
                bar = "#" * int(val * scale)
                lines.append(f"- {d} | {bar} {val}")

        return "\n".join(lines)

    def save_markdown(self, journal: str) -> None:
        """Write the markdown summary to ``summary.md`` in ``journal``."""
        path = os.path.join(journal, "summary.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a sunstone journal and print overall statistics"
    )
    args = parser.parse_args()

    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal or not os.path.isdir(journal):
        parser.error("JOURNAL_PATH not set or invalid")

    js = JournalStats()
    js.scan(journal)
    js.report()
    try:
        js.save_markdown(journal)
    except Exception as e:
        print(f"Error writing summary: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
