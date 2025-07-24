import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import soundfile as sf

from hear.transcribe import Transcriber
from see.describe import Describer
from see.reduce import scan_day as reduce_scan_day
from think.entity_roll import scan_day as entity_scan_day
from think.ponder import scan_day as ponder_scan_day
from think.utils import setup_cli

DATE_RE = re.compile(r"\d{8}")


class JournalStats:
    def __init__(self) -> None:
        self.days: Dict[str, Dict[str, float | int]] = {}
        self.totals: Counter[str] = Counter()
        self.total_audio_sec = 0.0
        self.total_audio_bytes = 0
        self.total_image_bytes = 0
        self.topic_counts: Counter[str] = Counter()
        self.topic_minutes: Counter[str] = Counter()
        self.heatmap: list[list[float]] = [[0.0 for _ in range(24)] for _ in range(7)]

    def scan_day(self, day: str, path: str) -> None:
        stats: Counter[str] = Counter()
        audio_sec = 0.0
        audio_bytes = 0
        image_bytes = 0
        day_dir = Path(path)

        # --- hear ---
        audio_info = Transcriber.scan_day(day_dir)
        audio_files = [(day_dir / n, n) for n in audio_info["raw"]]
        for path, name in audio_files:
            if name.endswith(".flac"):
                stats["audio_flac"] += 1
                try:
                    info = sf.info(path)
                    audio_sec += float(info.frames) / float(info.samplerate)
                except Exception:
                    pass
                try:
                    audio_bytes += os.path.getsize(path)
                except OSError:
                    pass
        stats["audio_json"] = len(audio_info["processed"])
        stats["repair_hear"] = len(audio_info["repairable"])

        # --- see ---
        diff_info = Describer.scan_day(day_dir)
        stats["diff_png"] = len(diff_info["raw"])
        stats["desc_json"] = len(diff_info["processed"])
        stats["repair_see"] = len(diff_info["repairable"])
        for img_name in diff_info["raw"]:
            img_path = day_dir / img_name
            try:
                image_bytes += os.path.getsize(img_path)
            except OSError:
                pass

        screen_info = reduce_scan_day(day)
        stats["screen_md"] = len(screen_info["processed"])
        stats["repair_reduce"] = len(screen_info["repairable"])

        # --- think ---
        entity_info = entity_scan_day(day)
        stats["entities"] = len(entity_info["processed"])
        stats["repair_entity"] = len(entity_info["repairable"])

        ponder_info = ponder_scan_day(day)
        stats["ponder_processed"] = len(ponder_info["processed"])
        stats["repair_ponder"] = len(ponder_info["repairable"])

        # --- occurrences ---
        topics_dir = day_dir / "topics"
        if topics_dir.is_dir():
            weekday = datetime.strptime(day, "%Y%m%d").weekday()
            for fname in os.listdir(topics_dir):
                base, ext = os.path.splitext(fname)
                if ext != ".json":
                    continue
                try:
                    with open(topics_dir / fname, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                items = data.get("occurrences", []) if isinstance(data, dict) else data
                if not isinstance(items, list):
                    continue
                for occ in items:
                    self.topic_counts[base] += 1
                    start = occ.get("start")
                    end = occ.get("end")
                    try:
                        sh, sm, ss = map(int, start.split(":"))
                        eh, em, es = map(int, end.split(":"))
                    except Exception:
                        continue
                    start_sec = sh * 3600 + sm * 60 + ss
                    end_sec = eh * 3600 + em * 60 + es
                    if end_sec < start_sec:
                        duration = 0
                    else:
                        duration = end_sec - start_sec
                    self.topic_minutes[base] += duration / 60
                    cur = start_sec
                    while cur < end_sec:
                        hour = cur // 3600
                        if hour >= 24:
                            break
                        next_tick = min((hour + 1) * 3600, end_sec)
                        self.heatmap[weekday][hour] += (next_tick - cur) / 60
                        cur = next_tick

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

    def scan(self, journal: str, verbose: bool = False) -> None:
        day_dirs = [d for d in os.listdir(journal) if DATE_RE.fullmatch(d)]
        day_dirs.sort()
        for idx, day in enumerate(day_dirs, 1):
            path = os.path.join(journal, day)
            if not os.path.isdir(path):
                continue
            if verbose:
                print(
                    f"[{idx}/{len(day_dirs)}] Scanning {day}...",
                    end="\r",
                    flush=True,
                )
            self.scan_day(day, path)
        if verbose:
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
        print(
            f"Ponder processed: {self.totals.get('ponder_processed', 0)} | Repairable: {self.totals.get('repair_ponder', 0)}"
        )

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
        lines.append(
            f"- Ponder processed: {self.totals.get('ponder_processed', 0)}"
            f" | Repairable: {self.totals.get('repair_ponder', 0)}"
        )

        if self.topic_counts:
            lines.append("")
            lines.append("## Topic activity")
            for topic in sorted(self.topic_counts):
                minutes = self.topic_minutes.get(topic, 0.0)
                lines.append(
                    f"- {topic}: {self.topic_counts[topic]} occurrences, {minutes:.1f}m"
                )

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

        if any(any(row) for row in self.heatmap):
            lines.append("")
            lines.append("## Activity heat map (minutes)")
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            for idx, row in enumerate(self.heatmap):
                values = " ".join(f"{int(v):2d}" for v in row)
                lines.append(f"- {days[idx]}: {values}")

        return "\n".join(lines)

    def save_markdown(self, journal: str) -> None:
        """Write the markdown summary to ``summary.md`` in ``journal``."""
        path = os.path.join(journal, "summary.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown() + "\n")

    def to_dict(self) -> dict:
        """Return a dictionary with all collected statistics."""
        return {
            "days": self.days,
            "totals": dict(self.totals),
            "total_audio_seconds": self.total_audio_sec,
            "total_audio_bytes": self.total_audio_bytes,
            "total_image_bytes": self.total_image_bytes,
            "topic_counts": dict(self.topic_counts),
            "topic_minutes": {k: round(v, 2) for k, v in self.topic_minutes.items()},
            "heatmap": self.heatmap,
        }

    def save_json(self, journal: str) -> None:
        """Write full statistics to ``stats.json`` in ``journal``."""
        path = os.path.join(journal, "stats.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a sunstone journal and print overall statistics"
    )
    args = setup_cli(parser)
    journal = os.getenv("JOURNAL_PATH")

    js = JournalStats()
    js.scan(journal, verbose=args.verbose)
    js.report()
    try:
        js.save_markdown(journal)
        js.save_json(journal)
    except Exception as e:
        print(f"Error writing summary or stats: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
