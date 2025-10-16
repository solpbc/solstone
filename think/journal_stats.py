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

from observe.sense import scan_day as observe_scan_day
from think.entity_roll import scan_day as entity_scan_day
from think.summarize import scan_day as summarize_scan_day
from think.utils import day_dirs, setup_cli

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
        # Token usage tracking: {day: {model: {token_type: count}}}
        self.token_usage: Dict[str, Dict[str, Dict[str, int]]] = {}
        # Total token usage by model: {model: {token_type: count}}
        self.token_totals: Dict[str, Dict[str, int]] = {}

    def scan_day(self, day: str, path: str) -> None:
        stats: Counter[str] = Counter()
        audio_sec = 0.0
        audio_bytes = 0
        image_bytes = 0
        day_dir = Path(path)

        # --- observe (hear + see) ---
        observe_info = observe_scan_day(day_dir)

        # Count audio and video files from raw (processed) list
        for file_name in observe_info["raw"]:
            file_path = day_dir / file_name
            if file_name.endswith((".flac", ".m4a")):
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
            elif file_name.endswith((".webm", ".mp4", ".png")):
                stats["diff_png"] += 1
                try:
                    image_bytes += os.path.getsize(file_path)
                except OSError:
                    pass

        # Count processed outputs
        for file_name in observe_info["processed"]:
            if file_name.endswith("_audio.json"):
                stats["audio_json"] += 1
            elif file_name.endswith("_screen.jsonl"):
                stats["desc_json"] += 1

        # Single repair count for all unprocessed observe files
        stats["repair_observe"] = len(observe_info["repairable"])

        # --- think ---
        entity_info = entity_scan_day(day)
        stats["entities"] = len(entity_info["processed"])
        stats["repair_entity"] = len(entity_info["repairable"])

        summary_info = summarize_scan_day(day)
        stats["summaries_processed"] = len(summary_info["processed"])
        stats["repair_summaries"] = len(summary_info["repairable"])

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

    def scan_all_tokens(self, journal_path: Path) -> None:
        """Scan all token usage files in the tokens directory.

        Supports both old format (individual *.json files) and new format (daily *.jsonl files).
        """
        tokens_dir = journal_path / "tokens"
        if not tokens_dir.is_dir():
            return

        # Scan both old JSON files and new JSONL files
        all_token_files = list(tokens_dir.glob("*.json")) + list(tokens_dir.glob("*.jsonl"))

        for token_file in all_token_files:
            try:
                with open(token_file, "r", encoding="utf-8") as f:
                    # Handle JSONL files (one JSON object per line)
                    if token_file.suffix == ".jsonl":
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                self._process_token_entry(data)
                            except json.JSONDecodeError:
                                continue
                    # Handle old individual JSON files
                    else:
                        data = json.load(f)
                        self._process_token_entry(data)

            except (OSError, KeyError):
                continue

    def _process_token_entry(self, data: dict) -> None:
        """Process a single token usage entry (expects normalized format)."""
        from datetime import datetime

        # Extract date from timestamp
        timestamp = data.get("timestamp")
        if not timestamp:
            return

        file_date = datetime.fromtimestamp(timestamp).strftime("%Y%m%d")
        model = data.get("model", "unknown")
        usage = data.get("usage", {})

        # Initialize day's token usage if not exists
        if file_date not in self.token_usage:
            self.token_usage[file_date] = {}

        # Initialize model entry if not exists
        if model not in self.token_usage[file_date]:
            self.token_usage[file_date][model] = {}
        if model not in self.token_totals:
            self.token_totals[model] = {}

        # Add token counts (all fields are already normalized by migration)
        for token_type, count in usage.items():
            if not isinstance(count, int):
                continue

            # Add to day's model totals
            if token_type not in self.token_usage[file_date][model]:
                self.token_usage[file_date][model][token_type] = 0
            self.token_usage[file_date][model][token_type] += count

            # Add to overall model totals
            if token_type not in self.token_totals[model]:
                self.token_totals[model][token_type] = 0
            self.token_totals[model][token_type] += count

    def scan(self, journal: str, verbose: bool = False) -> None:
        days_map = day_dirs()
        sorted_days = sorted(days_map.items())
        for idx, (day, path) in enumerate(sorted_days, 1):
            if not os.path.isdir(path):
                continue
            if verbose:
                print(
                    f"[{idx}/{len(sorted_days)}] Scanning {day}...",
                    end="\r",
                    flush=True,
                )
            self.scan_day(day, path)

        # Scan tokens directory once after all days are processed
        self.scan_all_tokens(Path(journal))

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
        print(f"Days with entities.md: {self.totals.get('entities', 0)}")
        print(
            f"Ponder processed: {self.totals.get('ponder_processed', 0)} | Repairable: {self.totals.get('repair_ponder', 0)}"
        )

        # Token usage report
        if self.token_totals:
            print("\nToken Usage by Model:")
            for model in sorted(self.token_totals.keys()):
                tokens = self.token_totals[model]
                total = tokens.get("total_tokens", 0)

                # Use normalized field names (old formats already converted during scan)
                input_tokens = tokens.get("input_tokens", 0)
                output_tokens = tokens.get("output_tokens", 0)

                print(f"  {model}:")
                print(
                    f"    Total: {total:,} | Input: {input_tokens:,} | Output: {output_tokens:,}"
                )

                # Show optional fields if present
                cached = tokens.get("cached_tokens", 0)
                reasoning = tokens.get("reasoning_tokens", 0)
                requests = tokens.get("requests", 0)

                if cached > 0 or reasoning > 0 or requests > 0:
                    parts = []
                    if cached > 0:
                        parts.append(f"Cached: {cached:,}")
                    if reasoning > 0:
                        parts.append(f"Reasoning: {reasoning:,}")
                    if requests > 0:
                        parts.append(f"Requests: {requests:,}")
                    print(f"    {' | '.join(parts)}")

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
        lines.append(f"- Days with entities.md: {self.totals.get('entities', 0)}")
        lines.append(
            f"- Ponder processed: {self.totals.get('ponder_processed', 0)}"
            f" | Repairable: {self.totals.get('repair_ponder', 0)}"
        )

        if self.token_totals:
            lines.append("")
            lines.append("## Token Usage by Model")
            lines.append("")
            for model in sorted(self.token_totals.keys()):
                tokens = self.token_totals[model]
                total = tokens.get("total_tokens", 0)

                # Use normalized field names (old formats already converted during scan)
                input_tokens = tokens.get("input_tokens", 0)
                output_tokens = tokens.get("output_tokens", 0)
                cached = tokens.get("cached_tokens", 0)
                reasoning = tokens.get("reasoning_tokens", 0)
                requests = tokens.get("requests", 0)

                lines.append(f"### {model}")
                lines.append(f"- Total: {total:,} tokens")
                lines.append(f"- Input: {input_tokens:,} | Output: {output_tokens:,}")

                # Show optional fields if present
                if cached > 0 or reasoning > 0 or requests > 0:
                    parts = []
                    if cached > 0:
                        parts.append(f"Cached: {cached:,}")
                    if reasoning > 0:
                        parts.append(f"Reasoning: {reasoning:,}")
                    if requests > 0:
                        parts.append(f"Requests: {requests:,}")
                    lines.append(f"- {' | '.join(parts)}")

                lines.append("")

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
            "token_usage_by_day": self.token_usage,
            "token_totals_by_model": self.token_totals,
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
