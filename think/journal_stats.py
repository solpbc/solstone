import argparse
import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict

from observe.utils import load_analysis_frames
from think.insight import scan_day as insight_scan_day
from think.utils import day_dirs, setup_cli

logger = logging.getLogger(__name__)


class JournalStats:
    def __init__(self) -> None:
        self.days: Dict[str, Dict[str, float | int]] = {}
        self.totals: Counter[str] = Counter()
        self.total_audio_duration = 0.0
        self.total_screen_duration = 0.0
        self.topic_counts: Counter[str] = Counter()
        self.topic_minutes: Counter[str] = Counter()
        self.facet_counts: Counter[str] = Counter()
        self.facet_minutes: Counter[str] = Counter()
        self.heatmap: list[list[float]] = [[0.0 for _ in range(24)] for _ in range(7)]
        # Token usage tracking: {day: {model: {token_type: count}}}
        self.token_usage: Dict[str, Dict[str, Dict[str, int]]] = {}
        # Total token usage by model: {model: {token_type: count}}
        self.token_totals: Dict[str, Dict[str, int]] = {}
        # Per-day topic counts: {day: {topic: count}}
        self.topic_counts_by_day: Dict[str, Dict[str, int]] = {}
        # Per-day facet counts: {day: {facet: count}}
        self.facet_counts_by_day: Dict[str, Dict[str, int]] = {}

    def _get_day_mtime(self, day_dir: Path) -> float:
        """Get latest modification time of files we scan."""
        files = []
        # Check timestamp subdirectories for processed files
        files.extend(day_dir.glob("*/audio.jsonl"))
        files.extend(day_dir.glob("*/*_audio.jsonl"))  # Split audio files
        files.extend(day_dir.glob("*/screen.jsonl"))
        files.extend(day_dir.glob("*/*_screen.jsonl"))  # Split screen files
        files.extend(day_dir.glob("*/raw.flac"))
        files.extend(day_dir.glob("*/screen.webm"))
        # Check day root for unprocessed files
        files.extend(day_dir.glob("*_raw.flac"))
        files.extend(day_dir.glob("*_raw.m4a"))
        files.extend(day_dir.glob("*_screen.webm"))
        files.extend(day_dir.glob("*_screen.mp4"))
        files.extend(day_dir.glob("*_screen.mov"))

        insights = day_dir / "insights"
        if insights.is_dir():
            files.extend(insights.glob("*.json"))
            files.extend(insights.glob("*.md"))

        if not files:
            return 0.0
        return max(f.stat().st_mtime for f in files)

    def _load_day_cache(self, day: str, day_dir: Path) -> dict | None:
        """Load cached day stats if fresh."""
        cache_file = day_dir / "stats.json"
        if not cache_file.exists():
            return None

        try:
            cache_mtime = cache_file.stat().st_mtime
            day_mtime = self._get_day_mtime(day_dir)

            if cache_mtime > day_mtime:
                with open(cache_file, encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Cache load failed for {day}: {e}")

        return None

    def _save_day_cache(self, day_dir: Path, stats: dict) -> None:
        """Save day stats to cache."""
        try:
            cache_file = day_dir / "stats.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")

    def _parse_timestamp(self, ts: str) -> float:
        """Parse HH:MM:SS timestamp to seconds since midnight."""
        try:
            h, m, s = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + int(s)
        except Exception:
            return 0.0

    def _calculate_audio_duration(self, segments: list) -> float:
        """Calculate audio duration from min/max timestamps."""
        timestamps = [seg.get("start") for seg in segments if seg.get("start")]
        if not timestamps:
            return 0.0

        times_seconds = [self._parse_timestamp(t) for t in timestamps]
        return max(times_seconds) - min(times_seconds)

    def _calculate_screen_duration(self, frames: list) -> float:
        """Calculate screen duration from min/max frame timestamps."""
        # Skip header (first element if it has no frame_id)
        frame_timestamps = [
            f["timestamp"] for f in frames if "timestamp" in f and "frame_id" in f
        ]
        if not frame_timestamps:
            return 0.0

        return max(frame_timestamps) - min(frame_timestamps)

    def _apply_day_stats(self, day: str, cached_data: dict) -> None:
        """Apply cached day stats to instance state."""
        # Extract components from cache
        stats = cached_data.get("stats", {})
        topic_data = cached_data.get("topic_data", {})
        heatmap_data = cached_data.get("heatmap_data", {})

        # Apply day stats
        self.days[day] = stats

        # Update totals (excluding per-day durations)
        counts_for_totals = {
            k: v
            for k, v in stats.items()
            if k not in ("audio_duration", "screen_duration")
        }
        self.totals.update(counts_for_totals)

        # Accumulate durations
        self.total_audio_duration += stats.get("audio_duration", 0.0)
        self.total_screen_duration += stats.get("screen_duration", 0.0)

        # Apply topic data
        day_topic_counts: Dict[str, int] = {}
        for topic, data in topic_data.items():
            count = data.get("count", 0)
            self.topic_counts[topic] += count
            self.topic_minutes[topic] += data.get("minutes", 0.0)
            if count > 0:
                day_topic_counts[topic] = count
        if day_topic_counts:
            self.topic_counts_by_day[day] = day_topic_counts

        # Apply facet data
        facet_data = cached_data.get("facet_data", {})
        day_facet_counts: Dict[str, int] = {}
        for facet, data in facet_data.items():
            count = data.get("count", 0)
            self.facet_counts[facet] += count
            self.facet_minutes[facet] += data.get("minutes", 0.0)
            if count > 0:
                day_facet_counts[facet] = count
        if day_facet_counts:
            self.facet_counts_by_day[day] = day_facet_counts

        # Apply heatmap data
        weekday = heatmap_data.get("weekday")
        hours = heatmap_data.get("hours", {})
        if weekday is not None:
            for hour_str, minutes in hours.items():
                hour = int(hour_str)
                self.heatmap[weekday][hour] += minutes

    def scan_day(self, day: str, path: str) -> dict:
        """Scan a single day and return stats dict for caching."""
        stats: Counter[str] = Counter()
        audio_duration = 0.0
        screen_duration = 0.0
        day_dir = Path(path)

        # Track topic data for cache
        topic_data = {}
        facet_data = {}
        heatmap_hours = {}

        # --- Audio sessions ---
        # Check timestamp subdirectories for audio files
        audio_files = list(day_dir.glob("*/audio.jsonl"))
        audio_files.extend(day_dir.glob("*/*_audio.jsonl"))  # Split audio files
        for jsonl_file in sorted(audio_files):
            stats["audio_sessions"] += 1

            try:
                with open(jsonl_file, encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]

                if not lines:
                    logger.debug(f"Empty audio file: {jsonl_file}")
                    continue

                # First line is metadata, rest are segments
                segments = []
                for i, line in enumerate(lines[1:], start=2):
                    try:
                        segments.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.debug(f"Invalid JSON at line {i} in {jsonl_file}: {e}")
                        continue

                stats["audio_segments"] += len(segments)

                # Calculate duration from timestamps
                if segments:
                    duration = self._calculate_audio_duration(segments)
                    audio_duration += duration

            except (OSError, IOError) as e:
                logger.warning(f"Error reading audio file {jsonl_file}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error processing {jsonl_file}: {e}")

        # --- Screen sessions ---
        # Check timestamp subdirectories for screen files (screen.jsonl, *_screen.jsonl)
        screen_files = list(day_dir.glob("*/screen.jsonl"))
        screen_files.extend(day_dir.glob("*/*_screen.jsonl"))
        for jsonl_file in sorted(screen_files):
            stats["screen_sessions"] += 1

            try:
                frames = load_analysis_frames(jsonl_file)
                if not frames:
                    logger.debug(f"No valid frames in: {jsonl_file}")
                    continue

                # Count frames (excluding header)
                frame_count = sum(1 for f in frames if "frame_id" in f)
                stats["screen_frames"] += frame_count

                # Calculate duration from timestamps
                if frame_count > 0:
                    duration = self._calculate_screen_duration(frames)
                    screen_duration += duration

            except (OSError, IOError) as e:
                logger.warning(f"Error reading screen file {jsonl_file}: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error processing {jsonl_file}: {e}")

        # --- Unprocessed files ---
        unprocessed = list(day_dir.glob("*_raw.flac"))
        unprocessed.extend(day_dir.glob("*_raw.m4a"))
        unprocessed.extend(day_dir.glob("*_screen.webm"))
        unprocessed.extend(day_dir.glob("*_screen.mp4"))
        unprocessed.extend(day_dir.glob("*_screen.mov"))
        stats["unprocessed_files"] = len(unprocessed)

        # --- Insight summaries ---
        insight_info = insight_scan_day(day)
        stats["insights_processed"] = len(insight_info["processed"])
        stats["insights_pending"] = len(insight_info["repairable"])

        # --- Events and heatmap from facets/*/events/YYYYMMDD.jsonl ---
        weekday = datetime.strptime(day, "%Y%m%d").weekday()
        journal_root = day_dir.parent
        facets_dir = journal_root / "facets"

        if facets_dir.is_dir():
            for facet_name in os.listdir(facets_dir):
                events_dir = facets_dir / facet_name / "events"
                if not events_dir.is_dir():
                    continue
                events_file = events_dir / f"{day}.jsonl"
                if not events_file.exists():
                    continue

                try:
                    with open(events_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            topic = event.get("topic", "unknown")
                            if topic not in topic_data:
                                topic_data[topic] = {"count": 0, "minutes": 0.0}
                            topic_data[topic]["count"] += 1

                            start = event.get("start")
                            end = event.get("end")
                            try:
                                sh, sm, ss = map(int, start.split(":"))
                                eh, em, es = map(int, end.split(":"))
                            except (ValueError, AttributeError, TypeError):
                                continue

                            start_sec = sh * 3600 + sm * 60 + ss
                            end_sec = eh * 3600 + em * 60 + es
                            duration = max(0, end_sec - start_sec)
                            topic_data[topic]["minutes"] += duration / 60

                            # Track facet stats
                            facet = event.get("facet", facet_name)
                            if facet not in facet_data:
                                facet_data[facet] = {"count": 0, "minutes": 0.0}
                            facet_data[facet]["count"] += 1
                            facet_data[facet]["minutes"] += duration / 60

                            # Build heatmap hours for this day
                            cur = start_sec
                            while cur < end_sec:
                                hour = cur // 3600
                                if hour >= 24:
                                    break
                                next_tick = min((hour + 1) * 3600, end_sec)
                                minutes = (next_tick - cur) / 60
                                heatmap_hours[str(hour)] = (
                                    heatmap_hours.get(str(hour), 0.0) + minutes
                                )
                                cur = next_tick
                except (OSError, IOError) as e:
                    logger.warning(f"Error reading {events_file}: {e}")

        # --- Build return dict ---
        stats["audio_duration"] = audio_duration
        stats["screen_duration"] = screen_duration

        return {
            "stats": dict(stats),
            "topic_data": topic_data,
            "facet_data": facet_data,
            "heatmap_data": {"weekday": weekday, "hours": heatmap_hours},
        }

    def scan_all_tokens(self, journal_path: Path) -> None:
        """Scan all token usage files in the tokens directory.

        Reads daily *.jsonl files (one JSON object per line).
        """
        tokens_dir = journal_path / "tokens"
        if not tokens_dir.is_dir():
            return

        # Scan JSONL files only
        for token_file in tokens_dir.glob("*.jsonl"):
            try:
                with open(token_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            self._process_token_entry(data)
                        except json.JSONDecodeError as e:
                            logger.debug(f"Invalid JSON in {token_file}: {e}")
                            continue

            except (OSError, IOError) as e:
                logger.warning(f"Error reading token file {token_file}: {e}")
                continue

    def _process_token_entry(self, data: dict) -> None:
        """Process a single token usage entry (expects normalized format)."""
        from datetime import datetime, timezone

        # Extract date from timestamp
        timestamp = data.get("timestamp")
        if not timestamp:
            return

        # Use UTC for consistent date extraction (timestamps are in UTC from time.time())
        file_date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(
            "%Y%m%d"
        )
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

    def scan(self, journal: str, verbose: bool = False, use_cache: bool = True) -> None:
        days_map = day_dirs()
        sorted_days = sorted(days_map.items())
        cache_hits = 0
        cache_misses = 0

        for idx, (day, path) in enumerate(sorted_days, 1):
            if not os.path.isdir(path):
                continue

            day_dir = Path(path)

            # Try cache first
            cached_data = None
            if use_cache:
                cached_data = self._load_day_cache(day, day_dir)

            if cached_data:
                # Cache hit - apply cached data
                self._apply_day_stats(day, cached_data)
                cache_hits += 1
                if verbose:
                    print(
                        f"[{idx}/{len(sorted_days)}] {day} (cached)",
                        end="\r",
                        flush=True,
                    )
            else:
                # Cache miss - scan and save
                cache_misses += 1
                if verbose:
                    print(
                        f"[{idx}/{len(sorted_days)}] Scanning {day}...",
                        end="\r",
                        flush=True,
                    )
                day_data = self.scan_day(day, path)
                self._apply_day_stats(day, day_data)

                if use_cache:
                    self._save_day_cache(day_dir, day_data)

        # Scan tokens directory once after all days are processed
        self.scan_all_tokens(Path(journal))

        if verbose:
            cache_status = (
                f" (cache: {cache_hits} hits, {cache_misses} misses)"
                if use_cache
                else ""
            )
            logger.info(
                f"Scanned {len(self.days)} days, "
                f"{self.totals.get('audio_sessions', 0)} audio sessions, "
                f"{self.totals.get('screen_sessions', 0)} screen sessions"
                f"{cache_status}"
            )

    def to_dict(self) -> dict:
        """Return a dictionary with all collected statistics."""
        return {
            "days": self.days,
            "totals": dict(self.totals),
            "total_audio_duration": self.total_audio_duration,
            "total_screen_duration": self.total_screen_duration,
            "topic_counts": dict(self.topic_counts),
            "topic_minutes": {k: round(v, 2) for k, v in self.topic_minutes.items()},
            "topic_counts_by_day": self.topic_counts_by_day,
            "facet_counts": dict(self.facet_counts),
            "facet_minutes": {k: round(v, 2) for k, v in self.facet_minutes.items()},
            "facet_counts_by_day": self.facet_counts_by_day,
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
        description="Scan a sunstone journal and generate statistics"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable per-day caching (force re-scan all days)",
    )
    args = setup_cli(parser)
    journal = os.getenv("JOURNAL_PATH")

    js = JournalStats()
    js.scan(journal, verbose=args.verbose, use_cache=not args.no_cache)

    try:
        js.save_json(journal)
        logger.info(f"Statistics saved to {journal}/stats.json")
    except Exception as e:
        logger.error(f"Error writing stats.json: {e}")
        raise


if __name__ == "__main__":
    main()
