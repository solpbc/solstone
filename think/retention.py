# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Media retention service for solstone journals.

Manages the lifecycle of raw media files (layer 1 captures) in journal segments.
Three retention modes:
- keep: retain raw media indefinitely (default)
- days: delete raw media after N days, once processing is complete
- processed: delete raw media as soon as processing completes

Safety invariant: never delete raw media from segments that haven't finished
processing. All completion checks must pass before any deletion.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from think.utils import day_dirs, get_journal, iter_segments

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Raw media file identification
# ---------------------------------------------------------------------------

RAW_AUDIO_EXTENSIONS = frozenset({".flac", ".opus", ".ogg", ".m4a"})
RAW_VIDEO_EXTENSIONS = frozenset({".webm", ".mov", ".mp4"})
RAW_MEDIA_EXTENSIONS = RAW_AUDIO_EXTENSIONS | RAW_VIDEO_EXTENSIONS


def is_raw_media(path: Path) -> bool:
    """Check if a file is raw media (layer 1 capture).

    Raw media: *.flac, *.opus, *.ogg, *.m4a (audio),
    *.webm, *.mov, *.mp4 (video), monitor_*_diff.png (screen diffs).
    """
    if path.suffix.lower() in RAW_MEDIA_EXTENSIONS:
        return True
    if (
        path.suffix.lower() == ".png"
        and path.name.startswith("monitor_")
        and "_diff" in path.name
    ):
        return True
    return False


def get_raw_media_files(segment_path: Path) -> list[Path]:
    """Return all raw media files in a segment directory."""
    if not segment_path.is_dir():
        return []
    return [f for f in segment_path.iterdir() if f.is_file() and is_raw_media(f)]


# ---------------------------------------------------------------------------
# Completion detection (safety invariant)
# ---------------------------------------------------------------------------


def is_segment_complete(segment_path: Path) -> bool:
    """Check if a segment has finished all processing.

    Completion checks (ALL must pass):
    1. No _active.jsonl files in agents/
    2. audio.jsonl exists if any audio raw media was captured
    3. screen.jsonl exists if any video raw media was captured
    4. agents/speaker_labels.json exists if embeddings (.npz) are present
    """
    agents_dir = segment_path / "agents"

    # Check 1: no active agent files
    if agents_dir.is_dir():
        for f in agents_dir.iterdir():
            if f.is_file() and f.name.endswith("_active.jsonl"):
                return False

    files = [f for f in segment_path.iterdir() if f.is_file()]
    file_names = {f.name for f in files}
    file_suffixes = {f.suffix.lower() for f in files}

    # Check 2: audio transcript exists if audio was captured
    if file_suffixes & RAW_AUDIO_EXTENSIONS:
        has_audio_extract = "audio.jsonl" in file_names or any(
            n.endswith("_audio.jsonl") for n in file_names
        )
        if not has_audio_extract:
            return False

    # Check 3: screen extract exists if video was captured
    if file_suffixes & RAW_VIDEO_EXTENSIONS:
        has_screen_extract = "screen.jsonl" in file_names or any(
            n.endswith("_screen.jsonl") for n in file_names
        )
        if not has_screen_extract:
            return False

    # Check 4: speaker labels exist if embeddings are present
    if ".npz" in file_suffixes:
        if not agents_dir.is_dir() or not (agents_dir / "speaker_labels.json").exists():
            return False

    return True


def _get_completion_files(segment_path: Path) -> list[Path]:
    """Return existing completion-indicating files for a segment."""
    completion_files: list[Path] = []

    for name in ("audio.jsonl", "screen.jsonl"):
        path = segment_path / name
        if path.exists():
            completion_files.append(path)

    completion_files.extend(
        path
        for pattern in ("*_audio.jsonl", "*_screen.jsonl")
        for path in segment_path.glob(pattern)
        if path.is_file()
    )

    speaker_labels = segment_path / "agents" / "speaker_labels.json"
    if speaker_labels.exists():
        completion_files.append(speaker_labels)

    return completion_files


# ---------------------------------------------------------------------------
# Retention configuration
# ---------------------------------------------------------------------------


@dataclass
class RetentionPolicy:
    """Retention policy for a single scope (global or per-stream)."""

    mode: str = "keep"  # "keep", "days", or "processed"
    days: int | None = None

    def is_eligible(self, segment_age_days: int) -> bool:
        """Check if a segment's raw media should be purged under this policy."""
        if self.mode == "keep":
            return False
        if self.mode == "processed":
            return True
        if self.mode == "days" and self.days is not None:
            return segment_age_days >= self.days
        return False


@dataclass
class RetentionConfig:
    """Retention configuration from journal.json."""

    default: RetentionPolicy = field(default_factory=RetentionPolicy)
    per_stream: dict[str, RetentionPolicy] = field(default_factory=dict)

    def policy_for_stream(self, stream: str) -> RetentionPolicy:
        """Return the effective policy for a stream."""
        return self.per_stream.get(stream, self.default)


def load_retention_config() -> RetentionConfig:
    """Load retention configuration from journal.json."""
    from think.utils import get_config

    config = get_config()
    retention = config.get("retention", {})

    mode = retention.get("raw_media", "keep")
    days = retention.get("raw_media_days")
    default = RetentionPolicy(mode=mode, days=days)

    per_stream: dict[str, RetentionPolicy] = {}
    for stream_name, stream_config in retention.get("per_stream", {}).items():
        per_stream[stream_name] = RetentionPolicy(
            mode=stream_config.get("raw_media", mode),
            days=stream_config.get("raw_media_days", days),
        )

    return RetentionConfig(default=default, per_stream=per_stream)


# ---------------------------------------------------------------------------
# Storage summary
# ---------------------------------------------------------------------------


def _human_bytes(size: int) -> str:
    """Format byte count as human-readable string."""
    n = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            if unit == "B":
                return f"{int(n)} B"
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


@dataclass
class StorageSummary:
    """Storage usage summary for a journal."""

    raw_media_bytes: int = 0
    derived_bytes: int = 0
    total_segments: int = 0
    segments_with_raw: int = 0
    segments_purged: int = 0

    @property
    def raw_media_human(self) -> str:
        return _human_bytes(self.raw_media_bytes)

    @property
    def derived_human(self) -> str:
        return _human_bytes(self.derived_bytes)


def compute_storage_summary() -> StorageSummary:
    """Compute storage summary across all journal segments."""
    summary = StorageSummary()

    for day_name in sorted(day_dirs().keys()):
        for _stream, _seg_key, seg_path in iter_segments(day_name):
            summary.total_segments += 1

            raw_files = get_raw_media_files(seg_path)
            summary.raw_media_bytes += sum(f.stat().st_size for f in raw_files)

            if raw_files:
                summary.segments_with_raw += 1
            elif (seg_path / "audio.jsonl").exists() or (
                seg_path / "screen.jsonl"
            ).exists():
                summary.segments_purged += 1

            for f in seg_path.rglob("*"):
                if f.is_file() and not is_raw_media(f):
                    summary.derived_bytes += f.stat().st_size

    return summary


# ---------------------------------------------------------------------------
# Retention purge
# ---------------------------------------------------------------------------


@dataclass
class PurgeResult:
    """Result of a purge operation."""

    files_deleted: int = 0
    bytes_freed: int = 0
    segments_processed: int = 0
    segments_skipped_incomplete: int = 0
    segments_skipped_policy: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)


def purge(
    *,
    older_than_days: int | None = None,
    stream_filter: str | None = None,
    dry_run: bool = False,
    config: RetentionConfig | None = None,
) -> PurgeResult:
    """Run retention purge across the journal.

    Parameters
    ----------
    older_than_days
        Override: purge raw media older than this many days.
        If None, uses the configured retention policy.
    stream_filter
        Only process segments from this stream.
    dry_run
        If True, report what would be deleted without deleting.
    config
        Retention config. If None, loads from journal.json.
    """
    if config is None:
        config = load_retention_config()

    result = PurgeResult()
    today = datetime.now().date()
    journal_path = Path(get_journal())

    for day_name in sorted(day_dirs().keys()):
        try:
            day_date = datetime.strptime(day_name, "%Y%m%d").date()
        except ValueError:
            continue
        age_days = (today - day_date).days

        for stream_name, seg_key, seg_path in iter_segments(day_name):
            if stream_filter and stream_name != stream_filter:
                continue

            raw_files = get_raw_media_files(seg_path)
            if not raw_files:
                continue

            result.segments_processed += 1

            # Safety invariant: never delete from incomplete segments
            if not is_segment_complete(seg_path):
                result.segments_skipped_incomplete += 1
                logger.debug(
                    "Skipping incomplete: %s/%s/%s", day_name, stream_name, seg_key
                )
                continue

            # Check eligibility
            if older_than_days is not None:
                eligible = age_days >= older_than_days
            else:
                policy = config.policy_for_stream(stream_name)
                eligible = policy.is_eligible(age_days)

            if not eligible:
                result.segments_skipped_policy += 1
                continue

            # Delete raw media
            segment_bytes = 0
            segment_files = []
            for f in raw_files:
                size = f.stat().st_size
                digest = hashlib.sha256()
                with open(f, "rb") as handle:
                    while chunk := handle.read(64 * 1024):
                        digest.update(chunk)
                hex_digest = digest.hexdigest()
                segment_bytes += size
                segment_files.append(
                    {"name": f.name, "bytes": size, "hash": hex_digest}
                )
                if not dry_run:
                    f.unlink()
                    logger.info("Deleted: %s (%s)", f, _human_bytes(size))

            completion_files = _get_completion_files(seg_path)
            processed_at = None
            if completion_files:
                latest_mtime = max(f.stat().st_mtime for f in completion_files)
                processed_at = datetime.fromtimestamp(latest_mtime).isoformat()

            result.files_deleted += len(raw_files)
            result.bytes_freed += segment_bytes
            result.details.append(
                {
                    "day": day_name,
                    "stream": stream_name,
                    "segment": seg_key,
                    "files": segment_files,
                    "bytes_freed": segment_bytes,
                    "processed_at": processed_at,
                }
            )

    if not dry_run and result.files_deleted > 0:
        _write_retention_log(journal_path, result)

    return result


def _write_retention_log(journal_path: Path, result: PurgeResult) -> None:
    """Append retention activity to health/retention.log."""
    health_dir = journal_path / "health"
    health_dir.mkdir(parents=True, exist_ok=True)
    log_path = health_dir / "retention.log"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "files_deleted": result.files_deleted,
        "bytes_freed": result.bytes_freed,
        "segments_processed": result.segments_processed,
        "segments_skipped_incomplete": result.segments_skipped_incomplete,
        "details": result.details,
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
