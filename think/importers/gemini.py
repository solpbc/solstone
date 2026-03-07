# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Gemini activity importer — imports from Google Takeout My Activity export.

Gemini/Bard exports via Google Takeout are activity logs, NOT threaded
conversations. Each activity is a standalone record with the user prompt
and (often truncated) AI response.

Expected paths in ZIP:
    Takeout/My Activity/Gemini Apps/MyActivity.json
    My Activity/Gemini Apps/MyActivity.json
Or a directory/JSON file with the same structure.
"""

import datetime as dt
import html
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any, Callable

from think.importers.file_importer import ImportPreview, ImportResult
from think.importers.shared import _window_messages, write_segment
from think.utils import day_path

logger = logging.getLogger(__name__)

# Paths to look for inside a Takeout ZIP
_ACTIVITY_PATHS = [
    "Takeout/My Activity/Gemini Apps/MyActivity.json",
    "My Activity/Gemini Apps/MyActivity.json",
    "Takeout/My Activity/Bard/MyActivity.json",
    "My Activity/Bard/MyActivity.json",
]

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Strip HTML tags and decode entities."""
    text = _HTML_TAG_RE.sub("", text)
    return html.unescape(text).strip()


def _load_activities(path: Path) -> list[dict[str, Any]]:
    """Load Gemini activity records from a ZIP, directory, or JSON file."""
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            for activity_path in _ACTIVITY_PATHS:
                if activity_path in names:
                    with zf.open(activity_path) as f:
                        return json.loads(f.read())
            raise ValueError(f"No Gemini activity file found in {path.name}")

    if path.is_file() and path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if path.is_dir():
        # Check for activity file in directory
        for activity_path in _ACTIVITY_PATHS:
            candidate = path / activity_path
            if candidate.exists():
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
        # Try MyActivity.json directly in the directory
        candidate = path / "MyActivity.json"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError(f"No Gemini activity file found in {path}")

    raise ValueError(f"Unsupported file type: {path.suffix}")


def _parse_activity(activity: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse a single Gemini activity record into timestamped messages."""
    # Extract user prompt from subtitles
    subtitles = activity.get("subtitles", [])
    prompt = ""
    for sub in subtitles:
        text = sub.get("value", sub.get("name", ""))
        if text:
            prompt = text
            break

    # Extract AI response from safeHtmlItem
    response = ""
    # safeHtmlItem can be at activity level or nested
    safe_html_items = activity.get("safeHtmlItem", [])
    for item in safe_html_items:
        html_content = item.get("html", "")
        if html_content:
            response = _strip_html(html_content)
            break

    # If no content at all, skip
    if not prompt and not response:
        return []

    # Parse timestamp
    time_str = activity.get("time", "")
    if not time_str:
        return []
    try:
        create_time = dt.datetime.fromisoformat(
            time_str.replace("Z", "+00:00")
        ).timestamp()
    except (ValueError, TypeError):
        return []

    messages: list[dict[str, Any]] = []
    if prompt:
        messages.append(
            {
                "create_time": create_time,
                "speaker": "Human",
                "text": prompt,
                "model_slug": None,
            }
        )
    if response:
        messages.append(
            {
                "create_time": create_time,
                "speaker": "Assistant",
                "text": response,
                "model_slug": None,
            }
        )

    return messages


class GeminiImporter:
    name = "gemini"
    display_name = "Gemini Activity History"
    file_patterns = ["*.zip", "*.json"]
    description = "Import activity from Google Takeout Gemini/Bard export"

    def detect(self, path: Path) -> bool:
        if path.is_file() and path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    names = zf.namelist()
                    return any(p in names for p in _ACTIVITY_PATHS)
            except zipfile.BadZipFile:
                return False

        if path.is_file() and path.suffix.lower() == ".json":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list) or len(data) == 0:
                    return False
                first = data[0]
                # Gemini activity records have "header" and "time" fields
                return "header" in first and "time" in first
            except (json.JSONDecodeError, KeyError):
                return False

        if path.is_dir():
            for activity_path in _ACTIVITY_PATHS:
                if (path / activity_path).exists():
                    return True
            if (path / "MyActivity.json").exists():
                return True

        return False

    def preview(self, path: Path) -> ImportPreview:
        activities = _load_activities(path)
        if not activities:
            return ImportPreview(
                date_range=("", ""),
                item_count=0,
                entity_count=0,
                summary="Empty export — no Gemini activity found",
            )

        dates: list[str] = []
        valid_count = 0
        bard_count = 0

        for act in activities:
            messages = _parse_activity(act)
            if not messages:
                continue
            valid_count += 1
            products = [p.lower() for p in act.get("products", [])]
            header = str(act.get("header", "")).lower()
            if any("bard" in product for product in products) or "bard" in header:
                bard_count += 1
            try:
                day = dt.datetime.fromtimestamp(messages[0]["create_time"]).strftime(
                    "%Y%m%d"
                )
                dates.append(day)
            except (ValueError, OSError):
                pass

        dates.sort()
        date_range = (dates[0], dates[-1]) if dates else ("", "")

        bard_info = f" ({bard_count} Bard-era)" if bard_count else ""
        return ImportPreview(
            date_range=date_range,
            item_count=valid_count,
            entity_count=0,
            summary=f"{valid_count} activities from Gemini export{bard_info}",
        )

    def process(
        self,
        path: Path,
        journal_root: Path,
        *,
        facet: str | None = None,
        progress_callback: Callable | None = None,
    ) -> ImportResult:
        activities = _load_activities(path)
        import_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        messages: list[dict[str, Any]] = []
        errors: list[str] = []
        skipped = 0
        bard_count = 0
        valid_count = 0

        for i, act in enumerate(activities):
            activity_messages = _parse_activity(act)
            if not activity_messages:
                skipped += 1
                continue
            valid_count += 1

            products = [p.lower() for p in act.get("products", [])]
            header = str(act.get("header", "")).lower()
            if any("bard" in product for product in products) or "bard" in header:
                bard_count += 1

            messages.extend(activity_messages)

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, len(activities))

        if not messages:
            return ImportResult(
                entries_written=0,
                entities_seeded=0,
                files_created=[],
                errors=errors,
                summary="No activities found to import",
            )

        messages.sort(key=lambda msg: msg["create_time"])

        windows = _window_messages(messages)
        created_files: list[str] = []
        segments: list[tuple[str, str]] = []
        written_count = 0

        for day, seg_key, model_slug, entries in windows:
            day_dir = str(day_path(day))
            try:
                json_path = write_segment(
                    day_dir,
                    "import.gemini",
                    seg_key,
                    entries,
                    import_id=import_id,
                    facet=facet,
                    model=model_slug,
                )
                created_files.append(json_path)
                segments.append((day, seg_key))
                written_count += len(entries)
            except Exception as exc:
                errors.append(f"Failed to write segment {day}/{seg_key}: {exc}")
                logger.warning("Failed to write segment %s/%s: %s", day, seg_key, exc)

        segment_days = {day for day, _ in segments}

        if skipped:
            logger.info("Skipped %d activities with no content", skipped)

        bard_info = f" ({bard_count} Bard-era)" if bard_count else ""

        return ImportResult(
            entries_written=written_count,
            entities_seeded=0,
            files_created=created_files,
            errors=errors,
            summary=(
                f"Imported {len(messages)} messages from {valid_count} Gemini activities{bard_info} across "
                f"{len(segment_days)} days into {len(segments)} segments"
            ),
            segments=segments,
        )


importer = GeminiImporter()
