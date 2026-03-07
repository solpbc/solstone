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
from think.importers.shared import window_items, write_markdown_segments

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


def _parse_activity(activity: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a single Gemini activity record into an import entry."""
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
    products = activity.get("products", [])
    # safeHtmlItem can be at activity level or nested
    safe_html_items = activity.get("safeHtmlItem", [])
    for item in safe_html_items:
        html_content = item.get("html", "")
        if html_content:
            response = _strip_html(html_content)
            break

    # If no content at all, skip
    if not prompt and not response:
        return None

    # Parse timestamp
    time_str = activity.get("time", "")
    if not time_str:
        return None
    try:
        ts = dt.datetime.fromisoformat(time_str.replace("Z", "+00:00")).isoformat()
    except (ValueError, TypeError):
        return None

    # Build title from prompt (truncated)
    title = activity.get("title", prompt)
    # Clean up "Asked Gemini" / "Talked to Bard" prefixes
    for prefix in ("Asked Gemini", "Talked to Bard", "Asked Bard"):
        if title.startswith(prefix):
            title = title[len(prefix) :].strip().lstrip(":").strip()
            break
    if not title:
        title = prompt[:80] if prompt else "Gemini activity"

    # Build readable content
    parts: list[str] = []
    if prompt:
        parts.append(f"Human: {prompt}")
    if response:
        parts.append(f"Assistant: {response}")
    content = "\n\n".join(parts)

    # Detect source variant
    source_products = [p.lower() for p in products] if products else []
    is_bard = any("bard" in p for p in source_products)
    header = activity.get("header", "")
    if "bard" in header.lower():
        is_bard = True

    entry: dict[str, Any] = {
        "type": "ai_chat",
        "ts": ts,
        "title": title,
        "source": "gemini",
        "message_count": (1 if prompt else 0) + (1 if response else 0),
        "content": content,
    }
    if is_bard:
        entry["variant"] = "bard"

    return entry


def _render_activity_markdown(activity: dict) -> str:
    """Render a Gemini activity as markdown."""
    title = activity.get("title", "Gemini activity")
    lines = [f"## {title}"]

    content = activity.get("content", "")
    if content:
        # Content already has "Human: ..." and "Assistant: ..." format
        # Convert to bold labels
        for part in content.split("\n\n"):
            part = part.strip()
            if part.startswith("Human: "):
                lines.append(f"**Human:** {part[7:]}")
            elif part.startswith("Assistant: "):
                lines.append(f"**Assistant:** {part[11:]}")
            elif part:
                lines.append(part)

    return "\n\n".join(lines)


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
            entry = _parse_activity(act)
            if entry is None:
                continue
            valid_count += 1
            if entry.get("variant") == "bard":
                bard_count += 1
            try:
                day = dt.datetime.fromisoformat(entry["ts"]).strftime("%Y%m%d")
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

        entries: list[dict[str, Any]] = []
        errors: list[str] = []
        skipped = 0

        for i, act in enumerate(activities):
            entry = _parse_activity(act)
            if entry is None:
                skipped += 1
                continue

            # Add epoch timestamp for windowing
            entry["create_ts"] = dt.datetime.fromisoformat(entry["ts"]).timestamp()
            entries.append(entry)

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, len(activities))

        if not entries:
            return ImportResult(
                entries_written=0,
                entities_seeded=0,
                files_created=[],
                errors=errors,
                summary="No activities found to import",
            )

        entries.sort(key=lambda e: e["create_ts"])

        windows = window_items(entries, "create_ts")
        created_files, segments = write_markdown_segments(
            "gemini",
            windows,
            lambda items: "\n\n".join(_render_activity_markdown(a) for a in items),
        )

        segment_days = {day for day, _ in segments}

        if skipped:
            logger.info("Skipped %d activities with no content", skipped)

        bard_count = sum(1 for e in entries if e.get("variant") == "bard")
        bard_info = f" ({bard_count} Bard-era)" if bard_count else ""

        return ImportResult(
            entries_written=len(entries),
            entities_seeded=0,
            files_created=created_files,
            errors=errors,
            summary=(
                f"Imported {len(entries)} Gemini activities{bard_info} across "
                f"{len(segment_days)} days into {len(segments)} segments"
            ),
            segments=segments,
        )


importer = GeminiImporter()
