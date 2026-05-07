# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Kindle My Clippings.txt importer."""

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Callable

from solstone.think.entities.seeding import seed_entities
from solstone.think.importers.file_importer import ImportPreview, ImportResult
from solstone.think.importers.shared import (
    map_items_to_segments,
    window_items,
    write_content_manifest,
    write_markdown_segments,
)

logger = logging.getLogger(__name__)

DELIMITER = "=========="

# Extract author from title line: "Book Title (Author Name)"
TITLE_AUTHOR_RE = re.compile(r"^(.+?)\s*\(([^)]+)\)\s*$")

# Extract date from metadata line
DATE_RE = re.compile(r"Added on\s+(.+)$")

# Extract page number
PAGE_RE = re.compile(r"page\s+(\d+)")

# Extract location
LOCATION_RE = re.compile(r"location\s+([\d-]+)", re.IGNORECASE)

# Detect clip type
CLIP_TYPE_RE = re.compile(r"Your\s+(Highlight|Note|Bookmark)", re.IGNORECASE)

# Date formats to try when parsing "Added on ..." strings
_DATE_FORMATS = [
    "%A, %B %d, %Y %I:%M:%S %p",  # English: Saturday, March 15, 2025 10:30:00 AM
    "%A, %d %B %Y %H:%M:%S",  # UK-style: Saturday, 15 March 2025 10:30:00
    "%A %d %B %Y %H:%M:%S",  # No comma after day name
    "%A, %B %d, %Y %I:%M %p",  # Without seconds
]


def _parse_date(date_str: str) -> dt.datetime | None:
    """Try multiple date formats to parse the Added on date string."""
    date_str = date_str.strip()
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _parse_block(block: str) -> dict | None:
    """Parse a single clipping block into a structured dict.

    Returns None for empty/unparseable blocks.
    """
    lines = block.strip().split("\n")
    if len(lines) < 2:
        return None

    # Line 1: Title (and optionally author)
    title_line = lines[0].strip()
    # Strip UTF-8 BOM if present on first block
    title_line = title_line.lstrip("\ufeff")
    if not title_line:
        return None

    m = TITLE_AUTHOR_RE.match(title_line)
    if m:
        book_title = m.group(1).strip()
        author = m.group(2).strip()
    else:
        book_title = title_line
        author = ""

    # Line 2: Metadata (type, page, location, date)
    meta_line = lines[1].strip()

    # Clip type
    clip_match = CLIP_TYPE_RE.search(meta_line)
    clip_type = clip_match.group(1).lower() if clip_match else "highlight"

    # Page
    page_match = PAGE_RE.search(meta_line)
    page = int(page_match.group(1)) if page_match else None

    # Location
    loc_match = LOCATION_RE.search(meta_line)
    location = loc_match.group(1) if loc_match else None

    # Date
    date_match = DATE_RE.search(meta_line)
    ts = None
    if date_match:
        parsed = _parse_date(date_match.group(1))
        if parsed:
            ts = parsed.isoformat()

    if ts is None:
        return None

    # Lines 3+: content (line 3 is usually blank)
    content_lines = lines[2:]
    # Strip leading blank line
    if content_lines and not content_lines[0].strip():
        content_lines = content_lines[1:]
    content = "\n".join(content_lines).strip()

    # Skip bookmarks with no content (they're just position markers)
    if clip_type == "bookmark" and not content:
        return None

    entry: dict = {
        "type": "highlight",
        "ts": ts,
        "book_title": book_title,
        "author": author,
        "content": content,
        "clip_type": clip_type,
    }
    if page is not None:
        entry["page"] = page
    if location is not None:
        entry["location"] = location

    return entry


def _render_highlight_markdown(highlights: list[dict]) -> str:
    """Render highlights grouped by book as markdown."""
    # Group by book
    by_book: dict[str, list[dict]] = {}
    for h in highlights:
        key = h["book_title"]
        by_book.setdefault(key, []).append(h)

    sections: list[str] = []
    for book_title, book_highlights in by_book.items():
        # Use first highlight's author (all from same book share author)
        author = book_highlights[0].get("author", "")
        if author:
            heading = f"## {book_title} by {author}"
        else:
            heading = f"## {book_title}"

        lines = [heading]
        for h in book_highlights:
            content = h.get("content", "")
            clip_type = h.get("clip_type", "highlight")

            if clip_type == "note":
                lines.append(f"Note: {content}")
            else:
                lines.append(f"> {content}")

            # Page / location metadata
            meta_parts: list[str] = []
            if h.get("page") is not None:
                meta_parts.append(f"Page {h['page']}")
            if h.get("location"):
                meta_parts.append(f"Location {h['location']}")
            if meta_parts:
                lines.append(" | ".join(meta_parts))

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


class KindleImporter:
    name = "kindle"
    display_name = "Kindle Highlights"
    file_patterns = ["*.txt"]
    description = "Import highlights and notes from Kindle's My Clippings.txt"

    def detect(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if path.suffix.lower() != ".txt":
            return False
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            return False

        if DELIMITER not in text:
            return False

        # Check first few blocks for Kindle-style headers
        blocks = text.split(DELIMITER)[:5]
        for block in blocks:
            if CLIP_TYPE_RE.search(block):
                return True
        return False

    def preview(self, path: Path) -> ImportPreview:
        text = path.read_text(encoding="utf-8-sig")
        blocks = text.split(DELIMITER)

        entries = []
        books: set[str] = set()
        authors: set[str] = set()

        for block in blocks:
            entry = _parse_block(block)
            if entry is None:
                continue
            entries.append(entry)
            books.add(entry["book_title"])
            if entry["author"]:
                authors.add(entry["author"])

        if not entries:
            return ImportPreview(
                date_range=("", ""),
                item_count=0,
                entity_count=0,
                summary="No parseable clippings found",
            )

        dates = sorted(
            dt.datetime.fromisoformat(e["ts"]).strftime("%Y%m%d") for e in entries
        )

        # Count by type
        type_counts: dict[str, int] = {}
        for e in entries:
            ct = e["clip_type"]
            type_counts[ct] = type_counts.get(ct, 0) + 1
        type_parts = [
            f"{n} {t}s" for t, n in sorted(type_counts.items(), key=lambda x: -x[1])
        ]

        entity_count = len(books) + len(authors)

        return ImportPreview(
            date_range=(dates[0], dates[-1]),
            item_count=len(entries),
            entity_count=entity_count,
            summary=f"{', '.join(type_parts)} from {len(books)} books",
        )

    def process(
        self,
        path: Path,
        journal_root: Path,
        *,
        facet: str | None = None,
        import_id: str | None = None,
        progress_callback: Callable | None = None,
        dry_run: bool = False,
    ) -> ImportResult:
        text = path.read_text(encoding="utf-8-sig")
        blocks = text.split(DELIMITER)
        import_id = import_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        entries: list[dict] = []
        errors: list[str] = []
        books: set[str] = set()
        authors: set[str] = set()
        earliest_so_far: str | None = None
        latest_so_far: str | None = None

        for i, block in enumerate(blocks):
            if not block.strip():
                continue
            entry = _parse_block(block)
            if entry is None:
                stripped = block.strip()
                if stripped and "\n" in stripped:
                    errors.append(f"Failed to parse clipping block {i + 1}")
                continue

            # Add epoch timestamp for windowing
            entry["create_ts"] = dt.datetime.fromisoformat(entry["ts"]).timestamp()
            entry_day = dt.datetime.fromisoformat(entry["ts"]).strftime("%Y%m%d")
            if earliest_so_far is None or entry_day < earliest_so_far:
                earliest_so_far = entry_day
            if latest_so_far is None or entry_day > latest_so_far:
                latest_so_far = entry_day
            entries.append(entry)
            books.add(entry["book_title"])
            if entry["author"]:
                authors.add(entry["author"])

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(
                    i + 1,
                    len(blocks),
                    earliest_date=earliest_so_far,
                    latest_date=latest_so_far,
                    entities_found=len(books) + len(authors),
                )

        if not entries:
            return ImportResult(
                entries_written=0,
                entities_seeded=0,
                files_created=[],
                errors=errors,
                summary="No clippings found to import",
            )

        entries.sort(key=lambda e: e["create_ts"])
        date_range_val = (
            (earliest_so_far, latest_so_far)
            if earliest_so_far and latest_so_far
            else None
        )
        books_map: dict[str, list[int]] = {}
        for i, entry in enumerate(entries):
            books_map.setdefault(entry["book_title"], []).append(i)

        windows = window_items(entries, "create_ts", tz=None)
        created_files, segments = write_markdown_segments(
            "kindle",
            windows,
            _render_highlight_markdown,
            filename="highlights_transcript.md",
        )
        item_segments = map_items_to_segments(
            [entry["create_ts"] for entry in entries],
            tz=None,
        )
        entry_segment_map = {idx: segment for idx, segment in enumerate(item_segments)}
        manifest_entries: list[dict] = []
        book_obs: dict[str, list[str]] = {}
        author_obs: dict[str, list[str]] = {}
        for book_idx, (book_title, indices) in enumerate(sorted(books_map.items())):
            book_entries = [entries[i] for i in indices]
            author = book_entries[0].get("author", "")
            first_ts = min(entry["create_ts"] for entry in book_entries)
            first_dt = dt.datetime.fromtimestamp(first_ts)
            highlight_count = sum(
                1 for entry in book_entries if entry.get("clip_type") != "note"
            )
            note_count = sum(
                1 for entry in book_entries if entry.get("clip_type") == "note"
            )
            meta = {"author": author, "highlight_count": highlight_count}
            if note_count:
                meta["note_count"] = note_count
            segment_set = {entry_segment_map[i] for i in indices}
            manifest_entries.append(
                {
                    "id": f"book-{book_idx}",
                    "title": book_title + (f" by {author}" if author else ""),
                    "date": first_dt.strftime("%Y%m%d"),
                    "type": "highlight_group",
                    "preview": book_entries[0].get("content", "")[:200],
                    "meta": meta,
                    "segments": [
                        {"day": day, "key": key} for day, key in sorted(segment_set)
                    ],
                }
            )
            obs_date = first_dt.strftime("%Y-%m-%d")
            highlight_only = sum(
                1 for entry in book_entries if entry.get("clip_type") == "highlight"
            )
            observations: list[str] = []
            if author:
                observations.append(f"By {author} (via Kindle, {obs_date})")
            if highlight_only > 0 and note_count > 0:
                observations.append(
                    f"{highlight_only} highlights, {note_count} notes "
                    f"(via Kindle, {obs_date})"
                )
            elif highlight_only > 0:
                observations.append(
                    f"{highlight_only} highlights (via Kindle, {obs_date})"
                )
            elif note_count > 0:
                observations.append(f"{note_count} notes (via Kindle, {obs_date})")
            if observations:
                book_obs[book_title] = observations
            if author:
                author_obs.setdefault(author, []).append(
                    f"Author of {book_title} (via Kindle, {obs_date})"
                )
        write_content_manifest(import_id, manifest_entries)

        segment_days = {day for day, _ in segments}

        # Seed entities (books and authors)
        entities_seeded = 0
        if facet and entries:
            earliest_day = min(
                dt.datetime.fromisoformat(e["ts"]).strftime("%Y%m%d") for e in entries
            )
            entity_defs: list[dict] = []
            for book in sorted(books):
                d: dict = {"name": book, "type": "Book"}
                if book in book_obs:
                    d["observations"] = book_obs[book]
                entity_defs.append(d)
            for author in sorted(authors):
                if author:
                    d = {"name": author, "type": "Person"}
                    if author in author_obs:
                        d["observations"] = author_obs[author]
                    entity_defs.append(d)

            resolved = seed_entities(facet, earliest_day, entity_defs)
            entities_seeded = len(resolved)

        return ImportResult(
            entries_written=len(entries),
            entities_seeded=entities_seeded,
            files_created=created_files,
            errors=errors,
            summary=(
                f"Imported {len(entries)} Kindle clippings from {len(books)} books "
                f"across {len(segment_days)} days into {len(segments)} segments"
            ),
            segments=segments,
            date_range=date_range_val,
        )


importer = KindleImporter()
