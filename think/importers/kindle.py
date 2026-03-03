# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Kindle My Clippings.txt importer."""

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Callable

from think.importers.file_importer import ImportPreview, ImportResult
from think.importers.shared import seed_entities, write_structured_import

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
        type_parts = [f"{n} {t}s" for t, n in sorted(type_counts.items(), key=lambda x: -x[1])]

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
        progress_callback: Callable | None = None,
    ) -> ImportResult:
        text = path.read_text(encoding="utf-8-sig")
        blocks = text.split(DELIMITER)
        import_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        entries: list[dict] = []
        errors: list[str] = []
        books: set[str] = set()
        authors: set[str] = set()

        for i, block in enumerate(blocks):
            if not block.strip():
                continue
            entry = _parse_block(block)
            if entry is None:
                # Only log as error if block had real content (not just whitespace)
                stripped = block.strip()
                if stripped and "\n" in stripped:
                    errors.append(f"Failed to parse clipping block {i + 1}")
                continue

            entries.append(entry)
            books.add(entry["book_title"])
            if entry["author"]:
                authors.add(entry["author"])

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, len(blocks))

        # Write to journal
        created_files = write_structured_import(
            "kindle",
            entries,
            import_id=import_id,
            facet=facet,
        )

        # Seed entities (books and authors)
        entities_seeded = 0
        if facet and entries:
            earliest_day = min(
                dt.datetime.fromisoformat(e["ts"]).strftime("%Y%m%d") for e in entries
            )
            entity_defs: list[dict] = []
            for book in sorted(books):
                entity_defs.append({"name": book, "type": "Book"})
            for author in sorted(authors):
                if author:
                    entity_defs.append({"name": author, "type": "Person"})

            resolved = seed_entities(facet, earliest_day, entity_defs)
            entities_seeded = len(resolved)

        return ImportResult(
            entries_written=len(entries),
            entities_seeded=entities_seeded,
            files_created=created_files,
            errors=errors,
            summary=f"Imported {len(entries)} Kindle clippings from {len(books)} books across {len(created_files)} days",
        )


importer = KindleImporter()
