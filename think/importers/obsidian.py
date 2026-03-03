# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Obsidian and Logseq vault importer."""

import datetime as dt
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

from think.importers.file_importer import ImportPreview, ImportResult
from think.importers.shared import seed_entities, write_structured_import

logger = logging.getLogger(__name__)

# Wikilink extraction: [[Target]] or [[Target|Display Text]]
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

# Daily note date patterns
DAILY_NOTE_PATTERNS = [
    (re.compile(r"^(\d{4}-\d{2}-\d{2})\.md$"), "%Y-%m-%d"),  # YYYY-MM-DD.md
    (re.compile(r"^(\d{4}_\d{2}_\d{2})\.md$"), "%Y_%m_%d"),  # YYYY_MM_DD.md (Logseq)
    (re.compile(r"^(\d{8})\.md$"), "%Y%m%d"),  # YYYYMMDD.md
]

# Frontmatter regex: --- at start of file, then content, then ---
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Simple YAML tag extraction from frontmatter
TAGS_RE = re.compile(r"^tags:\s*\[([^\]]*)\]", re.MULTILINE)
TAGS_LIST_RE = re.compile(r"^  ?- (.+)$", re.MULTILINE)

# Binary/non-text extensions to skip
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp", ".ico",
    ".pdf", ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".m4a",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib",
    ".woff", ".woff2", ".ttf", ".eot",
}


def _parse_daily_note_date(filename: str) -> dt.date | None:
    """Try to parse a daily note date from filename. Returns None if not a daily note."""
    for pattern, fmt in DAILY_NOTE_PATTERNS:
        m = pattern.match(filename)
        if m:
            try:
                return dt.datetime.strptime(m.group(1), fmt).date()
            except ValueError:
                continue
    return None


def _parse_frontmatter_tags(content: str) -> list[str]:
    """Extract tags from YAML frontmatter if present."""
    fm = FRONTMATTER_RE.match(content)
    if not fm:
        return []
    fm_text = fm.group(1)

    # Try inline format: tags: [tag1, tag2]
    m = TAGS_RE.search(fm_text)
    if m:
        raw = m.group(1)
        return [t.strip().strip("\"'") for t in raw.split(",") if t.strip()]

    # Try list format:
    # tags:
    #   - tag1
    #   - tag2
    tags_start = fm_text.find("tags:")
    if tags_start == -1:
        return []
    after_tags = fm_text[tags_start:]
    return TAGS_LIST_RE.findall(after_tags)


def _read_file_safe(path: Path) -> str | None:
    """Read file as UTF-8 with BOM handling. Returns None on failure."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, OSError):
        return None


def _is_hidden(name: str) -> bool:
    """Check if a filename/dirname starts with a dot."""
    return name.startswith(".")


class ObsidianImporter:
    name = "obsidian"
    display_name = "Obsidian / Logseq Vault"
    file_patterns = ["*.md"]
    description = "Import notes from an Obsidian or Logseq vault"

    def detect(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        # Obsidian vault
        if (path / ".obsidian").is_dir():
            return True
        # Logseq vault
        if (path / "logseq").is_dir():
            return True
        # Heuristic: at least 3 markdown files
        md_count = 0
        for entry in path.rglob("*.md"):
            if not any(_is_hidden(p) for p in entry.relative_to(path).parts):
                md_count += 1
                if md_count >= 3:
                    return True
        return False

    def preview(self, path: Path) -> ImportPreview:
        md_files = list(self._walk_md_files(path))
        daily_count = 0
        knowledge_count = 0
        all_wikilinks: set[str] = set()
        dates: list[str] = []

        for md_path in md_files:
            rel = md_path.relative_to(path)
            date = _parse_daily_note_date(md_path.name)
            if date:
                daily_count += 1
                dates.append(date.strftime("%Y%m%d"))
            else:
                knowledge_count += 1
                try:
                    mtime = dt.datetime.fromtimestamp(md_path.stat().st_mtime)
                    dates.append(mtime.strftime("%Y%m%d"))
                except OSError:
                    pass

            content = _read_file_safe(md_path)
            if content:
                all_wikilinks.update(WIKILINK_RE.findall(content))

        dates.sort()
        date_range = (dates[0], dates[-1]) if dates else ("", "")

        parts = []
        if daily_count:
            parts.append(f"{daily_count} daily notes")
        if knowledge_count:
            parts.append(f"{knowledge_count} knowledge notes")
        if all_wikilinks:
            parts.append(f"{len(all_wikilinks)} unique wikilinks")
        summary = ", ".join(parts) if parts else "Empty vault"

        return ImportPreview(
            date_range=date_range,
            item_count=len(md_files),
            entity_count=len(all_wikilinks),
            summary=summary,
        )

    def process(
        self,
        path: Path,
        journal_root: Path,
        *,
        facet: str | None = None,
        progress_callback: Callable | None = None,
    ) -> ImportResult:
        import_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        md_files = list(self._walk_md_files(path))
        total = len(md_files)

        entries: list[dict[str, Any]] = []
        all_wikilinks: set[str] = set()
        errors: list[str] = []

        for i, md_path in enumerate(md_files):
            content = _read_file_safe(md_path)
            if content is None:
                errors.append(f"Failed to read: {md_path.relative_to(path)}")
                continue
            if not content.strip():
                continue

            rel_path = str(md_path.relative_to(path))
            title = md_path.stem

            # Classify as daily note or knowledge note
            date = _parse_daily_note_date(md_path.name)
            is_daily = date is not None
            if date:
                ts = dt.datetime.combine(date, dt.time()).isoformat()
            else:
                try:
                    mtime = md_path.stat().st_mtime
                    ts = dt.datetime.fromtimestamp(mtime).isoformat()
                except OSError:
                    ts = dt.datetime.now().isoformat()

            # Extract metadata
            tags = _parse_frontmatter_tags(content)
            wikilinks = WIKILINK_RE.findall(content)
            all_wikilinks.update(wikilinks)

            entries.append({
                "type": "note",
                "ts": ts,
                "title": title,
                "content": content,
                "source_path": rel_path,
                "is_daily": is_daily,
                "tags": tags,
                "wikilinks": wikilinks,
            })

            if progress_callback:
                progress_callback(i + 1, total)

        # Write to journal
        created_files = write_structured_import(
            "obsidian",
            entries,
            import_id=import_id,
            facet=facet,
        )

        # Seed entities from wikilinks
        entities_seeded = 0
        if all_wikilinks and facet:
            # Use the earliest date for seeding
            dates = sorted(e["ts"] for e in entries)
            day = dt.datetime.fromisoformat(dates[0]).strftime("%Y%m%d") if dates else dt.datetime.now().strftime("%Y%m%d")
            entity_dicts = [{"name": link, "type": "Topic"} for link in sorted(all_wikilinks)]
            resolved = seed_entities(facet, day, entity_dicts)
            entities_seeded = len(resolved)

        return ImportResult(
            entries_written=len(entries),
            entities_seeded=entities_seeded,
            files_created=created_files,
            errors=errors,
            summary=f"Imported {len(entries)} notes ({sum(1 for e in entries if e['is_daily'])} daily, {sum(1 for e in entries if not e['is_daily'])} knowledge) across {len(created_files)} days",
        )

    def _walk_md_files(self, root: Path) -> list[Path]:
        """Walk vault directory, yielding markdown files. Skips hidden dirs and logseq recycle."""
        md_files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Filter out hidden directories and logseq recycle in-place
            dirnames[:] = [
                d for d in dirnames
                if not _is_hidden(d) and not (d == ".recycle" and Path(dirpath).name == "logseq")
            ]
            # Also skip the logseq/.recycle path explicitly
            rel = Path(dirpath).relative_to(root)
            if len(rel.parts) >= 2 and rel.parts[0] == "logseq" and rel.parts[1] == ".recycle":
                continue

            for fname in filenames:
                if _is_hidden(fname):
                    continue
                fpath = Path(dirpath) / fname
                ext = fpath.suffix.lower()
                if ext != ".md":
                    continue
                if ext in SKIP_EXTENSIONS:
                    continue
                md_files.append(fpath)
        return md_files


importer = ObsidianImporter()
