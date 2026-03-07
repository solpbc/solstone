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
from think.importers.shared import seed_entities, window_items, write_markdown_segments

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
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".svg",
    ".webp",
    ".ico",
    ".pdf",
    ".mp3",
    ".mp4",
    ".wav",
    ".ogg",
    ".flac",
    ".m4a",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
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


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter block from content."""
    return FRONTMATTER_RE.sub("", content)


def _is_hidden(name: str) -> bool:
    """Check if a filename/dirname starts with a dot."""
    return name.startswith(".")


def _render_note_markdown(note: dict[str, Any]) -> str:
    """Render a note as markdown for imported.md output."""
    title = note.get("title", "Untitled")
    lines = [f"## {title}"]

    source_path = note.get("source_path", "")
    if source_path:
        lines.append(f"Source: {source_path}")

    tags = note.get("tags", [])
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")

    wikilinks = note.get("wikilinks", [])
    if wikilinks:
        lines.append("Links: " + ", ".join(f"[[{link}]]" for link in wikilinks))

    content = note.get("content", "")
    if content:
        stripped = _strip_frontmatter(content).strip()
        if stripped:
            lines.append("")
            lines.append(stripped)

    return "\n".join(lines)


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
            date = _parse_daily_note_date(md_path.name)
            if date:
                daily_count += 1
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
        md_files = list(self._walk_md_files(path))
        total = len(md_files)

        notes: list[dict[str, Any]] = []
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
            is_daily = _parse_daily_note_date(md_path.name) is not None

            # All notes use file mtime for segment placement (creation-moment principle)
            try:
                mtime = md_path.stat().st_mtime
            except OSError:
                mtime = dt.datetime.now().timestamp()

            tags = _parse_frontmatter_tags(content)
            wikilinks = WIKILINK_RE.findall(content)
            all_wikilinks.update(wikilinks)

            notes.append(
                {
                    "mtime": mtime,
                    "title": title,
                    "content": content,
                    "source_path": rel_path,
                    "is_daily": is_daily,
                    "tags": tags,
                    "wikilinks": wikilinks,
                }
            )

            if progress_callback:
                progress_callback(i + 1, total)

        if not notes:
            return ImportResult(
                entries_written=0,
                entities_seeded=0,
                files_created=[],
                errors=errors,
                summary="No notes found to import",
            )

        notes.sort(key=lambda n: n["mtime"])

        windows = window_items(notes, "mtime", tz=None)
        created_files, segments = write_markdown_segments(
            "obsidian",
            windows,
            lambda items: "\n\n".join(_render_note_markdown(n) for n in items),
            filename="note_transcript.md",
        )

        # Seed entities from wikilinks
        entities_seeded = 0
        if all_wikilinks and facet:
            day = segments[0][0] if segments else dt.datetime.now().strftime("%Y%m%d")
            entity_dicts = [
                {"name": link, "type": "Topic"} for link in sorted(all_wikilinks)
            ]
            resolved = seed_entities(facet, day, entity_dicts)
            entities_seeded = len(resolved)

        daily_count = sum(1 for n in notes if n["is_daily"])
        knowledge_count = len(notes) - daily_count

        return ImportResult(
            entries_written=len(notes),
            entities_seeded=entities_seeded,
            files_created=created_files,
            errors=errors,
            summary=(
                f"Imported {len(notes)} notes ({daily_count} daily, "
                f"{knowledge_count} knowledge) across "
                f"{len({d for d, _ in segments})} days into {len(segments)} segments"
            ),
            segments=segments,
        )

    def _walk_md_files(self, root: Path) -> list[Path]:
        """Walk vault directory, yielding markdown files. Skips hidden dirs and logseq recycle."""
        md_files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Filter out hidden directories and logseq recycle in-place
            dirnames[:] = [
                d
                for d in dirnames
                if not _is_hidden(d)
                and not (d == ".recycle" and Path(dirpath).name == "logseq")
            ]
            # Also skip the logseq/.recycle path explicitly
            rel = Path(dirpath).relative_to(root)
            if (
                len(rel.parts) >= 2
                and rel.parts[0] == "logseq"
                and rel.parts[1] == ".recycle"
            ):
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
