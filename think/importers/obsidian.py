# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Obsidian and Logseq vault importer."""

import datetime as dt
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

from think.importers.file_importer import ImportPreview, ImportResult
from think.importers.shared import (
    map_items_to_segments,
    seed_entities,
    window_items,
    write_content_manifest,
    write_markdown_segments,
)
from think.importers.sync import load_sync_state, save_sync_state

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
        import_id: str | None = None,
        progress_callback: Callable | None = None,
    ) -> ImportResult:
        md_files = list(self._walk_md_files(path))
        total = len(md_files)
        import_id = import_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        notes: list[dict[str, Any]] = []
        all_wikilinks: set[str] = set()
        errors: list[str] = []
        earliest_ts: float | None = None
        latest_ts: float | None = None

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
            if earliest_ts is None or mtime < earliest_ts:
                earliest_ts = mtime
            if latest_ts is None or mtime > latest_ts:
                latest_ts = mtime

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
                progress_callback(
                    i + 1,
                    total,
                    earliest_date=(
                        dt.datetime.fromtimestamp(earliest_ts).strftime("%Y%m%d")
                        if earliest_ts
                        else None
                    ),
                    latest_date=(
                        dt.datetime.fromtimestamp(latest_ts).strftime("%Y%m%d")
                        if latest_ts
                        else None
                    ),
                    entities_found=len(all_wikilinks),
                )

        if not notes:
            return ImportResult(
                entries_written=0,
                entities_seeded=0,
                files_created=[],
                errors=errors,
                summary="No notes found to import",
            )
        if earliest_ts is not None and latest_ts is not None:
            earliest = dt.datetime.fromtimestamp(earliest_ts).strftime("%Y%m%d")
            latest = dt.datetime.fromtimestamp(latest_ts).strftime("%Y%m%d")
        else:
            earliest = latest = dt.datetime.now().strftime("%Y%m%d")

        notes.sort(key=lambda n: n["mtime"])
        note_manifest: list[dict[str, Any]] = []
        for i, note in enumerate(notes):
            note_dt = dt.datetime.fromtimestamp(note["mtime"])
            meta: dict[str, Any] = {}
            if note.get("tags"):
                meta["tags"] = note["tags"]
            if note.get("is_daily"):
                meta["is_daily"] = True
            raw_preview = _strip_frontmatter(note.get("content", "")).strip()[:300]
            # Strip markdown syntax for clean plain-text preview
            clean_preview = re.sub(r"^#{1,6}\s+", "", raw_preview, flags=re.MULTILINE)
            clean_preview = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean_preview)
            clean_preview = re.sub(r"\*([^*]+)\*", r"\1", clean_preview)
            clean_preview = re.sub(r"^[-*+]\s+", "", clean_preview, flags=re.MULTILINE)
            clean_preview = re.sub(
                r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]",
                lambda m: m.group(2) or m.group(1),
                clean_preview,
            )
            clean_preview = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_preview)
            clean_preview = re.sub(r"`([^`]+)`", r"\1", clean_preview)
            clean_preview = re.sub(r"^>\s+", "", clean_preview, flags=re.MULTILINE)
            clean_preview = " ".join(clean_preview.split())[:200]
            note_manifest.append(
                {
                    "id": f"note-{i}",
                    "title": note["title"],
                    "date": note_dt.strftime("%Y%m%d"),
                    "type": "note",
                    "preview": clean_preview,
                    "meta": meta,
                    "segments": [],
                }
            )

        windows = window_items(notes, "mtime", tz=None)
        created_files, segments = write_markdown_segments(
            "obsidian",
            windows,
            lambda items: "\n\n".join(_render_note_markdown(n) for n in items),
            filename="note_transcript.md",
        )
        note_segments = map_items_to_segments(
            [note["mtime"] for note in notes],
            tz=None,
        )
        for manifest_entry, (day, key) in zip(
            note_manifest, note_segments, strict=False
        ):
            manifest_entry["segments"] = [{"day": day, "key": key}]
        write_content_manifest(import_id, note_manifest)

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
            date_range=(earliest, latest),
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


# Common Obsidian vault locations for auto-detection
DEFAULT_VAULT_PATHS = [
    Path.home() / "Documents" / "Obsidian",
    Path.home() / "Obsidian",
]


def _content_hash(content: str) -> str:
    """SHA-256 hash of file content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _find_vault_path(
    source_path: Path | None, state: dict[str, Any] | None
) -> Path:
    """Resolve vault path from explicit arg, sync state, or auto-detection."""
    if source_path:
        if not source_path.is_dir():
            raise ValueError(f"Vault path does not exist: {source_path}")
        return source_path

    # Check existing sync state
    if state and state.get("source_path"):
        saved = Path(state["source_path"])
        if saved.is_dir():
            return saved

    # Auto-detect common locations
    for p in DEFAULT_VAULT_PATHS:
        if p.is_dir():
            return p

    raise ValueError(
        "No Obsidian vault found.\n"
        "Specify a vault path: sol import --sync obsidian --path /path/to/vault"
    )


class ObsidianSyncBackend:
    """Syncable backend for Obsidian vault notes with edit-as-activity model."""

    name: str = "obsidian"

    def sync(
        self,
        journal_root: Path,
        *,
        dry_run: bool = True,
        source_path: Path | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Sync Obsidian vault notes incrementally.

        Scans the vault for markdown files, compares against sync state,
        and imports new notes and captures edits as new journal segments.

        Args:
            journal_root: Path to the journal root directory.
            dry_run: If True, catalog only (no import). If False, import.
            source_path: Override vault directory path.
            force: If True, clear sync state and re-import everything.

        Returns:
            Summary dict with total, imported, available, skipped, downloaded, errors.
        """
        state = load_sync_state(journal_root, "obsidian")

        vault_path = _find_vault_path(source_path, state)

        if state is None:
            state = {
                "backend": "obsidian",
                "source_path": str(vault_path),
                "files": {},
            }

        if force:
            state["files"] = {}

        known_files: dict[str, dict[str, Any]] = state.get("files", {})

        # Walk vault using existing importer logic
        md_files = importer._walk_md_files(vault_path)

        current_rel_paths: set[str] = set()
        to_import: list[tuple[Path, str, str]] = []  # (path, rel_path, change_type)

        for md_path in md_files:
            rel_path = str(md_path.relative_to(vault_path))
            current_rel_paths.add(rel_path)

            content = _read_file_safe(md_path)
            if content is None or not content.strip():
                continue

            try:
                mtime = md_path.stat().st_mtime
            except OSError:
                continue

            hash_val = _content_hash(content)

            if rel_path in known_files and not force:
                existing = known_files[rel_path]
                if existing.get("status") == "imported":
                    # Fast path: mtime unchanged — skip
                    if mtime == existing.get("mtime"):
                        continue
                    # Correctness: hash unchanged — mtime-only change, skip
                    if hash_val == existing.get("content_hash"):
                        existing["mtime"] = mtime
                        continue
                    # Content actually changed — edit
                    change_type = "edited"
                elif existing.get("status") == "removed":
                    change_type = "new"
                else:
                    change_type = "new"
            else:
                change_type = "new"

            known_files.setdefault(rel_path, {})
            known_files[rel_path].update(
                {
                    "filename": md_path.name,
                    "title": md_path.stem,
                    "mtime": mtime,
                    "content_hash": hash_val,
                    "status": "available",
                    "_change_type": change_type,
                }
            )
            to_import.append((md_path, rel_path, change_type))

        # Detect deleted files
        for rel_path, info in known_files.items():
            if rel_path not in current_rel_paths and info.get("status") not in (
                "removed",
                "skipped",
            ):
                info["status"] = "removed"

        total = len(known_files)
        imported = sum(
            1 for f in known_files.values() if f.get("status") == "imported"
        )
        available = len(to_import)
        skipped_total = sum(
            1
            for f in known_files.values()
            if f.get("status") in ("skipped", "removed")
        )

        result: dict[str, Any] = {
            "total": total,
            "imported": imported,
            "available": available,
            "skipped": skipped_total,
            "downloaded": 0,
            "errors": [],
        }

        if not dry_run and to_import:
            downloaded = 0
            errors: list[str] = []

            for md_path, rel_path, change_type in to_import:
                try:
                    content = _read_file_safe(md_path)
                    if content is None:
                        errors.append(f"Failed to read: {rel_path}")
                        continue

                    mtime = md_path.stat().st_mtime
                    tags = _parse_frontmatter_tags(content)
                    wikilinks = WIKILINK_RE.findall(content)

                    note = {
                        "mtime": mtime,
                        "title": md_path.stem,
                        "content": content,
                        "source_path": rel_path,
                        "is_daily": _parse_daily_note_date(md_path.name) is not None,
                        "tags": tags,
                        "wikilinks": wikilinks,
                    }

                    windows = window_items([note], "mtime", tz=None)
                    created_files, segments = write_markdown_segments(
                        "obsidian",
                        windows,
                        lambda items: "\n\n".join(
                            _render_note_markdown(n) for n in items
                        ),
                        filename="note_transcript.md",
                    )

                    # Seed entities from wikilinks
                    if wikilinks and segments:
                        entity_dicts = [
                            {"name": link, "type": "Topic"}
                            for link in sorted(set(wikilinks))
                        ]
                        try:
                            seed_entities(
                                "import.obsidian", segments[0][0], entity_dicts
                            )
                        except Exception as exc:
                            logger.warning(
                                "Entity seeding failed for %s: %s", rel_path, exc
                            )

                    # Update sync state for this file
                    file_state = known_files[rel_path]
                    edit_count = file_state.get("edit_count", 0)
                    if change_type == "edited":
                        edit_count += 1

                    file_state.update(
                        {
                            "status": "imported",
                            "mtime": mtime,
                            "content_hash": _content_hash(content),
                            "edit_count": edit_count,
                            "imported_at": dt.datetime.now().isoformat(),
                            "segments": len(segments),
                        }
                    )
                    file_state.pop("_change_type", None)

                    downloaded += 1
                    action = "Re-imported (edit)" if change_type == "edited" else "Imported"
                    logger.info("%s %s: %d segments", action, rel_path, len(segments))

                except Exception as exc:
                    msg = f"{rel_path}: {exc}"
                    logger.warning("Import failed: %s", msg)
                    errors.append(msg)

            result["downloaded"] = downloaded
            result["errors"] = errors
            result["imported"] = sum(
                1 for f in known_files.values() if f.get("status") == "imported"
            )
            result["available"] = sum(
                1 for f in known_files.values() if f.get("status") == "available"
            )

        # Clean transient fields before persisting
        for info in known_files.values():
            info.pop("_change_type", None)

        state["files"] = known_files
        state["source_path"] = str(vault_path)
        state["last_sync"] = dt.datetime.now().isoformat()
        save_sync_state(journal_root, "obsidian", state)

        return result


importer = ObsidianImporter()

# Module-level backend instance for registry discovery
backend = ObsidianSyncBackend()
