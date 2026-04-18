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

from think.entities.seeding import seed_entities
from think.importers.file_importer import ImportPreview, ImportResult
from think.importers.shared import (
    map_items_to_segments,
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

# Folder name to entity type mapping (case-insensitive, after stripping numeric prefixes)
FOLDER_TYPE_MAP: dict[str, str] = {
    "people": "Person",
    "contacts": "Person",
    "projects": "Project",
    "companies": "Organization",
    "organizations": "Organization",
    "places": "Place",
    "locations": "Place",
}

# Numeric prefix pattern (e.g., "00 knowledge" → "knowledge")
NUMERIC_PREFIX_RE = re.compile(r"^\d+\s+")

# Default excluded folders — system/template artifacts, not user content.
# Matched case-insensitively. Hidden dirs (.obsidian, .trash) are already
# excluded by _is_hidden(); these cover non-hidden system folders.
DEFAULT_EXCLUDED_FOLDERS = frozenset(
    {
        "templates",
        "_templates",
    }
)


def infer_entity_type_from_path(rel_path: str) -> str | None:
    """Infer entity type from a note's relative folder path.

    Checks each folder component (after stripping numeric prefixes) against
    known entity-typed folder names. Returns the entity type or None.
    """
    parts = Path(rel_path).parent.parts
    for part in parts:
        cleaned = NUMERIC_PREFIX_RE.sub("", part).lower()
        entity_type = FOLDER_TYPE_MAP.get(cleaned)
        if entity_type:
            return entity_type
    return None


def _clean_at_prefix(name: str) -> tuple[str, bool]:
    """Strip @ prefix from an entity name.

    Returns (cleaned_name, had_at_prefix). Handles both '@Name' and '@ Name'.
    """
    if name.startswith("@"):
        return name[1:].lstrip(), True
    return name, False


def _build_entity_dicts(
    wikilinks: set[str],
    title_type_map: dict[str, str],
    at_filenames: set[str] | None = None,
) -> list[dict[str, str]]:
    """Build entity dicts from wikilinks with type inference.

    Precedence: @ prefix > folder-path type > "Topic" default.
    Also includes @-prefixed filenames as Person entities.
    """
    entities: dict[str, dict[str, str]] = {}

    for link in wikilinks:
        name, is_at = _clean_at_prefix(link)
        if not name:
            continue
        if is_at:
            entity_type = "Person"
        elif name in title_type_map:
            entity_type = title_type_map[name]
        else:
            entity_type = "Topic"
        # @ prefix wins if we've already seen this name without @
        if name not in entities or (is_at and entities[name]["type"] != "Person"):
            entities[name] = {"name": name, "type": entity_type}

    if at_filenames:
        for name in at_filenames:
            if name not in entities:
                entities[name] = {"name": name, "type": "Person"}

    return [entities[k] for k in sorted(entities)]


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


def _walk_md_files(root: Path) -> list[Path]:
    """Walk vault directory, collecting markdown files.

    Skips hidden dirs, default excluded folders (case-insensitive), and logseq recycle.
    """
    md_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if not _is_hidden(d)
            and d.lower() not in DEFAULT_EXCLUDED_FOLDERS
            and not (d == ".recycle" and Path(dirpath).name == "logseq")
        ]
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


class ObsidianImporter:
    name = "obsidian"
    display_name = "Obsidian / Logseq Vault"
    file_patterns = ["*.md"]
    description = "Import notes from an Obsidian or Logseq vault"

    def _walk_md_files(self, root: Path) -> list[Path]:
        return _walk_md_files(root)

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

        # Build title → entity type mapping from folder paths and @ filenames
        title_type_map: dict[str, str] = {}
        at_filenames: set[str] = set()
        for note in notes:
            folder_type = infer_entity_type_from_path(note["source_path"])
            if folder_type:
                title_type_map[note["title"]] = folder_type
            name, is_at = _clean_at_prefix(note["title"])
            if is_at and name:
                at_filenames.add(name)

        # Seed entities from wikilinks with type inference
        entities_seeded = 0
        if (all_wikilinks or at_filenames) and facet:
            day = segments[0][0] if segments else dt.datetime.now().strftime("%Y%m%d")
            entity_dicts = _build_entity_dicts(
                all_wikilinks, title_type_map, at_filenames
            )
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


class ObsidianSyncBackend:
    """Syncable backend for Obsidian vault incremental sync.

    Edits are activities -- when a note's content changes, a new journal segment
    is created at the edit timestamp. Old segments are preserved.
    """

    name: str = "obsidian"

    def sync(
        self,
        journal_root: Path,
        *,
        dry_run: bool = True,
        source_path: Path | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        state = load_sync_state(journal_root, "obsidian")

        vault_path: Path | None = None
        if source_path is not None:
            vault_path = source_path
        elif state and state.get("source_path"):
            vault_path = Path(str(state["source_path"]))
        else:
            for candidate in (
                Path.home() / "Documents" / "Obsidian",
                Path.home() / "Obsidian",
            ):
                if candidate.exists() and candidate.is_dir():
                    vault_path = candidate
                    break

        if vault_path is None:
            raise ValueError(
                "No Obsidian vault found. Use --path to specify your vault location."
            )
        if not vault_path.exists() or not vault_path.is_dir():
            raise ValueError(
                f"Obsidian vault not found at {vault_path}. "
                "Use --path to specify your vault location."
            )

        state = state or {
            "backend": "obsidian",
            "source_path": str(vault_path),
            "files": {},
        }
        if force:
            state["files"] = {}

        known_files: dict[str, dict[str, Any]] = state.get("files", {})
        to_import: list[dict[str, Any]] = []
        current_paths: set[str] = set()
        title_type_map: dict[str, str] = {}

        for md_path in _walk_md_files(vault_path):
            rel_path = str(md_path.relative_to(vault_path))
            current_paths.add(rel_path)

            # Track folder-path types for entity type inference
            folder_type = infer_entity_type_from_path(rel_path)
            if folder_type:
                title_type_map[md_path.stem] = folder_type

            content = _read_file_safe(md_path)
            if content is None or not content.strip():
                continue

            try:
                mtime = md_path.stat().st_mtime
            except OSError:
                continue

            existing = known_files.get(rel_path)
            if existing and existing.get("status") == "imported" and not force:
                if existing.get("mtime") == mtime:
                    continue
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if existing.get("content_hash") == content_hash:
                    existing["mtime"] = mtime
                    continue
            else:
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if (
                    existing
                    and existing.get("content_hash") == content_hash
                    and not force
                ):
                    continue

            title = md_path.stem
            tags = _parse_frontmatter_tags(content)
            wikilinks = WIKILINK_RE.findall(content)
            is_daily = _parse_daily_note_date(md_path.name) is not None

            known_files[rel_path] = {
                **(known_files.get(rel_path) or {}),
                "filename": md_path.name,
                "title": title,
                "mtime": mtime,
                "content_hash": content_hash,
                "status": "available",
                "edit_count": known_files.get(rel_path, {}).get("edit_count", 0),
            }
            to_import.append(
                {
                    "mtime": mtime,
                    "title": title,
                    "content": content,
                    "source_path": rel_path,
                    "is_daily": is_daily,
                    "tags": tags,
                    "wikilinks": wikilinks,
                    "rel_path": rel_path,
                    "content_hash": content_hash,
                }
            )

        for rel_path, info in known_files.items():
            if rel_path not in current_paths and info.get("status") not in ("removed",):
                info["status"] = "removed"

        result: dict[str, Any] = {
            "total": len(known_files),
            "imported": sum(
                1 for f in known_files.values() if f.get("status") == "imported"
            ),
            "available": len(to_import),
            "skipped": 0,
            "downloaded": 0,
            "errors": [],
        }

        if not dry_run and to_import:
            downloaded = 0
            errors: list[str] = []

            def render_fn(items: list[dict[str, Any]]) -> str:
                return "\n\n".join(_render_note_markdown(n) for n in items)

            for note in to_import:
                try:
                    windows = window_items([note], "mtime", tz=None)
                    _created_files, segs = write_markdown_segments(
                        "obsidian",
                        windows,
                        render_fn,
                        filename="note_transcript.md",
                    )

                    if segs:
                        day = segs[0][0]
                        note_at_filenames: set[str] = set()
                        name, is_at = _clean_at_prefix(note["title"])
                        if is_at and name:
                            note_at_filenames.add(name)
                        wikilink_set = (
                            set(note["wikilinks"]) if note["wikilinks"] else set()
                        )
                        if wikilink_set or note_at_filenames:
                            entity_dicts = _build_entity_dicts(
                                wikilink_set,
                                title_type_map,
                                note_at_filenames or None,
                            )
                            if entity_dicts:
                                try:
                                    seed_entities("import.obsidian", day, entity_dicts)
                                except Exception as exc:
                                    logger.warning(
                                        "Entity seeding failed for %s: %s",
                                        note["rel_path"],
                                        exc,
                                    )

                    known_files[note["rel_path"]].update(
                        {
                            "status": "imported",
                            "imported_at": dt.datetime.now().isoformat(),
                            "segments": len(segs),
                            "edit_count": known_files[note["rel_path"]].get(
                                "edit_count", 0
                            )
                            + 1,
                        }
                    )
                    downloaded += 1
                except Exception as exc:
                    msg = f"{note['rel_path']}: {exc}"
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

        state["files"] = known_files
        state["source_path"] = str(vault_path)
        state["last_sync"] = dt.datetime.now().isoformat()
        save_sync_state(journal_root, "obsidian", state)

        return result


importer = ObsidianImporter()
backend = ObsidianSyncBackend()
