# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Granola meeting transcript importer via muesli.

Reads local markdown files produced by muesli (harper reed's Granola extractor)
from ~/.local/share/muesli/transcripts/ and imports them into the journal as
speaker-labeled transcript segments.

Architecture: SyncableBackend registered as "granola" in the sync registry.
No network calls — reads local files only.
"""

import datetime as dt
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import frontmatter

from think.entities.seeding import seed_entities
from think.importers.shared import (
    _window_messages,
    write_segment,
)
from think.importers.sync import load_sync_state, save_sync_state
from think.utils import day_path

logger = logging.getLogger(__name__)

DEFAULT_MUESLI_PATH = Path.home() / ".local" / "share" / "muesli" / "transcripts"

# **Name (HH:MM:SS):** text
TRANSCRIPT_RE = re.compile(
    r"^\*\*(.+?)\s*\((\d{1,2}:\d{2}:\d{2})\):\*\*\s*(.+)",
    re.MULTILINE,
)

# - **Name**, Title, Company, (email)
PARTICIPANT_RE = re.compile(
    r"^\s*-\s*\*\*(.+?)\*\*(?:,\s*(.+))?$",
    re.MULTILINE,
)

EMAIL_RE = re.compile(r"\(([^)]+@[^)]+)\)")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/([^\s,)]+)", re.IGNORECASE)

# YYYY-MM-DD at the start of a filename
FILENAME_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def _parse_muesli_file(path: Path) -> tuple[dict[str, Any], str]:
    """Parse a muesli markdown file into (frontmatter_dict, body_text)."""
    post = frontmatter.load(str(path))
    return dict(post.metadata), post.content


def _parse_participants(body: str) -> list[dict[str, Any]]:
    """Extract enriched participant info from the ## Participants section.

    Returns list of dicts with keys: name, email, title, company, linkedin.
    """
    section_match = re.search(
        r"^##\s*Participants\s*\n(.*?)(?=^##|\Z)",
        body,
        re.MULTILINE | re.DOTALL,
    )
    if not section_match:
        return []

    section = section_match.group(1)
    participants: list[dict[str, Any]] = []

    for match in PARTICIPANT_RE.finditer(section):
        name = match.group(1).strip()
        # Strip [[wikilink]] brackets
        name = name.replace("[[", "").replace("]]", "")

        rest = match.group(2) or ""

        # Extract email: (user@domain.com)
        email = ""
        email_match = EMAIL_RE.search(rest)
        if email_match:
            email = email_match.group(1).strip()
            rest = rest[: email_match.start()] + rest[email_match.end() :]

        # Extract LinkedIn handle
        linkedin = ""
        linkedin_match = LINKEDIN_RE.search(rest)
        if linkedin_match:
            linkedin = linkedin_match.group(1).strip()
            rest = rest[: linkedin_match.start()] + rest[linkedin_match.end() :]

        # Remaining comma-separated parts: title, company
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        title = parts[0] if len(parts) >= 1 else ""
        company = parts[1] if len(parts) >= 2 else ""

        participants.append(
            {
                "name": name,
                "email": email,
                "title": title,
                "company": company,
                "linkedin": linkedin,
            }
        )

    return participants


def _date_from_filename(filename: str) -> dt.date | None:
    """Extract date from muesli filename like 2025-10-28_q1-planning.md."""
    match = FILENAME_DATE_RE.match(filename)
    if match:
        return dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def _parse_created_at(
    fm: dict[str, Any], filename: str
) -> tuple[dt.datetime | None, dt.timezone | None]:
    """Parse created_at from frontmatter, falling back to filename date.

    Returns (datetime, timezone) where timezone is UTC if created_at had Z/+00:00,
    or None for naive/local time.
    """
    created_at_str = fm.get("created_at", "")
    if created_at_str:
        try:
            created_dt = dt.datetime.fromisoformat(
                str(created_at_str).replace("Z", "+00:00")
            )
            tz = created_dt.tzinfo
            return created_dt, tz
        except (ValueError, TypeError):
            pass

    # Fall back to filename date
    base_date = _date_from_filename(filename)
    if base_date:
        return dt.datetime(base_date.year, base_date.month, base_date.day), None
    return None, None


def _parse_transcript_entries(
    body: str,
    base_dt: dt.datetime,
    tz: dt.timezone | None,
) -> list[dict[str, Any]]:
    """Parse speaker-labeled transcript entries into message dicts.

    Each message has: create_time (float epoch), speaker, text, model_slug.
    """
    base_date = base_dt.date() if tz is None else base_dt.astimezone(tz).date()
    messages: list[dict[str, Any]] = []

    for match in TRANSCRIPT_RE.finditer(body):
        speaker = match.group(1).strip()
        time_str = match.group(2)
        text = match.group(3).strip()

        h, m, s = (int(p) for p in time_str.split(":"))

        if tz is not None:
            msg_dt = dt.datetime(
                base_date.year, base_date.month, base_date.day, h, m, s, tzinfo=tz
            )
        else:
            msg_dt = dt.datetime(
                base_date.year, base_date.month, base_date.day, h, m, s
            )

        messages.append(
            {
                "create_time": msg_dt.timestamp(),
                "speaker": speaker,
                "text": text,
                "model_slug": None,
            }
        )

    return messages


def _import_transcript(
    md_file: Path,
    fm: dict[str, Any],
    body: str,
    journal_root: Path,
) -> tuple[list[str], list[tuple[str, str]], int]:
    """Import a single muesli transcript into the journal.

    Returns (created_files, segments, entities_seeded).
    """
    doc_id = fm.get("doc_id", md_file.stem)
    title = fm.get("title", "")
    duration_seconds = fm.get("duration_seconds")
    summary_text = fm.get("summary_text", "")

    created_dt, tz = _parse_created_at(fm, md_file.name)
    if created_dt is None:
        logger.warning("No date for %s -- skipping", md_file.name)
        return [], [], 0

    # Parse transcript entries
    messages = _parse_transcript_entries(body, created_dt, tz)
    if not messages:
        return [], [], 0

    messages.sort(key=lambda m: m["create_time"])

    # Estimate duration if missing
    if not duration_seconds and len(messages) >= 2:
        duration_seconds = int(messages[-1]["create_time"] - messages[0]["create_time"])
    if not duration_seconds:
        duration_seconds = 300

    # Window into 5-min segments
    windows = _window_messages(messages)

    created_files: list[str] = []
    segments: list[tuple[str, str]] = []
    source_copied = False

    for day, seg_key, _model, entries in windows:
        day_dir = str(day_path(day))

        # Build setting string with summary if available
        setting = "meeting"
        if summary_text:
            setting = f"meeting -- {summary_text}"

        json_path = write_segment(
            day_dir,
            "import.granola",
            seg_key,
            entries,
            import_id=doc_id,
            topics=title,
            detected_setting=setting,
        )
        created_files.append(json_path)
        segments.append((day, seg_key))

        # Copy source markdown into first segment for ground-truth preservation
        if not source_copied:
            import os

            seg_dir = os.path.dirname(json_path)
            source_dest = os.path.join(seg_dir, "source.md")
            shutil.copy2(str(md_file), source_dest)
            created_files.append(source_dest)
            source_copied = True

    # Seed entities from participants
    entities_seeded = 0
    participants = _parse_participants(body)
    if participants and segments:
        first_day = segments[0][0]
        # Format day for observation attribution: YYYYMMDD -> YYYY-MM-DD
        obs_date = f"{first_day[:4]}-{first_day[4:6]}-{first_day[6:8]}"
        entity_dicts = []
        for p in participants:
            if not p["name"]:
                continue
            d: dict[str, Any] = {
                "name": p["name"],
                "type": "Person",
                "email": p["email"],
            }
            observations: list[str] = []
            title = p.get("title", "")
            company = p.get("company", "")
            linkedin = p.get("linkedin", "")
            if title and company:
                observations.append(f"{title} at {company} (via Granola, {obs_date})")
            elif title:
                observations.append(f"{title} (via Granola, {obs_date})")
            elif company:
                observations.append(f"Works at {company} (via Granola, {obs_date})")
            if linkedin:
                observations.append(
                    f"LinkedIn: linkedin.com/in/{linkedin} (via Granola, {obs_date})"
                )
            if observations:
                d["observations"] = observations
            entity_dicts.append(d)
        if entity_dicts:
            try:
                seeded = seed_entities("import.granola", first_day, entity_dicts)
                entities_seeded = len(seeded)
            except Exception as exc:
                logger.warning("Entity seeding failed for %s: %s", md_file.name, exc)

    return created_files, segments, entities_seeded


class GranolaBackend:
    """Syncable backend for Granola meeting transcripts via muesli."""

    name: str = "granola"

    def sync(
        self,
        journal_root: Path,
        *,
        dry_run: bool = True,
        source_path: Path | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Sync Granola transcripts from muesli local files.

        Scans the muesli transcripts directory for markdown files, compares
        against sync state, and imports new/updated transcripts into the journal.

        Args:
            journal_root: Path to the journal root directory.
            dry_run: If True, catalog only (no import). If False, import.
            source_path: Override muesli transcripts directory path.
            force: If True, ignore sync state and re-import everything.

        Returns:
            Summary dict with total, imported, available, skipped, downloaded, errors.
        """
        muesli_dir = source_path or DEFAULT_MUESLI_PATH

        # Three-state setup detection
        if not muesli_dir.exists():
            parent = muesli_dir.parent
            if parent.exists():
                raise ValueError(
                    "Muesli is installed but no transcripts found.\n"
                    "Run `muesli sync` to extract your Granola transcripts, "
                    "then try again."
                )
            raise ValueError(
                "Granola transcripts require muesli to extract.\n"
                "Install: cargo install --git "
                "https://github.com/harperreed/muesli.git --all-features\n"
                "Then run: muesli sync"
            )

        # Load or initialize sync state
        state = load_sync_state(journal_root, "granola") or {
            "backend": "granola",
            "source_path": str(muesli_dir),
            "files": {},
        }

        if force:
            state["files"] = {}

        known_files: dict[str, dict[str, Any]] = state.get("files", {})

        # Scan muesli transcripts directory
        md_files = sorted(muesli_dir.glob("*.md"))

        # Parse each file and determine sync status
        to_import: list[tuple[Path, dict[str, Any], str]] = []
        current_doc_ids: set[str] = set()
        skipped_count = 0

        for md_file in md_files:
            try:
                fm, body = _parse_muesli_file(md_file)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", md_file.name, exc)
                continue

            doc_id = fm.get("doc_id")
            if not doc_id:
                logger.debug("No doc_id in %s -- skipping", md_file.name)
                skipped_count += 1
                continue

            current_doc_ids.add(doc_id)
            remote_updated_at = str(fm.get("remote_updated_at", ""))
            title = fm.get("title", md_file.stem)
            duration = fm.get("duration_seconds", 0)

            # Check for transcript content
            has_transcript = bool(TRANSCRIPT_RE.search(body))
            if not has_transcript:
                if doc_id not in known_files:
                    known_files[doc_id] = {
                        "filename": md_file.name,
                        "title": title,
                        "remote_updated_at": remote_updated_at,
                        "status": "skipped",
                        "skip_reason": "no_transcript",
                    }
                skipped_count += 1
                continue

            # Check sync state for this doc
            if doc_id in known_files and not force:
                existing = known_files[doc_id]
                if existing.get("status") == "imported":
                    existing_updated = existing.get("remote_updated_at", "")
                    if remote_updated_at <= existing_updated:
                        # Unchanged -- skip
                        continue
                    # Updated -- will re-import

            # Mark as available for import
            known_files.setdefault(doc_id, {})
            known_files[doc_id].update(
                {
                    "filename": md_file.name,
                    "title": title,
                    "remote_updated_at": remote_updated_at,
                    "duration_seconds": duration,
                    "status": "available",
                }
            )
            to_import.append((md_file, fm, doc_id))

        # Detect deleted files (in state but gone from muesli dir)
        for doc_id, info in known_files.items():
            if doc_id not in current_doc_ids and info.get("status") not in (
                "removed",
                "skipped",
            ):
                info["status"] = "removed"

        # Compute summary
        total = len(known_files)
        imported = sum(1 for f in known_files.values() if f.get("status") == "imported")
        available = len(to_import)
        skipped_total = sum(
            1 for f in known_files.values() if f.get("status") == "skipped"
        )

        result: dict[str, Any] = {
            "total": total,
            "imported": imported,
            "available": available,
            "skipped": skipped_total,
            "downloaded": 0,
            "errors": [],
        }

        # Import transcripts if not dry-run
        if not dry_run and to_import:
            downloaded = 0
            errors: list[str] = []

            for md_file, _fm_partial, doc_id in to_import:
                try:
                    fm_full, body = _parse_muesli_file(md_file)
                    files, segs, entities = _import_transcript(
                        md_file, fm_full, body, journal_root
                    )

                    if files:
                        known_files[doc_id]["status"] = "imported"
                        known_files[doc_id]["imported_at"] = (
                            dt.datetime.now().isoformat()
                        )
                        known_files[doc_id]["segments"] = len(segs)
                        known_files[doc_id]["entities_seeded"] = entities
                        downloaded += 1
                        logger.info(
                            "Imported %s: %d segments, %d entities",
                            md_file.name,
                            len(segs),
                            entities,
                        )
                    else:
                        known_files[doc_id]["status"] = "skipped"
                        known_files[doc_id]["skip_reason"] = "no_content"
                except Exception as exc:
                    msg = f"{md_file.name}: {exc}"
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

        # Save updated state
        state["files"] = known_files
        state["source_path"] = str(muesli_dir)
        state["last_sync"] = dt.datetime.now().isoformat()
        save_sync_state(journal_root, "granola", state)

        return result


# Module-level backend instance for discovery
backend = GranolaBackend()
