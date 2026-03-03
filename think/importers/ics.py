# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Google Calendar ICS file importer."""

import datetime as dt
import logging
import zipfile
from pathlib import Path
from typing import Any, Callable

from think.importers.file_importer import ImportPreview, ImportResult
from think.importers.shared import seed_entities, write_structured_import

logger = logging.getLogger(__name__)

# How far back to expand recurring events (from now)
_RECURRENCE_LOOKBACK_YEARS = 2


def _extract_ics_data(path: Path) -> list[bytes]:
    """Extract ICS file content(s) from a single .ics file or a ZIP archive.

    Returns a list of raw ICS bytes (one per .ics file found).
    """
    if path.suffix.lower() == ".ics":
        return [path.read_bytes()]

    if path.suffix.lower() == ".zip":
        result = []
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".ics"):
                    result.append(zf.read(name))
        return result

    return []


def _parse_attendee(attendee: Any) -> dict[str, str] | None:
    """Extract name and email from a vCalAddress (ATTENDEE or ORGANIZER).

    Handles formats like:
        ATTENDEE;CN=Name;RSVP=TRUE:mailto:email@example.com
        ORGANIZER;CN=Name:mailto:email@example.com
    """
    try:
        email = str(attendee).replace("mailto:", "").replace("MAILTO:", "").strip()
        if not email or "@" not in email:
            return None
        name = str(attendee.params.get("CN", "")) if hasattr(attendee, "params") else ""
        return {"name": name.strip(), "email": email.lower()}
    except Exception:
        return None


def _dt_to_iso(val: Any) -> str | None:
    """Convert an icalendar date/datetime value to ISO 8601 string.

    Handles date (all-day) and datetime (with or without timezone).
    All-day events use midnight UTC.
    """
    if val is None:
        return None

    d = val.dt if hasattr(val, "dt") else val

    if isinstance(d, dt.datetime):
        # If naive, assume UTC
        if d.tzinfo is None:
            return d.isoformat()
        return d.isoformat()
    elif isinstance(d, dt.date):
        # All-day event: use midnight
        return dt.datetime(d.year, d.month, d.day).isoformat()

    return None


def _duration_minutes(dtstart: Any, dtend: Any) -> int | None:
    """Calculate event duration in minutes from DTSTART/DTEND."""
    try:
        start = dtstart.dt if hasattr(dtstart, "dt") else dtstart
        end = dtend.dt if hasattr(dtend, "dt") else dtend

        # Normalize date to datetime for subtraction
        if isinstance(start, dt.date) and not isinstance(start, dt.datetime):
            start = dt.datetime(start.year, start.month, start.day)
        if isinstance(end, dt.date) and not isinstance(end, dt.datetime):
            end = dt.datetime(end.year, end.month, end.day)

        # Strip timezone info for delta calc if mismatched
        if hasattr(start, "tzinfo") and hasattr(end, "tzinfo"):
            if (start.tzinfo is None) != (end.tzinfo is None):
                start = start.replace(tzinfo=None)
                end = end.replace(tzinfo=None)

        delta = end - start
        return max(0, int(delta.total_seconds() / 60))
    except Exception:
        return None


def _expand_rrule(component: Any, dtstart_val: Any) -> list[dt.datetime]:
    """Expand RRULE into concrete datetimes within the lookback window.

    Returns list of occurrence datetimes (excluding the original DTSTART).
    """
    from dateutil import rrule as du_rrule

    rrule_prop = component.get("RRULE")
    if not rrule_prop:
        return []

    try:
        start = dtstart_val.dt if hasattr(dtstart_val, "dt") else dtstart_val

        # Normalize date → datetime
        if isinstance(start, dt.date) and not isinstance(start, dt.datetime):
            start = dt.datetime(start.year, start.month, start.day)

        # Ensure timezone-aware for bounds comparison
        now = dt.datetime.now(dt.timezone.utc)
        if start.tzinfo is None:
            now = now.replace(tzinfo=None)

        window_start = now - dt.timedelta(days=_RECURRENCE_LOOKBACK_YEARS * 365)

        # Convert rrule dict to string for dateutil
        rrule_dict = dict(rrule_prop)
        rrule_str = ";".join(f"{k}={v}" for k, v in rrule_dict.items())

        rule = du_rrule.rrulestr(
            f"RRULE:{rrule_str}",
            dtstart=start,
            ignoretz=start.tzinfo is None,
        )

        # Collect occurrences within bounds (cap at 1000 to avoid runaway)
        occurrences = []
        for occ in rule:
            if occ > now:
                break
            if occ >= window_start and occ != start:
                occurrences.append(occ)
            if len(occurrences) >= 1000:
                break

        return occurrences
    except Exception as exc:
        logger.debug("Failed to expand RRULE: %s", exc)
        return []


def _parse_events(ics_bytes: bytes) -> list[dict[str, Any]]:
    """Parse VEVENT components from raw ICS bytes into structured entries."""
    import icalendar

    entries: list[dict[str, Any]] = []

    try:
        cal = icalendar.Calendar.from_ical(ics_bytes)
    except Exception as exc:
        logger.warning("Failed to parse ICS data: %s", exc)
        return entries

    for component in cal.walk():
        if component.name != "VEVENT":
            continue

        try:
            dtstart = component.get("DTSTART")
            ts = _dt_to_iso(dtstart)
            if not ts:
                continue

            dtend = component.get("DTEND")
            duration = _duration_minutes(dtstart, dtend) if dtend else None

            title = str(component.get("SUMMARY", "")) or "Untitled event"
            description = str(component.get("DESCRIPTION", "")) or ""
            location = str(component.get("LOCATION", "")) or ""

            # Collect attendees
            attendees: list[dict[str, str]] = []
            seen_emails: set[str] = set()

            # Organizer
            organizer = component.get("ORGANIZER")
            if organizer:
                parsed = _parse_attendee(organizer)
                if parsed and parsed["email"] not in seen_emails:
                    attendees.append(parsed)
                    seen_emails.add(parsed["email"])

            # Attendees (may be a list or a single value)
            raw_attendees = component.get("ATTENDEE")
            if raw_attendees:
                if not isinstance(raw_attendees, list):
                    raw_attendees = [raw_attendees]
                for att in raw_attendees:
                    parsed = _parse_attendee(att)
                    if parsed and parsed["email"] not in seen_emails:
                        attendees.append(parsed)
                        seen_emails.add(parsed["email"])

            # Build base entry
            entry: dict[str, Any] = {
                "type": "calendar_event",
                "ts": ts,
                "title": title,
                "content": description,
            }
            if duration is not None:
                entry["duration_minutes"] = duration
            if location:
                entry["location"] = location
            if attendees:
                entry["attendees"] = attendees

            entries.append(entry)

            # Expand recurring events
            recurrences = _expand_rrule(component, dtstart)
            for occ_dt in recurrences:
                occ_ts = occ_dt.isoformat()
                occ_entry = {**entry, "ts": occ_ts}
                entries.append(occ_entry)

        except Exception as exc:
            summary = component.get("SUMMARY", "<unknown>")
            logger.warning("Skipping event %r: %s", summary, exc)

    return entries


class ICSImporter:
    name = "ics"
    display_name = "Google Calendar (ICS)"
    file_patterns = ["*.ics", "*.zip"]
    description = "Import events from ICS calendar files or Google Calendar export ZIP"

    def detect(self, path: Path) -> bool:
        if not path.is_file():
            return False

        suffix = path.suffix.lower()

        if suffix == ".ics":
            return True

        if suffix == ".zip":
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    return any(n.lower().endswith(".ics") for n in zf.namelist())
            except zipfile.BadZipFile:
                return False

        return False

    def preview(self, path: Path) -> ImportPreview:
        ics_blobs = _extract_ics_data(path)
        if not ics_blobs:
            return ImportPreview(
                date_range=("", ""),
                item_count=0,
                entity_count=0,
                summary="No ICS data found",
            )

        all_entries: list[dict[str, Any]] = []
        for blob in ics_blobs:
            all_entries.extend(_parse_events(blob))

        if not all_entries:
            return ImportPreview(
                date_range=("", ""),
                item_count=0,
                entity_count=0,
                summary="No events found in ICS data",
            )

        # Date range
        dates = sorted(
            dt.datetime.fromisoformat(e["ts"]).strftime("%Y%m%d")
            for e in all_entries
            if e.get("ts")
        )
        date_range = (dates[0], dates[-1]) if dates else ("", "")

        # Unique attendees by email
        unique_emails: set[str] = set()
        for e in all_entries:
            for att in e.get("attendees", []):
                if att.get("email"):
                    unique_emails.add(att["email"])

        return ImportPreview(
            date_range=date_range,
            item_count=len(all_entries),
            entity_count=len(unique_emails),
            summary=f"{len(all_entries)} events, {len(unique_emails)} unique attendees",
        )

    def process(
        self,
        path: Path,
        journal_root: Path,
        *,
        facet: str | None = None,
        progress_callback: Callable | None = None,
    ) -> ImportResult:
        ics_blobs = _extract_ics_data(path)
        import_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        all_entries: list[dict[str, Any]] = []
        errors: list[str] = []

        for i, blob in enumerate(ics_blobs):
            try:
                all_entries.extend(_parse_events(blob))
            except Exception as exc:
                errors.append(f"Failed to parse ICS file {i}: {exc}")

            if progress_callback:
                progress_callback(i + 1, len(ics_blobs))

        if not all_entries:
            return ImportResult(
                entries_written=0,
                entities_seeded=0,
                files_created=[],
                errors=errors,
                summary="No events found to import",
            )

        # Write structured entries
        created_files = write_structured_import(
            "ics",
            all_entries,
            import_id=import_id,
            facet=facet,
        )

        # Seed entities from attendees
        entities_seeded = 0
        if facet:
            # Collect unique attendees across all entries, grouped by day
            by_day: dict[str, list[dict[str, str]]] = {}
            seen_emails: set[str] = set()

            for entry in all_entries:
                day = dt.datetime.fromisoformat(entry["ts"]).strftime("%Y%m%d")
                for att in entry.get("attendees", []):
                    email = att.get("email", "")
                    name = att.get("name", "")
                    if not email or not name:
                        continue
                    if email not in seen_emails:
                        seen_emails.add(email)
                        by_day.setdefault(day, []).append(
                            {"name": name, "type": "Person", "email": email}
                        )

            for day, day_entities in by_day.items():
                try:
                    resolved = seed_entities(facet, day, day_entities)
                    entities_seeded += len(resolved)
                except Exception as exc:
                    errors.append(f"Failed to seed entities for {day}: {exc}")

        return ImportResult(
            entries_written=len(all_entries),
            entities_seeded=entities_seeded,
            files_created=created_files,
            errors=errors,
            summary=(
                f"Imported {len(all_entries)} calendar events across "
                f"{len(created_files)} days, {entities_seeded} entities seeded"
            ),
        )


importer = ICSImporter()
