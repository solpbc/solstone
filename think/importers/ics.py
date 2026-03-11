# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Google Calendar ICS file importer."""

import datetime as dt
import logging
import zipfile
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

logger = logging.getLogger(__name__)


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


def _creation_timestamp(component: Any) -> float | None:
    """Extract a creation timestamp from VEVENT metadata."""

    for field in ("LAST-MODIFIED", "CREATED", "DTSTART"):
        value = component.get(field)
        if value is None:
            continue

        try:
            parsed = value.dt if hasattr(value, "dt") else value
            if isinstance(parsed, dt.date) and not isinstance(parsed, dt.datetime):
                parsed = dt.datetime(parsed.year, parsed.month, parsed.day)
            if not isinstance(parsed, dt.datetime):
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            if field == "DTSTART":
                logger.debug(
                    "VEVENT missing CREATED/LAST-MODIFIED; falling back to DTSTART"
                )
            return parsed.timestamp()
        except Exception as exc:
            logger.debug("Failed to parse %s timestamp: %s", field, exc)

    return None


def _describe_rrule(rrule: dict[str, Any]) -> str:
    """Convert an icalendar vRecur dict to a human-readable description."""
    freq_list = rrule.get("FREQ", [])
    if not freq_list:
        return ""
    freq = str(freq_list[0])

    interval = int(rrule.get("INTERVAL", [1])[0])

    day_names = {
        "MO": "Mon",
        "TU": "Tue",
        "WE": "Wed",
        "TH": "Thu",
        "FR": "Fri",
        "SA": "Sat",
        "SU": "Sun",
    }

    freq_map = {
        "DAILY": ("day", "days", "Daily"),
        "WEEKLY": ("week", "weeks", "Weekly"),
        "MONTHLY": ("month", "months", "Monthly"),
        "YEARLY": ("year", "years", "Yearly"),
    }
    if freq not in freq_map:
        return ""

    _singular, plural, adjective = freq_map[freq]

    if interval == 1:
        desc = adjective
    else:
        desc = f"Every {interval} {plural}"

    by_day = rrule.get("BYDAY", [])
    if by_day:
        names = [day_names.get(str(d).lstrip("+-0123456789"), str(d)) for d in by_day]
        desc += f" on {', '.join(names)}"

    by_monthday = rrule.get("BYMONTHDAY", [])
    if by_monthday:
        days_str = ", ".join(str(d) for d in by_monthday)
        desc += f" on day {days_str}"

    by_month = rrule.get("BYMONTH", [])
    if by_month:
        month_names = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        names = [month_names.get(int(m), str(m)) for m in by_month]
        desc += f" in {', '.join(names)}"

    count = rrule.get("COUNT", [])
    if count:
        desc += f", {int(count[0])} times"

    until = rrule.get("UNTIL", [])
    if until:
        until_val = until[0]
        if hasattr(until_val, "strftime"):
            desc += f", until {until_val.strftime('%Y-%m-%d')}"

    return desc


def _render_event_markdown(event: dict[str, Any]) -> str:
    """Render a calendar event as markdown."""
    title = event.get("title", "Untitled event")
    lines = [f"## {title}"]

    ts = event.get("ts")
    end_ts = event.get("end_ts")
    duration = event.get("duration_minutes")
    if ts:
        try:
            start_dt = dt.datetime.fromisoformat(ts)
            time_line = start_dt.strftime("%Y-%m-%d %I:%M %p")
            if end_ts:
                end_dt = dt.datetime.fromisoformat(end_ts)
                time_line = f"{time_line} – {end_dt.strftime('%I:%M %p')}"
            time_line = f"**{time_line}**"
            if duration is not None:
                time_line += f" ({duration} min)"
            lines.append(time_line)
        except ValueError:
            pass

    recurrence = event.get("recurrence", "")
    if recurrence:
        lines.append(f"🔁 {recurrence}")

    location = event.get("location", "")
    if location:
        lines.append(f"📍 {location}")

    attendees = event.get("attendees", [])
    attendee_names = []
    for attendee in attendees:
        if not isinstance(attendee, dict):
            attendee_names.append(str(attendee))
            continue
        attendee_name = attendee.get("name") or attendee.get("email", "")
        if attendee_name:
            attendee_names.append(attendee_name)
    if attendee_names:
        lines.append(f"👥 {', '.join(attendee_names)}")

    description = event.get("content", "")
    if description:
        lines.append("")
        lines.append(description)

    return "\n".join(lines)


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
            dtend = component.get("DTEND")
            ts = _dt_to_iso(dtstart)
            end_ts = _dt_to_iso(dtend) if dtend else None
            create_ts = _creation_timestamp(component)
            if create_ts is None:
                continue

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

            # Recurrence rule
            rrule = component.get("RRULE")
            recurrence = ""
            if rrule:
                recurrence = _describe_rrule(dict(rrule))

            # Build base entry
            entry: dict[str, Any] = {
                "type": "calendar_event",
                "title": title,
                "content": description,
                "create_ts": create_ts,
            }
            if ts:
                entry["ts"] = ts
            if end_ts:
                entry["end_ts"] = end_ts
            if duration is not None:
                entry["duration_minutes"] = duration
            if location:
                entry["location"] = location
            if attendees:
                entry["attendees"] = attendees
            if recurrence:
                entry["recurrence"] = recurrence

            entries.append(entry)

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
            dt.datetime.fromtimestamp(e["create_ts"], tz=dt.timezone.utc).strftime(
                "%Y%m%d"
            )
            for e in all_entries
            if e.get("create_ts") is not None
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
        import_id: str | None = None,
        progress_callback: Callable | None = None,
    ) -> ImportResult:
        ics_blobs = _extract_ics_data(path)
        import_id = import_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        all_entries: list[dict[str, Any]] = []
        errors: list[str] = []
        earliest_so_far: str | None = None
        latest_so_far: str | None = None

        for i, blob in enumerate(ics_blobs):
            try:
                parsed_entries = _parse_events(blob)
                all_entries.extend(parsed_entries)
                if parsed_entries:
                    parsed_dates = sorted(
                        dt.datetime.fromtimestamp(
                            entry["create_ts"], tz=dt.timezone.utc
                        ).strftime("%Y%m%d")
                        for entry in parsed_entries
                    )
                    if earliest_so_far is None or parsed_dates[0] < earliest_so_far:
                        earliest_so_far = parsed_dates[0]
                    if latest_so_far is None or parsed_dates[-1] > latest_so_far:
                        latest_so_far = parsed_dates[-1]
            except Exception as exc:
                errors.append(f"Failed to parse ICS file {i}: {exc}")

            if progress_callback:
                progress_callback(
                    i + 1,
                    len(ics_blobs),
                    earliest_date=earliest_so_far,
                    latest_date=latest_so_far,
                    entities_found=0,
                )

        if not all_entries:
            return ImportResult(
                entries_written=0,
                entities_seeded=0,
                files_created=[],
                errors=errors,
                summary="No events found to import",
            )

        all_entries.sort(key=lambda entry: entry["create_ts"])
        manifest_entries: list[dict[str, Any]] = []
        for i, entry in enumerate(all_entries):
            create_dt = dt.datetime.fromtimestamp(
                entry["create_ts"], tz=dt.timezone.utc
            )
            meta: dict[str, Any] = {}
            if entry.get("ts") and entry.get("end_ts"):
                try:
                    start_dt = dt.datetime.fromisoformat(entry["ts"])
                    end_dt = dt.datetime.fromisoformat(entry["end_ts"])
                    meta["time_range"] = (
                        f"{start_dt.strftime('%I:%M %p').lstrip('0')}"
                        f"–{end_dt.strftime('%I:%M %p').lstrip('0')}"
                    )
                except ValueError:
                    pass
            if entry.get("location"):
                meta["location"] = entry["location"]
            if entry.get("duration_minutes") is not None:
                meta["duration_minutes"] = entry["duration_minutes"]
            if entry.get("attendees"):
                meta["attendee_count"] = len(entry["attendees"])
                meta["attendee_names"] = [
                    a.get("name") or a.get("email", "")
                    for a in entry["attendees"][:5]
                ]
            if entry.get("recurrence"):
                meta["recurrence"] = entry["recurrence"]

            # Build plain-text preview from structured data if description is empty
            preview = entry.get("content", "").strip()
            if not preview:
                parts: list[str] = []
                if meta.get("time_range"):
                    parts.append(meta["time_range"])
                if entry.get("location"):
                    parts.append(entry["location"])
                if meta.get("attendee_names"):
                    parts.append(", ".join(meta["attendee_names"]))
                if entry.get("recurrence"):
                    parts.append(entry["recurrence"])
                preview = " · ".join(parts)
            manifest_entries.append(
                {
                    "id": f"event-{i}",
                    "title": entry.get("title", "Untitled event"),
                    "date": create_dt.strftime("%Y%m%d"),
                    "type": "event",
                    "preview": preview[:200],
                    "meta": meta,
                    "segments": [],
                }
            )
        earliest = dt.datetime.fromtimestamp(
            all_entries[0]["create_ts"], tz=dt.timezone.utc
        ).strftime("%Y%m%d")
        latest = dt.datetime.fromtimestamp(
            all_entries[-1]["create_ts"], tz=dt.timezone.utc
        ).strftime("%Y%m%d")

        windows = window_items(all_entries, "create_ts")
        created_files, segments = write_markdown_segments(
            "ics",
            windows,
            lambda items: "\n\n".join(_render_event_markdown(e) for e in items),
            filename="event_transcript.md",
        )
        item_segments = map_items_to_segments(
            [entry["create_ts"] for entry in all_entries],
            tz=dt.timezone.utc,
        )
        for manifest_entry, (day, key) in zip(
            manifest_entries,
            item_segments,
            strict=False,
        ):
            manifest_entry["segments"] = [{"day": day, "key": key}]
        write_content_manifest(import_id, manifest_entries)

        segment_days = {day for day, _ in segments}

        # Seed entities from attendees
        entities_seeded = 0
        if facet:
            # Collect unique attendees across all entries, grouped by day
            by_day: dict[str, list[dict[str, str]]] = {}
            seen_emails: set[str] = set()

            for entry in all_entries:
                day = dt.datetime.fromtimestamp(
                    entry["create_ts"], tz=dt.timezone.utc
                ).strftime("%Y%m%d")
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
                f"{len(segment_days)} days into {len(segments)} segments, "
                f"{entities_seeded} entities seeded"
            ),
            segments=segments,
            date_range=(earliest, latest),
        )


importer = ICSImporter()
