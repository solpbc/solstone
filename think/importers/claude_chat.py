# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Claude chat history importer — imports conversations from Claude export archives."""

import datetime as dt
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Callable

from think.importers.file_importer import ImportPreview, ImportResult
from think.importers.shared import (
    _window_messages,
    map_items_to_segments,
    write_content_manifest,
    write_segment,
)
from think.utils import day_path

logger = logging.getLogger(__name__)


def _open_conversations(path: Path) -> list[dict[str, Any]]:
    """Extract and parse conversations.json from a .dms or .zip file."""
    if path.suffix.lower() == ".dms" or path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            if "conversations.json" not in zf.namelist():
                raise ValueError(f"No conversations.json found in {path.name}")
            with zf.open("conversations.json") as f:
                return json.loads(f.read())
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _extract_messages(
    conversations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Extract timestamped messages from Claude conversations."""
    messages: list[dict[str, Any]] = []
    skipped = 0

    for conv_idx, conv in enumerate(conversations):
        chat_messages = conv.get("chat_messages", [])
        if not chat_messages:
            skipped += 1
            continue

        conv_created = conv.get("created_at", "")
        conv_ts: float | None = None
        if conv_created:
            try:
                conv_ts = dt.datetime.fromisoformat(conv_created).timestamp()
            except (ValueError, TypeError):
                pass

        conv_has_content = False
        for msg in chat_messages:
            sender = msg.get("sender", "")
            text = msg.get("text", "")
            if not text:
                continue

            created_at = msg.get("created_at", "")
            create_time: float | None = None
            if created_at:
                try:
                    create_time = dt.datetime.fromisoformat(created_at).timestamp()
                except (ValueError, TypeError):
                    pass
            if create_time is None:
                create_time = conv_ts
            if create_time is None:
                continue

            messages.append(
                {
                    "create_time": create_time,
                    "speaker": "Human" if sender == "human" else "Assistant",
                    "text": text,
                    "model_slug": None,
                    "conv_id": conv_idx,
                    "conv_title": conv.get("name", ""),
                }
            )
            conv_has_content = True

        if not conv_has_content:
            skipped += 1

    return messages, skipped


class ClaudeChatImporter:
    name = "claude"
    display_name = "Claude Chat History"
    file_patterns = ["*.zip", "*.dms"]
    description = "Import conversations from Claude chat export"

    def detect(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if path.suffix.lower() not in {".zip", ".dms"}:
            return False
        try:
            with zipfile.ZipFile(path, "r") as zf:
                if "conversations.json" not in zf.namelist():
                    return False
                # Peek at structure to confirm it's Claude format (flat array)
                with zf.open("conversations.json") as f:
                    data = json.loads(f.read())
                if not isinstance(data, list) or len(data) == 0:
                    return False
                first = data[0]
                # Claude exports have chat_messages, ChatGPT has mapping
                return "chat_messages" in first and "mapping" not in first
        except (zipfile.BadZipFile, json.JSONDecodeError, KeyError):
            return False

    def preview(self, path: Path) -> ImportPreview:
        conversations = _open_conversations(path)
        if not conversations:
            return ImportPreview(
                date_range=("", ""),
                item_count=0,
                entity_count=0,
                summary="Empty export — no conversations found",
            )

        dates: list[str] = []
        message_count = 0
        for conv in conversations:
            for msg in conv.get("chat_messages", []):
                text = msg.get("text", "")
                if not text:
                    continue
                message_count += 1
                created_at = msg.get("created_at", "")
                if not created_at:
                    created_at = conv.get("created_at", "")
                if created_at:
                    try:
                        day = dt.datetime.fromisoformat(created_at).strftime("%Y%m%d")
                        dates.append(day)
                    except ValueError:
                        pass

        dates.sort()
        date_range = (dates[0], dates[-1]) if dates else ("", "")

        return ImportPreview(
            date_range=date_range,
            item_count=message_count,
            entity_count=0,
            summary=f"{message_count} messages from Claude chat export",
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
        conversations = _open_conversations(path)
        import_id = import_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        messages, skipped = _extract_messages(conversations)
        if not messages:
            return ImportResult(
                entries_written=0,
                entities_seeded=0,
                files_created=[],
                errors=[],
                summary="No messages found in Claude export",
            )

        messages.sort(key=lambda m: m["create_time"])
        conv_meta: dict[int, dict[str, Any]] = {}
        for msg in messages:
            conv_id = msg.get("conv_id")
            if conv_id is None:
                continue
            meta = conv_meta.setdefault(
                conv_id,
                {
                    "title": msg.get("conv_title", ""),
                    "first_ts": msg["create_time"],
                    "preview": "",
                    "message_count": 0,
                },
            )
            meta["message_count"] += 1
            if not meta["preview"] and msg["speaker"] == "Human":
                meta["preview"] = msg["text"][:200]
        earliest = dt.datetime.fromtimestamp(
            messages[0]["create_time"], tz=dt.timezone.utc
        ).strftime("%Y%m%d")
        latest = dt.datetime.fromtimestamp(
            messages[-1]["create_time"], tz=dt.timezone.utc
        ).strftime("%Y%m%d")

        if progress_callback:
            progress_callback(
                len(conversations),
                len(conversations),
                earliest_date=earliest,
                latest_date=latest,
                entities_found=0,
            )

        windows = _window_messages(messages)
        created_files: list[str] = []
        segments: list[tuple[str, str]] = []
        errors: list[str] = []
        written_count = 0

        for day, seg_key, _model, entries in windows:
            day_dir = str(day_path(day))
            try:
                json_path = write_segment(
                    day_dir,
                    "import.claude",
                    seg_key,
                    entries,
                    import_id=import_id,
                    facet=facet,
                    model=None,
                )
                created_files.append(json_path)
                segments.append((day, seg_key))
                written_count += len(entries)
            except Exception as exc:
                errors.append(f"Failed to write segment {day}/{seg_key}: {exc}")
                logger.warning("Failed to write segment %s/%s: %s", day, seg_key, exc)

        if skipped:
            logger.info("Skipped %d conversations with no content", skipped)

        conv_segments: dict[int, set[tuple[str, str]]] = {}
        msg_segments = map_items_to_segments(
            [msg["create_time"] for msg in messages],
            tz=None,
        )
        for msg, segment in zip(messages, msg_segments, strict=False):
            conv_id = msg.get("conv_id")
            if conv_id is None:
                continue
            conv_segments.setdefault(conv_id, set()).add(segment)

        manifest_entries: list[dict[str, Any]] = []
        for conv_id, meta in sorted(
            conv_meta.items(), key=lambda item: item[1]["first_ts"]
        ):
            first_dt = dt.datetime.fromtimestamp(meta["first_ts"], tz=dt.timezone.utc)
            manifest_entries.append(
                {
                    "id": f"conv-{conv_id}",
                    "title": meta["title"] or f"Conversation {conv_id + 1}",
                    "date": first_dt.strftime("%Y%m%d"),
                    "type": "conversation",
                    "preview": meta["preview"],
                    "meta": {"message_count": meta["message_count"]},
                    "segments": [
                        {"day": day, "key": key}
                        for day, key in sorted(conv_segments.get(conv_id, set()))
                    ],
                }
            )
        write_content_manifest(import_id, manifest_entries)

        days = sorted({day for day, _ in segments})

        return ImportResult(
            entries_written=written_count,
            entities_seeded=0,
            files_created=created_files,
            errors=errors,
            summary=(
                f"Imported {len(messages)} messages across {len(days)} days into "
                f"{len(segments)} segments"
            ),
            segments=segments,
            date_range=(earliest, latest),
        )


importer = ClaudeChatImporter()
