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
from think.importers.shared import write_structured_import
from think.utils import get_journal

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


def _format_messages(messages: list[dict[str, Any]]) -> str:
    """Format chat messages into readable text with role prefixes."""
    lines: list[str] = []
    for msg in messages:
        sender = msg.get("sender", "unknown")
        text = msg.get("text", "")
        if not text:
            continue
        role = "Human" if sender == "human" else "Assistant"
        lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


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
        valid_count = 0
        for conv in conversations:
            messages = conv.get("chat_messages", [])
            if not messages:
                continue
            valid_count += 1
            created = conv.get("created_at", "")
            if created:
                try:
                    day = dt.datetime.fromisoformat(created).strftime("%Y%m%d")
                    dates.append(day)
                except ValueError:
                    pass

        dates.sort()
        date_range = (dates[0], dates[-1]) if dates else ("", "")

        return ImportPreview(
            date_range=date_range,
            item_count=valid_count,
            entity_count=0,
            summary=f"{valid_count} conversations from Claude chat export",
        )

    def process(
        self,
        path: Path,
        journal_root: Path,
        *,
        facet: str | None = None,
        progress_callback: Callable | None = None,
    ) -> ImportResult:
        conversations = _open_conversations(path)
        import_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        entries: list[dict[str, Any]] = []
        errors: list[str] = []
        skipped = 0

        for i, conv in enumerate(conversations):
            messages = conv.get("chat_messages", [])
            if not messages:
                skipped += 1
                continue

            title = conv.get("name", "Untitled")
            created = conv.get("created_at", "")

            # Parse timestamp
            try:
                ts = dt.datetime.fromisoformat(created).isoformat()
            except (ValueError, TypeError):
                errors.append(f"Bad timestamp for conversation: {title!r}")
                continue

            content = _format_messages(messages)
            if not content:
                skipped += 1
                continue

            entries.append({
                "type": "ai_chat",
                "ts": ts,
                "title": title,
                "source": "claude",
                "message_count": len(messages),
                "content": content,
            })

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, len(conversations))

        # Write to journal
        created_files = write_structured_import(
            "claude",
            entries,
            import_id=import_id,
            facet=facet,
        )

        if skipped:
            logger.info("Skipped %d conversations with no messages", skipped)

        return ImportResult(
            entries_written=len(entries),
            entities_seeded=0,
            files_created=created_files,
            errors=errors,
            summary=f"Imported {len(entries)} Claude conversations across {len(created_files)} days",
        )


importer = ClaudeChatImporter()
