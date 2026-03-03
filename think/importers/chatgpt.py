# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""ChatGPT history importer — imports conversations from ChatGPT export archives."""

import datetime as dt
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Callable

from think.importers.file_importer import ImportPreview, ImportResult
from think.importers.shared import write_structured_import

logger = logging.getLogger(__name__)


def _open_conversations(path: Path) -> list[dict[str, Any]]:
    """Extract and parse conversations.json from a ChatGPT export ZIP."""
    with zipfile.ZipFile(path, "r") as zf:
        if "conversations.json" not in zf.namelist():
            raise ValueError(f"No conversations.json found in {path.name}")
        with zf.open("conversations.json") as f:
            return json.loads(f.read())


def _walk_message_tree(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk the ChatGPT message tree from current_node back to root, return messages in order."""
    mapping = conversation.get("mapping", {})
    if not mapping:
        return []

    current_node = conversation.get("current_node")
    if not current_node or current_node not in mapping:
        return []

    # Walk parent chain from current_node to root
    chain: list[dict[str, Any]] = []
    node_id = current_node
    while node_id and node_id in mapping:
        node = mapping[node_id]
        msg = node.get("message")
        if msg is not None:
            chain.append(msg)
        node_id = node.get("parent")

    # Reverse to get chronological order
    chain.reverse()
    return chain


def _format_messages(messages: list[dict[str, Any]]) -> tuple[str, int, str | None]:
    """Format ChatGPT messages into readable text.

    Returns (formatted_content, message_count, model_slug).
    """
    lines: list[str] = []
    count = 0
    model_slug: str | None = None

    for msg in messages:
        author = msg.get("author", {})
        role = author.get("role", "")

        # Skip system and tool messages
        if role in {"system", "tool"}:
            continue

        content = msg.get("content", {})
        parts = content.get("parts", [])
        # Filter out non-string parts (e.g., image refs)
        text_parts = [p for p in parts if isinstance(p, str)]
        text = "\n".join(text_parts).strip()
        if not text:
            continue

        # Extract model info from assistant messages
        if role == "assistant" and model_slug is None:
            meta = msg.get("metadata", {})
            model_slug = meta.get("model_slug")

        label = "Human" if role == "user" else "Assistant"
        lines.append(f"{label}: {text}")
        count += 1

    return "\n\n".join(lines), count, model_slug


class ChatGPTImporter:
    name = "chatgpt"
    display_name = "ChatGPT History"
    file_patterns = ["*.zip"]
    description = "Import conversations from ChatGPT export"

    def detect(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if path.suffix.lower() != ".zip":
            return False
        try:
            with zipfile.ZipFile(path, "r") as zf:
                if "conversations.json" not in zf.namelist():
                    return False
                with zf.open("conversations.json") as f:
                    data = json.loads(f.read())
                if not isinstance(data, list) or len(data) == 0:
                    return False
                # ChatGPT format has mapping dict; Claude has chat_messages
                return "mapping" in data[0]
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
        model_counts: dict[str, int] = {}

        for conv in conversations:
            mapping = conv.get("mapping", {})
            if not mapping:
                continue
            valid_count += 1

            create_time = conv.get("create_time")
            if create_time is not None:
                try:
                    day = dt.datetime.fromtimestamp(create_time).strftime("%Y%m%d")
                    dates.append(day)
                except (ValueError, OSError):
                    pass

            # Scan for model info
            for node in mapping.values():
                msg = node.get("message")
                if msg and msg.get("author", {}).get("role") == "assistant":
                    meta = msg.get("metadata", {})
                    slug = meta.get("model_slug")
                    if slug:
                        model_counts[slug] = model_counts.get(slug, 0) + 1
                        break

        dates.sort()
        date_range = (dates[0], dates[-1]) if dates else ("", "")

        model_info = ""
        if model_counts:
            top_models = sorted(model_counts.items(), key=lambda x: -x[1])[:3]
            model_info = "; models: " + ", ".join(
                f"{m} ({n})" for m, n in top_models
            )

        return ImportPreview(
            date_range=date_range,
            item_count=valid_count,
            entity_count=0,
            summary=f"{valid_count} conversations from ChatGPT export{model_info}",
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
        model_counts: dict[str, int] = {}

        for i, conv in enumerate(conversations):
            mapping = conv.get("mapping", {})
            if not mapping:
                skipped += 1
                continue

            title = conv.get("title", "Untitled")
            create_time = conv.get("create_time")

            # Parse timestamp (Unix epoch float)
            if create_time is None:
                errors.append(f"Missing create_time for conversation: {title!r}")
                continue
            try:
                ts = dt.datetime.fromtimestamp(create_time).isoformat()
            except (ValueError, OSError):
                errors.append(f"Bad timestamp for conversation: {title!r}")
                continue

            # Walk message tree
            messages = _walk_message_tree(conv)
            content, message_count, model_slug = _format_messages(messages)
            if not content:
                skipped += 1
                continue

            if model_slug:
                model_counts[model_slug] = model_counts.get(model_slug, 0) + 1

            entry: dict[str, Any] = {
                "type": "ai_chat",
                "ts": ts,
                "title": title,
                "source": "chatgpt",
                "message_count": message_count,
                "content": content,
            }
            if model_slug:
                entry["model"] = model_slug

            entries.append(entry)

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, len(conversations))

        # Write to journal
        created_files = write_structured_import(
            "chatgpt",
            entries,
            import_id=import_id,
            facet=facet,
        )

        if skipped:
            logger.info("Skipped %d conversations with no content", skipped)

        # Build summary with model distribution
        model_info = ""
        if model_counts:
            top_models = sorted(model_counts.items(), key=lambda x: -x[1])[:5]
            model_info = " — models: " + ", ".join(
                f"{m} ({n})" for m, n in top_models
            )

        return ImportResult(
            entries_written=len(entries),
            entities_seeded=0,
            files_created=created_files,
            errors=errors,
            summary=f"Imported {len(entries)} ChatGPT conversations across {len(created_files)} days{model_info}",
        )


importer = ChatGPTImporter()
