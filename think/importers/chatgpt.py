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
from think.importers.shared import _window_messages, write_segment
from think.utils import day_path

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


def _extract_messages(
    conversations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Extract timestamped user and assistant messages from conversations."""
    messages: list[dict[str, Any]] = []
    model_counts: dict[str, int] = {}
    skipped = 0

    for conv in conversations:
        mapping = conv.get("mapping", {})
        if not mapping:
            skipped += 1
            continue

        conv_messages = _walk_message_tree(conv)
        conv_has_content = False

        for msg in conv_messages:
            author = msg.get("author", {})
            role = author.get("role", "")
            if role not in ("user", "assistant"):
                continue

            content = msg.get("content", {})
            parts = content.get("parts", [])
            text_parts = [p for p in parts if isinstance(p, str)]
            text = "\n".join(text_parts).strip()
            if not text:
                continue

            create_time = msg.get("create_time")
            if create_time is None or not isinstance(create_time, (int, float)):
                continue

            model_slug = None
            if role == "assistant":
                meta = msg.get("metadata", {})
                model_slug = meta.get("model_slug")
                if model_slug:
                    model_counts[model_slug] = model_counts.get(model_slug, 0) + 1

            messages.append(
                {
                    "create_time": float(create_time),
                    "speaker": "Human" if role == "user" else "Assistant",
                    "text": text,
                    "model_slug": model_slug,
                }
            )
            conv_has_content = True

        if not conv_has_content:
            skipped += 1

    return messages, model_counts, skipped


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
            model_info = "; models: " + ", ".join(f"{m} ({n})" for m, n in top_models)

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
        messages, model_counts, skipped = _extract_messages(conversations)
        if not messages:
            return ImportResult(
                entries_written=0,
                entities_seeded=0,
                files_created=[],
                errors=[],
                summary="No messages found in ChatGPT export",
            )

        messages.sort(key=lambda m: m["create_time"])
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

        for day, seg_key, model_slug, entries in windows:
            day_dir = str(day_path(day))
            try:
                json_path = write_segment(
                    day_dir,
                    "import.chatgpt",
                    seg_key,
                    entries,
                    import_id=import_id,
                    facet=facet,
                    model=model_slug,
                )
                created_files.append(json_path)
                segments.append((day, seg_key))
                written_count += len(entries)
            except Exception as exc:
                errors.append(f"Failed to write segment {day}/{seg_key}: {exc}")
                logger.warning("Failed to write segment %s/%s: %s", day, seg_key, exc)

        if skipped:
            logger.info("Skipped %d conversations with no content", skipped)

        days = sorted({day for day, _ in segments})
        model_info = ""
        if model_counts:
            top_models = sorted(model_counts.items(), key=lambda x: -x[1])[:5]
            model_info = " — models: " + ", ".join(f"{m} ({n})" for m, n in top_models)

        return ImportResult(
            entries_written=written_count,
            entities_seeded=0,
            files_created=created_files,
            errors=errors,
            summary=(
                f"Imported {len(messages)} messages across {len(days)} days into "
                f"{len(segments)} segments{model_info}"
            ),
            segments=segments,
            date_range=(earliest, latest),
        )


importer = ChatGPTImporter()
