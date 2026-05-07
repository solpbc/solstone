# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from flask import Blueprint, abort, g, jsonify, render_template, request
from werkzeug.utils import secure_filename

from solstone.apps.utils import log_app_action
from solstone.convey import emit, state
from solstone.think.detect_created import detect_created
from solstone.think.importers.utils import (
    build_import_info,
    generate_content_manifest,
    get_import_details,
    list_import_timestamps,
    read_import_metadata,
    save_import_file,
    save_import_text,
    update_import_metadata_fields,
    write_import_metadata,
)
from solstone.think.media import MEDIA_EXTENSIONS
from solstone.think.utils import day_path, now_ms

from .journal_sources import (
    STATE_AREAS,
    create_state_directory,
    find_journal_source_by_name,
    generate_key,
    get_state_directory,
    is_valid_journal_source_name,
    list_journal_sources,
    require_journal_source,
    save_journal_source,
)

import_bp = Blueprint(
    "app:import",
    __name__,
    url_prefix="/app/import",
)

SOURCE_METADATA = [
    {
        "name": "ics",
        "display_name": "Calendar",
        "emoji": "📅",
        "icon": "calendar",
        "description": "Import events from Google Calendar, Apple Calendar, or Outlook",
        "input_type": "file",
        "upload_prompt": "Upload your .ics file or .zip export",
        "has_guide": True,
        "accept": ".ics,.zip",
    },
    {
        "name": "chatgpt",
        "display_name": "ChatGPT",
        "emoji": "💬",
        "icon": "message-square",
        "description": "Import your conversation history from ChatGPT",
        "input_type": "file",
        "upload_prompt": "Upload your ChatGPT export .zip file",
        "has_guide": True,
        "accept": ".zip",
    },
    {
        "name": "claude",
        "display_name": "Claude",
        "emoji": "🤖",
        "icon": "message-circle",
        "description": "Import your conversation history from Claude",
        "input_type": "file",
        "upload_prompt": "Upload your Claude export .zip file",
        "has_guide": True,
        "accept": ".zip",
    },
    {
        "name": "gemini",
        "display_name": "Gemini",
        "emoji": "✨",
        "icon": "sparkles",
        "description": "Import your activity history from Google Gemini",
        "input_type": "file",
        "upload_prompt": "Upload your Google Takeout .zip file",
        "has_guide": True,
        "accept": ".zip,.json",
    },
    {
        "name": "obsidian",
        "display_name": "Notes",
        "emoji": "📝",
        "icon": "file-text",
        "description": "Import notes from Obsidian, Logseq, or any markdown vault",
        "input_type": "path_input",
        "upload_prompt": "Paste the full path to your vault folder",
        "has_guide": True,
        "accept": "",
    },
    {
        "name": "kindle",
        "display_name": "Kindle",
        "emoji": "📚",
        "icon": "book-open",
        "description": "Import highlights and clippings from your Kindle",
        "input_type": "file",
        "upload_prompt": "Upload your My Clippings.txt file",
        "has_guide": True,
        "accept": ".txt",
    },
    {
        "name": "journal_archive",
        "display_name": "Journal",
        "emoji": "📓",
        "icon": "book",
        "description": "Import a full journal export from another solstone journal",
        "input_type": "file",
        "upload_prompt": "Upload your journal export .zip file",
        "has_guide": True,
        "accept": ".zip",
    },
    {
        "name": "granola",
        "display_name": "Granola",
        "emoji": "🌾",
        "icon": "mic",
        "description": "Import meeting transcripts from Granola via muesli",
        "input_type": "path_input",
        "upload_prompt": "Path to muesli transcripts folder",
        "has_guide": True,
        "accept": "",
    },
    {
        "name": "recording",
        "display_name": "meeting audio",
        "emoji": "🎙️",
        "icon": "mic",
        "description": "import audio observations of meetings or conversations",
        "input_type": "file",
        "upload_prompt": "upload an audio file (.m4a, .mp3, .wav)",
        "has_guide": False,
        "accept": ",".join(sorted(MEDIA_EXTENSIONS)),
    },
    {
        "name": "document",
        "display_name": "Document",
        "emoji": "📄",
        "icon": "file",
        "description": "Import a PDF document",
        "input_type": "file",
        "upload_prompt": "Upload a PDF file",
        "has_guide": False,
        "accept": ".pdf",
    },
    {
        "name": "quick",
        "display_name": "Quick Import",
        "emoji": "⚡",
        "icon": "zap",
        "description": "Paste text or drop any file for quick import",
        "input_type": "text",
        "upload_prompt": "Paste text or drag and drop a file",
        "has_guide": False,
        "accept": "",
    },
]


@import_bp.route("/api/save", methods=["POST"])
def import_save() -> Any:
    from datetime import datetime

    upload = request.files.get("file")
    text = request.form.get("text", "").strip()
    facet = request.form.get("facet", "").strip() or None
    setting = request.form.get("setting", "").strip() or None

    # Generate timestamp for folder name
    timestamp_ms = now_ms()

    # Determine filename
    if upload and upload.filename:
        filename = secure_filename(upload.filename)
    elif text:
        filename = "paste.txt"
    else:
        return jsonify({"error": "No input"}), 400

    # Detect timestamp from content first (need temporary save for detection)
    ts = None
    detection_result = None

    # Create temporary file for detection if needed
    if upload:
        import tempfile

        # Preserve original filename structure in temp file name for timestamp detection
        # Use prefix to include original filename (minus extension)
        original_stem = Path(filename).stem
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(
            delete=False, prefix=f"{original_stem}_", suffix=suffix
        ) as tmp:
            upload.save(tmp.name)
            temp_path = tmp.name
            upload.seek(0)  # Reset file pointer for later save
    else:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp.write(text)
            temp_path = tmp.name

    try:
        # Pass original filename for better timestamp detection
        original_name = upload.filename if upload else None
        detection_result = detect_created(temp_path, original_filename=original_name)
        if (
            detection_result
            and detection_result.get("day")
            and detection_result.get("time")
        ):
            ts = f"{detection_result['day']}_{detection_result['time']}"
    except Exception:
        ts = None
    finally:
        # Clean up temporary file
        Path(temp_path).unlink(missing_ok=True)

    # Use detected timestamp or fall back to upload timestamp
    folder_timestamp = (
        ts
        if ts
        else f"{datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y%m%d_%H%M%S')}"
    )

    # Save the actual file using utility function
    if upload:
        # Save uploaded file to temp location first, then move to import dir
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            upload.save(tmp.name)
            temp_source = Path(tmp.name)

        try:
            file_path = save_import_file(
                journal_root=Path(state.journal_root),
                timestamp=folder_timestamp,
                source_path=temp_source,
                filename=filename,
            )
        finally:
            temp_source.unlink(missing_ok=True)
    else:
        file_path = save_import_text(
            journal_root=Path(state.journal_root),
            timestamp=folder_timestamp,
            content=text,
            filename=filename,
        )

    # Build metadata dict
    metadata = {
        "original_filename": upload.filename if upload else "paste.txt",
        "upload_timestamp": timestamp_ms,
        "upload_datetime": datetime.fromtimestamp(timestamp_ms / 1000).isoformat(),
        "detection_result": detection_result,
        "detected_timestamp": ts,
        "user_timestamp": folder_timestamp,  # The timestamp used for the folder
        "file_size": file_path.stat().st_size if file_path.exists() else 0,
        "mime_type": upload.content_type if upload else "text/plain",
        "facet": facet,  # Include selected facet
        "setting": setting,
        "file_path": str(file_path),  # Store the actual file path
    }

    # Write metadata using utility function
    write_import_metadata(
        journal_root=Path(state.journal_root),
        timestamp=folder_timestamp,
        metadata=metadata,
    )

    # Check for dedup — has this exact file been imported before?
    dedup = None
    try:
        from solstone.think.importers.shared import find_manifest_by_hash, hash_source

        source_hash = hash_source(file_path)
        existing = find_manifest_by_hash(Path(state.journal_root), source_hash)
        if existing:
            dedup = {
                "imported_at": existing.get("imported_at", "unknown"),
                "entry_count": existing.get("entry_count", 0),
                "import_id": existing.get("import_id", ""),
            }
    except OSError as exc:
        logging.warning("Dedup check failed for %s: %s", file_path, exc)

    result: dict[str, Any] = {
        "path": str(file_path),
        "timestamp": folder_timestamp,
        "facet": facet,
        "setting": setting,
    }
    if dedup:
        result["dedup"] = dedup

    return jsonify(result)


@import_bp.route("/api/save-path", methods=["POST"])
def import_save_path() -> Any:
    """Register a local filesystem path for import (e.g. Obsidian vault)."""
    from datetime import datetime

    data = request.get_json(force=True)
    local_path = data.get("path", "").strip()
    facet = data.get("facet", "").strip() or None
    setting = data.get("setting", "").strip() or None

    if not local_path:
        return jsonify({"error": "Missing path"}), 400

    local = Path(local_path)
    if not local.exists():
        return jsonify({"error": f"Path not found: {local_path}"}), 404

    timestamp_ms = now_ms()
    folder_timestamp = (
        f"{datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y%m%d_%H%M%S')}"
    )

    # Create import directory and metadata
    journal_root = Path(state.journal_root)
    import_dir = journal_root / "imports" / folder_timestamp
    import_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "original_filename": local.name,
        "upload_timestamp": timestamp_ms,
        "upload_datetime": datetime.fromtimestamp(timestamp_ms / 1000).isoformat(),
        "user_timestamp": folder_timestamp,
        "file_path": local_path,
        "facet": facet,
        "setting": setting,
        "is_local_path": True,
    }

    write_import_metadata(
        journal_root=journal_root,
        timestamp=folder_timestamp,
        metadata=metadata,
    )

    return jsonify(
        {
            "path": local_path,
            "timestamp": folder_timestamp,
            "facet": facet,
            "setting": setting,
        }
    )


@import_bp.route("/api/facet", methods=["POST"])
def import_update_metadata() -> Any:
    """Update stored metadata (facet/setting) for a saved import."""
    data = request.get_json(force=True)
    raw_path = data.get("path", "").strip()
    if not raw_path:
        return jsonify({"error": "Missing import path"}), 400

    facet = data.get("facet", "").strip() or None
    setting = data.get("setting", "").strip() or None

    # Extract timestamp from path
    # Path format: .../imports/{timestamp}/{filename}
    file_path = Path(raw_path)
    timestamp = file_path.parent.name

    try:
        # Use utility function to update metadata
        metadata, updated = update_import_metadata_fields(
            journal_root=Path(state.journal_root),
            timestamp=timestamp,
            updates={"facet": facet, "setting": setting},
        )
    except FileNotFoundError:
        return jsonify({"error": "Import metadata not found"}), 404
    except Exception as exc:
        return jsonify({"error": f"Failed to update metadata: {exc}"}), 500

    return jsonify(
        {
            "status": "ok",
            "facet": facet,
            "setting": setting,
            "updated": updated,
        }
    )


@import_bp.route("/api/list")
def import_list() -> Any:
    """Get list of all imports with their metadata."""
    # Get all import timestamps using utility function
    timestamps = list_import_timestamps(journal_root=Path(state.journal_root))

    # Build info for each import using utility function
    imports = []
    for timestamp in timestamps:
        import_data = build_import_info(
            journal_root=Path(state.journal_root),
            timestamp=timestamp,
        )

        # Calculate status based on processing state
        # Default status
        import_data["status"] = "pending"

        task_id = import_data.get("task_id")
        current_time = time.time()
        import_age_seconds = current_time - import_data.get("imported_at", current_time)
        import_timeout_seconds = 3600  # 1 hour

        # Check for error state first (imported.json exists with error field)
        if import_data.get("error"):
            import_data["status"] = "failed"
        # If we have processing results without error, it's successful
        elif import_data.get("processed"):
            import_data["status"] = "success"
        # If task was started but no results yet, check if it timed out
        elif task_id:
            if import_age_seconds > import_timeout_seconds:
                # Import is stuck/crashed - mark as failed
                import_data["status"] = "failed"
                import_data["error"] = "Import never completed"
                import_data["error_stage"] = "timeout"
            else:
                import_data["status"] = "running"

        imports.append(import_data)

    # Sort by imported_at (newest first)
    imports.sort(key=lambda x: x.get("imported_at", 0), reverse=True)

    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    try:
        per_page = min(100, max(1, int(request.args.get("per_page", 25))))
    except ValueError:
        per_page = 25

    total = len(imports)
    total_entries_written = sum(imp.get("entries_written") or 0 for imp in imports)
    total_entities_seeded = sum(imp.get("entities_seeded") or 0 for imp in imports)

    start = (page - 1) * per_page
    page_imports = imports[start : start + per_page]

    return jsonify(
        {
            "imports": page_imports,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page if total > 0 else 0,
            "total_entries_written": total_entries_written,
            "total_entities_seeded": total_entities_seeded,
        }
    )


@import_bp.route("/api/sources")
def import_sources() -> Any:
    """Return available import source metadata."""
    return jsonify(SOURCE_METADATA)


@import_bp.route("/api/check-path/<source>")
def import_check_path(source: str) -> Any:
    """Check default path for a path_input source and return info."""
    if source == "granola":
        default_path = Path.home() / ".local" / "share" / "muesli" / "transcripts"
        if default_path.is_dir():
            count = sum(1 for f in default_path.glob("*.md"))
            return jsonify(
                {
                    "found": True,
                    "path": str(default_path),
                    "count": count,
                    "message": f"Found {count} Granola transcript{'s' if count != 1 else ''}.",
                }
            )
        # Check if muesli is installed but no transcripts yet
        muesli_dir = default_path.parent
        if muesli_dir.is_dir():
            return jsonify(
                {
                    "found": False,
                    "path": str(default_path),
                    "message": "Muesli is installed but no transcripts found. Run `muesli sync` first.",
                }
            )
        return jsonify(
            {
                "found": False,
                "path": "",
                "message": "No muesli installation found. Follow the guide above to install.",
            }
        )
    return jsonify({"found": False, "path": "", "message": ""}), 404


@import_bp.route("/api/guide/<source>")
def import_guide(source: str) -> Any:
    """Return export guide markdown for a given source."""
    if not re.fullmatch(r"[a-z_]+", source):
        return jsonify({"error": "Invalid source name"}), 400
    guide_path = Path(__file__).parent / "guides" / f"{source}.md"
    if not guide_path.is_file():
        return jsonify({"error": f"No guide available for '{source}'"}), 404
    return (
        guide_path.read_text(encoding="utf-8"),
        200,
        {"Content-Type": "text/markdown; charset=utf-8"},
    )


@import_bp.route("/<timestamp>")
def import_detail(timestamp: str) -> str:
    """Show detailed view of a specific import."""
    import_dir = Path(state.journal_root) / "imports" / timestamp
    if not import_dir.exists():
        return render_template("error.html", error="Import not found"), 404

    return render_template(
        "app.html",
        view="detail",
        timestamp=timestamp,
    )


@import_bp.route("/api/<timestamp>")
def import_detail_api(timestamp: str) -> Any:
    """Get detailed data for a specific import."""
    try:
        # Use utility function to get all details
        result = get_import_details(
            journal_root=Path(state.journal_root),
            timestamp=timestamp,
        )
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Import not found"}), 404


@import_bp.route("/api/<timestamp>/content")
def import_content_list(timestamp: str) -> Any:
    """Get paginated content items for an import."""
    journal_root = Path(state.journal_root)
    import_dir = journal_root / "imports" / timestamp
    if not import_dir.exists():
        return jsonify({"error": "Import not found"}), 404

    manifest_path = import_dir / "content_manifest.jsonl"
    if (
        not manifest_path.exists()
        and generate_content_manifest(journal_root, timestamp) is None
    ):
        return jsonify({"error": "No content available"}), 404

    items: list[dict] = []
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return jsonify({"error": "Failed to read manifest"}), 500

    source_type = ""
    imported_path = import_dir / "imported.json"
    if imported_path.exists():
        try:
            imported = json.loads(imported_path.read_text(encoding="utf-8"))
            source_type = imported.get("source_type", "")
        except (OSError, json.JSONDecodeError):
            pass

    source_meta = next((s for s in SOURCE_METADATA if s["name"] == source_type), None)

    month_counts: dict[str, int] = {}
    for item in items:
        date = item.get("date", "")
        if len(date) >= 6:
            month = date[:6]
            month_counts[month] = month_counts.get(month, 0) + 1

    q = request.args.get("q", "").strip().lower()
    month = request.args.get("month", "").strip()

    filtered = items
    if month:
        filtered = [item for item in filtered if item.get("date", "").startswith(month)]
    if q:
        filtered = [
            item
            for item in filtered
            if q in item.get("title", "").lower()
            or q in item.get("preview", "").lower()
        ]

    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    try:
        per_page = min(100, max(1, int(request.args.get("per_page", 50))))
    except ValueError:
        per_page = 50
    total = len(filtered)
    start = (page - 1) * per_page
    page_items = filtered[start : start + per_page]

    return jsonify(
        {
            "items": page_items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page if total > 0 else 0,
            "months": dict(sorted(month_counts.items())),
            "source_type": source_type,
            "source_display": source_meta["display_name"]
            if source_meta
            else source_type,
            "source_emoji": source_meta["emoji"] if source_meta else "",
        }
    )


@import_bp.route("/api/<timestamp>/content/<item_id>")
def import_content_detail(timestamp: str, item_id: str) -> Any:
    """Get full content for a specific imported item."""
    journal_root = Path(state.journal_root)
    import_dir = journal_root / "imports" / timestamp
    if not import_dir.exists():
        return jsonify({"error": "Import not found"}), 404

    manifest_path = import_dir / "content_manifest.jsonl"
    if not manifest_path.exists():
        generate_content_manifest(journal_root, timestamp)
    if not manifest_path.exists():
        return jsonify({"error": "No content available"}), 404

    item = None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("id") == item_id:
                    item = entry
                    break
    except OSError:
        return jsonify({"error": "Failed to read manifest"}), 500

    if item is None:
        return jsonify({"error": "Item not found"}), 404

    source_type = ""
    imported_path = import_dir / "imported.json"
    if imported_path.exists():
        try:
            imported = json.loads(imported_path.read_text(encoding="utf-8"))
            source_type = imported.get("source_type", "")
        except (OSError, json.JSONDecodeError):
            pass

    content_parts: list[dict] = []
    for seg in item.get("segments", []):
        day = seg.get("day", "")
        key = seg.get("key", "")
        if not day or not key:
            continue
        seg_dir = day_path(day, create=False) / f"import.{source_type}" / key
        if not seg_dir.exists():
            continue

        jsonl_path = seg_dir / "conversation_transcript.jsonl"
        if jsonl_path.exists():
            try:
                lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
            except OSError:
                continue
            for line in lines[1:]:
                try:
                    content_parts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            continue

        for md_file in seg_dir.glob("*_transcript.md"):
            try:
                md_content = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            title = item.get("title", "")
            if title:
                sections = re.split(r"(?m)^## ", md_content)
                for section in sections:
                    stripped = section.strip()
                    if stripped.startswith(title):
                        content_parts.append(
                            {"type": "markdown", "content": "## " + stripped}
                        )
                        break
                else:
                    content_parts.append(
                        {"type": "markdown", "content": md_content.strip()}
                    )
            else:
                content_parts.append(
                    {"type": "markdown", "content": md_content.strip()}
                )

    return jsonify({"item": item, "content": content_parts})


@import_bp.route("/api/start", methods=["POST"])
def import_start() -> Any:
    data = request.get_json(force=True)
    path = data.get("path")
    ts = data.get("timestamp")
    source = data.get("source")
    force = data.get("force", False)
    if not path or not ts:
        return jsonify({"error": "missing params"}), 400

    # Generate task ID
    task_id = str(now_ms())

    # Extract original timestamp from path and handle timestamp changes
    file_path = Path(path)
    journal_root = Path(state.journal_root)
    imports_dir = journal_root / "imports"
    is_local_path = not str(file_path).startswith(str(imports_dir))
    original_timestamp = file_path.parent.name if not is_local_path else ts

    # If timestamp changed, rename the import directory
    if not is_local_path and original_timestamp != ts:
        old_import_dir = journal_root / "imports" / original_timestamp
        new_import_dir = journal_root / "imports" / ts

        # Check if old directory exists
        if not old_import_dir.exists():
            return (
                jsonify(
                    {"error": f"Import directory not found for {original_timestamp}"}
                ),
                404,
            )

        # Check if target directory already exists
        if new_import_dir.exists():
            return jsonify({"error": f"Import already exists for timestamp {ts}"}), 409

        # Rename the directory
        try:
            old_import_dir.rename(new_import_dir)
        except Exception as e:
            return (
                jsonify({"error": f"Failed to rename import directory: {str(e)}"}),
                500,
            )

        # Update path to point to new location
        path = str(new_import_dir / file_path.name)

        # Update file_path in metadata (need to update after reading)
        # We'll handle this after reading the metadata below

    # Read import metadata to get facet and setting
    try:
        metadata = read_import_metadata(journal_root=journal_root, timestamp=ts)
    except FileNotFoundError:
        return jsonify({"error": f"Import metadata not found for {ts}"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to read metadata: {str(e)}"}), 500

    # Update file_path in metadata if timestamp changed
    if not is_local_path and original_timestamp != ts:
        try:
            update_import_metadata_fields(
                journal_root=journal_root,
                timestamp=ts,
                updates={"file_path": path},
            )
        except Exception as e:
            return (
                jsonify({"error": f"Failed to update file path in metadata: {str(e)}"}),
                500,
            )

    facet = metadata.get("facet")
    setting = metadata.get("setting")

    # Build command
    cmd = ["sol", "import", path, ts]
    if facet:
        cmd.extend(["--facet", facet])
    if setting:
        cmd.extend(["--setting", setting])
    if source:
        cmd.extend(["--source", source])
    if force:
        cmd.append("--force")

    # Store task_id in metadata
    try:
        update_import_metadata_fields(
            journal_root=journal_root,
            timestamp=ts,
            updates={"task_id": task_id, "source": source},
        )
    except Exception as e:
        return jsonify({"error": f"Failed to update metadata: {str(e)}"}), 500

    # Emit task request to Callosum (non-blocking, drops if disconnected)
    emit("supervisor", "request", ref=task_id, cmd=cmd)

    return jsonify({"status": "ok", "task_id": task_id})


@import_bp.route("/api/journal-sources/create", methods=["POST"])
def api_journal_source_create() -> Any:
    data = request.get_json(force=True) if request.is_json else {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not is_valid_journal_source_name(name):
        return jsonify({"error": "Invalid journal source name"}), 400
    if find_journal_source_by_name(name):
        return jsonify({"error": f"Journal source '{name}' already exists"}), 409
    key = generate_key()
    source_data = {
        "key": key,
        "name": name,
        "created_at": now_ms(),
        "enabled": True,
        "revoked": False,
        "revoked_at": None,
        "stats": {
            "segments_received": 0,
            "entities_received": 0,
            "facets_received": 0,
            "imports_received": 0,
            "config_received": 0,
        },
    }
    if not save_journal_source(source_data):
        return jsonify({"error": "Failed to save journal source"}), 500
    create_state_directory(Path(state.journal_root), key[:8])
    log_app_action(
        app="import",
        facet=None,
        action="journal_source_create",
        params={"name": name, "key_prefix": key[:8]},
    )
    return jsonify({"key": key, "key_prefix": key[:8], "name": name})


@import_bp.route("/api/journal-sources/list")
def api_journal_source_list() -> Any:
    sources = list_journal_sources()
    result = []
    for s in sources:
        result.append(
            {
                "name": s.get("name", ""),
                "prefix": s.get("key", "")[:8],
                "status": "revoked" if s.get("revoked") else "active",
                "created_at": s.get("created_at"),
            }
        )
    return jsonify(result)


@import_bp.route("/api/journal-sources/<name>/revoke", methods=["POST"])
def api_journal_source_revoke(name: str) -> Any:
    source = find_journal_source_by_name(name)
    if not source:
        return jsonify({"error": f"Journal source '{name}' not found"}), 404
    if source.get("revoked"):
        return jsonify({"error": f"Journal source '{name}' is already revoked"}), 409
    source["revoked"] = True
    source["revoked_at"] = now_ms()
    if not save_journal_source(source):
        return jsonify({"error": "Failed to save journal source"}), 500
    log_app_action(
        app="import",
        facet=None,
        action="journal_source_revoke",
        params={"name": name, "key_prefix": source["key"][:8]},
    )
    return jsonify({"name": name, "prefix": source["key"][:8], "revoked": True})


@import_bp.route("/api/journal-sources/<name>/status")
def api_journal_source_status(name: str) -> Any:
    source = find_journal_source_by_name(name)
    if not source:
        return jsonify({"error": f"Journal source '{name}' not found"}), 404
    key = source.get("key", "")
    return jsonify(
        {
            "name": source.get("name", ""),
            "prefix": key[:8],
            "status": "revoked" if source.get("revoked") else "active",
            "created_at": source.get("created_at"),
            "revoked": source.get("revoked", False),
            "revoked_at": source.get("revoked_at"),
            "stats": source.get("stats", {}),
        }
    )


@import_bp.route("/journal/<key_prefix>/manifest/<area>")
@require_journal_source
def journal_source_manifest(key_prefix: str, area: str) -> Any:
    if g.journal_source["key"][:8] != key_prefix:
        abort(403, description="Key prefix mismatch")
    if area not in STATE_AREAS:
        abort(404, description="Unknown manifest area")
    state_path = get_state_directory(key_prefix) / area / "state.json"
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    return jsonify(data)


# Segment ingest routes (separate module to keep routes.py manageable)
from .ingest import register_ingest_routes

register_ingest_routes(import_bp)
