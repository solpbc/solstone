from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Blueprint, jsonify, render_template, request

bp = Blueprint("inbox", __name__, template_folder="../templates")


def _inbox_dir() -> Path | None:
    """Get the inbox directory path."""
    load_dotenv()
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        return None
    return Path(journal) / "inbox"


def _read_messages(status: str = "active") -> list[dict[str, Any]]:
    """Read all messages from active or archived folder."""
    inbox_dir = _inbox_dir()
    if not inbox_dir:
        return []

    folder = inbox_dir / status
    if not folder.exists():
        return []

    messages = []
    for msg_file in sorted(folder.glob("msg_*.json"), reverse=True):
        try:
            with open(msg_file, "r", encoding="utf-8") as f:
                message = json.load(f)
                messages.append(message)
        except Exception:
            continue

    return messages


def _log_activity(action: str, message_id: str, **kwargs: Any) -> None:
    """Log an activity to the inbox activity log."""
    inbox_dir = _inbox_dir()
    if not inbox_dir:
        return

    inbox_dir.mkdir(exist_ok=True)
    log_file = inbox_dir / "activity_log.jsonl"

    import time

    entry = {
        "timestamp": int(time.time() * 1000),
        "action": action,
        "message_id": message_id,
        **kwargs,
    }

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


@bp.route("/inbox")
def inbox_page() -> str:
    """Render the inbox view."""
    return render_template("inbox.html", active="inbox")


@bp.route("/inbox/api/messages")
def get_messages() -> Any:
    """Get messages from active or archived folder."""
    status = request.args.get("status", "active")
    if status not in ["active", "archived"]:
        return jsonify({"error": "Invalid status"}), 400

    messages = _read_messages(status)

    # Calculate unread count for active messages
    unread_count = (
        sum(1 for m in messages if m.get("status") == "unread")
        if status == "active"
        else 0
    )

    return jsonify(
        {"messages": messages, "unread_count": unread_count, "total": len(messages)}
    )


@bp.route("/inbox/api/message/<message_id>", methods=["GET"])
def get_message(message_id: str) -> Any:
    """Get a specific message by ID."""
    inbox_dir = _inbox_dir()
    if not inbox_dir:
        return jsonify({"error": "Inbox not configured"}), 500

    # Check both active and archived folders
    for status in ["active", "archived"]:
        msg_path = inbox_dir / status / f"{message_id}.json"
        if msg_path.exists():
            try:
                with open(msg_path, "r", encoding="utf-8") as f:
                    message = json.load(f)
                return jsonify(message)
            except Exception as e:
                return jsonify({"error": f"Failed to read message: {str(e)}"}), 500

    return jsonify({"error": "Message not found"}), 404


@bp.route("/inbox/api/message/<message_id>/read", methods=["POST"])
def mark_read(message_id: str) -> Any:
    """Mark a message as read."""
    inbox_dir = _inbox_dir()
    if not inbox_dir:
        return jsonify({"error": "Inbox not configured"}), 500

    msg_path = inbox_dir / "active" / f"{message_id}.json"
    if not msg_path.exists():
        return jsonify({"error": "Message not found"}), 404

    try:
        with open(msg_path, "r", encoding="utf-8") as f:
            message = json.load(f)

        if message.get("status") != "read":
            message["status"] = "read"
            with open(msg_path, "w", encoding="utf-8") as f:
                json.dump(message, f, indent=2)

            _log_activity("read", message_id)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": f"Failed to mark as read: {str(e)}"}), 500


@bp.route("/inbox/api/message/<message_id>/archive", methods=["POST"])
def archive_message(message_id: str) -> Any:
    """Archive a message by moving it from active to archived folder."""
    inbox_dir = _inbox_dir()
    if not inbox_dir:
        return jsonify({"error": "Inbox not configured"}), 500

    active_path = inbox_dir / "active" / f"{message_id}.json"
    if not active_path.exists():
        return jsonify({"error": "Message not found"}), 404

    archived_dir = inbox_dir / "archived"
    archived_dir.mkdir(parents=True, exist_ok=True)
    archived_path = archived_dir / f"{message_id}.json"

    try:
        # Read the message and update status
        with open(active_path, "r", encoding="utf-8") as f:
            message = json.load(f)

        message["status"] = "archived"

        # Write to archived location
        with open(archived_path, "w", encoding="utf-8") as f:
            json.dump(message, f, indent=2)

        # Remove from active
        active_path.unlink()

        _log_activity("archived", message_id)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": f"Failed to archive: {str(e)}"}), 500


@bp.route("/inbox/api/message/<message_id>/unarchive", methods=["POST"])
def unarchive_message(message_id: str) -> Any:
    """Unarchive a message by moving it from archived to active folder."""
    inbox_dir = _inbox_dir()
    if not inbox_dir:
        return jsonify({"error": "Inbox not configured"}), 500

    archived_path = inbox_dir / "archived" / f"{message_id}.json"
    if not archived_path.exists():
        return jsonify({"error": "Message not found"}), 404

    active_dir = inbox_dir / "active"
    active_dir.mkdir(parents=True, exist_ok=True)
    active_path = active_dir / f"{message_id}.json"

    try:
        # Read the message and update status
        with open(archived_path, "r", encoding="utf-8") as f:
            message = json.load(f)

        message["status"] = "read"  # Set to read when unarchiving

        # Write to active location
        with open(active_path, "w", encoding="utf-8") as f:
            json.dump(message, f, indent=2)

        # Remove from archived
        archived_path.unlink()

        _log_activity("unarchived", message_id)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": f"Failed to unarchive: {str(e)}"}), 500


@bp.route("/inbox/api/stats")
def get_stats() -> Any:
    """Get inbox statistics."""
    active_messages = _read_messages("active")
    archived_messages = _read_messages("archived")

    unread_count = sum(1 for m in active_messages if m.get("status") == "unread")

    return jsonify(
        {
            "active_count": len(active_messages),
            "archived_count": len(archived_messages),
            "unread_count": unread_count,
            "total_count": len(active_messages) + len(archived_messages),
        }
    )
