from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, render_template, request

from think import messages

bp = Blueprint("inbox", __name__, template_folder="../templates")


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

    try:
        message_list = messages.list_messages(status)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    # Calculate unread count for active messages
    unread_count = (
        sum(1 for m in message_list if m.get("status") == "unread")
        if status == "active"
        else 0
    )

    return jsonify(
        {
            "messages": message_list,
            "unread_count": unread_count,
            "total": len(message_list),
        }
    )


@bp.route("/inbox/api/message/<message_id>", methods=["GET"])
def get_message(message_id: str) -> Any:
    """Get a specific message by ID."""
    try:
        message = messages.get_message(message_id)
        if message is None:
            return jsonify({"error": "Message not found"}), 404
        return jsonify(message)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/inbox/api/message/<message_id>/read", methods=["POST"])
def mark_read(message_id: str) -> Any:
    """Mark a message as read."""
    try:
        if messages.mark_read(message_id):
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Message not found"}), 404
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/inbox/api/message/<message_id>/archive", methods=["POST"])
def archive_message(message_id: str) -> Any:
    """Archive a message by moving it from active to archived folder."""
    try:
        if messages.archive_message(message_id):
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Message not found"}), 404
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/inbox/api/message/<message_id>/unarchive", methods=["POST"])
def unarchive_message(message_id: str) -> Any:
    """Unarchive a message by moving it from archived to active folder."""
    try:
        if messages.unarchive_message(message_id):
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Message not found"}), 404
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/inbox/api/stats")
def get_stats() -> Any:
    """Get inbox statistics."""
    try:
        active_messages = messages.list_messages("active")
        archived_messages = messages.list_messages("archived")
        unread_count = messages.get_unread_count()

        return jsonify(
            {
                "active_count": len(active_messages),
                "archived_count": len(archived_messages),
                "unread_count": unread_count,
                "total_count": len(active_messages) + len(archived_messages),
            }
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
