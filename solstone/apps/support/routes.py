# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Flask routes for the support app.

Provides API endpoints consumed by workspace.html and the background service.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from solstone.convey.utils import error_response

logger = logging.getLogger(__name__)

support_bp = Blueprint(
    "app:support",
    __name__,
    url_prefix="/app/support",
)


def _get_client():
    """Lazy-import portal client."""
    from solstone.apps.support.portal import get_client

    return get_client()


def _enabled() -> bool:
    from solstone.apps.support.portal import is_enabled

    return is_enabled()


# -- Tickets -----------------------------------------------------------------


@support_bp.route("/api/tickets", methods=["GET"])
def list_tickets() -> Any:
    """List user's tickets."""
    if not _enabled():
        return error_response("Support is disabled", 403)

    try:
        status = request.args.get("status")
        client = _get_client()
        tickets = client.list_tickets(status=status)
        return jsonify(tickets)
    except Exception as exc:
        logger.exception("Failed to list tickets")
        return error_response(str(exc))


@support_bp.route("/api/tickets/<int:ticket_id>", methods=["GET"])
def get_ticket(ticket_id: int) -> Any:
    """Get a single ticket with thread."""
    if not _enabled():
        return error_response("Support is disabled", 403)

    try:
        client = _get_client()
        ticket = client.get_ticket(ticket_id)
        return jsonify(ticket)
    except Exception as exc:
        logger.exception("Failed to get ticket %d", ticket_id)
        return error_response(str(exc))


@support_bp.route("/api/tickets", methods=["POST"])
def create_ticket() -> Any:
    """Create a support ticket."""
    if not _enabled():
        return error_response("Support is disabled", 403)

    payload = request.get_json(force=True)
    subject = payload.get("subject")
    description = payload.get("description")

    if not subject or not description:
        return error_response("subject and description are required")

    try:
        from solstone.apps.support.tools import support_create

        result = support_create(
            subject=subject,
            description=description,
            product=payload.get("product", "solstone"),
            severity=payload.get("severity", "medium"),
            category=payload.get("category"),
            user_context=payload.get("user_context"),
            auto_context=payload.get("auto_context", True),
            anonymous=payload.get("anonymous", False),
        )
        return jsonify(result), 201
    except Exception as exc:
        logger.exception("Failed to create ticket")
        return error_response(str(exc))


@support_bp.route("/api/tickets/<int:ticket_id>/reply", methods=["POST"])
def reply_to_ticket(ticket_id: int) -> Any:
    """Reply to a ticket."""
    if not _enabled():
        return error_response("Support is disabled", 403)

    payload = request.get_json(force=True)
    content = payload.get("content", "")
    if not content:
        return error_response("content is required")

    try:
        client = _get_client()
        result = client.reply_to_ticket(ticket_id, content)
        return jsonify(result), 201
    except Exception as exc:
        logger.exception("Failed to reply to ticket %d", ticket_id)
        return error_response(str(exc))


# -- Attachments -------------------------------------------------------------


@support_bp.route("/api/tickets/<int:ticket_id>/attachments", methods=["POST"])
def upload_attachment(ticket_id: int) -> Any:
    """Upload a file attachment to a ticket."""
    if not _enabled():
        return error_response("Support is disabled", 403)

    if "file" not in request.files:
        return error_response("No file provided")

    uploaded = request.files["file"]
    if not uploaded.filename:
        return error_response("No filename")

    try:
        import tempfile
        from pathlib import Path

        from solstone.apps.support.portal import PortalClient

        # Validate content type by extension
        suffix = Path(uploaded.filename).suffix.lower()
        if suffix not in PortalClient.ALLOWED_CONTENT_TYPES:
            return error_response(
                f"Unsupported file type: {suffix}. "
                f"Allowed: {', '.join(sorted(PortalClient.ALLOWED_CONTENT_TYPES))}"
            )

        # Save to temp file, then upload via portal client
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            uploaded.save(tmp)
            tmp_path = Path(tmp.name)

        try:
            from solstone.apps.support.tools import support_attach

            result = support_attach(
                ticket_id,
                str(tmp_path),
                filename=uploaded.filename,
            )
            return jsonify(result), 201
        finally:
            tmp_path.unlink(missing_ok=True)

    except ValueError as exc:
        return error_response(str(exc))
    except Exception as exc:
        logger.exception("Failed to upload attachment to ticket %d", ticket_id)
        return error_response(str(exc))


# -- Feedback ----------------------------------------------------------------


@support_bp.route("/api/feedback", methods=["POST"])
def submit_feedback() -> Any:
    """Submit feedback."""
    if not _enabled():
        return error_response("Support is disabled", 403)

    payload = request.get_json(force=True)
    body = payload.get("body", "")
    if not body:
        return error_response("body is required")

    try:
        from solstone.apps.support.tools import support_feedback

        product = payload.get("product", "solstone")
        anonymous = bool(payload.get("anonymous"))
        feedback_kwargs: dict[str, object] = {
            "body": body,
            "product": product,
            "anonymous": anonymous,
        }
        if not anonymous:
            raw_email = (payload.get("user_email") or "").strip()
            if raw_email:
                feedback_kwargs["user_email"] = raw_email

        result = support_feedback(**feedback_kwargs)
        return jsonify(result), 201
    except Exception as exc:
        logger.exception("Failed to submit feedback")
        return error_response(str(exc))


# -- KB & Announcements ------------------------------------------------------


@support_bp.route("/api/articles", methods=["GET"])
def search_articles() -> Any:
    """Search KB articles."""
    if not _enabled():
        return error_response("Support is disabled", 403)

    try:
        query = request.args.get("q")
        client = _get_client()
        articles = client.search_articles(query=query)
        return jsonify(articles)
    except Exception as exc:
        logger.exception("Failed to search articles")
        return error_response(str(exc))


@support_bp.route("/api/announcements", methods=["GET"])
def list_announcements() -> Any:
    """List active announcements."""
    if not _enabled():
        return error_response("Support is disabled", 403)

    try:
        client = _get_client()
        items = client.list_announcements()
        return jsonify(items)
    except Exception as exc:
        logger.exception("Failed to list announcements")
        return error_response(str(exc))


# -- Diagnostics -------------------------------------------------------------


@support_bp.route("/api/diagnostics", methods=["GET"])
def diagnostics() -> Any:
    """Run local diagnostics."""
    from solstone.apps.support.diagnostics import collect_all

    return jsonify(collect_all())


# -- Badge -------------------------------------------------------------------


@support_bp.route("/api/badge-count", methods=["GET"])
def badge_count() -> Any:
    """Return count of tickets with new responses (for app badge)."""
    if not _enabled():
        return error_response("Support is not enabled", 403)

    try:
        client = _get_client()
        tickets = client.list_tickets(status="open")
        count = sum(
            1 for t in tickets if t.get("updated_at", "") > t.get("created_at", "")
        )
        return jsonify({"count": count})
    except Exception as exc:
        logger.exception("Failed to fetch badge count")
        return error_response(str(exc), 500)
