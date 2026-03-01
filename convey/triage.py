# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Triage endpoint for universal chat bar queries from non-chat apps."""

from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, request

from convey.utils import error_response

logger = logging.getLogger(__name__)

bp = Blueprint("triage", __name__, url_prefix="/api/triage")


@bp.route("", methods=["POST"])
def triage() -> Any:
    """Accept a message from the universal chat bar and return a response.

    Expects JSON: {message, app, path, facet}
    Returns JSON: {response}
    """
    payload = request.get_json(force=True)
    message = payload.get("message", "").strip()

    if not message:
        return error_response("message is required", 400)

    app_name = payload.get("app", "")
    path = payload.get("path", "")
    facet = payload.get("facet")

    try:
        from think.models import generate

        system_prompt = (
            "You are a helpful assistant in solstone, a journaling toolkit. "
            "The user is asking from the app bar. "
            "Give a brief, one-sentence answer."
        )

        ctx = f"App: {app_name}, Path: {path}"
        if facet:
            ctx += f", Facet: {facet}"

        full_prompt = f"[Context: {ctx}]\n\n{message}"

        response_text = generate(
            full_prompt, context="convey.triage", system_instruction=system_prompt
        )
        return {"response": response_text}
    except Exception:
        logger.exception("Triage generation failed")
        return error_response("Failed to generate response", 500)
