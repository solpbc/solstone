# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Root-level voice API."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from flask import Blueprint, current_app, jsonify, request
from openai import AsyncOpenAI
from werkzeug.exceptions import BadRequest

from think.voice import brain
from think.voice.config import get_openai_api_key, get_voice_model
from think.voice.nav_queue import get_nav_queue
from think.voice.runtime import get_runtime_state
from think.voice.sideband import _run_sideband, register_voice_task
from think.voice.tools import get_tool_manifest

logger = logging.getLogger(__name__)

voice_bp = Blueprint("voice", __name__, url_prefix="/api/voice")


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def _optional_json_object() -> tuple[dict[str, Any], Any | None]:
    if not request.get_data(cache=True):
        return {}, None
    try:
        data = request.get_json(silent=False)
    except BadRequest:
        return {}, _error("request body must be valid JSON", 400)
    if not isinstance(data, dict):
        return {}, _error("request body must be a JSON object", 400)
    return data, None


def _required_json_object() -> tuple[dict[str, Any], Any | None]:
    try:
        data = request.get_json(silent=False)
    except BadRequest:
        return {}, _error("request body must be valid JSON", 400)
    if not isinstance(data, dict):
        return {}, _error("request body must be a JSON object", 400)
    return data, None


def _require_runtime(app: Any):
    if not getattr(app, "voice_runtime_started", False):
        return None, _error("voice runtime unavailable", 500)
    runtime = get_runtime_state()
    if runtime.loop is None:
        return None, _error("voice runtime unavailable", 500)
    return runtime, None


@voice_bp.post("/session")
async def create_voice_session():
    _, error = _optional_json_object()
    if error is not None:
        return error

    app = current_app._get_current_object()
    _, runtime_error = _require_runtime(app)
    if runtime_error is not None:
        return runtime_error

    openai_key = get_openai_api_key()
    if openai_key is None:
        return _error("voice unavailable — openai key not configured", 503)

    ready = await asyncio.to_thread(brain.wait_until_ready, app, 10.0)
    if not ready:
        return _error("voice unavailable — brain not ready", 503)

    if brain.brain_is_stale(app):
        brain.schedule_refresh(app)

    client = AsyncOpenAI(api_key=openai_key)
    try:
        response = await client.realtime.client_secrets.create(
            session={
                "type": "realtime",
                "model": get_voice_model(),
                "instructions": app.voice_brain_instruction,
                "tool_choice": "auto",
                "tools": get_tool_manifest(),
                "output_modalities": ["audio"],
            }
        )
    except Exception:
        logger.exception("voice session mint failed")
        return _error("voice session unavailable", 500)

    return jsonify({"ephemeral_key": response.value})


@voice_bp.post("/connect")
def connect_voice_sideband():
    body, error = _required_json_object()
    if error is not None:
        return error

    app = current_app._get_current_object()
    runtime, runtime_error = _require_runtime(app)
    if runtime_error is not None:
        return runtime_error

    if get_openai_api_key() is None:
        return _error("voice unavailable — openai key not configured", 503)

    call_id = body.get("call_id")
    if not isinstance(call_id, str) or not call_id.strip():
        return _error("call_id is required", 400)

    future = asyncio.run_coroutine_threadsafe(
        _run_sideband(call_id.strip(), app), runtime.loop
    )
    register_voice_task(app, future)
    return jsonify({"status": "connected"})


@voice_bp.post("/refresh-brain")
def refresh_voice_brain():
    _, error = _optional_json_object()
    if error is not None:
        return error

    app = current_app._get_current_object()
    _, runtime_error = _require_runtime(app)
    if runtime_error is not None:
        return runtime_error

    future = brain.schedule_refresh(app, force=True)
    try:
        _, instruction = future.result(timeout=30)
    except FutureTimeoutError:
        return jsonify({"status": "refreshing"}), 202
    except Exception:
        logger.exception("voice brain refresh failed")
        return _error("brain refresh failed", 500)

    return jsonify(
        {
            "status": "refreshed",
            "instruction_preview": instruction[:240],
            "brain_ready": bool(app.voice_brain_instruction),
            "brain_age_seconds": brain.brain_age_seconds(app),
        }
    )


@voice_bp.get("/nav-hints")
def nav_hints():
    call_id = request.args.get("call_id", "").strip()
    if not call_id:
        return _error("call_id is required", 400)
    hints = get_nav_queue().drain(call_id)
    return jsonify({"hints": hints, "consumed": True})


@voice_bp.get("/status")
def voice_status():
    app = current_app._get_current_object()
    active_sessions = sum(
        1
        for future in getattr(app, "voice_tasks", set())
        if hasattr(future, "done") and not future.done()
    )
    return jsonify(
        {
            "brain_ready": bool(getattr(app, "voice_brain_instruction", "")),
            "brain_age_seconds": brain.brain_age_seconds(app),
            "openai_configured": get_openai_api_key() is not None,
            "active_sessions": active_sessions,
        }
    )


__all__ = ["voice_bp"]
