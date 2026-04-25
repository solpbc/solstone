# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

# Chat backend runs in a single Flask worker process. The threading.Lock plus
# module-level singleton state assumes one convey process per stack.

from __future__ import annotations

import atexit
import json
import logging
import os
import pprint
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

from convey.chat_stream import (
    append_chat_event,
    find_unresponded_trigger,
    read_chat_events,
    reduce_chat_state,
)
from convey.utils import error_response
from think.callosum import CallosumConnection, callosum_send
from think.utils import get_journal, now_ms

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")

MAX_ACTIVE_TALENTS = 2
MAX_LOOP_RETRIES = 3
DEFAULT_STREAM_LIMIT = 200
MAX_STREAM_LIMIT = 1000
_CHAT_WATCHDOG_SECONDS = 180
MAX_ACTIVE_REASON = "max active — waiting for one to finish"
CHAT_TROUBLE_REASON = "chat had trouble — try again"
CHAT_WATCHDOG_REASON = "chat took too long — try again"

_DAY_RE = re.compile(r"^\d{8}$")
_state_lock = threading.Lock()
_runtime_lock = threading.Lock()
_current_chat_use_id: str | None = None
_current_chat_state: dict[str, Any] | None = None
_queued_trigger: dict[str, Any] | None = None
_active_talents: dict[str, dict[str, Any]] = {}
_watchdog_timers: dict[str, threading.Timer] = {}
_last_use_id = 0
_runtime: "ChatRuntimeState | None" = None
_atexit_registered = False


@dataclass
class ChatRuntimeState:
    callosum: CallosumConnection
    apps: list[Any] = field(default_factory=list)


@chat_bp.route("", methods=["POST"])
def post_chat() -> Any:
    """Accept an owner message and schedule the chat singleton."""
    payload = request.get_json(force=True) or {}
    message = str(payload.get("message") or "").strip()
    if not message:
        return error_response("message is required", 400)

    from think.identity import ensure_identity_directory

    ensure_identity_directory()

    location = _normalize_location(
        payload.get("app"),
        payload.get("path"),
        payload.get("facet"),
    )
    append_chat_event(
        "owner_message",
        text=message,
        app=location["app"],
        path=location["path"],
        facet=location["facet"],
    )
    trigger = {
        "type": "owner_message",
        "message": message,
    }

    start_info: dict[str, Any] | None = None
    with _state_lock:
        if _current_chat_use_id is None:
            logical_use_id = _reserve_use_id_locked()
            start_info = _activate_current_locked(logical_use_id, trigger, location)
            queued = False
            response_use_id = logical_use_id
        else:
            response_use_id = _queue_trigger_locked(trigger, location)
            queued = True

    if start_info is not None and not _spawn_chat_generate(start_info):
        _handle_chat_failure(response_use_id, CHAT_TROUBLE_REASON)
        return error_response("Failed to connect to agent service", 503)

    return jsonify(use_id=response_use_id, queued=queued)


@chat_bp.route("/session", methods=["GET"])
def chat_session() -> Any:
    """Return reduced state for today's chat stream."""
    _recover_chat_if_needed()
    return jsonify(reduce_chat_state(_today_day()))


@chat_bp.route("/stream/<day>", methods=["GET"])
def chat_stream(day: str) -> Any:
    """Return ordered chat events for a day."""
    if not _DAY_RE.fullmatch(day):
        return error_response("day must be YYYYMMDD", 400)

    limit_raw = request.args.get("limit", str(DEFAULT_STREAM_LIMIT))
    try:
        limit = int(limit_raw)
    except (TypeError, ValueError):
        limit = DEFAULT_STREAM_LIMIT
    limit = max(1, min(limit, MAX_STREAM_LIMIT))

    return jsonify(events=read_chat_events(day, limit=limit))


@chat_bp.route("/result/<use_id>", methods=["GET"])
def chat_result(use_id: str) -> Any:
    """Return chat or exec state from the chat stream."""
    result = _read_result_state(use_id)
    if result is None:
        return jsonify(error="not found"), 404
    return jsonify(result)


@chat_bp.route("/talent-log/<use_id>", methods=["GET"])
def get_talent_log(use_id: str) -> Any:
    """Return a talent-use timeline from the JSONL log."""
    result = _read_talent_log(use_id)
    if result is None:
        return jsonify(error=f"Talent log not found for use_id {use_id}"), 404
    return jsonify(result)


def start_chat_runtime(app: Any) -> None:
    """Start the chat backend runtime and subscribe to cortex events."""
    global _runtime, _atexit_registered

    if app.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        logger.info("skipping chat runtime startup in Werkzeug reloader parent")
        app.chat_runtime_started = False
        return

    with _runtime_lock:
        if _runtime is None:
            runtime = ChatRuntimeState(callosum=CallosumConnection())
            runtime.callosum.start(callback=_handle_callosum_message)
            _runtime = runtime
        runtime = _runtime
        if app not in runtime.apps:
            runtime.apps.append(app)
        app.chat_runtime_started = True
        if not _atexit_registered:
            atexit.register(stop_all_chat_runtime)
            _atexit_registered = True

    _recover_chat_if_needed()


def stop_chat_runtime(app: Any) -> None:
    """Detach an app from the shared runtime."""
    app.chat_runtime_started = False
    runtime = _runtime
    if runtime is None:
        return
    with _runtime_lock:
        if app in runtime.apps:
            runtime.apps.remove(app)
        remaining = list(runtime.apps)
    if not remaining:
        stop_all_chat_runtime()


def stop_all_chat_runtime() -> None:
    """Stop the shared runtime."""
    global _runtime

    with _state_lock:
        for timer in _watchdog_timers.values():
            timer.cancel()
        _watchdog_timers.clear()

    with _runtime_lock:
        runtime = _runtime
        _runtime = None
    if runtime is None:
        return
    for app in list(runtime.apps):
        try:
            app.chat_runtime_started = False
        except Exception:
            logger.exception("chat runtime app cleanup failed")
    runtime.callosum.stop()


def _handle_callosum_message(message: dict[str, Any]) -> None:
    if message.get("chat_proxy"):
        return
    if message.get("tract") != "cortex":
        return

    event_type = message.get("event")
    if event_type == "finish":
        _on_cortex_finish(message)
        return
    if event_type == "error":
        _on_cortex_error(message)
        return

    _proxy_progress(message)


def _proxy_progress(message: dict[str, Any]) -> None:
    logical_use_id: str | None = None
    use_id = str(message.get("use_id") or "")
    if not use_id:
        return

    with _state_lock:
        if _current_chat_state is None or _current_chat_use_id is None:
            return
        raw_chat_use_id = str(_current_chat_state.get("raw_use_id") or "")
        if use_id == raw_chat_use_id:
            logical_use_id = _current_chat_use_id
            _refresh_watchdog_locked(use_id, "chat", str(_current_chat_use_id))
        elif use_id in _active_talents:
            logical_use_id = str(_active_talents[use_id]["chat_use_id"])
            _refresh_watchdog_locked(use_id, "talent", logical_use_id)
        elif _is_superseded_raw_use_id_locked(use_id):
            logger.debug(
                "superseded raw cortex event use_id=%s event=%s reason=%s",
                use_id,
                str(message.get("event") or "progress"),
                "raw rotated",
            )

    if logical_use_id is None:
        return

    fields = {
        key: value
        for key, value in message.items()
        if key not in {"tract", "event", "use_id"}
    }
    fields["use_id"] = logical_use_id
    fields["chat_proxy"] = True
    _emit_cortex_event(message["event"], **fields)


def _on_cortex_finish(message: dict[str, Any]) -> None:
    use_id = str(message.get("use_id") or "")
    if not use_id:
        return

    next_info: dict[str, Any] | None = None
    finish_payload: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None = None

    with _state_lock:
        if _current_chat_state is not None and use_id == _current_chat_state.get(
            "raw_use_id"
        ):
            logical_use_id = str(_current_chat_use_id)
            _cancel_watchdog_locked(use_id)
            try:
                parsed = _parse_chat_result(message.get("result"))
            except ValueError:
                if int(_current_chat_state.get("retry_count", 0) or 0) < 1:
                    retry_use_id = _reserve_use_id_locked()
                    _set_current_raw_use_locked(logical_use_id, retry_use_id)
                    _current_chat_state["retry_count"] = (
                        int(_current_chat_state.get("retry_count", 0) or 0) + 1
                    )
                    next_info = _build_spawn_info_locked(logical_use_id)
                else:
                    append_chat_event(
                        "chat_error",
                        reason=CHAT_TROUBLE_REASON,
                        use_id=logical_use_id,
                    )
                    error_payload = {
                        "use_id": logical_use_id,
                        "reason": CHAT_TROUBLE_REASON,
                    }
                    next_info = _clear_current_locked()
            else:
                message_text = parsed["message"] or ""
                requested_target = (
                    parsed["talent_request"]["target"]
                    if parsed["talent_request"]
                    else None
                )
                requested_task = (
                    parsed["talent_request"]["task"]
                    if parsed["talent_request"]
                    else None
                )
                append_chat_event(
                    "sol_message",
                    use_id=logical_use_id,
                    text=message_text,
                    notes=parsed["notes"],
                    requested_target=requested_target,
                    requested_task=requested_task,
                )
                _current_chat_state["retry_count"] = 0
                _set_current_raw_use_locked(logical_use_id, None)
                if requested_target:
                    active_talent_count = _active_talent_count_for_today_locked()
                    if active_talent_count >= MAX_ACTIVE_TALENTS:
                        _current_chat_state["trigger"] = {
                            "type": "synthetic-max-active",
                            "reason": MAX_ACTIVE_REASON,
                        }
                        synthetic_use_id = _reserve_use_id_locked()
                        _set_current_raw_use_locked(logical_use_id, synthetic_use_id)
                        next_info = _build_spawn_info_locked(logical_use_id)
                    elif _talent_loop_count_locked() >= MAX_LOOP_RETRIES:
                        append_chat_event(
                            "chat_error",
                            reason=CHAT_TROUBLE_REASON,
                            use_id=logical_use_id,
                        )
                        error_payload = {
                            "use_id": logical_use_id,
                            "reason": CHAT_TROUBLE_REASON,
                        }
                        next_info = _clear_current_locked()
                    else:
                        talent_use_id = _reserve_use_id_locked()
                        _active_talents[talent_use_id] = {
                            "chat_use_id": logical_use_id,
                            "target": requested_target,
                            "task": requested_task,
                            "location": dict(_current_chat_state["location"]),
                        }
                        append_chat_event(
                            "talent_spawned",
                            use_id=talent_use_id,
                            name=requested_target,
                            task=requested_task,
                            started_at=int(talent_use_id),
                        )
                        next_info = {
                            "kind": "talent",
                            "logical_use_id": logical_use_id,
                            "target": requested_target,
                            "use_id": talent_use_id,
                            "task": requested_task,
                            "context": parsed["talent_request"].get("context") or {},
                            "location": dict(_current_chat_state["location"]),
                        }
                else:
                    if not message_text:
                        append_chat_event(
                            "chat_error",
                            reason=CHAT_TROUBLE_REASON,
                            use_id=logical_use_id,
                        )
                        error_payload = {
                            "use_id": logical_use_id,
                            "reason": CHAT_TROUBLE_REASON,
                        }
                    else:
                        finish_payload = {
                            "use_id": logical_use_id,
                            "message": message_text,
                        }
                    next_info = _clear_current_locked()

        elif use_id in _active_talents:
            _cancel_watchdog_locked(use_id)
            talent_state = _active_talents.pop(use_id)
            logical_use_id = str(talent_state["chat_use_id"])
            summary = str(message.get("result") or "").strip()
            append_chat_event(
                "talent_finished",
                use_id=use_id,
                name=str(talent_state["target"]),
                summary=summary,
            )
            if (
                _current_chat_use_id == logical_use_id
                and _current_chat_state is not None
            ):
                _current_chat_state["trigger"] = {
                    "type": "talent_finished",
                    "use_id": use_id,
                    "name": str(talent_state["target"]),
                    "summary": summary,
                }
                _set_current_raw_use_locked(
                    logical_use_id,
                    _reserve_use_id_locked(),
                )
                _current_chat_state["retry_count"] = 0
                next_info = _build_spawn_info_locked(logical_use_id)
        elif _is_superseded_raw_use_id_locked(use_id):
            logger.debug(
                "superseded raw cortex event use_id=%s event=%s reason=%s",
                use_id,
                "finish",
                "raw rotated",
            )
        else:
            logger.warning(
                "unrouteable cortex event use_id=%s event=%s reason=%s",
                use_id,
                "finish",
                "no matching active chat-generate or talent",
            )

    _run_next_action(next_info)
    if finish_payload is not None:
        _emit_finish(finish_payload["use_id"], finish_payload["message"])
    if error_payload is not None:
        _emit_error(error_payload["use_id"], error_payload["reason"])


def _on_cortex_error(message: dict[str, Any]) -> None:
    use_id = str(message.get("use_id") or "")
    if not use_id:
        return

    next_info: dict[str, Any] | None = None
    error_payload: dict[str, Any] | None = None

    with _state_lock:
        if _current_chat_state is not None and use_id == _current_chat_state.get(
            "raw_use_id"
        ):
            logical_use_id = str(_current_chat_use_id)
            _cancel_watchdog_locked(use_id)
            append_chat_event(
                "chat_error",
                reason=CHAT_TROUBLE_REASON,
                use_id=logical_use_id,
            )
            error_payload = {"use_id": logical_use_id, "reason": CHAT_TROUBLE_REASON}
            next_info = _clear_current_locked()
        elif use_id in _active_talents:
            _cancel_watchdog_locked(use_id)
            talent_state = _active_talents.pop(use_id)
            logical_use_id = str(talent_state["chat_use_id"])
            reason = str(message.get("error") or CHAT_TROUBLE_REASON)
            append_chat_event(
                "talent_errored",
                use_id=use_id,
                name=str(talent_state["target"]),
                reason=reason,
            )
            if (
                _current_chat_use_id == logical_use_id
                and _current_chat_state is not None
            ):
                _current_chat_state["trigger"] = {
                    "type": "talent_errored",
                    "use_id": use_id,
                    "name": str(talent_state["target"]),
                    "reason": reason,
                }
                _set_current_raw_use_locked(
                    logical_use_id,
                    _reserve_use_id_locked(),
                )
                _current_chat_state["retry_count"] = 0
                next_info = _build_spawn_info_locked(logical_use_id)
        elif _is_superseded_raw_use_id_locked(use_id):
            logger.debug(
                "superseded raw cortex event use_id=%s event=%s reason=%s",
                use_id,
                "error",
                "raw rotated",
            )
        else:
            logger.warning(
                "unrouteable cortex event use_id=%s event=%s reason=%s",
                use_id,
                "error",
                "no matching active chat-generate or talent",
            )

    _run_next_action(next_info)
    if error_payload is not None:
        _emit_error(error_payload["use_id"], error_payload["reason"])


def _run_next_action(action: dict[str, Any] | None) -> None:
    if action is None:
        return
    if action.get("kind") == "chat":
        if not _spawn_chat_generate(action):
            _handle_chat_failure(action["logical_use_id"], CHAT_TROUBLE_REASON)
        return
    if action.get("kind") == "talent":
        if not _spawn_talent(action):
            _handle_talent_spawn_failure(action)
            return
        with _state_lock:
            _arm_watchdog_locked(
                str(action["use_id"]),
                "talent",
                str(action["logical_use_id"]),
            )


def _spawn_chat_generate(action: dict[str, Any]) -> bool:
    logger.info(
        "starting chat generate logical=%s raw=%s trigger=%s",
        action["logical_use_id"],
        action["raw_use_id"],
        action["trigger"]["type"],
    )
    from convey.utils import spawn_agent

    config = {
        "app": action["location"]["app"],
        "path": action["location"]["path"],
        "facet": action["location"]["facet"],
        "trigger": action["trigger"],
        "chat_request_use_id": action["logical_use_id"],
    }
    use_id = spawn_agent(
        prompt="",
        name="chat",
        provider=None,
        config=config,
        use_id=action["raw_use_id"],
    )
    if use_id is None:
        return False
    _emit_cortex_event("thinking", use_id=action["logical_use_id"], chat_proxy=True)
    return True


def _spawn_talent(action: dict[str, Any]) -> bool:
    from convey.utils import spawn_agent

    prompt = _build_talent_prompt(
        action["target"],
        action["task"],
        action["context"],
        action["location"],
    )
    config = {
        "app": action["location"]["app"],
        "path": action["location"]["path"],
        "facet": action["location"]["facet"],
        "chat_parent_use_id": action["logical_use_id"],
    }
    use_id = spawn_agent(
        prompt=prompt,
        name=action["target"],
        provider=None,
        config=config,
        use_id=action["use_id"],
    )
    if use_id is None:
        return False
    _emit_cortex_event("thinking", use_id=action["logical_use_id"], chat_proxy=True)
    return True


def _handle_talent_spawn_failure(action: dict[str, Any]) -> None:
    next_info: dict[str, Any] | None = None
    with _state_lock:
        _cancel_watchdog_locked(str(action["use_id"]))
        _active_talents.pop(str(action["use_id"]), None)
        append_chat_event(
            "talent_errored",
            use_id=action["use_id"],
            name=action["target"],
            reason=CHAT_TROUBLE_REASON,
        )
        if _current_chat_use_id == action["logical_use_id"] and _current_chat_state:
            _current_chat_state["trigger"] = {
                "type": "talent_errored",
                "use_id": action["use_id"],
                "name": action["target"],
                "reason": CHAT_TROUBLE_REASON,
            }
            _set_current_raw_use_locked(
                str(action["logical_use_id"]),
                _reserve_use_id_locked(),
            )
            _current_chat_state["retry_count"] = 0
            next_info = _build_spawn_info_locked(action["logical_use_id"])
    _run_next_action(next_info)


def _handle_chat_failure(logical_use_id: str, reason: str) -> None:
    next_info: dict[str, Any] | None = None
    with _state_lock:
        append_chat_event("chat_error", reason=reason, use_id=logical_use_id)
        if _current_chat_use_id == logical_use_id:
            if _current_chat_state is not None:
                _cancel_watchdog_locked(
                    str(_current_chat_state.get("raw_use_id") or "")
                )
            next_info = _clear_current_locked()
    _emit_error(logical_use_id, reason)
    _run_next_action(next_info)


def _recover_active_talents_locked(day: str) -> None:
    events = read_chat_events(day)
    latest_owner_message: dict[str, Any] | None = None
    latest_sol_message: dict[str, Any] | None = None
    spawned: dict[str, dict[str, Any]] = {}

    for event in events:
        kind = event.get("kind")
        if kind == "owner_message":
            latest_owner_message = event
            continue
        if kind == "sol_message":
            latest_sol_message = event
            continue
        if kind == "talent_spawned":
            use_id = str(event.get("use_id") or "")
            if not use_id:
                continue
            if latest_sol_message is None or latest_owner_message is None:
                logger.warning(
                    "skipping active-talent recovery for %s: no parent chat turn",
                    use_id,
                )
                continue
            chat_use_id = str(latest_sol_message.get("use_id") or "")
            if not chat_use_id:
                logger.warning(
                    "skipping active-talent recovery for %s: sol_message missing use_id",
                    use_id,
                )
                continue
            spawned[use_id] = {
                "chat_use_id": chat_use_id,
                "target": str(event.get("name") or ""),
                "task": str(event.get("task") or ""),
                "location": _normalize_location(
                    latest_owner_message.get("app"),
                    latest_owner_message.get("path"),
                    latest_owner_message.get("facet"),
                ),
            }
            continue
        if kind in {"talent_finished", "talent_errored"}:
            spawned.pop(str(event.get("use_id") or ""), None)

    for use_id, state in spawned.items():
        if use_id in _active_talents:
            continue
        _active_talents[use_id] = state
        if use_id not in _watchdog_timers:
            _arm_watchdog_locked(use_id, "talent", state["chat_use_id"])


def _recover_chat_if_needed() -> None:
    day = _today_day()
    start_info: dict[str, Any] | None = None

    with _state_lock:
        _recover_active_talents_locked(day)
        if _current_chat_use_id is not None:
            return
        unresolved = find_unresponded_trigger(day)
        if unresolved is None:
            return
        location = _location_for_trigger(day, unresolved)
        logical_use_id = _reserve_use_id_locked()
        trigger = _trigger_from_stream_event(unresolved)
        start_info = _activate_current_locked(logical_use_id, trigger, location)

    if start_info is not None and not _spawn_chat_generate(start_info):
        _handle_chat_failure(start_info["logical_use_id"], CHAT_TROUBLE_REASON)


def _activate_current_locked(
    logical_use_id: str,
    trigger: dict[str, Any],
    location: dict[str, str],
) -> dict[str, Any]:
    global _current_chat_use_id, _current_chat_state

    raw_use_id = _reserve_use_id_locked()
    _current_chat_use_id = logical_use_id
    _current_chat_state = {
        "raw_use_id": None,
        "raw_use_ids_seen": set(),
        "trigger": dict(trigger),
        "location": dict(location),
        "retry_count": 0,
    }
    _set_current_raw_use_locked(logical_use_id, raw_use_id)
    return _build_spawn_info_locked(logical_use_id)


def _build_spawn_info_locked(logical_use_id: str) -> dict[str, Any]:
    assert _current_chat_state is not None
    return {
        "kind": "chat",
        "logical_use_id": logical_use_id,
        "raw_use_id": str(_current_chat_state["raw_use_id"]),
        "trigger": dict(_current_chat_state["trigger"]),
        "location": dict(_current_chat_state["location"]),
    }


def _queue_trigger_locked(trigger: dict[str, Any], location: dict[str, str]) -> str:
    global _queued_trigger
    if _queued_trigger is None:
        _queued_trigger = {
            "use_id": _reserve_use_id_locked(),
            "trigger": dict(trigger),
            "location": dict(location),
        }
    return str(_queued_trigger["use_id"])


def _clear_current_locked() -> dict[str, Any] | None:
    global _current_chat_use_id, _current_chat_state, _queued_trigger

    _current_chat_use_id = None
    _current_chat_state = None
    if _queued_trigger is None:
        return None

    queued = _queued_trigger
    _queued_trigger = None
    return _activate_current_locked(
        str(queued["use_id"]),
        dict(queued["trigger"]),
        dict(queued["location"]),
    )


def _arm_watchdog_locked(use_id: str, kind: str, logical_use_id: str) -> None:
    _cancel_watchdog_locked(use_id)
    timer = threading.Timer(
        _CHAT_WATCHDOG_SECONDS,
        _on_watchdog_timeout,
        args=(use_id, kind, logical_use_id),
    )
    timer.daemon = True
    _watchdog_timers[use_id] = timer
    timer.start()


def _cancel_watchdog_locked(use_id: str | None) -> None:
    if not use_id:
        return
    timer = _watchdog_timers.pop(str(use_id), None)
    if timer is not None:
        timer.cancel()


def _refresh_watchdog_locked(use_id: str, kind: str, logical_use_id: str) -> None:
    if not use_id or use_id not in _watchdog_timers:
        return
    _arm_watchdog_locked(use_id, kind, logical_use_id)


def _set_current_raw_use_locked(logical_use_id: str, raw_use_id: str | None) -> None:
    assert _current_chat_state is not None
    _cancel_watchdog_locked(str(_current_chat_state.get("raw_use_id") or ""))
    if raw_use_id is not None:
        _current_chat_state["raw_use_ids_seen"].add(str(raw_use_id))
    _current_chat_state["raw_use_id"] = raw_use_id
    if raw_use_id is not None:
        _arm_watchdog_locked(str(raw_use_id), "chat", logical_use_id)


def _is_superseded_raw_use_id_locked(use_id: str) -> bool:
    if _current_chat_state is None:
        return False
    raw_chat_use_id = str(_current_chat_state.get("raw_use_id") or "")
    if use_id == raw_chat_use_id:
        return False
    return use_id in _current_chat_state["raw_use_ids_seen"]


def _on_watchdog_timeout(use_id: str, kind: str, logical_use_id: str) -> None:
    next_info: dict[str, Any] | None = None
    should_emit = False

    with _state_lock:
        _watchdog_timers.pop(use_id, None)

        if kind == "chat":
            if _current_chat_use_id != logical_use_id or _current_chat_state is None:
                return
            if str(_current_chat_state.get("raw_use_id") or "") != use_id:
                return
            logger.warning(
                "chat watchdog timed out use_id=%s kind=%s logical_use_id=%s",
                use_id,
                kind,
                logical_use_id,
            )
            append_chat_event(
                "chat_error",
                reason=CHAT_WATCHDOG_REASON,
                use_id=logical_use_id,
            )
            next_info = _clear_current_locked()
            should_emit = True
        elif kind == "talent":
            talent_state = _active_talents.get(use_id)
            if (
                talent_state is None
                or str(talent_state.get("chat_use_id")) != logical_use_id
            ):
                return
            logger.warning(
                "chat watchdog timed out use_id=%s kind=%s logical_use_id=%s",
                use_id,
                kind,
                logical_use_id,
            )
            append_chat_event(
                "talent_errored",
                use_id=use_id,
                name=str(talent_state["target"]),
                reason=CHAT_WATCHDOG_REASON,
            )
            _active_talents.pop(use_id, None)
            append_chat_event(
                "chat_error",
                reason=CHAT_WATCHDOG_REASON,
                use_id=logical_use_id,
            )
            if (
                _current_chat_use_id == logical_use_id
                and _current_chat_state is not None
                and not _current_chat_state.get("raw_use_id")
            ):
                next_info = _clear_current_locked()
            should_emit = True
        else:
            return

    if should_emit:
        _emit_error(logical_use_id, CHAT_WATCHDOG_REASON)
        _run_next_action(next_info)


def _active_talent_count_for_today_locked() -> int:
    return len(reduce_chat_state(_today_day())["active_talents"])


def _talent_loop_count_locked() -> int:
    events = read_chat_events(_today_day())
    count = 0
    for index in range(len(events) - 1, -1, -1):
        event = events[index]
        kind = event.get("kind")
        if kind == "owner_message":
            break
        if kind != "sol_message":
            continue
        if not event.get("requested_target"):
            continue

        previous = events[index - 1] if index > 0 else None
        if previous and previous.get("kind") in {"talent_finished", "talent_errored"}:
            count += 1
        else:
            break
    return count


def _parse_chat_result(result: Any) -> dict[str, Any]:
    if isinstance(result, str):
        payload = json.loads(result)
    elif isinstance(result, dict):
        payload = result
    else:
        raise ValueError("chat result must be JSON text")

    if not isinstance(payload, dict):
        raise ValueError("chat result must be an object")
    if not isinstance(payload.get("notes"), str):
        raise ValueError("chat result notes must be a string")

    message = payload.get("message")
    if message is not None and not isinstance(message, str):
        raise ValueError("chat result message must be a string or null")

    talent_request = payload.get("talent_request")
    if talent_request is None:
        return {"message": message, "notes": payload["notes"], "talent_request": None}
    if not isinstance(talent_request, dict):
        raise ValueError("chat talent_request must be an object or null")
    target = talent_request.get("target")
    if target is None:
        # Why: keep one release of compatibility for older chat outputs.
        target = "exec"
    if not isinstance(target, str):
        raise ValueError("chat talent_request.target must be a string")
    if target not in {"exec", "reflection"}:
        raise ValueError(f"unknown talent target: {target}")
    task = talent_request.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("chat talent_request.task must be a non-empty string")
    context = talent_request.get("context") or {}
    if not isinstance(context, dict):
        raise ValueError("chat talent_request.context must be an object")
    return {
        "message": message,
        "notes": payload["notes"],
        "talent_request": {
            "target": target,
            "task": task.strip(),
            "context": context,
        },
    }


def _build_talent_prompt(
    target: str,
    task: str,
    context_hints: dict[str, Any],
    location: dict[str, str],
) -> str:
    parts = [f"Task: {task}"]
    if context_hints:
        parts.append(
            "Context hints:\n" + pprint.pformat(context_hints, sort_dicts=True)
        )
    parts.append(
        "Location: "
        f"app={location['app']} path={location['path']} facet={location['facet']}"
    )

    history_lines: list[str] = []
    for event in read_chat_events(_today_day()):
        kind = event.get("kind")
        if kind == "owner_message":
            history_lines.append(f"**Owner**: {event['text']}")
        elif kind == "sol_message":
            history_lines.append(f"**Sol**: {event['text']}")
    if history_lines:
        parts.append("Recent chat:\n" + "\n".join(history_lines[-6:]))

    if target != "exec":
        parts.append(f"Target: {target}")

    return "\n\n".join(parts)


def _emit_finish(use_id: str, message: str) -> None:
    _emit_cortex_event(
        "finish",
        use_id=use_id,
        result=message,
        chat_proxy=True,
    )


def _emit_error(use_id: str, reason: str) -> None:
    _emit_cortex_event(
        "error",
        use_id=use_id,
        error=reason,
        chat_proxy=True,
    )


def _emit_cortex_event(event: str, **fields: Any) -> None:
    runtime = _runtime
    if runtime is not None and runtime.callosum.emit("cortex", event, **fields):
        return
    callosum_send("cortex", event, **fields)


def _normalize_location(app_name: Any, path: Any, facet: Any) -> dict[str, str]:
    return {
        "app": str(app_name or ""),
        "path": str(path or ""),
        "facet": str(facet or ""),
    }


def _location_for_trigger(day: str, trigger: dict[str, Any]) -> dict[str, str]:
    if trigger.get("kind") == "owner_message":
        return _normalize_location(
            trigger.get("app"),
            trigger.get("path"),
            trigger.get("facet"),
        )
    for event in reversed(read_chat_events(day)):
        if event.get("kind") == "owner_message":
            return _normalize_location(
                event.get("app"),
                event.get("path"),
                event.get("facet"),
            )
    return _normalize_location("", "", "")


def _trigger_from_stream_event(event: dict[str, Any]) -> dict[str, Any]:
    kind = event.get("kind")
    if kind == "owner_message":
        return {"type": "owner_message", "message": event.get("text", "")}
    if kind == "talent_finished":
        return {
            "type": "talent_finished",
            "use_id": event.get("use_id"),
            "name": event.get("name", "exec"),
            "summary": event.get("summary", ""),
        }
    if kind == "talent_errored":
        return {
            "type": "talent_errored",
            "use_id": event.get("use_id"),
            "name": event.get("name", "exec"),
            "reason": event.get("reason", ""),
        }
    raise ValueError(f"unsupported trigger event: {kind}")


def _read_result_state(use_id: str) -> dict[str, Any] | None:
    day = _day_for_use_id(use_id)
    if day is None:
        return None

    latest_sol: dict[str, Any] | None = None
    talent_state: dict[str, Any] | None = None
    chat_error: dict[str, Any] | None = None
    spawned_task: str | None = None

    for event in read_chat_events(day):
        kind = event.get("kind")
        if kind == "sol_message" and str(event.get("use_id")) == use_id:
            latest_sol = event
        elif kind == "chat_error" and str(event.get("use_id") or "") == use_id:
            chat_error = event
        elif kind == "talent_spawned" and str(event.get("use_id")) == use_id:
            spawned_task = event.get("task")
            talent_state = {"state": "active", "task": spawned_task}
        elif kind == "talent_finished" and str(event.get("use_id")) == use_id:
            talent_state = {
                "state": "finished",
                "summary": event.get("summary", ""),
                "task": spawned_task,
            }
        elif kind == "talent_errored" and str(event.get("use_id")) == use_id:
            talent_state = {
                "state": "errored",
                "reason": event.get("reason", ""),
                "task": spawned_task,
            }

    with _state_lock:
        if _current_chat_use_id == use_id:
            task = None
            if latest_sol and latest_sol.get("requested_target"):
                task = latest_sol.get("requested_task")
            return {"state": "active", "task": task}

    if chat_error is not None:
        return {
            "state": "errored",
            "reason": chat_error.get("reason", CHAT_TROUBLE_REASON),
        }
    if latest_sol is not None:
        return {
            "state": "finished",
            "summary": latest_sol.get("text", ""),
        }
    return talent_state


def _read_talent_log(use_id: str) -> dict[str, Any] | None:
    log_path = _find_talent_log_path(use_id)
    if log_path is None:
        return None

    request_event: dict[str, Any] | None = None
    events: list[dict[str, Any]] = []
    started_at: int | None = None
    finished_at: int | None = None

    for index, event in enumerate(_read_jsonl_events(log_path)):
        event_type = str(event.get("event") or "").strip()
        if index == 0 and event_type == "request":
            request_event = event
            continue
        if request_event is None and event_type == "request":
            request_event = event
            continue

        event.pop("raw", None)
        events.append(event)

        event_ts = _event_ts(event)
        if event_type == "start" and started_at is None:
            started_at = event_ts
        elif event_type == "finish":
            finished_at = event_ts
        elif event_type == "error":
            finished_at = event_ts

    request_ts = _event_ts(request_event)
    task = None
    if request_event is not None:
        task = request_event.get("task") or request_event.get("prompt")
    if started_at is None:
        started_at = request_ts

    last_event_type = str(events[-1].get("event") or "").strip() if events else ""
    if last_event_type == "finish":
        status = "completed"
    elif last_event_type == "error":
        status = "errored"
    else:
        status = "running"

    return {
        "use_id": use_id,
        "status": status,
        "task": task,
        "started_at": started_at,
        "finished_at": finished_at,
        "events": events,
    }


def _find_talent_log_path(use_id: str) -> Path | None:
    talents_dir = Path(get_journal()) / "talents"
    if not talents_dir.is_dir():
        return None

    for pattern in (f"*/{use_id}_active.jsonl", f"*/{use_id}.jsonl"):
        matches = sorted(talents_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return parsed


def _event_ts(event: dict[str, Any] | None) -> int | None:
    if event is None:
        return None
    value = event.get("ts")
    return value if isinstance(value, int) else None


def _reserve_use_id_locked() -> str:
    global _last_use_id

    ts = now_ms()
    if ts <= _last_use_id:
        ts = _last_use_id + 1
    _last_use_id = ts
    return str(ts)


def _today_day() -> str:
    return datetime.now().strftime("%Y%m%d")


def _day_for_use_id(use_id: str) -> str | None:
    if not use_id.isdigit():
        return None
    try:
        return datetime.fromtimestamp(int(use_id) / 1000).strftime("%Y%m%d")
    except (OSError, OverflowError, ValueError):
        return None
