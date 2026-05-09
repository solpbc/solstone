# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Any

from flask import Blueprint, abort, jsonify, redirect, render_template, url_for

from solstone.convey.chat_stream import read_chat_events
from solstone.convey.sol_initiated import record_owner_chat_open
from solstone.convey.sol_initiated import copy as sol_voice_copy
from solstone.convey.sol_initiated.copy import (
    KIND_OWNER_CHAT_OPEN,
    KIND_SOL_CHAT_REQUEST,
    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
    SURFACE_CONVEY,
)
from solstone.convey.sol_initiated.state import latest_unresolved_sol_chat_request
from solstone.convey.utils import DATE_RE
from solstone.think.utils import get_config

chat_bp = Blueprint(
    "app:chat",
    __name__,
    url_prefix="/app/chat",
)


@chat_bp.route("/")
def index() -> Any:
    today = date.today().strftime("%Y%m%d")
    return redirect(url_for("app:chat.day", day=today))


@chat_bp.route("/<day>")
def day(day: str) -> str:
    if not DATE_RE.fullmatch(day):
        abort(404)

    today_day = date.today().strftime("%Y%m%d")
    owner_name, agent_name = _resolve_identity()
    events = read_chat_events(day)
    if day == today_day:
        # Page loads are engagement signals in Lode 2, so prior open facts do not
        # suppress another page-load open. Dismiss and supersede facts still do.
        openable_events = [
            event for event in events if event.get("kind") != KIND_OWNER_CHAT_OPEN
        ]
        unresolved_request = latest_unresolved_sol_chat_request(openable_events)
        if unresolved_request is not None:
            record_owner_chat_open(
                unresolved_request["request_id"],
                surface=SURFACE_CONVEY,
            )
    sol_message_origins = _build_sol_message_origins(events)

    return render_template(
        "app.html",
        app="chat",
        events=events,
        day=day,
        today_day=today_day,
        owner_name=owner_name,
        agent_name=agent_name,
        sol_message_origins=sol_message_origins,
        sol_voice_copy=sol_voice_copy,
    )


@chat_bp.route("/api/stats/<month>")
def stats(month: str) -> Any:
    if len(month) != 6 or not month.isdigit():
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    try:
        return jsonify(_month_chat_counts(month))
    except ValueError:
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400


def _month_chat_counts(month: str) -> dict[str, int]:
    year = int(month[:4])
    month_num = int(month[4:6])
    _, days_in_month = calendar.monthrange(year, month_num)
    stats: dict[str, int] = {}

    for day_num in range(1, days_in_month + 1):
        day = f"{month}{day_num:02d}"
        count = len(read_chat_events(day))
        if count:
            stats[day] = count

    return stats


def _resolve_identity() -> tuple[str, str]:
    config = get_config()
    identity = config.get("identity", {})
    owner_name = str(identity.get("preferred") or identity.get("name") or "").strip()
    agent_name = str(config.get("agent", {}).get("name") or "").strip()
    return owner_name or "Owner", agent_name or "Sol"


def _build_sol_message_origins(
    events: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    origins: dict[int, dict[str, Any]] = {}
    origins_by_request_id: dict[str, dict[str, Any]] = {}
    pending_request: dict[str, Any] | None = None

    for index, event in enumerate(events):
        kind = event.get("kind")
        if kind == KIND_SOL_CHAT_REQUEST:
            pending_request = {
                "request_id": event.get("request_id"),
                "summary": event.get("summary"),
                "trigger_talent": event.get("trigger_talent"),
                "dedupe": event.get("dedupe"),
                "since_ts": event.get("since_ts"),
                "ts": event.get("ts"),
                "time": _format_origin_time(event.get("ts")),
                "category": event.get("category"),
            }
            continue

        if kind == "sol_message" and pending_request is not None:
            origin = dict(pending_request)
            origins[index] = origin
            request_id = str(origin.get("request_id") or "")
            if request_id:
                origins_by_request_id[request_id] = origin
            pending_request = None
            continue

        if kind == KIND_SOL_CHAT_REQUEST_SUPERSEDED:
            request_id = str(event.get("request_id") or "")
            if (
                pending_request is not None
                and str(pending_request.get("request_id") or "") == request_id
            ):
                pending_request = None
            origin = origins_by_request_id.get(request_id)
            if origin is not None:
                origin["superseded_by_id"] = event.get("replaced_by")
                origin["superseded_at"] = event.get("ts")
                origin["superseded_time"] = _format_origin_time(event.get("ts"))

    return origins


def _format_origin_time(raw_ts: object) -> str:
    try:
        ts = int(raw_ts or 0)
    except (TypeError, ValueError):
        return ""
    if ts <= 0:
        return ""
    return datetime.fromtimestamp(ts / 1000).strftime("%I:%M %p").lstrip("0")
