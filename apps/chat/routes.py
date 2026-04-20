# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import calendar
from datetime import date
from typing import Any

from flask import Blueprint, abort, jsonify, redirect, render_template, url_for

from convey.chat_stream import read_chat_events
from convey.utils import DATE_RE
from think.utils import get_config

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

    return render_template(
        "app.html",
        app="chat",
        events=read_chat_events(day),
        day=day,
        today_day=today_day,
        owner_name=owner_name,
        agent_name=agent_name,
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
