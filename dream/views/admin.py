from __future__ import annotations

import os
import re
from typing import Any

from flask import Blueprint, jsonify, render_template

from .. import state
from ..task_runner import run_task
from ..utils import DATE_RE, adjacent_days, format_date

bp = Blueprint("admin", __name__, template_folder="../templates")


@bp.route("/admin")
def admin_page() -> str:
    return render_template("admin.html", active="admin")


@bp.route("/admin/api/reindex", methods=["POST"])
def reindex() -> Any:
    run_task("reindex")
    return jsonify({"status": "ok"})


@bp.route("/admin/api/summary", methods=["POST"])
def refresh_summary() -> Any:
    run_task("summary")
    return jsonify({"status": "ok"})


@bp.route("/admin/api/reload_entities", methods=["POST"])
def reload_entities_view() -> Any:
    run_task("reload_entities")
    return jsonify({"status": "ok"})


def _valid_day(day: str) -> bool:
    if not re.fullmatch(DATE_RE, day):
        return False
    if not state.journal_root:
        return False
    return os.path.isdir(os.path.join(state.journal_root, day))


@bp.route("/admin/<day>")
def admin_day_page(day: str) -> str:
    if not _valid_day(day):
        return "", 404
    title = format_date(day)
    prev_day, next_day = adjacent_days(state.journal_root, day)
    return render_template(
        "admin_day.html",
        active="admin",
        day=day,
        title=f"Admin {title}",
        prev_day=prev_day,
        next_day=next_day,
    )


@bp.route("/admin/api/<day>/repairs", methods=["POST"])
def admin_repair(day: str) -> Any:
    if not _valid_day(day):
        return jsonify({"error": "invalid day"}), 404

    run_task("repairs", day)
    return jsonify({"status": "ok"})


@bp.route("/admin/api/<day>/ponder", methods=["POST"])
def admin_ponder(day: str) -> Any:
    if not _valid_day(day):
        return jsonify({"error": "invalid day"}), 404

    run_task("ponder", day)
    return jsonify({"status": "ok"})


@bp.route("/admin/api/<day>/entity", methods=["POST"])
def admin_entity(day: str) -> Any:
    if not _valid_day(day):
        return jsonify({"error": "invalid day"}), 404

    run_task("entity", day)
    return jsonify({"status": "ok"})


@bp.route("/admin/api/<day>/reduce", methods=["POST"])
def admin_reduce(day: str) -> Any:
    if not _valid_day(day):
        return jsonify({"error": "invalid day"}), 404

    run_task("reduce", day)
    return jsonify({"status": "ok"})
