from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template

from hear.transcribe import Transcriber
from see.describe import Describer
from see.reduce import scan_day as reduce_scan_day
from think.entity_roll import scan_day as entity_scan_day
from think.ponder import scan_day as ponder_scan_day

from .. import state
from ..task_runner import run_task
from ..utils import DATE_RE, adjacent_days, format_date, time_since

bp = Blueprint("admin", __name__, template_folder="../templates")


@bp.route("/admin")
def admin_page() -> str:
    repair_by_cat: dict[str, list[dict[str, Any]]] = {
        "hear": [],
        "see": [],
        "reduce": [],
        "ponder": [],
        "entity": [],
    }
    if state.journal_root:
        stats_path = Path(state.journal_root) / "stats.json"
        if stats_path.is_file():
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for day in sorted(data.get("days", {})):
                    d = data["days"].get(day, {})
                    day_info = {
                        "hear": d.get("repair_hear", 0),
                        "see": d.get("repair_see", 0),
                        "reduce": d.get("repair_reduce", 0),
                        "ponder": d.get("repair_ponder", 0),
                        "entity": d.get("repair_entity", 0),
                    }
                    for cat, count in day_info.items():
                        if count > 0:
                            repair_by_cat[cat].append({"day": day, "count": count})
            except Exception:
                repair_by_cat = {k: [] for k in repair_by_cat}
    return render_template("admin.html", active="admin", repair_data=repair_by_cat)


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
    hear_rep = hear_proc = 0
    see_rep = see_proc = 0
    ponder_rep = ponder_proc = 0
    entity_rep = entity_proc = 0
    reduce_rep = reduce_proc = 0
    try:
        day_dir = Path(state.journal_root) / day
        hear_info = Transcriber.scan_day(day_dir)
        hear_rep = len(hear_info.get("repairable", []))
        hear_proc = len(hear_info.get("processed", []))
        see_info = Describer.scan_day(day_dir)
        see_rep = len(see_info.get("repairable", []))
        see_proc = len(see_info.get("processed", []))
        if state.journal_root:
            os.environ["JOURNAL_PATH"] = state.journal_root
        reduce_info = reduce_scan_day(day)
        reduce_rep = len(reduce_info.get("repairable", []))
        reduce_proc = len(reduce_info.get("processed", []))
        ponder_info = ponder_scan_day(day)
        ponder_rep = len(ponder_info.get("repairable", []))
        ponder_proc = len(ponder_info.get("processed", []))
        entity_info = entity_scan_day(day)
        entity_rep = len(entity_info.get("repairable", []))
        entity_proc = len(entity_info.get("processed", []))
    except Exception:
        pass
    return render_template(
        "admin_day.html",
        active="admin",
        day=day,
        title=f"Admin {title}",
        prev_day=prev_day,
        next_day=next_day,
        hear_rep=hear_rep,
        hear_proc=hear_proc,
        see_rep=see_rep,
        see_proc=see_proc,
        ponder_rep=ponder_rep,
        ponder_proc=ponder_proc,
        entity_rep=entity_rep,
        entity_proc=entity_proc,
        reduce_rep=reduce_rep,
        reduce_proc=reduce_proc,
    )


@bp.route("/admin/api/<day>/repair_hear", methods=["POST"])
def admin_repair_hear(day: str) -> Any:
    if not _valid_day(day):
        return jsonify({"error": "invalid day"}), 404

    run_task("hear_repair", day)
    return jsonify({"status": "ok"})


@bp.route("/admin/api/<day>/repair_see", methods=["POST"])
def admin_repair_see(day: str) -> Any:
    if not _valid_day(day):
        return jsonify({"error": "invalid day"}), 404

    run_task("see_repair", day)
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


@bp.route("/admin/api/<day>/process", methods=["POST"])
def admin_process(day: str) -> Any:
    if not _valid_day(day):
        return jsonify({"error": "invalid day"}), 404

    run_task("process_day", day)
    return jsonify({"status": "ok"})


@bp.route("/admin/api/task_log")
@bp.route("/admin/api/<day>/task_log")
def task_log(day: str | None = None) -> Any:
    """Return task log entries for the journal or a specific day."""
    path = None
    if state.journal_root:
        base = Path(state.journal_root)
        if day:
            if not _valid_day(day):
                return jsonify([])
            path = base / day / "task_log.txt"
        else:
            path = base / "task_log.txt"
    entries: list[dict[str, Any]] = []
    if path and path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t", 1)
                    if len(parts) != 2:
                        continue
                    try:
                        ts = int(parts[0])
                    except ValueError:
                        continue
                    entries.append({"time": ts, "message": parts[1]})
        except Exception:
            entries = []
    entries.sort(key=lambda e: e["time"], reverse=True)
    for e in entries:
        e["since"] = time_since(e["time"])
    return jsonify(entries)
