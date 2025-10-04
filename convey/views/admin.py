from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from think.utils import get_config as get_journal_config

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
        "summaries": [],
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
                        "summaries": d.get("repair_summaries", 0),
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


@bp.route("/admin/api/reset_indexes", methods=["POST"])
def reset_indexes() -> Any:
    run_task("reset_indexes")
    return jsonify({"status": "ok"})


@bp.route("/admin/api/summary", methods=["POST"])
def refresh_summary() -> Any:
    run_task("summary")
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
    summaries_rep = summaries_proc = 0
    entity_rep = entity_proc = 0
    reduce_rep = reduce_proc = 0

    # Read stats from stats.json instead of scanning on demand
    if state.journal_root:
        stats_path = Path(state.journal_root) / "stats.json"
        if stats_path.is_file():
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                day_stats = data.get("days", {}).get(day, {})

                # Extract repair counts
                hear_rep = day_stats.get("repair_hear", 0)
                see_rep = day_stats.get("repair_see", 0)
                reduce_rep = day_stats.get("repair_reduce", 0)
                summaries_rep = day_stats.get("repair_summaries", 0)
                entity_rep = day_stats.get("repair_entity", 0)

                # Extract processed counts
                # For hear: audio_json indicates processed transcripts
                hear_proc = day_stats.get("audio_json", 0)
                # For see: desc_json indicates processed descriptions
                see_proc = day_stats.get("desc_json", 0)
                # For reduce: screen_md indicates processed screen summaries
                reduce_proc = day_stats.get("screen_md", 0)
                # For summaries: summaries_processed is directly available
                summaries_proc = day_stats.get("summaries_processed", 0)
                # For entity: entities indicates days with entities.md (1 or 0)
                entity_proc = day_stats.get("entities", 0)
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
        summaries_rep=summaries_rep,
        summaries_proc=summaries_proc,
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


@bp.route("/admin/api/<day>/summarize", methods=["POST"])
def admin_summarize(day: str) -> Any:
    if not _valid_day(day):
        return jsonify({"error": "invalid day"}), 404

    run_task("summarize", day)
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


@bp.route("/admin/api/config")
def get_config() -> Any:
    """Return the journal configuration."""
    try:
        config = get_journal_config()
        return jsonify(config)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/admin/api/config", methods=["PUT"])
def update_config() -> Any:
    """Update the journal configuration."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Get journal path from environment
        import os

        from dotenv import load_dotenv

        load_dotenv()
        journal = os.getenv("JOURNAL_PATH")
        if not journal:
            return jsonify({"error": "JOURNAL_PATH not set"}), 500

        config_dir = Path(journal) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_path = config_dir / "journal.json"

        # Load existing config using shared utility
        config = get_journal_config()

        # Update the identity section with provided data
        if "identity" in data:
            config["identity"].update(data["identity"])

        # Write back to file
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")

        return jsonify({"success": True, "config": config})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
