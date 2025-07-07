from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
import threading
import uuid
from typing import Any

from flask import Blueprint, jsonify, render_template

from think import entity_roll
from think.indexer import load_cache, save_cache, scan_entities, scan_occurrences, scan_ponders
from think.journal_stats import JournalStats
from think.reduce_screen import reduce_day

from .. import state
from ..utils import DATE_RE
from ..views.entities import reload_entities
from ..wslog import capture_logs, ws_server

bp = Blueprint("admin", __name__, template_folder="../templates")

_current_job: str | None = None


def _start_job(func) -> str:
    job_id = uuid.uuid4().hex

    def runner():
        global _current_job
        _current_job = job_id
        with capture_logs(ws_server, job_id):
            try:
                func()
            finally:
                ws_server.broadcast(job_id, "__DONE__")
                _current_job = None

    if "PYTEST_CURRENT_TEST" in os.environ:
        runner()
    else:
        threading.Thread(target=runner, daemon=True).start()
    return job_id


@bp.route("/admin")
def admin_page() -> str:
    return render_template("admin.html", active="admin")


@bp.route("/admin/api/reindex", methods=["POST"])
def reindex() -> Any:
    def task() -> None:
        journal = state.journal_root
        cache = load_cache(journal)
        changed = False
        changed |= scan_entities(journal, cache)
        changed |= scan_ponders(journal, cache)
        changed |= scan_occurrences(journal, cache)
        if changed:
            save_cache(journal, cache)

    if "PYTEST_CURRENT_TEST" in os.environ:
        task()
        return jsonify({"status": "ok"})

    job = _start_job(task)
    return jsonify({"status": "started", "job_id": job})


@bp.route("/admin/api/summary", methods=["POST"])
def refresh_summary() -> Any:
    def task() -> None:
        js = JournalStats()
        js.scan(state.journal_root)
        js.save_markdown(state.journal_root)

    if "PYTEST_CURRENT_TEST" in os.environ:
        task()
        return jsonify({"status": "ok"})

    job = _start_job(task)
    return jsonify({"status": "started", "job_id": job})


@bp.route("/admin/api/reload_entities", methods=["POST"])
def reload_entities_view() -> Any:
    def task() -> None:
        reload_entities()

    if "PYTEST_CURRENT_TEST" in os.environ:
        task()
        return jsonify({"status": "ok"})

    job = _start_job(task)
    return jsonify({"status": "started", "job_id": job})


def _valid_day(day: str) -> bool:
    if not re.fullmatch(DATE_RE, day):
        return False
    if not state.journal_root:
        return False
    return os.path.isdir(os.path.join(state.journal_root, day))


def _run(cmd: list[str]) -> int:
    env = os.environ.copy()
    if state.journal_root:
        env["JOURNAL_PATH"] = state.journal_root
    if _current_job:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            ws_server.broadcast(_current_job, line.rstrip())
            logging.getLogger().info(line.rstrip())
        proc.wait()
        return proc.returncode
    result = subprocess.run(cmd, env=env)
    return result.returncode


@bp.route("/admin/<day>")
def admin_day_page(day: str) -> str:
    if not _valid_day(day):
        return "", 404
    return render_template("admin_day.html", active="admin", day=day)


@bp.route("/admin/api/<day>/repairs", methods=["POST"])
def admin_repair(day: str) -> Any:
    if not _valid_day(day):
        return jsonify({"error": "invalid day"}), 404

    def task() -> None:
        _run(["gemini-transcribe", "--repair", day])
        _run(["screen-describe", "--repair", day])

    if "PYTEST_CURRENT_TEST" in os.environ:
        task()
        return jsonify({"status": "ok"})

    job = _start_job(task)
    return jsonify({"status": "started", "job_id": job})


@bp.route("/admin/api/<day>/ponder", methods=["POST"])
def admin_ponder(day: str) -> Any:
    if not _valid_day(day):
        return jsonify({"error": "invalid day"}), 404

    def task() -> None:
        think_dir = os.path.dirname(entity_roll.__file__)
        prompt_paths = sorted(glob.glob(os.path.join(think_dir, "ponder", "*.txt")))
        for prompt in prompt_paths:
            _run(["ponder", day, "-f", prompt, "-p"])

    if "PYTEST_CURRENT_TEST" in os.environ:
        task()
        return jsonify({"status": "ok"})

    job = _start_job(task)
    return jsonify({"status": "started", "job_id": job})


@bp.route("/admin/api/<day>/entity", methods=["POST"])
def admin_entity(day: str) -> Any:
    if not _valid_day(day):
        return jsonify({"error": "invalid day"}), 404

    def task() -> None:
        day_dirs = entity_roll.find_day_dirs(state.journal_root)
        entity_roll.process_day(day, day_dirs, True)

    if "PYTEST_CURRENT_TEST" in os.environ:
        task()
        return jsonify({"status": "ok"})

    job = _start_job(task)
    return jsonify({"status": "started", "job_id": job})


@bp.route("/admin/api/<day>/reduce", methods=["POST"])
def admin_reduce(day: str) -> Any:
    if not _valid_day(day):
        return jsonify({"error": "invalid day"}), 404

    def task() -> None:
        reduce_day(day)

    if "PYTEST_CURRENT_TEST" in os.environ:
        task()
        return jsonify({"status": "ok"})

    job = _start_job(task)
    return jsonify({"status": "started", "job_id": job})
