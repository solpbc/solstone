from __future__ import annotations

import time

from flask import Blueprint, jsonify, render_template

from ..tasks import task_manager
from ..utils import time_since

bp = Blueprint("tasks", __name__, template_folder="../templates")


@bp.route("/tasks")
def tasks_page() -> str:
    return render_template("tasks.html", active="tasks")


@bp.route("/tasks/api/list")
def tasks_list() -> object:
    items = task_manager.list_tasks()
    for t in items:
        t["since"] = time_since(t["start"])
        if t.get("end"):
            t["duration"] = int(t["end"] - t["start"])
        else:
            t["duration"] = int(time.time() - t["start"])
    return jsonify(items)


@bp.route("/tasks/api/clear", methods=["POST"])
def clear_old() -> object:
    removed = task_manager.clear_old(7)
    return jsonify({"removed": removed})
