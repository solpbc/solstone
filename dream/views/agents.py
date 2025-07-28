from __future__ import annotations

import json
import os

from flask import Blueprint, jsonify, render_template

from .. import state
from ..utils import time_since

bp = Blueprint("agents", __name__, template_folder="../templates")


@bp.route("/agents")
def agents_page() -> str:
    """Render the Agents view."""
    return render_template("agents.html", active="agents")


def _agents_dir() -> str:
    if not state.journal_root:
        return ""
    path = os.path.join(state.journal_root, "agents")
    os.makedirs(path, exist_ok=True)
    return path


@bp.route("/agents/api/list")
def agents_list() -> object:
    path = _agents_dir()
    items: list[dict[str, object]] = []
    if path and os.path.isdir(path):
        for name in os.listdir(path):
            if not name.endswith(".jsonl"):
                continue
            full = os.path.join(path, name)
            start_ms = 0
            try:
                start_ms = int(os.path.splitext(name)[0])
            except ValueError:
                try:
                    start_ms = int(os.stat(full).st_mtime * 1000)
                except Exception:
                    start_ms = 0
            start = start_ms / 1000
            prompt = ""
            persona = ""
            model = ""
            try:
                with open(full, "r", encoding="utf-8") as f:
                    for line in f:
                        j = json.loads(line)
                        if j.get("event") == "start":
                            prompt = j.get("prompt", "")
                            persona = j.get("persona", "")
                            model = j.get("model", "")
                            break
            except Exception:
                continue
            items.append(
                {
                    "id": os.path.splitext(name)[0],
                    "start": start,
                    "since": time_since(start),
                    "model": model,
                    "persona": persona,
                    "prompt": prompt,
                }
            )
    items.sort(key=lambda x: float(x.get("start", 0)), reverse=True)
    return jsonify(items)
