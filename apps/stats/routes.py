from __future__ import annotations

import json
import os
from typing import Any

from flask import Blueprint, jsonify, render_template

from convey import state
from think.utils import get_insights

stats_bp = Blueprint(
    "app:stats",
    __name__,
    url_prefix="/app/stats",
    static_folder=".",
    static_url_path="/static",
)


@stats_bp.route("/api/stats")
def stats_data() -> Any:
    """Return statistics from stats.json."""
    response = {
        "stats": {},
    }

    # Load stats.json
    stats_path = os.path.join(state.journal_root, "stats.json")
    if os.path.isfile(stats_path):
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                response["stats"] = json.load(f)
        except Exception:
            pass

    response["insights"] = get_insights()

    return jsonify(response)
