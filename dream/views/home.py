from __future__ import annotations

import json
import os
from typing import Any

from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from think.utils import get_topics

from .. import state

bp = Blueprint(
    "review",
    __name__,
    template_folder="../templates",
    static_folder="../static",
)


@bp.before_app_request
def require_login() -> Any:
    if request.endpoint in {"review.login", "review.static", "review.stats_data"}:
        return None
    if not session.get("logged_in"):
        return redirect(url_for("review.login"))


@bp.route("/login", methods=["GET", "POST"])
def login() -> Any:
    error = None
    if request.method == "POST":
        if request.form.get("password") == current_app.config.get("PASSWORD"):
            session["logged_in"] = True
            return redirect(url_for("review.home"))
        error = "Invalid password"
    return render_template("login.html", error=error)


@bp.route("/logout")
def logout() -> Any:
    session.pop("logged_in", None)
    return redirect(url_for("review.login"))


@bp.route("/")
def home() -> str:
    # Just render the template - all data loading happens client-side
    return render_template("home.html", active="home")


@bp.route("/api/stats")
def stats_data() -> Any:
    """Return statistics from stats.json and summary.md if available."""
    response = {
        "stats": {},
        "summary_html": "",
    }

    if state.journal_root:
        # Load stats.json
        stats_path = os.path.join(state.journal_root, "stats.json")
        if os.path.isfile(stats_path):
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    response["stats"] = json.load(f)
            except Exception:
                pass

        # Load and render summary.md
        summary_path = os.path.join(state.journal_root, "summary.md")
        if os.path.isfile(summary_path):
            try:
                import markdown  # type: ignore

                with open(summary_path, "r", encoding="utf-8") as f:
                    response["summary_html"] = markdown.markdown(
                        f.read(), extensions=["extra"]
                    )
            except Exception:
                pass

        response["topics"] = get_topics()

    return jsonify(response)
