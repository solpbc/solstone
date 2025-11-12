from __future__ import annotations

import json
import os
from typing import Any

from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

from think.utils import get_config, get_topics

from .. import state


def _get_password() -> str:
    """Get current password from config, reloading on each call."""
    try:
        config = get_config()
        convey_config = config.get("convey", {})
        return convey_config.get("password", "")
    except Exception:
        return ""


bp = Blueprint(
    "home",
    __name__,
    template_folder="../templates",
    static_folder="../static",
)


@bp.before_app_request
def require_login() -> Any:
    if request.endpoint in {
        "home.login",
        "home.static",
        "home.stats_data",
        "home.favicon",
    }:
        return None

    # Auto-bypass for localhost requests WITHOUT proxy headers
    remote_addr = request.remote_addr
    is_localhost = remote_addr in ("127.0.0.1", "::1", "localhost")

    # Detect proxy headers that might indicate forwarded external request
    proxy_headers = (
        request.headers.get("X-Forwarded-For")
        or request.headers.get("X-Real-IP")
        or request.headers.get("X-Forwarded-Host")
    )

    if is_localhost and not proxy_headers:
        # Genuine localhost request - auto-bypass
        return None

    # Otherwise require session authentication
    if not session.get("logged_in"):
        return redirect(url_for("home.login"))


@bp.route("/login", methods=["GET", "POST"])
def login() -> Any:
    # Re-check password from config on each request
    password = _get_password()

    # If no password is configured, show error page
    if not password:
        error = (
            "No password configured. Please add a password to your journal "
            "config at config/journal.json:\n\n"
            '{\n  "convey": {\n    "password": "your-password-here"\n  }\n}'
        )
        return render_template("login.html", error=error, no_password=True)

    error = None
    if request.method == "POST":
        if request.form.get("password") == password:
            session["logged_in"] = True
            session.permanent = True
            return redirect(url_for("home.home"))
        error = "Invalid password"
    return render_template("login.html", error=error, no_password=False)


@bp.route("/logout")
def logout() -> Any:
    session.pop("logged_in", None)
    return redirect(url_for("home.login"))


@bp.route("/favicon.ico")
def favicon() -> Any:
    """Serve the favicon from the project root."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return send_from_directory(project_root, "favicon.ico", mimetype="image/x-icon")


@bp.route("/")
def home() -> str:
    # Just render the template - all data loading happens client-side
    return render_template("home.html", active="home")


@bp.route("/api/stats")
def stats_data() -> Any:
    """Return statistics from stats.json."""
    response = {
        "stats": {},
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

        response["topics"] = get_topics()

    return jsonify(response)
