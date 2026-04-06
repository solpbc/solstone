# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Root blueprint: authentication and core routes."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
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
from werkzeug.security import check_password_hash, generate_password_hash

from think.cluster import cluster_segments
from think.utils import day_dirs, get_config, get_journal


def _get_password_hash() -> str:
    """Get current password hash from config, reloading on each call."""
    try:
        config = get_config()
        convey_config = config.get("convey", {})
        return convey_config.get("password_hash", "")
    except Exception:
        return ""


def _save_config_section(section: str, data: dict) -> dict:
    """Merge data into a config section and write back to journal.json."""
    config = get_config()
    config.setdefault(section, {}).update(data)
    config_path = Path(get_journal()) / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.chmod(config_path, 0o600)
    return config


bp = Blueprint(
    "root",
    __name__,
    template_folder="templates",
    static_folder="static",
)


@bp.before_app_request
def require_login() -> Any:
    if request.endpoint in {
        "root.init",
        "root.init_password",
        "root.init_identity",
        "root.init_provider",
        "root.init_observers",
        "root.init_finalize",
        "root.login",
        "root.static",
        "root.favicon",
        # Remote ingest endpoints use key-based auth, not session
        "app:remote.ingest_upload",
        "app:remote.ingest_event",
        "app:remote.ingest_segments",
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
        if not _get_password_hash():
            return redirect(url_for("root.init"))
        return redirect(url_for("root.login"))


@bp.route("/login", methods=["GET", "POST"])
def login() -> Any:
    # Re-check password from config on each request
    password_hash = _get_password_hash()

    # If no password is configured, show error page
    if not password_hash:
        error = "No password configured. Run 'sol password set' to set one."
        return render_template("login.html", error=error, no_password=True)

    error = None
    if request.method == "POST":
        if check_password_hash(password_hash, request.form.get("password", "")):
            session["logged_in"] = True
            session.permanent = True
            return redirect(url_for("root.index"))
        error = "Invalid password"
    return render_template("login.html", error=error, no_password=False)


@bp.route("/init")
def init() -> Any:
    if _get_password_hash():
        return redirect(url_for("root.index"))

    config_path = str(Path(get_journal()) / "config" / "journal.json")
    repo_path = str(Path(__file__).resolve().parent.parent)
    return render_template("init.html", config_path=config_path, repo_path=repo_path)


@bp.route("/init/password", methods=["POST"])
def init_password() -> Any:
    if _get_password_hash():
        return jsonify({"error": "Already configured"}), 400

    data = request.get_json(silent=True) or {}
    password = data.get("password", "")
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    hashed = generate_password_hash(password)
    _save_config_section("convey", {"password_hash": hashed})
    return jsonify({"success": True})


@bp.route("/init/identity", methods=["POST"])
def init_identity() -> Any:
    if not _get_password_hash():
        return jsonify({"error": "Password required first"}), 403

    data = request.get_json(silent=True) or {}
    allowed = {k: data[k] for k in ("name", "preferred", "timezone") if k in data}
    _save_config_section("identity", allowed)
    return jsonify({"success": True})


@bp.route("/init/provider", methods=["POST"])
def init_provider() -> Any:
    if not _get_password_hash():
        return jsonify({"error": "Password required first"}), 403

    data = request.get_json(silent=True) or {}
    key = data.get("key", "")
    _save_config_section("env", {"GOOGLE_API_KEY": key})

    from think.providers import validate_key

    try:
        result = validate_key("google", key)
    except Exception as e:
        result = {"valid": False, "error": str(e)}
    return jsonify({"success": True, "validation": result})


@bp.route("/init/observers")
def init_observers() -> Any:
    if not _get_password_hash():
        return jsonify({"error": "Password required first"}), 403

    from apps.remote.utils import list_remotes

    remotes_list = []
    for remote in list_remotes():
        if remote.get("revoked", False):
            continue
        remotes_list.append(
            {
                "key_prefix": remote.get("key", "")[:8],
                "name": remote.get("name", ""),
                "created_at": remote.get("created_at", 0),
                "last_seen": remote.get("last_seen"),
                "last_segment": remote.get("last_segment"),
                "enabled": remote.get("enabled", True),
                "revoked": remote.get("revoked", False),
                "revoked_at": remote.get("revoked_at"),
                "stats": remote.get("stats", {}),
            }
        )
    return jsonify(remotes_list)


@bp.route("/init/finalize", methods=["POST"])
def init_finalize() -> Any:
    if not _get_password_hash():
        return jsonify({"error": "Password required first"}), 403

    from think.utils import now_ms

    data = request.get_json(silent=True) or {}
    coding_agent = data.get("coding_agent", "")
    _save_config_section(
        "setup",
        {"coding_agent": coding_agent, "completed_at": now_ms()},
    )
    session["logged_in"] = True
    session.permanent = True
    return jsonify({"success": True, "redirect": url_for("root.index")})


@bp.route("/logout")
def logout() -> Any:
    session.pop("logged_in", None)
    return redirect(url_for("root.login"))


@bp.route("/favicon.ico")
def favicon() -> Any:
    """Serve the favicon from the project root."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return send_from_directory(project_root, "favicon.ico", mimetype="image/x-icon")


@bp.route("/app/today")
def app_today() -> Any:
    """Redirect /app/today to the most recent day with journal data."""
    today = date.today().strftime("%Y%m%d")
    for day in sorted(day_dirs().keys(), reverse=True):
        if cluster_segments(day):
            return redirect(url_for("app:transcripts.transcripts_day", day=day))
    return redirect(url_for("app:transcripts.transcripts_day", day=today))


@bp.route("/")
def index() -> Any:
    """Root redirect — always to home; the app handles new journals there."""
    return redirect(url_for("app:home.index"))
