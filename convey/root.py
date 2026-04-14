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


def _is_setup_complete() -> bool:
    """Check if initial setup has been completed."""
    try:
        config = get_config()
        return bool(config.get("setup", {}).get("completed_at"))
    except Exception:
        return False


def _check_basic_auth() -> bool:
    """Check Basic Auth credentials against stored password hash."""
    auth = request.authorization
    if not auth or auth.type != "basic":
        return False
    password_hash = _get_password_hash()
    if not password_hash:
        return False
    return check_password_hash(password_hash, auth.password or "")


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
    if request.endpoint is None:
        return None

    if request.endpoint in {
        "root.init",
        "root.init_validate_provider",
        "root.init_observers",
        "root.init_finalize",
        "root.login",
        "root.static",
        "root.favicon",
        # Observer ingest endpoints use key-based auth, not session
        "app:observer.ingest_upload",
        "app:observer.ingest_event",
        "app:observer.ingest_segments",
        "app:observer.ingest_transfer",
        "app:observer.ingest_manifest",
        "app:observer.ingest_manifest_day",
        # Journal-source manifest and ingest endpoints use key-based auth, not session
        "app:import.journal_source_manifest",
        "app:import.ingest_segments",
        "app:import.ingest_entities",
    }:
        return None

    # Session cookie
    if session.get("logged_in"):
        return None

    # Basic Auth (per-request, no session creation)
    if _check_basic_auth():
        return None

    # Check setup state
    setup_complete = _is_setup_complete()

    # Opt-in localhost bypass (requires completed setup + trust_localhost flag)
    if setup_complete:
        config = get_config()
        if config.get("convey", {}).get("trust_localhost", False):
            remote_addr = request.remote_addr
            is_localhost = remote_addr in ("127.0.0.1", "::1", "localhost")
            proxy_headers = (
                request.headers.get("X-Forwarded-For")
                or request.headers.get("X-Real-IP")
                or request.headers.get("X-Forwarded-Host")
            )
            if is_localhost and not proxy_headers:
                return None

    # Not authenticated — redirect based on setup state
    if not setup_complete:
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
        error = "incorrect password. passwords are case-sensitive. if you've forgotten it, you can reset via sol password set on the command line."
    return render_template("login.html", error=error, no_password=False)


@bp.route("/init")
def init() -> Any:
    if _is_setup_complete():
        return redirect(url_for("root.index"))

    config_path = str(Path(get_journal()) / "config" / "journal.json")
    repo_path = str(Path(__file__).resolve().parent.parent)
    return render_template(
        "init.html",
        config_path=config_path,
        repo_path=repo_path,
    )


@bp.route("/init/validate-provider", methods=["POST"])
def init_validate_provider() -> Any:
    data = request.get_json(silent=True) or {}
    key = data.get("key", "")

    from think.providers import validate_key

    try:
        result = validate_key("google", key)
    except Exception as e:
        result = {"valid": False, "error": str(e)}
    return jsonify(result)


@bp.route("/init/observers")
def init_observers() -> Any:
    from apps.observer.utils import list_observers

    observers_list = []
    for observer in list_observers():
        if observer.get("revoked", False):
            continue
        observers_list.append(
            {
                "key_prefix": observer.get("key", "")[:8],
                "name": observer.get("name", ""),
                "created_at": observer.get("created_at", 0),
                "last_seen": observer.get("last_seen"),
                "last_segment": observer.get("last_segment"),
                "enabled": observer.get("enabled", True),
                "revoked": observer.get("revoked", False),
                "revoked_at": observer.get("revoked_at"),
                "stats": observer.get("stats", {}),
            }
        )
    return jsonify(observers_list)


@bp.route("/init/finalize", methods=["POST"])
def init_finalize() -> Any:
    data = request.get_json(silent=True) or {}

    password = data.get("password", "")
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    from think.utils import now_ms

    hashed = generate_password_hash(password)

    config = get_config()
    config.setdefault("convey", {}).update(
        {
            "password_hash": hashed,
            "trust_localhost": True,
        }
    )
    config.setdefault("identity", {}).update(
        {
            k: v
            for k, v in {
                "name": data.get("name"),
                "preferred": data.get("preferred"),
                "timezone": data.get("timezone"),
            }.items()
            if v
        }
    )
    gemini_key = data.get("gemini_key")
    if gemini_key:
        config.setdefault("env", {})["GOOGLE_API_KEY"] = gemini_key
    config.setdefault("setup", {})["completed_at"] = now_ms()
    retention_mode = data.get("retention_mode", "days")
    retention_days = data.get("retention_days", 7)
    if isinstance(retention_days, str):
        try:
            retention_days = int(retention_days)
        except (ValueError, TypeError):
            retention_days = 7
    config.setdefault("retention", {}).update(
        {
            "raw_media": retention_mode,
            "raw_media_days": retention_days if retention_mode == "days" else None,
        }
    )

    config_path = Path(get_journal()) / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.chmod(config_path, 0o600)

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
