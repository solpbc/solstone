# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Web interface for navigating and interacting with journal data."""

from __future__ import annotations

import json
import os
import secrets
from datetime import timedelta
from pathlib import Path

from flask import Flask
from flask_sock import Sock
from jinja2 import ChoiceLoader, FileSystemLoader

from apps import AppRegistry

from . import state
from . import system
from .apps import register_app_context
from .bridge import emit, register_websocket
from .config import bp as config_bp
from .root import bp as root_bp
from .triage import bp as triage_bp

__all__ = [
    "create_app",
    "emit",
]


def _get_or_create_secret() -> str:
    """Load convey.secret from journal.json, generating one if absent."""
    from think.utils import get_config, get_journal

    config = get_config()
    secret = config.get("convey", {}).get("secret")
    if secret:
        return secret

    secret = secrets.token_hex(32)

    config.setdefault("convey", {})["secret"] = secret
    config_path = Path(get_journal()) / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.chmod(config_path, 0o600)

    return secret


def _migrate_password_hash() -> None:
    """Migrate plaintext convey.password to hashed password_hash."""
    from werkzeug.security import generate_password_hash

    from think.utils import get_config, get_journal

    config = get_config()
    convey = config.get("convey", {})

    if "password_hash" in convey or "password" not in convey:
        return

    plaintext = convey.pop("password")
    if plaintext:
        convey["password_hash"] = generate_password_hash(plaintext)

    config["convey"] = convey
    config_path = Path(get_journal()) / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.chmod(config_path, 0o600)


def _migrate_setup_completed() -> None:
    """Infer setup.completed_at and set trust_localhost for existing installs.

    Legacy migration: handles journals where password_hash was set via
    'sol password set' CLI before web onboarding existed. Web onboarding
    now writes all config atomically in init_finalize(), so this path is
    only reached for pre-existing journals.
    """
    from think.utils import get_config, get_journal

    config = get_config()

    if not config.get("convey", {}).get("password_hash"):
        return
    if config.get("setup", {}).get("completed_at"):
        return

    from think.utils import now_ms

    config.setdefault("setup", {})["completed_at"] = now_ms()
    config.setdefault("convey", {})["trust_localhost"] = True

    config_path = Path(get_journal()) / "config" / "journal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.chmod(config_path, 0o600)


def create_app(journal: str = "") -> Flask:
    """Create and configure the Convey Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )

    # Add apps directory to template search path so apps can have their templates
    # in apps/{name}/workspace.html instead of needing a templates/ subfolder
    convey_templates = os.path.join(os.path.dirname(__file__), "templates")
    apps_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "apps")
    app.jinja_loader = ChoiceLoader(
        [
            FileSystemLoader(convey_templates),
            FileSystemLoader(apps_root),
        ]
    )

    app.secret_key = _get_or_create_secret()
    _migrate_password_hash()
    _migrate_setup_completed()
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

    # Register root blueprint (login, logout, /, favicon)
    app.register_blueprint(root_bp)

    # Register config API blueprint
    app.register_blueprint(config_bp)

    # Register triage API blueprint (universal chat bar)
    app.register_blueprint(triage_bp)

    # Register system health API blueprint
    app.register_blueprint(system.bp)

    # Initialize and register app system
    registry = AppRegistry()
    registry.discover()
    registry.register_blueprints(app)

    # Register app system context processors
    register_app_context(app, registry)

    sock = Sock(app)
    register_websocket(sock)

    if journal:
        state.journal_root = journal
    return app
