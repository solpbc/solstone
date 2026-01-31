# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Web interface for navigating and interacting with journal data."""

from __future__ import annotations

import os
from datetime import timedelta

from flask import Flask
from flask_sock import Sock
from jinja2 import ChoiceLoader, FileSystemLoader

from apps import AppRegistry

from . import state
from .apps import register_app_context
from .bridge import emit, register_websocket
from .config import bp as config_bp
from .root import bp as root_bp

__all__ = [
    "create_app",
    "emit",
]


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

    app.secret_key = os.getenv("CONVEY_SECRET", "solstone-secret")
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

    # Register root blueprint (login, logout, /, favicon)
    app.register_blueprint(root_bp)

    # Register config API blueprint
    app.register_blueprint(config_bp)

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
