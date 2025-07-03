"""Utility web apps for reviewing dream data."""

from __future__ import annotations

import argparse
import os
import sys
import types

from flask import Flask

from . import routes, state
from .utils import (
    build_index,
    format_date,
    generate_top_summary,
    modify_entity_file,
    modify_entity_in_file,
    update_top_entry,
)


def create_app(journal: str = "", password: str = "") -> Flask:
    """Create and configure the review Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.secret_key = os.getenv("DREAM_SECRET", "sunstone-secret")
    app.config["PASSWORD"] = password
    app.register_blueprint(routes.bp)

    if journal:
        state.journal_root = journal
        routes.reload_entities()
        state.meetings_index = build_index(journal)
    return app


# Default application used by tests
app = create_app()

# Re-export commonly used callables
reload_entities = routes.reload_entities

home = routes.home
entities = routes.entities
calendar = routes.calendar
calendar_day = routes.calendar_day
entities_data = routes.entities_data
api_top_generate = routes.api_top_generate
api_top_update = routes.api_top_update
api_modify_entity = routes.api_modify_entity
calendar_meetings = routes.calendar_meetings
login = routes.login
logout = routes.logout

__all__ = [
    "app",
    "create_app",
    "home",
    "entities",
    "calendar",
    "calendar_day",
    "entities_data",
    "api_top_generate",
    "api_top_update",
    "api_modify_entity",
    "calendar_meetings",
    "login",
    "logout",
    "format_date",
    "modify_entity_in_file",
    "modify_entity_file",
    "update_top_entry",
    "generate_top_summary",
    "reload_entities",
    "journal_root",
    "entities_index",
    "meetings_index",
]


def __getattr__(name: str):
    if name == "journal_root":
        return state.journal_root
    if name == "entities_index":
        return state.entities_index
    if name == "meetings_index":
        return state.meetings_index
    raise AttributeError(name)


def __setattr__(name: str, value) -> None:
    if name == "journal_root":
        state.journal_root = value
    elif name == "entities_index":
        state.entities_index = value
    elif name == "meetings_index":
        state.meetings_index = value
    globals()[name] = value


class _Module(types.ModuleType):
    def __setattr__(self, key, value):
        if key in {"journal_root", "entities_index", "meetings_index"}:
            setattr(state, key, value)
        super().__setattr__(key, value)


sys.modules[__name__].__class__ = _Module


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined review web service")
    parser.add_argument("journal", help="Journal directory containing YYYYMMDD folders")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument(
        "--password",
        help="Password required for login (can also set DREAM_PASSWORD)",
        default=os.getenv("DREAM_PASSWORD"),
    )
    args = parser.parse_args()

    app = create_app(args.journal, args.password)
    if not app.config["PASSWORD"]:
        raise ValueError("Password must be provided via --password or DREAM_PASSWORD")

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
