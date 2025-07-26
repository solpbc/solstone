"""Utility web apps for reviewing dream data."""

from __future__ import annotations

import argparse
import os
import sys
import types
from importlib import import_module

from flask import Flask
from flask_sock import Sock

from think.utils import setup_cli

from . import state
from .push import push_server
from .utils import (
    adjacent_days,
    build_occurrence_index,
    format_date,
    generate_top_summary,
    list_day_folders,
    modify_entity_file,
    modify_entity_in_file,
    time_since,
    update_top_entry,
)
from .views import admin as admin_view
from .views import agents as agents_view
from .views import calendar as calendar_view
from .views import chat as chat_view
from .views import entities as entities_view
from .views import home as home_view
from .views import live as live_view
from .views import register_views
from .views import search as search_view
from .views import tasks as tasks_view

# isort: off
from .task_runner import task_runner
from .tasks import task_manager

# isort: on

import_page_view = import_module(".import", "dream.views")


def create_app(journal: str = "", password: str = "") -> Flask:
    """Create and configure the review Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.secret_key = os.getenv("DREAM_SECRET", "sunstone-secret")
    app.config["PASSWORD"] = password
    register_views(app)
    sock = Sock(app)
    task_runner.register(sock)
    push_server.register(sock)

    if journal:
        state.journal_root = journal
        task_manager.load_cached()
        entities_view.reload_entities()
        state.occurrences_index = build_occurrence_index(journal)
    return app


# Default application used by tests
app = create_app()

# Re-export commonly used callables
reload_entities = entities_view.reload_entities

home = home_view.home
entities = entities_view.entities
calendar = calendar_view.calendar_page
calendar_day = calendar_view.calendar_day
chat_page = chat_view.chat_page
agents_page = agents_view.agents_page
send_message = chat_view.send_message
chat_history = chat_view.chat_history
clear_history = chat_view.clear_history
live_page = live_view.live_page
live_join = live_view.live_join
live_leave = live_view.live_leave
search_page = search_view.search_page
import_page = import_page_view.import_page
tasks_page = tasks_view.tasks_page
tasks_list = tasks_view.tasks_list
clear_old = tasks_view.clear_old
admin_page = admin_view.admin_page
admin_day_page = admin_view.admin_day_page
admin_repair_hear = admin_view.admin_repair_hear
admin_repair_see = admin_view.admin_repair_see
admin_ponder = admin_view.admin_ponder
admin_entity = admin_view.admin_entity
admin_reduce = admin_view.admin_reduce
admin_process = admin_view.admin_process
task_log = admin_view.task_log
reindex = admin_view.reindex
refresh_summary = admin_view.refresh_summary
reload_entities_view = admin_view.reload_entities_view
entities_data = entities_view.entities_data
api_top_generate = entities_view.api_top_generate
api_top_update = entities_view.api_top_update
api_modify_entity = entities_view.api_modify_entity
calendar_occurrences = calendar_view.calendar_occurrences
calendar_days = calendar_view.calendar_days
login = home_view.login
logout = home_view.logout
stats_data = home_view.stats_data

__all__ = [
    "app",
    "create_app",
    "home",
    "entities",
    "calendar",
    "calendar_day",
    "chat_page",
    "agents_page",
    "send_message",
    "chat_history",
    "clear_history",
    "live_page",
    "live_join",
    "live_leave",
    "search_page",
    "import_page",
    "entities_data",
    "api_top_generate",
    "api_top_update",
    "api_modify_entity",
    "calendar_occurrences",
    "calendar_days",
    "adjacent_days",
    "login",
    "logout",
    "admin_page",
    "admin_day_page",
    "admin_repair_hear",
    "admin_repair_see",
    "admin_ponder",
    "admin_entity",
    "admin_reduce",
    "admin_process",
    "task_log",
    "reindex",
    "refresh_summary",
    "reload_entities_view",
    "tasks_page",
    "tasks_list",
    "clear_old",
    "format_date",
    "modify_entity_in_file",
    "modify_entity_file",
    "update_top_entry",
    "generate_top_summary",
    "time_since",
    "list_day_folders",
    "reload_entities",
    "stats_data",
    "journal_root",
    "entities_index",
    "occurrences_index",
]


def __getattr__(name: str):
    if name == "journal_root":
        return state.journal_root
    if name == "entities_index":
        return state.entities_index
    if name == "occurrences_index":
        return state.occurrences_index
    raise AttributeError(name)


def __setattr__(name: str, value) -> None:
    if name == "journal_root":
        state.journal_root = value
    elif name == "entities_index":
        state.entities_index = value
    elif name == "occurrences_index":
        state.occurrences_index = value
    globals()[name] = value


class _Module(types.ModuleType):
    def __setattr__(self, key, value):
        if key in {"journal_root", "entities_index", "occurrences_index"}:
            setattr(state, key, value)
        super().__setattr__(key, value)


sys.modules[__name__].__class__ = _Module


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined review web service")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument(
        "--password",
        help="Password required for login (can also set DREAM_PASSWORD)",
        default=os.getenv("DREAM_PASSWORD"),
    )
    args = setup_cli(parser)
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise SystemExit("JOURNAL_PATH not set")

    app = create_app(journal, args.password)
    if not app.config["PASSWORD"]:
        raise ValueError("Password must be provided via --password or DREAM_PASSWORD")

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
