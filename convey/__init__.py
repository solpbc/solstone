"""Utility web apps for reviewing convey data."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import types
from datetime import datetime, timedelta
from importlib import import_module
from typing import Callable

from flask import Flask
from flask_sock import Sock

from think import messages as message_store
from think import todo as todo_store
from think.utils import setup_cli

from . import state
from .cortex_utils import start_cortex_event_watcher
from .push import push_server
from .utils import (
    adjacent_days,
    build_occurrence_index,
    format_date,
    generate_top_summary,
    modify_entity_file,
    modify_entity_in_file,
    time_since,
    update_top_entry,
)
from .views import admin as admin_view
from .views import agents as agents_view
from .views import calendar as calendar_view
from .views import chat as chat_view
from .views import domains as domains_view
from .views import entities as entities_view
from .views import home as home_view
from .views import inbox as inbox_view
from .views import register_views
from .views import search as search_view
from .views import tasks as tasks_view

# isort: off
from .task_runner import task_runner
from .tasks import task_manager

# isort: on

import_page_view = import_module(".import", "convey.views")


logger = logging.getLogger(__name__)


def _resolve_config_password() -> str:
    """Return the configured Convey password from journal config."""
    from think.utils import get_config

    try:
        config = get_config()
        convey_config = config.get("convey", {})
        return convey_config.get("password", "")
    except Exception:
        return ""


def _count_pending_todos_today() -> int:
    """Return count of unfinished todos for the current day."""

    today = datetime.now().strftime("%Y%m%d")
    try:
        todos = todo_store.get_todos(today, ensure_day=False)
    except (FileNotFoundError, RuntimeError, ValueError):
        return 0
    if not todos:
        return 0
    return sum(
        1
        for todo in todos
        if not bool(todo.get("completed")) and not bool(todo.get("cancelled"))
    )


BadgeProvider = Callable[[], int]

NAV_BADGE_PROVIDERS: dict[str, BadgeProvider] = {
    "inbox": message_store.get_unread_count,
    "todos": _count_pending_todos_today,
}


def _resolve_nav_badges() -> dict[str, int]:
    """Run registered badge providers and gather non-zero counts."""
    badges: dict[str, int] = {}
    for key, provider in NAV_BADGE_PROVIDERS.items():
        try:
            count = int(provider())
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("nav badge provider %s failed: %s", key, exc)
            continue
        if count > 0:
            badges[key] = count
    return badges


def create_app(journal: str = "") -> Flask:
    """Create and configure the review Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.secret_key = os.getenv("CONVEY_SECRET", "sunstone-secret")
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
    register_views(app)

    @app.context_processor
    def inject_nav_badges() -> dict[str, dict[str, int]]:
        """Expose nav badge counts to all templates."""
        return {"nav_badges": _resolve_nav_badges()}

    sock = Sock(app)
    task_runner.register(sock)
    push_server.register(sock)

    if journal:
        state.journal_root = journal
        os.environ.setdefault("JOURNAL_PATH", journal)
        task_manager.load_cached()
        state.occurrences_index = build_occurrence_index(journal)
    return app


# Default application used by tests
app = create_app()

# Re-export commonly used callables
reload_entities = entities_view.reload_entities

home = home_view.home
domains_page = domains_view.domains_page
domains_list = domains_view.domains_list
inbox_page = inbox_view.inbox_page
get_messages = inbox_view.get_messages
entities = entities_view.entities
calendar = calendar_view.calendar_page
calendar_day = calendar_view.calendar_day
chat_page = chat_view.chat_page
agents_page = agents_view.agents_page
agents_list = agents_view.agents_list
send_message = chat_view.send_message
chat_history = chat_view.chat_history
clear_history = chat_view.clear_history
search_page = search_view.search_page
import_page = import_page_view.import_page
tasks_page = tasks_view.tasks_page
tasks_list = tasks_view.tasks_list
clear_old = tasks_view.clear_old
admin_page = admin_view.admin_page
admin_day_page = admin_view.admin_day_page
admin_repair_hear = admin_view.admin_repair_hear
admin_repair_see = admin_view.admin_repair_see
admin_summarize = admin_view.admin_summarize
admin_entity = admin_view.admin_entity
admin_reduce = admin_view.admin_reduce
admin_process = admin_view.admin_process
task_log = admin_view.task_log
reindex = admin_view.reindex
reset_indexes = admin_view.reset_indexes
refresh_summary = admin_view.refresh_summary
entities_types = entities_view.entities_types
entities_list = entities_view.entities_list
entities_details = entities_view.entities_details
api_top_generate = entities_view.api_top_generate
api_top_update = entities_view.api_top_update
api_modify_entity = entities_view.api_modify_entity
calendar_occurrences = calendar_view.calendar_occurrences
calendar_days = calendar_view.calendar_days
calendar_stats = calendar_view.calendar_stats
calendar_transcript_page = calendar_view.calendar_transcript_page
calendar_transcript_ranges = calendar_view.calendar_transcript_ranges
calendar_transcript_range = calendar_view.calendar_transcript_range
login = home_view.login
logout = home_view.logout
stats_data = home_view.stats_data

__all__ = [
    "app",
    "create_app",
    "home",
    "domains_page",
    "domains_list",
    "inbox_page",
    "get_messages",
    "entities",
    "calendar",
    "calendar_day",
    "chat_page",
    "agents_page",
    "agents_list",
    "send_message",
    "chat_history",
    "clear_history",
    "search_page",
    "import_page",
    "entities_types",
    "entities_list",
    "entities_details",
    "api_top_generate",
    "api_top_update",
    "api_modify_entity",
    "calendar_occurrences",
    "calendar_days",
    "calendar_stats",
    "calendar_transcript_page",
    "calendar_transcript_ranges",
    "calendar_transcript_range",
    "adjacent_days",
    "login",
    "logout",
    "admin_page",
    "admin_day_page",
    "admin_repair_hear",
    "admin_repair_see",
    "admin_summarize",
    "admin_entity",
    "admin_reduce",
    "admin_process",
    "task_log",
    "reindex",
    "reset_indexes",
    "refresh_summary",
    "tasks_page",
    "tasks_list",
    "clear_old",
    "format_date",
    "modify_entity_in_file",
    "modify_entity_file",
    "update_top_entry",
    "generate_top_summary",
    "time_since",
    "reload_entities",
    "stats_data",
    "journal_root",
    "occurrences_index",
    "run_service",
]


def __getattr__(name: str):
    if name == "journal_root":
        return state.journal_root
    if name == "occurrences_index":
        return state.occurrences_index
    raise AttributeError(name)


def __setattr__(name: str, value) -> None:
    if name == "journal_root":
        state.journal_root = value
    elif name == "occurrences_index":
        state.occurrences_index = value
    globals()[name] = value


class _Module(types.ModuleType):
    def __setattr__(self, key, value):
        if key in {"journal_root", "occurrences_index"}:
            setattr(state, key, value)
        super().__setattr__(key, value)


sys.modules[__name__].__class__ = _Module


def run_service(
    app: Flask,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False,
    start_watcher: bool = True,
) -> None:
    """Run the Convey service, optionally starting the Cortex watcher."""

    if start_watcher:
        start_cortex_event_watcher()
    app.run(host=host, port=port, debug=debug)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined review web service")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = setup_cli(parser)
    journal = os.getenv("JOURNAL_PATH")
    if not journal:
        raise SystemExit("JOURNAL_PATH not set")

    app = create_app(journal)
    password = _resolve_config_password()
    if password:
        logger.info("Password authentication enabled")
    else:
        logger.warning("No password configured - add to config/journal.json to enable authentication")

    run_service(app, host="0.0.0.0", port=args.port, debug=args.verbose)


if __name__ == "__main__":
    main()
