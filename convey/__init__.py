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
from .callosum_bridge import register_websocket, start_callosum_bridge
from .utils import (
    adjacent_days,
    build_occurrence_index,
    format_date,
    time_since,
)
from .views import admin as admin_view
from .views import agents as agents_view
from .views import calendar as calendar_view
from .views import chat as chat_view
from .views import domains as domains_view
from .views import home as home_view
from .views import inbox as inbox_view
from .views import register_views
from .views import search as search_view

# Old task system removed - now using Callosum for task execution

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
        # Get all domains that have todos for today
        domains = todo_store.get_domains_with_todos(today)
    except (FileNotFoundError, RuntimeError, ValueError):
        return 0

    if not domains:
        return 0

    # Count pending todos across all domains
    count = 0
    for domain in domains:
        try:
            todos = todo_store.get_todos(today, domain)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        if not todos:
            continue
        count += sum(
            1
            for todo in todos
            if not bool(todo.get("completed")) and not bool(todo.get("cancelled"))
        )

    return count


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
    register_websocket(sock)

    if journal:
        state.journal_root = journal
        os.environ.setdefault("JOURNAL_PATH", journal)
        state.occurrences_index = build_occurrence_index(journal)
    return app


# Default application used by tests
app = create_app()

# Re-export commonly used callables
home = home_view.home
domains_page = domains_view.domains_page
domains_list = domains_view.domains_list
inbox_page = inbox_view.inbox_page
get_messages = inbox_view.get_messages
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
admin_page = admin_view.admin_page
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
    "format_date",
    "time_since",
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
        # In debug mode with reloader, only start in child process
        # In non-debug mode, always start (no reloader)
        # WERKZEUG_RUN_MAIN is set to 'true' only in the child/main process
        should_start = not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
        if should_start:
            logger.info("Starting Callosum bridge")
            start_callosum_bridge()
        else:
            logger.debug("Skipping bridge start in reloader parent process")
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
        logger.warning(
            "No password configured - add to config/journal.json to enable authentication"
        )

    run_service(app, host="0.0.0.0", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
