"""Web interface for navigating and interacting with journal data."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import types
from datetime import datetime, timedelta
from importlib import import_module
from typing import Callable

from flask import Flask, request, url_for
from flask_sock import Sock
from jinja2 import ChoiceLoader, FileSystemLoader

from apps import AppRegistry
from think import messages as message_store
from think import todo as todo_store
from think.utils import setup_cli

from . import state
from .bridge import register_websocket, start_bridge
from .views import agents as agents_view
from .views import calendar as calendar_view
from .views import facets as facets_view
from .views import home as home_view
from .views import register_views

# Old task system removed - now using Callosum for task execution

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
        # Get all facets that have todos for today
        facets = todo_store.get_facets_with_todos(today)
    except (FileNotFoundError, RuntimeError, ValueError):
        return 0

    if not facets:
        return 0

    # Count pending todos across all facets
    count = 0
    for facet in facets:
        try:
            todos = todo_store.get_todos(today, facet)
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


def _get_facets_data() -> list[dict]:
    """Get facets data for templates."""
    from think.facets import get_facets

    all_facets = get_facets()
    active_facets = []

    for name, data in all_facets.items():
        if not data.get("disabled", False):
            active_facets.append(
                {
                    "name": name,
                    "title": data.get("title", name),
                    "color": data.get("color", ""),
                    "emoji": data.get("emoji", ""),
                }
            )

    return active_facets


def _get_selected_facet() -> str | None:
    """Get the currently selected facet from cookie."""
    return request.cookies.get("selectedFacet")


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

    app.secret_key = os.getenv("CONVEY_SECRET", "sunstone-secret")
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

    # Register legacy views
    register_views(app)

    # Initialize and register app system
    registry = AppRegistry()
    registry.discover()
    registry.register_blueprints(app)

    @app.context_processor
    def inject_nav_badges() -> dict[str, dict[str, int]]:
        """Expose nav badge counts to all templates."""
        return {"nav_badges": _resolve_nav_badges()}

    @app.context_processor
    def inject_app_context() -> dict:
        """Inject app registry and facets context for new app system."""
        facets = _get_facets_data()
        selected_facet = _get_selected_facet()

        # Build apps dict for menu-bar (includes submenu items)
        apps_dict = {}
        for app_instance in registry.apps.values():
            submenu = app_instance.get_submenu_items(facets, selected_facet)
            apps_dict[app_instance.name] = {
                "icon": app_instance.icon,
                "label": app_instance.label,
                "submenu": submenu if submenu else None,
            }

        return {
            "app_registry": registry,
            "apps": apps_dict,
            "facets": facets,
            "selected_facet": selected_facet,
        }

    @app.context_processor
    def inject_vendor_helper() -> dict:
        """Provide convenient vendor library helper for templates."""

        def vendor_lib(library_name: str, file: str | None = None) -> str:
            """Generate URL for vendor library.

            Args:
                library_name: Name of vendor library (e.g., 'marked')
                file: Optional specific file, defaults to {library}.min.js

            Returns:
                URL to the vendor library file

            Example:
                {{ vendor_lib('marked') }}
                → /static/vendor/marked/marked.min.js
            """
            if file is None:
                file = f"{library_name}.min.js"
            return url_for("static", filename=f"vendor/{library_name}/{file}")

        return {"vendor_lib": vendor_lib}

    sock = Sock(app)
    register_websocket(sock)

    if journal:
        state.journal_root = journal
    return app


# Default application used by tests
app = create_app()

# Re-export commonly used callables
home = home_view.home
facets_page = facets_view.facets_page
facets_list = facets_view.facets_list
calendar = calendar_view.calendar_page
calendar_day = calendar_view.calendar_day
agents_page = agents_view.agents_page
agents_list = agents_view.agents_list
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
    "facets_page",
    "facets_list",
    "calendar",
    "calendar_day",
    "agents_page",
    "agents_list",
    "calendar_days",
    "calendar_stats",
    "calendar_transcript_page",
    "calendar_transcript_ranges",
    "calendar_transcript_range",
    "login",
    "logout",
    "stats_data",
    "journal_root",
    "run_service",
]


def __getattr__(name: str):
    if name == "journal_root":
        return state.journal_root
    raise AttributeError(name)


def __setattr__(name: str, value) -> None:
    if name == "journal_root":
        state.journal_root = value
    globals()[name] = value


class _Module(types.ModuleType):
    def __setattr__(self, key, value):
        if key == "journal_root":
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
            start_bridge()
        else:
            logger.debug("Skipping bridge start in reloader parent process")
    app.run(host=host, port=port, debug=debug)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convey web interface")
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
