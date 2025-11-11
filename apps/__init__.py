"""App plugin system for Sunstone.

Apps are self-contained modules that can extend any part of the system:
- routes.py: Convey web routes and handlers
- templates/: Jinja2 templates for web views
- agents/: Muse agent prompts (future)
- topics/: Think topic templates (future)
- tasks/: Background task definitions (future)

Each app provides a Flask blueprint for web routes and metadata for
navigation. Apps are automatically discovered and registered.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from flask import Blueprint

logger = logging.getLogger(__name__)


class BaseApp(ABC):
    """Base class for all Sunstone apps.

    Apps must inherit from this class and implement the required methods.
    All apps automatically get facet integration (facet pills, selected_facet
    context, and facet cookie handling).
    """

    # Required metadata - subclasses must set these
    name: str
    icon: str
    label: str

    @abstractmethod
    def get_blueprint(self) -> Blueprint:
        """Return Flask Blueprint with app routes.

        The blueprint should handle all routing for the app. Typically,
        it will have at least one route that renders app.html with
        app=self.name.

        Returns:
            Flask Blueprint instance with routes registered
        """
        pass

    def get_workspace_template(self) -> str:
        """Return path to workspace template.

        This template is included in the main content area and should
        contain the primary UI for the app.

        Returns:
            Template path relative to Flask template directory
            Default: apps/{name}/templates/workspace.html
        """
        return f"apps/{self.name}/templates/workspace.html"

    def get_app_bar_template(self) -> Optional[str]:
        """Return path to custom app-bar template, or None for empty.

        The app-bar is the bottom fixed bar where apps can place actions,
        inputs, controls, etc. Return None to leave it empty.

        Returns:
            Template path or None for no app-bar
            Default: None (empty app-bar)
        """
        return None

    def get_submenu_items(
        self, facets: list[dict], selected_facet: Optional[str] = None
    ) -> list[dict]:
        """Return submenu items for the menu-bar.

        Fully custom implementation per app. Apps can create submenu items
        based on facets, dates, static lists, or any other pattern.

        Args:
            facets: List of active facet dicts with name, title, color, emoji
            selected_facet: Currently selected facet name, or None

        Returns:
            List of dicts with:
                - label: Display text
                - path: URL path
                - count: Optional badge count (int)
                - facet: Optional facet name for data-facet attribute

        Example:
            [
                {"label": "Personal", "path": "/app/todos", "facet": "personal", "count": 7},
                {"label": "Work", "path": "/app/todos", "facet": "work", "count": 5},
            ]
        """
        return []

    def get_facet_counts(
        self, facets: list[dict], selected_facet: Optional[str] = None
    ) -> dict[str, int]:
        """Optional: Return badge counts for facet pills.

        Only implement if app wants to show counts on facet pills.
        Leave unimplemented or return empty dict to show no counts.

        Args:
            facets: List of active facet dicts
            selected_facet: Currently selected facet name, or None

        Returns:
            Dict mapping facet name to count, e.g.:
            {"work": 5, "personal": 3, "acme": 12}
        """
        return {}


class AppRegistry:
    """Registry for discovering and managing Sunstone apps."""

    def __init__(self):
        self.apps: dict[str, BaseApp] = {}

    def discover(self) -> None:
        """Auto-discover apps in apps/ directory.

        Scans the apps/ directory for subdirectories and attempts to import
        and instantiate an App class from each one. Apps must follow the
        naming convention: {name}App (e.g., TodosApp, InboxApp, HomeApp).

        The app module should be at apps/{name}/__init__.py and contain
        a class named {Name}App that inherits from BaseApp.
        """
        apps_dir = Path(__file__).parent

        for app_path in sorted(apps_dir.iterdir()):
            # Skip non-directories and private/internal directories
            if not app_path.is_dir() or app_path.name.startswith("_"):
                continue

            app_name = app_path.name

            # Skip if __init__.py doesn't exist
            if not (app_path / "__init__.py").exists():
                continue

            try:
                # Import app module
                module = importlib.import_module(f"apps.{app_name}")

                # Find App class (e.g., TodosApp, InboxApp, HomeApp)
                # Convert kebab-case or snake_case to PascalCase
                app_class_name = f"{app_name.title().replace('_', '')}App"

                if hasattr(module, app_class_name):
                    app_class = getattr(module, app_class_name)
                    app_instance = app_class()

                    # Validate it's a BaseApp subclass
                    if not isinstance(app_instance, BaseApp):
                        logger.warning(
                            f"App {app_class_name} is not a BaseApp subclass, skipping"
                        )
                        continue

                    # Register the app
                    self.apps[app_instance.name] = app_instance
                    logger.info(f"Discovered app: {app_instance.name}")
                else:
                    logger.debug(
                        f"App directory {app_name}/ missing {app_class_name} class, skipping"
                    )

            except Exception as e:
                logger.error(f"Failed to load app {app_name}: {e}", exc_info=True)

    def register_blueprints(self, flask_app) -> None:
        """Register all app blueprints with Flask.

        Args:
            flask_app: Flask application instance
        """
        for app in self.apps.values():
            try:
                blueprint = app.get_blueprint()
                flask_app.register_blueprint(blueprint)
                logger.info(f"Registered blueprint: {blueprint.name}")
            except Exception as e:
                logger.error(
                    f"Failed to register blueprint for app {app.name}: {e}",
                    exc_info=True,
                )
