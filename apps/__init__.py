"""App plugin system for Sunstone.

Convention-based app discovery with minimal configuration:

Directory Structure:
    apps/my_app/           # Use underscores, not hyphens!
      routes.py            # Required: Flask blueprint
      workspace.html       # Required: Main template
      service.html         # Optional: Background service
      app_bar.html         # Optional: Bottom bar
      app.json             # Optional: Metadata overrides
      hooks.py             # Optional: Dynamic logic

Naming Rules:
    - App directory names must use underscores (my_app), not hyphens (my-app)
    - App name = directory name (e.g., "my_app")
    - Blueprint variable must be named {app_name}_bp (e.g., my_app_bp)
    - Blueprint names are automatically prefixed with "app:" to avoid conflicts
      (e.g., "home" becomes "app:home", use url_for('app:home.index'))
    - URL prefix convention: /app/{app_name}

app.json format (all optional):
    {
      "icon": "🏠",
      "label": "Custom Label"
    }

hooks.py format (all functions optional):
    def get_submenu_items(facets, selected_facet):
        return [{"label": "...", "path": "...", "count": 0, "facet": "..."}]

    def get_facet_counts(facets, selected_facet):
        return {"facet_name": count}

Apps are automatically discovered and registered.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from flask import Blueprint

logger = logging.getLogger(__name__)


@dataclass
class App:
    """Convention-based app configuration."""

    name: str
    icon: str
    label: str
    blueprint: Blueprint

    # Template paths (relative to Flask template root)
    workspace_template: str
    app_bar_template: Optional[str] = None
    service_template: Optional[str] = None

    # Dynamic hooks (optional)
    hooks: dict[str, Callable] = field(default_factory=dict)

    def get_blueprint(self) -> Blueprint:
        """Return Flask Blueprint with app routes."""
        return self.blueprint

    def get_workspace_template(self) -> str:
        """Return path to workspace template."""
        return self.workspace_template

    def get_app_bar_template(self) -> Optional[str]:
        """Return path to custom app-bar template, or None."""
        return self.app_bar_template

    def get_service_template(self) -> Optional[str]:
        """Return path to background service template, or None."""
        return self.service_template

    def get_submenu_items(
        self, facets: list[dict], selected_facet: Optional[str] = None
    ) -> list[dict]:
        """Return submenu items for the menu-bar.

        Calls hooks.get_submenu_items() if defined, otherwise returns empty list.

        Args:
            facets: List of active facet dicts with name, title, color, emoji
            selected_facet: Currently selected facet name, or None

        Returns:
            List of dicts with:
                - label: Display text
                - path: URL path
                - count: Optional badge count (int)
                - facet: Optional facet name for data-facet attribute
        """
        hook = self.hooks.get("get_submenu_items")
        if hook:
            return hook(facets, selected_facet)
        return []

    def get_facet_counts(
        self, facets: list[dict], selected_facet: Optional[str] = None
    ) -> dict[str, int]:
        """Return badge counts for facet pills.

        Calls hooks.get_facet_counts() if defined, otherwise returns empty dict.

        Args:
            facets: List of active facet dicts
            selected_facet: Currently selected facet name, or None

        Returns:
            Dict mapping facet name to count, e.g.:
            {"work": 5, "personal": 3, "acme": 12}
        """
        hook = self.hooks.get("get_facet_counts")
        if hook:
            return hook(facets, selected_facet)
        return {}


class AppRegistry:
    """Registry for discovering and managing Sunstone apps."""

    def __init__(self):
        self.apps: dict[str, App] = {}

    def discover(self) -> None:
        """Auto-discover apps using convention over configuration.

        For each directory in apps/:
        1. Load app.json if present (for icon, label overrides)
        2. Import routes.py and get blueprint
        3. Check for workspace.html (required)
        4. Check for service.html, app_bar.html (optional)
        5. Import hooks.py if present (for dynamic logic)
        """
        apps_dir = Path(__file__).parent

        for app_path in sorted(apps_dir.iterdir()):
            # Skip non-directories and private/internal directories
            if not app_path.is_dir() or app_path.name.startswith("_"):
                continue

            app_name = app_path.name

            # Skip if routes.py doesn't exist (required)
            if not (app_path / "routes.py").exists():
                logger.debug(f"Skipping {app_name}/ - no routes.py found")
                continue

            # Skip if workspace.html doesn't exist (required)
            if not (app_path / "workspace.html").exists():
                logger.debug(f"Skipping {app_name}/ - no workspace.html found")
                continue

            try:
                app = self._load_app(app_name, app_path)
                self.apps[app_name] = app
                logger.info(f"Discovered app: {app_name}")
            except Exception as e:
                logger.error(f"Failed to load app {app_name}: {e}", exc_info=True)

    def _load_app(self, app_name: str, app_path: Path) -> App:
        """Load a single app from its directory.

        Args:
            app_name: Name of the app (directory name)
            app_path: Path to app directory

        Returns:
            App instance

        Raises:
            Exception: If app cannot be loaded
        """
        # Validate app name
        if "-" in app_name:
            logger.warning(
                f"App '{app_name}' uses hyphens. Use underscores instead (e.g., 'my_app')"
            )

        # Load metadata from app.json (optional)
        metadata = self._load_metadata(app_path)

        # Get icon and label (with defaults)
        icon = metadata.get("icon", "📦")
        label = metadata.get("label", app_name.replace("_", " ").title())

        # Import routes module and get blueprint
        routes_module = importlib.import_module(f"apps.{app_name}.routes")

        # Find blueprint - look for *_bp attribute
        blueprint = None
        expected_bp_var = f"{app_name}_bp"

        for attr_name in dir(routes_module):
            if attr_name.endswith("_bp"):
                bp = getattr(routes_module, attr_name)
                if isinstance(bp, Blueprint):
                    blueprint = bp

                    # Warn if variable name doesn't match convention
                    if attr_name != expected_bp_var:
                        logger.warning(
                            f"App '{app_name}': Blueprint variable '{attr_name}' should be '{expected_bp_var}'"
                        )

                    # Warn if blueprint name doesn't match app name
                    if blueprint.name != app_name:
                        logger.warning(
                            f"App '{app_name}': Blueprint name '{blueprint.name}' should match app name '{app_name}' "
                            f"for url_for() consistency"
                        )

                    break

        if not blueprint:
            raise ValueError(
                f"No blueprint found in apps.{app_name}.routes - "
                f"expected variable named '{expected_bp_var}'"
            )

        # Automatically prefix blueprint name with "app:" to avoid conflicts with core blueprints
        original_name = blueprint.name
        blueprint.name = f"app:{app_name}"
        logger.debug(
            f"Prefixed blueprint name: '{original_name}' -> '{blueprint.name}'"
        )

        # Resolve template paths (relative to apps/ directory since that's in the loader)
        workspace_template = f"{app_name}/workspace.html"

        service_template = None
        if (app_path / "service.html").exists():
            service_template = f"{app_name}/service.html"

        app_bar_template = None
        if (app_path / "app_bar.html").exists():
            app_bar_template = f"{app_name}/app_bar.html"

        # Load hooks (optional)
        hooks = self._load_hooks(app_name, app_path)

        return App(
            name=app_name,
            icon=icon,
            label=label,
            blueprint=blueprint,
            workspace_template=workspace_template,
            app_bar_template=app_bar_template,
            service_template=service_template,
            hooks=hooks,
        )

    def _load_metadata(self, app_path: Path) -> dict[str, Any]:
        """Load app.json metadata file if it exists.

        Args:
            app_path: Path to app directory

        Returns:
            Dict with metadata, or empty dict if no app.json
        """
        metadata_file = app_path / "app.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {metadata_file}: {e}")
        return {}

    def _load_hooks(self, app_name: str, app_path: Path) -> dict[str, Callable]:
        """Load hooks.py module if it exists.

        Args:
            app_name: Name of the app
            app_path: Path to app directory

        Returns:
            Dict mapping hook name to callable
        """
        hooks_file = app_path / "hooks.py"
        if not hooks_file.exists():
            return {}

        try:
            hooks_module = importlib.import_module(f"apps.{app_name}.hooks")
            hooks = {}

            # Look for known hook functions
            for hook_name in ["get_submenu_items", "get_facet_counts"]:
                if hasattr(hooks_module, hook_name):
                    hooks[hook_name] = getattr(hooks_module, hook_name)

            return hooks
        except Exception as e:
            logger.warning(f"Failed to load hooks for {app_name}: {e}")
            return {}

    def register_blueprints(self, flask_app) -> None:
        """Register all app blueprints with Flask.

        Args:
            flask_app: Flask application instance
        """
        for app in self.apps.values():
            try:
                flask_app.register_blueprint(app.blueprint)
                logger.info(f"Registered blueprint: {app.blueprint.name}")
            except Exception as e:
                logger.error(
                    f"Failed to register blueprint for app {app.name}: {e}",
                    exc_info=True,
                )
