"""App plugin system for Sunstone.

Convention-based app discovery with minimal configuration:

Directory Structure:
    apps/my_app/           # Use underscores, not hyphens!
      workspace.html       # Required: Main template
      routes.py            # Optional: Flask blueprint (for custom routes beyond index)
      background.html      # Optional: Background service
      app_bar.html         # Optional: Bottom bar
      app.json             # Optional: Metadata overrides

Naming Rules:
    - App directory names must use underscores (my_app), not hyphens (my-app)
    - App name = directory name (e.g., "my_app")
    - Blueprint variable must be named {app_name}_bp (e.g., my_app_bp)
    - Blueprint name must use "app:{name}" pattern for consistency
      (e.g., Blueprint("app:home", ...), use url_for('app:home.index'))
    - URL prefix convention: /app/{app_name}

app.json format (all optional):
    {
      "icon": "🏠",
      "label": "Custom Label"
    }

Apps are automatically discovered and registered.
All apps are served at /app/{name} via shared handler.
Apps with routes.py can define custom routes beyond the index route.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from flask import Blueprint

logger = logging.getLogger(__name__)


@dataclass
class App:
    """Convention-based app configuration."""

    name: str
    icon: str
    label: str
    blueprint: Optional[Blueprint] = None

    # Template paths (relative to Flask template root)
    workspace_template: str = ""
    app_bar_template: Optional[str] = None
    background_template: Optional[str] = None

    # Facet configuration (optional, default {})
    # Can be bool (backwards compat) or dict with options:
    #   - muted: Include facets marked as disabled in facet.json
    facets_config: bool | dict = field(default_factory=dict)

    def facets_enabled(self) -> bool:
        """Check if facets are enabled for this app."""
        if isinstance(self.facets_config, bool):
            return self.facets_config
        if isinstance(self.facets_config, dict):
            return not self.facets_config.get("disabled", False)
        return True

    def show_muted_facets(self) -> bool:
        """Check if muted/disabled facets should be shown."""
        if isinstance(self.facets_config, dict):
            return self.facets_config.get("muted", False)
        return False

    def get_blueprint(self) -> Optional[Blueprint]:
        """Return Flask Blueprint with app routes, or None if app has no custom routes."""
        return self.blueprint

    def get_workspace_template(self) -> str:
        """Return path to workspace template."""
        return self.workspace_template

    def get_app_bar_template(self) -> Optional[str]:
        """Return path to custom app-bar template, or None."""
        return self.app_bar_template

    def get_background_template(self) -> Optional[str]:
        """Return path to background service template, or None."""
        return self.background_template


class AppRegistry:
    """Registry for discovering and managing Sunstone apps."""

    def __init__(self):
        self.apps: dict[str, App] = {}

    def discover(self) -> None:
        """Auto-discover apps using convention over configuration.

        For each directory in apps/:
        1. Check for workspace.html (required)
        2. Load app.json if present (for icon, label overrides)
        3. Import routes.py and get blueprint (optional - for custom routes)
        4. Check for background.html, app_bar.html (optional)
        """
        apps_dir = Path(__file__).parent

        for app_path in sorted(apps_dir.iterdir()):
            # Skip non-directories and private/internal directories
            if not app_path.is_dir() or app_path.name.startswith("_"):
                continue

            app_name = app_path.name

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

        # Parse facets config: can be bool or dict
        facets_raw = metadata.get("facets", {})
        if isinstance(facets_raw, bool):
            # Backwards compat: true means enabled, false means disabled
            facets_config = facets_raw
        elif isinstance(facets_raw, dict):
            facets_config = facets_raw
        else:
            facets_config = {}

        # Import routes module and get blueprint (optional)
        blueprint = None
        routes_module = None
        routes_file = app_path / "routes.py"

        if routes_file.exists():
            routes_module = importlib.import_module(f"apps.{app_name}.routes")

            # Find blueprint - look for *_bp attribute
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

                        break

            if not blueprint:
                raise ValueError(
                    f"No blueprint found in apps.{app_name}.routes - "
                    f"expected variable named '{expected_bp_var}'"
                )

            # Verify blueprint name uses "app:{name}" pattern for consistency
            expected_name = f"app:{app_name}"
            if blueprint.name != expected_name:
                raise ValueError(
                    f"App '{app_name}': Blueprint name must be '{expected_name}', "
                    f"got '{blueprint.name}'. Update Blueprint() declaration in routes.py"
                )
        else:
            # No routes.py - create a minimal blueprint
            blueprint = self._create_minimal_blueprint(app_name)
            logger.debug(
                f"Created minimal blueprint for app '{app_name}' (no routes.py)"
            )

        # Inject default index route if app doesn't define one
        self._inject_index_if_needed(blueprint, routes_module, app_name)

        # Resolve template paths (relative to apps/ directory since that's in the loader)
        workspace_template = f"{app_name}/workspace.html"

        background_template = None
        if (app_path / "background.html").exists():
            background_template = f"{app_name}/background.html"

        app_bar_template = None
        if (app_path / "app_bar.html").exists():
            app_bar_template = f"{app_name}/app_bar.html"

        return App(
            name=app_name,
            icon=icon,
            label=label,
            blueprint=blueprint,
            workspace_template=workspace_template,
            app_bar_template=app_bar_template,
            background_template=background_template,
            facets_config=facets_config,
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

    def _create_minimal_blueprint(self, app_name: str) -> Blueprint:
        """Create a minimal blueprint for apps without routes.py.

        Args:
            app_name: Name of the app

        Returns:
            Blueprint with proper naming and URL prefix
        """
        blueprint = Blueprint(
            f"app:{app_name}",
            __name__,
            url_prefix=f"/app/{app_name}",
        )
        return blueprint

    def _inject_index_if_needed(
        self, blueprint: Blueprint, routes_module: Any, app_name: str
    ) -> None:
        """Inject default index route if app doesn't define one.

        Checks if routes module has an 'index' function. If not, adds a
        default index route that renders app.html using blueprint.record()
        to support multiple app registrations.

        Args:
            blueprint: The Flask blueprint to inject into
            routes_module: The imported routes module (or None if no routes.py)
            app_name: Name of the app
        """
        import inspect

        has_index = False

        if routes_module:
            # Get functions defined in this module (not imported)
            module_functions = [
                name
                for name, obj in inspect.getmembers(routes_module)
                if inspect.isfunction(obj) and obj.__module__ == routes_module.__name__
            ]
            has_index = "index" in module_functions

        if not has_index:
            # No index function, inject default one using record() for deferred setup
            # Only inject if blueprint hasn't been registered yet
            if not blueprint._got_registered_once:

                def index():
                    from flask import render_template

                    return render_template("app.html", app=app_name)

                def setup_index(state):
                    """Deferred setup function called when blueprint is registered."""
                    state.app.add_url_rule(
                        f"{blueprint.url_prefix}/",
                        endpoint=f"{blueprint.name}.index",
                        view_func=index,
                    )

                blueprint.record(setup_index)
                logger.debug(f"Injected default index route for app '{app_name}'")

    def register_blueprints(self, flask_app) -> None:
        """Register all app blueprints with Flask.

        Args:
            flask_app: Flask application instance
        """
        for app in self.apps.values():
            if not app.blueprint:
                logger.error(
                    f"App '{app.name}' has no blueprint - this should not happen"
                )
                continue

            try:
                flask_app.register_blueprint(app.blueprint)
                logger.info(f"Registered blueprint: {app.blueprint.name}")
            except Exception as e:
                logger.error(
                    f"Failed to register blueprint for app {app.name}: {e}",
                    exc_info=True,
                )
