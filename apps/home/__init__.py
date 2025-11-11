"""Home app - main dashboard and overview."""

from __future__ import annotations

from apps import BaseApp


class HomeApp(BaseApp):
    """Home app implementation."""

    name = "home"
    icon = "🏠"
    label = "Home"

    def get_blueprint(self):
        from .routes import home_bp

        return home_bp

    def get_workspace_template(self):
        return "workspace.html"
