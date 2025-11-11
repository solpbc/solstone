"""Dev app - testing and development tools."""

from __future__ import annotations

from apps import BaseApp


class DevApp(BaseApp):
    """Dev app for testing notification system and other features."""

    name = "dev"
    icon = "🛠️"
    label = "Dev Tools"

    def get_blueprint(self):
        from .routes import dev_bp

        return dev_bp

    def get_workspace_template(self):
        return "workspace.html"
