from __future__ import annotations

from flask import Flask

from . import home


def register_views(app: Flask) -> None:
    app.register_blueprint(home.bp)
