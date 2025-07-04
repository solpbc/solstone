from __future__ import annotations

from flask import Flask

from importlib import import_module

from . import calendar, entities, home, search

import_view = import_module(".import", __name__)


def register_views(app: Flask) -> None:
    for bp in [home.bp, search.bp, entities.bp, calendar.bp, import_view.bp]:
        app.register_blueprint(bp)
