from __future__ import annotations

from importlib import import_module

from flask import Flask

from . import calendar, entities, home, search

chat_view = import_module(".chat", __name__)

import_view = import_module(".import", __name__)


def register_views(app: Flask) -> None:
    for bp in [home.bp, search.bp, entities.bp, calendar.bp, chat_view.bp, import_view.bp]:
        app.register_blueprint(bp)
