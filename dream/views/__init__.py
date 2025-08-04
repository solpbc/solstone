from __future__ import annotations

from importlib import import_module

from flask import Flask

from . import admin, agents, calendar, domains, entities, home, search, tasks

live_view = import_module(".live", __name__)

chat_view = import_module(".chat", __name__)

import_view = import_module(".import", __name__)


def register_views(app: Flask) -> None:
    for bp in [
        home.bp,
        domains.bp,
        search.bp,
        entities.bp,
        calendar.bp,
        admin.bp,
        chat_view.bp,
        agents.bp,
        import_view.bp,
        live_view.bp,
        tasks.bp,
    ]:
        app.register_blueprint(bp)
