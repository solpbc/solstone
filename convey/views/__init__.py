from __future__ import annotations

from flask import Flask

from . import (
    calendar,
    facets,
    home,
)


def register_views(app: Flask) -> None:
    for bp in [
        home.bp,
        facets.bp,
        calendar.bp,
    ]:
        app.register_blueprint(bp)
