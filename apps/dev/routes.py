"""Dev app routes."""

from __future__ import annotations

from flask import Blueprint, render_template

dev_bp = Blueprint(
    "app:dev",
    __name__,
    url_prefix="/app/dev",
)


@dev_bp.route("/")
def index():
    """Render the dev tools view."""
    return render_template("app.html", app="dev")
