from __future__ import annotations

from flask import Blueprint, render_template

live_bp = Blueprint(
    "app:live",
    __name__,
    url_prefix="/app/live",
)


@live_bp.route("/")
def index() -> str:
    """Render the live events dashboard."""
    return render_template("app.html", app="live")
