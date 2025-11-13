"""Home app routes and handlers."""

from __future__ import annotations

from flask import Blueprint, render_template

home_bp = Blueprint(
    "app:home",
    __name__,
    url_prefix="/app/home",
)


@home_bp.route("/")
def index():
    """Home main view."""
    return render_template("app.html", app="home")
