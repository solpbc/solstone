from __future__ import annotations

from flask import Blueprint, render_template

bp = Blueprint("live", __name__, template_folder="../templates")


@bp.route("/live")
def live_page() -> str:
    return render_template("live.html", active="live")
