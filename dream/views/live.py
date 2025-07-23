from __future__ import annotations

import os

from flask import Blueprint, jsonify, render_template, request

from hear.live import start_thread, stop_thread

from ..push import push_server

bp = Blueprint("live", __name__, template_folder="../templates")

_thread = None


def _push(event: dict) -> None:
    push_server.push({"view": "live", **event})


@bp.route("/live")
def live_page() -> str:
    return render_template("live.html", active="live")


@bp.route("/live/api/join", methods=["POST"])
def live_join() -> object:
    global _thread
    ws_url = request.json.get("ws_url") or os.getenv(
        "LIVE_WS_URL", "ws://localhost:9987"
    )
    if _thread is None or not _thread.is_alive():
        _thread = start_thread(ws_url, _push, True)
    return jsonify(status="started")


@bp.route("/live/api/leave", methods=["POST"])
def live_leave() -> object:
    global _thread
    if _thread is not None:
        stop_thread(_thread)
        _thread = None
        return jsonify(status="stopped")
    return jsonify(status="not_running")
