from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, render_template

from think.indexer import (
    load_cache,
    save_cache,
    scan_entities,
    scan_occurrences,
    scan_ponders,
)
from think.journal_stats import JournalStats

from .. import state
from ..views.entities import reload_entities

bp = Blueprint("admin", __name__, template_folder="../templates")


@bp.route("/admin")
def admin_page() -> str:
    return render_template("admin.html", active="admin")


@bp.route("/admin/api/reindex", methods=["POST"])
def reindex() -> Any:
    journal = state.journal_root
    cache = load_cache(journal)
    changed = False
    changed |= scan_entities(journal, cache)
    changed |= scan_ponders(journal, cache)
    changed |= scan_occurrences(journal, cache)
    if changed:
        save_cache(journal, cache)
    return jsonify({"status": "ok", "changed": bool(changed)})


@bp.route("/admin/api/summary", methods=["POST"])
def refresh_summary() -> Any:
    js = JournalStats()
    js.scan(state.journal_root)
    js.save_markdown(state.journal_root)
    return jsonify({"status": "ok"})


@bp.route("/admin/api/reload_entities", methods=["POST"])
def reload_entities_view() -> Any:
    reload_entities()
    return jsonify({"status": "ok"})
