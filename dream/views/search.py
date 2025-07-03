from __future__ import annotations

import json
import os
from typing import Any

from flask import Blueprint, jsonify, render_template, request

from think.indexer import search_occurrences, search_ponders

from .. import state

bp = Blueprint("search", __name__, template_folder="../templates")


@bp.route("/search")
def search_page() -> str:
    return render_template("search.html", active="search")


@bp.route("/search/api/ponder")
def search_ponder_api() -> Any:
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    results = search_ponders(state.journal_root, query, 10)
    return jsonify(results)


@bp.route("/search/api/occurrence")
def search_occurrence_api() -> Any:
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])
    results = search_occurrences(state.journal_root, query, 10)
    return jsonify(results)


@bp.route("/search/api/ponder_detail")
def ponder_detail() -> Any:
    path = request.args.get("path")
    if not path:
        return jsonify({}), 400
    full = os.path.join(state.journal_root, path)
    text = ""
    if os.path.isfile(full):
        try:
            with open(full, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            pass
    return jsonify({"text": text})


@bp.route("/search/api/occurrence_detail")
def occurrence_detail() -> Any:
    path = request.args.get("path")
    idx = int(request.args.get("index", 0))
    if not path:
        return jsonify({}), 400
    full = os.path.join(state.journal_root, path)
    data = {}
    if os.path.isfile(full):
        try:
            with open(full, "r", encoding="utf-8") as f:
                jd = json.load(f)
            occs = jd.get("occurrences", jd if isinstance(jd, list) else [])
            if 0 <= idx < len(occs):
                data = occs[idx]
        except Exception:
            pass
    return jsonify(data)
