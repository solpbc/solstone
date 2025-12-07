from __future__ import annotations

import json
import os
import re
from typing import Any

import markdown  # type: ignore
from flask import Blueprint, jsonify, request

from convey import state
from convey.utils import format_date
from think.indexer.journal import search_journal

search_bp = Blueprint(
    "app:search",
    __name__,
    url_prefix="/app/search",
)


@search_bp.route("/api/search")
def search_journal_api() -> Any:
    """Unified journal search endpoint.

    Query parameters:
        q: Search query (required)
        limit: Max results (default 20)
        offset: Pagination offset (default 0)
        day: Filter by YYYYMMDD day
        topic: Filter by topic (e.g., "audio", "screen", "event", "flow")
        facet: Filter by facet name
    """
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"total": 0, "results": []})

    from convey.utils import parse_pagination_params
    from think.utils import get_insights

    limit, offset = parse_pagination_params(default_limit=20)

    insights = get_insights()
    day = request.args.get("day")
    topic = request.args.get("topic")
    facet = request.args.get("facet")

    total, rows = search_journal(query, limit, offset, day=day, topic=topic, facet=facet)

    results = []
    for r in rows:
        meta = r.get("metadata", {})
        result_topic = meta.get("topic", "")
        text = r.get("text", "")

        # Format text based on topic type
        if result_topic in ("audio", "screen"):
            # Transcript-style: clean preview
            preview = re.sub(r"[^A-Za-z0-9]+", " ", text)
            preview = re.sub(r"\s+", " ", preview).strip()
            results.append(
                {
                    "day": meta.get("day", ""),
                    "date": format_date(meta.get("day", "")),
                    "topic": result_topic,
                    "facet": meta.get("facet", ""),
                    "preview": preview,
                    "score": r.get("score", 0.0),
                }
            )
        else:
            # Insight/event-style: markdown rendered
            words = text.split()
            if len(words) > 100:
                text = " ".join(words[:100]) + " ..."
            results.append(
                {
                    "day": meta.get("day", ""),
                    "date": format_date(meta.get("day", "")),
                    "topic": result_topic,
                    "facet": meta.get("facet", ""),
                    "color": insights.get(result_topic, {}).get("color"),
                    "text": markdown.markdown(text, extensions=["extra"]),
                    "score": r.get("score", 0.0),
                }
            )

    return jsonify({"total": total, "results": results})


@search_bp.route("/api/topic_detail")
def topic_detail() -> Any:
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


@search_bp.route("/api/occurrence_detail")
def occurrence_detail() -> Any:
    """Return event details from a JSONL file by line index."""
    path = request.args.get("path")
    idx = int(request.args.get("index", 0))
    if not path:
        return jsonify({}), 400
    full = os.path.join(state.journal_root, path)
    data = {}
    if os.path.isfile(full):
        try:
            with open(full, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    if line_num == idx:
                        data = json.loads(line.strip())
                        break
        except Exception:
            pass
    return jsonify(data)
