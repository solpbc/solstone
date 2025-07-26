from __future__ import annotations

import json
import os
import re
from typing import Any

import markdown  # type: ignore
from flask import Blueprint, jsonify, render_template, request

from think.indexer import (
    search_events,
    search_summaries,
    search_transcripts,
)

from .. import state
from ..utils import format_date

bp = Blueprint("search", __name__, template_folder="../templates")


@bp.route("/search")
def search_page() -> str:
    return render_template("search.html", active="search")


@bp.route("/search/api/summaries")
def search_summaries_api() -> Any:
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"total": 0, "results": []})

    try:
        limit = int(request.args.get("limit", 20))
    except ValueError:
        limit = 20
    try:
        offset = int(request.args.get("offset", 0))
    except ValueError:
        offset = 0

    from think.utils import get_topics

    topics = get_topics()
    day = request.args.get("day")
    topic_filter = request.args.get("topic")
    total, rows = search_summaries(query, limit, offset, day=day, topic=topic_filter)
    results = []
    for r in rows:
        meta = r.get("metadata", {})
        topic = meta.get("topic", "")
        if topic.startswith("topics/"):
            topic = topic[len("topics/") :]  # noqa: E203
        if topic.endswith(".md"):
            topic = topic[:-3]
        text = r["text"]
        words = text.split()
        if len(words) > 100:
            text = " ".join(words[:100]) + " ..."
        results.append(
            {
                "day": meta.get("day", ""),
                "date": format_date(meta.get("day", "")),
                "topic": topic,
                "color": topics.get(topic, {}).get("color"),
                "text": markdown.markdown(text, extensions=["extra"]),
                "score": r.get("score", 0.0),
            }
        )

    return jsonify({"total": total, "results": results})


@bp.route("/search/api/events")
def search_events_api() -> Any:
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"total": 0, "results": []})

    try:
        limit = int(request.args.get("limit", 10))
    except ValueError:
        limit = 10
    try:
        offset = int(request.args.get("offset", 0))
    except ValueError:
        offset = 0

    from think.utils import get_topics

    topics = get_topics()
    day = request.args.get("day")
    topic_filter = request.args.get("topic")
    total, rows = search_events(query, limit, offset, day=day, topic=topic_filter)
    results = []
    for r in rows:
        meta = r.get("metadata", {})
        topic = meta.get("topic", "")
        if topic.startswith("topics/"):
            topic = topic[len("topics/") :]  # noqa: E203
        if topic.endswith(".md"):
            topic = topic[:-3]
        text = r.get("text", "")
        words = text.split()
        if len(words) > 100:
            text = " ".join(words[:100]) + " ..."
        start = meta.get("start", "")
        end = meta.get("end")
        length = 0
        if start and end:
            try:
                import datetime as _dt

                s = _dt.datetime.strptime(start, "%H:%M:%S")
                e = _dt.datetime.strptime(end, "%H:%M:%S")
                length = int((e - s).total_seconds() / 60)
            except Exception:
                length = 0
        results.append(
            {
                "day": meta.get("day", ""),
                "date": format_date(meta.get("day", "")),
                "topic": topic,
                "color": topics.get(topic, {}).get("color"),
                "start": start,
                "length": length,
                "text": markdown.markdown(text, extensions=["extra"]),
                "score": r.get("score", 0.0),
            }
        )

    return jsonify({"total": total, "results": results})


@bp.route("/search/api/transcripts")
def search_transcripts_api() -> Any:
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"total": 0, "results": []})

    try:
        limit = int(request.args.get("limit", 20))
    except ValueError:
        limit = 20
    try:
        offset = int(request.args.get("offset", 0))
    except ValueError:
        offset = 0

    day = request.args.get("day")
    total, rows = search_transcripts(query, limit, offset, day=day)
    results = []
    for r in rows:
        meta = r.get("metadata", {})
        text = r.get("text", "")
        preview = re.sub(r"[^A-Za-z0-9]+", " ", text)
        preview = re.sub(r"\s+", " ", preview).strip()
        results.append(
            {
                "day": meta.get("day", ""),
                "date": format_date(meta.get("day", "")),
                "time": meta.get("time", ""),
                "type": meta.get("type", ""),
                "preview": preview,
            }
        )

    return jsonify({"total": total, "results": results})


@bp.route("/search/api/topic_detail")
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


# Backwards compatibility aliases for tests and old URLs
def search_topic_api() -> Any:
    return search_summaries_api()


def search_occurrence_api() -> Any:
    return search_events_api()


def search_raw_api() -> Any:
    return search_transcripts_api()
