# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import frontmatter
from flask import Blueprint, Response, jsonify, redirect, render_template, url_for
from markdown import Markdown
from weasyprint import HTML, default_url_fetcher

from solstone.convey.utils import DATE_RE, format_date
from solstone.think.utils import get_journal, get_owner_timezone, sunday_of_week

reflections_bp = Blueprint(
    "app:reflections",
    __name__,
    url_prefix="/app/reflections",
)


def _reflections_dir() -> Path:
    return Path(get_journal()) / "reflections" / "weekly"


def _plain_not_found(
    message: str = "Reflection not found",
) -> tuple[str, int, dict[str, str]]:
    return (message, 404, {"Content-Type": "text/plain; charset=utf-8"})


def _parse_day_token(day: str) -> datetime | None:
    if not DATE_RE.fullmatch(day):
        return None
    try:
        return datetime.strptime(day, "%Y%m%d")
    except ValueError:
        return None


def _canonical_week_day(day: str) -> str | None:
    day_dt = _parse_day_token(day)
    if day_dt is None:
        return None
    return sunday_of_week(day_dt, get_owner_timezone())


def _reflection_path(day: str) -> Path:
    return _reflections_dir() / f"{day}.md"


def _list_reflection_days() -> list[str]:
    reflections_dir = _reflections_dir()
    if not reflections_dir.is_dir():
        return []
    days = [
        path.stem
        for path in reflections_dir.glob("*.md")
        if path.is_file() and DATE_RE.fullmatch(path.stem)
    ]
    return sorted(days, reverse=True)


def _load_reflection(day: str) -> tuple[Path, str, frontmatter.Post]:
    path = _reflection_path(day)
    if not path.is_file():
        raise FileNotFoundError(day)
    raw_markdown = path.read_text(encoding="utf-8")
    return path, raw_markdown, frontmatter.loads(raw_markdown)


def _safe_pdf_url_fetcher(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    scheme = urlsplit(url).scheme.lower()
    if scheme in {"http", "https"}:
        raise ValueError("Remote assets are disabled for reflection PDFs")
    return default_url_fetcher(url, *args, **kwargs)


def _render_reflection_pdf(path: Path, post: frontmatter.Post) -> bytes:
    markdown = Markdown(extensions=["extra", "sane_lists"])
    body_html = markdown.convert(post.content)
    html = render_template(
        "reflections/pdf.html",
        week_label=format_date(path.stem),
        reflection_html=body_html,
    )
    return HTML(
        string=html,
        base_url=path.parent.resolve().as_uri(),
        url_fetcher=_safe_pdf_url_fetcher,
    ).write_pdf()


def _canonical_redirect(endpoint: str, day: str) -> Response | None:
    canonical_day = _canonical_week_day(day)
    if canonical_day is None:
        return None
    if canonical_day == day:
        return None
    return redirect(url_for(endpoint, day=canonical_day), code=302)


@reflections_bp.route("/")
def index() -> str:
    weeks = [
        {
            "day": day,
            "label": format_date(day),
            "url": url_for("app:reflections.week_view", day=day),
        }
        for day in _list_reflection_days()
    ]
    return render_template(
        "app.html",
        app="reflections",
        view_mode="index",
        weeks=weeks,
    )


@reflections_bp.route("/<day>")
def week_view(day: str) -> Any:
    redirect_response = _canonical_redirect("app:reflections.week_view", day)
    if redirect_response is not None:
        return redirect_response

    canonical_day = _canonical_week_day(day)
    if canonical_day is None:
        return _plain_not_found("Reflection not found")

    try:
        _path, _raw_markdown, post = _load_reflection(canonical_day)
    except FileNotFoundError:
        return _plain_not_found("Reflection not found")

    return render_template(
        "app.html",
        app="reflections",
        day=canonical_day,
        view_mode="detail",
        reflection_day=canonical_day,
        reflection_week_label=format_date(canonical_day),
        reflection_markdown=post.content,
        raw_url=url_for("app:reflections.week_raw", day=canonical_day),
        pdf_url=url_for("app:reflections.week_pdf", day=canonical_day),
    )


@reflections_bp.route("/<day>/raw")
def week_raw(day: str) -> Any:
    redirect_response = _canonical_redirect("app:reflections.week_raw", day)
    if redirect_response is not None:
        return redirect_response

    canonical_day = _canonical_week_day(day)
    if canonical_day is None:
        return _plain_not_found("Reflection not found")

    try:
        _path, raw_markdown, _post = _load_reflection(canonical_day)
    except FileNotFoundError:
        return _plain_not_found("Reflection not found")

    return (
        raw_markdown,
        200,
        {"Content-Type": "text/markdown; charset=utf-8"},
    )


@reflections_bp.route("/<day>/pdf")
def week_pdf(day: str) -> Any:
    redirect_response = _canonical_redirect("app:reflections.week_pdf", day)
    if redirect_response is not None:
        return redirect_response

    canonical_day = _canonical_week_day(day)
    if canonical_day is None:
        return _plain_not_found("Reflection not found")

    try:
        path, _raw_markdown, post = _load_reflection(canonical_day)
        pdf_bytes = _render_reflection_pdf(path, post)
    except FileNotFoundError:
        return _plain_not_found("Reflection not found")
    except ValueError as exc:
        return (str(exc), 400, {"Content-Type": "text/plain; charset=utf-8"})

    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="reflection-{canonical_day}.pdf"'
            )
        },
    )


@reflections_bp.route("/api/stats/<month>")
def api_stats(month: str) -> Any:
    if len(month) != 6 or not month.isdigit():
        return jsonify({"error": "Invalid month format, expected YYYYMM"}), 400

    stats = {day: 1 for day in _list_reflection_days() if day.startswith(month)}
    return jsonify(stats)
