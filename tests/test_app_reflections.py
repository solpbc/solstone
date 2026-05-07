# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from solstone.convey import create_app

REFLECTION_FIXTURE = Path("tests/fixtures/journal/reflections/weekly/20260308.md")


def _seed_reflection(journal: Path, content: str | None = None) -> None:
    target = journal / "reflections" / "weekly" / "20260308.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        content
        if content is not None
        else REFLECTION_FIXTURE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def _make_client(journal: Path):
    app = create_app(str(journal))
    app.config["TESTING"] = True
    client = app.test_client()
    with client.session_transaction() as session:
        session["logged_in"] = True
        session.permanent = True
    return client


def test_reflections_index_lists_available_weeks(journal_copy):
    _seed_reflection(journal_copy)
    client = _make_client(journal_copy)

    response = client.get("/app/reflections/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Available weekly reflections" in html
    assert 'href="/app/reflections/20260308"' in html


def test_reflections_detail_renders_week(journal_copy):
    _seed_reflection(journal_copy)
    client = _make_client(journal_copy)

    response = client.get("/app/reflections/20260308")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "weekly reflection" in html
    assert "week of Sunday March 8th" in html
    assert ">copy<" in html
    assert ">download PDF<" in html


def test_reflections_detail_canonicalizes_to_sunday(journal_copy):
    _seed_reflection(journal_copy)
    client = _make_client(journal_copy)

    response = client.get("/app/reflections/20260310")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/app/reflections/20260308")


def test_reflections_missing_week_returns_plain_text_404(journal_copy):
    client = _make_client(journal_copy)

    response = client.get("/app/reflections/20260315")

    assert response.status_code == 404
    assert response.mimetype == "text/plain"
    assert "Reflection not found" in response.get_data(as_text=True)


def test_reflections_raw_returns_markdown(journal_copy):
    _seed_reflection(journal_copy)
    client = _make_client(journal_copy)

    response = client.get("/app/reflections/20260308/raw")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert response.headers["Content-Type"] == "text/markdown; charset=utf-8"
    assert text.startswith("---\ntype: weekly_reflection")


def test_reflections_pdf_returns_attachment(journal_copy):
    _seed_reflection(journal_copy)
    client = _make_client(journal_copy)

    response = client.get("/app/reflections/20260308/pdf")

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert (
        response.headers["Content-Disposition"]
        == 'attachment; filename="reflection-20260308.pdf"'
    )
    assert response.data.startswith(b"%PDF")


def test_reflections_pdf_rejects_remote_assets(journal_copy):
    _seed_reflection(
        journal_copy,
        """---
type: weekly_reflection
week: 20260308
generated: 2026-03-10T19:00:00Z
model: openai/gpt-5
sources:
  newsletters: 0
  activities: 0
  decisions: 0
  followups: 0
  todos: 0
  relationship_signals: 0
gaps: []
---

![remote](https://example.com/reflection.png)
""",
    )
    client = _make_client(journal_copy)

    with (
        patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("network disabled during reflection pdf render"),
        ),
        patch("solstone.apps.reflections.routes.default_url_fetcher") as mock_fetcher,
    ):
        response = client.get("/app/reflections/20260308/pdf")

    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert response.data.startswith(b"%PDF")
    mock_fetcher.assert_not_called()


def test_reflections_stats_returns_month_counts(journal_copy):
    _seed_reflection(journal_copy)
    client = _make_client(journal_copy)

    response = client.get("/app/reflections/api/stats/202603")

    assert response.status_code == 200
    assert response.get_json() == {"20260308": 1}
