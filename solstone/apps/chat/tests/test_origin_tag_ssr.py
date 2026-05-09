# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from solstone.apps.chat.routes import _format_origin_time
from solstone.convey import create_app
from solstone.convey.sol_initiated.copy import (
    CHAT_ORIGIN_SUPERSEDED_SUFFIX,
    CHAT_ORIGIN_TAG_WITH_TALENT,
    CHAT_ORIGIN_TAG_WITHOUT_TALENT,
    KIND_SOL_CHAT_REQUEST,
    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
)


def _ms(hour: int, minute: int) -> int:
    return int(datetime(2099, 1, 4, hour, minute).timestamp() * 1000)


@pytest.fixture
def chat_env(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    config_dir = journal / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "journal.json").write_text(
        json.dumps(
            {
                "setup": {"completed_at": "2026-05-09T00:00:00Z"},
                "convey": {"trust_localhost": True},
                "identity": {"preferred": "Owner"},
                "agent": {"name": "sol"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
    app = create_app(str(journal))
    app.config["TESTING"] = True
    client = app.test_client()
    with client.session_transaction() as session:
        session["logged_in"] = True
        session.permanent = True
    return client, journal


def _write_events(journal: Path, day: str, events: list[dict[str, Any]]) -> None:
    chat_dir = journal / "chronicle" / day / "chat" / "090000_300"
    chat_dir.mkdir(parents=True)
    (chat_dir / "chat.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )


def _sol_request(request_id: str, *, trigger_talent: str | None) -> dict[str, Any]:
    return {
        "kind": KIND_SOL_CHAT_REQUEST,
        "ts": _ms(9, 0),
        "request_id": request_id,
        "summary": "notice",
        "message": "",
        "category": "notice",
        "dedupe": "notice-key",
        "dedupe_window": "24h",
        "since_ts": 123,
        "trigger_talent": trigger_talent,
    }


def _sol_message(text: str, minute: int = 1) -> dict[str, Any]:
    return {
        "kind": "sol_message",
        "ts": _ms(9, minute),
        "use_id": "use-1",
        "text": text,
        "notes": "",
        "requested_target": None,
        "requested_task": None,
    }


def _render(client, day: str) -> str:
    response = client.get(f"/app/chat/{day}")
    assert response.status_code == 200
    return response.get_data(as_text=True)


def _transcript_html(html: str) -> str:
    start = html.index('<ol id="chatTranscript"')
    end = html.index("</ol>", start) + len("</ol>")
    return html[start:end]


def test_origin_tag_renders_with_trigger_talent(chat_env):
    client, journal = chat_env
    day = "20990104"
    _write_events(
        journal,
        day,
        [_sol_request("r1", trigger_talent="reflection"), _sol_message("hello")],
    )

    html = _transcript_html(_render(client, day))

    assert (
        CHAT_ORIGIN_TAG_WITH_TALENT.format(
            trigger_talent="reflection",
            time=_format_origin_time(_ms(9, 0)),
        )
        in html
    )
    assert "trigger talent" in html
    assert "notice-key" in html


def test_origin_tag_renders_without_trigger_talent(chat_env):
    client, journal = chat_env
    day = "20990105"
    _write_events(
        journal, day, [_sol_request("r1", trigger_talent=None), _sol_message("hello")]
    )

    html = _transcript_html(_render(client, day))

    assert (
        CHAT_ORIGIN_TAG_WITHOUT_TALENT.format(time=_format_origin_time(_ms(9, 0)))
        in html
    )
    assert "(from" not in html


def test_origin_tag_appends_superseded_marker(chat_env):
    client, journal = chat_env
    day = "20990106"
    _write_events(
        journal,
        day,
        [
            _sol_request("r1", trigger_talent="reflection"),
            _sol_message("hello"),
            {
                "kind": KIND_SOL_CHAT_REQUEST_SUPERSEDED,
                "ts": _ms(9, 2),
                "request_id": "r1",
                "replaced_by": "r2",
            },
        ],
    )

    html = _transcript_html(_render(client, day))

    assert (
        CHAT_ORIGIN_SUPERSEDED_SUFFIX.format(time=_format_origin_time(_ms(9, 2)))
        in html
    )


def test_unanswered_request_does_not_render_origin_tag(chat_env):
    client, journal = chat_env
    day = "20990107"
    _write_events(journal, day, [_sol_request("r1", trigger_talent="reflection")])

    html = _transcript_html(_render(client, day))

    assert "chat-origin-tag" not in html


def test_unmatched_supersede_does_not_render_origin_tag(chat_env):
    client, journal = chat_env
    day = "20990108"
    _write_events(
        journal,
        day,
        [
            {
                "kind": KIND_SOL_CHAT_REQUEST_SUPERSEDED,
                "ts": _ms(9, 2),
                "request_id": "r1",
                "replaced_by": "r2",
            }
        ],
    )

    html = _transcript_html(_render(client, day))

    assert "chat-origin-tag" not in html
