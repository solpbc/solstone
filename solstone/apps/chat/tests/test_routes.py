# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from solstone.convey import create_app
from solstone.convey.chat_stream import append_chat_event, read_chat_events
from solstone.convey.sol_initiated.copy import (
    CATEGORIES,
    KIND_OWNER_CHAT_OPEN,
    KIND_SOL_CHAT_REQUEST,
    SURFACE_CONVEY,
)


def _ms(year: int, month: int, day: int, hour: int, minute: int) -> int:
    return int(datetime(year, month, day, hour, minute).timestamp() * 1000)


@dataclass
class ChatTestEnv:
    client: Any
    journal: Any


@pytest.fixture
def journal_copy(tmp_path, monkeypatch):
    src = Path("tests/fixtures/journal").resolve()
    dst = tmp_path / "journal"
    copytree_tracked(src, dst)
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(dst.resolve()))
    return dst


def _make_env(journal, monkeypatch) -> ChatTestEnv:
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(journal))
    app = create_app(str(journal))
    app.config["TESTING"] = True
    client = app.test_client()
    with client.session_transaction() as session:
        session["logged_in"] = True
        session.permanent = True
    return ChatTestEnv(client=client, journal=journal)


def _set_today(monkeypatch, day: str) -> None:
    import solstone.apps.chat.routes as chat_routes

    class FixedDate(date):
        @classmethod
        def today(cls) -> date:
            return cls(int(day[:4]), int(day[4:6]), int(day[6:8]))

    monkeypatch.setattr(chat_routes, "date", FixedDate)


def _set_chat_stream_now(
    monkeypatch, day: str, hour: int = 10, minute: int = 1
) -> None:
    monkeypatch.setattr(
        "solstone.convey.chat_stream.time.time",
        lambda: _ms(int(day[:4]), int(day[4:6]), int(day[6:8]), hour, minute) / 1000,
    )


def copytree_tracked(src: Path, dst: Path) -> None:
    result = subprocess.run(
        ["git", "ls-files", "."],
        cwd=str(src),
        capture_output=True,
        text=True,
        check=True,
    )
    for rel in result.stdout.splitlines():
        if not rel:
            continue
        src_file = src / rel
        dst_file = dst / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if src_file.is_symlink():
            os.symlink(os.readlink(src_file), dst_file)
        else:
            shutil.copy2(src_file, dst_file)


def _append_sol_request(day: str, request_id: str = "req") -> None:
    append_chat_event(
        KIND_SOL_CHAT_REQUEST,
        ts=_ms(int(day[:4]), int(day[4:6]), int(day[6:8]), 10, 0),
        request_id=request_id,
        summary="Notice this",
        message=None,
        category=CATEGORIES[0],
        dedupe=request_id,
        dedupe_window="24h",
        since_ts=1,
        trigger_talent="reflection",
    )


def test_chat_index_redirects_to_today(journal_copy, monkeypatch):
    today = "20990101"
    _set_today(monkeypatch, today)
    env = _make_env(journal_copy, monkeypatch)

    response = env.client.get("/app/chat/")

    assert response.status_code == 302
    assert response.headers["Location"].endswith(f"/app/chat/{today}")


def test_chat_day_renders_empty_state_for_today(journal_copy, monkeypatch):
    today = "20990101"
    _set_today(monkeypatch, today)
    env = _make_env(journal_copy, monkeypatch)

    response = env.client.get(f"/app/chat/{today}")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "no chat yet on this day" in html
    assert 'id="chatBarForm"' in html


def test_chat_day_renders_all_event_kinds(journal_copy, monkeypatch):
    day = "20990102"
    _set_today(monkeypatch, "20990103")
    env = _make_env(journal_copy, monkeypatch)
    append_chat_event(
        "owner_message",
        ts=_ms(2099, 1, 2, 9, 0),
        text="owner hello",
        app="chat",
        path=f"/app/chat/{day}",
        facet="work",
    )
    append_chat_event(
        "sol_message",
        ts=_ms(2099, 1, 2, 9, 1),
        use_id="use-1",
        text="sol reply",
        notes="full note",
        requested_target=None,
        requested_task=None,
    )
    append_chat_event(
        "talent_spawned",
        ts=_ms(2099, 1, 2, 9, 2),
        use_id="use-2",
        name="search",
        task="find updates",
        started_at=_ms(2099, 1, 2, 9, 2),
    )
    append_chat_event(
        "talent_finished",
        ts=_ms(2099, 1, 2, 9, 3),
        use_id="use-2",
        name="search",
        summary="done",
    )
    append_chat_event(
        "talent_errored",
        ts=_ms(2099, 1, 2, 9, 4),
        use_id="use-3",
        name="exec",
        reason="bad args",
    )
    append_chat_event(
        "chat_error",
        ts=_ms(2026, 4, 20, 9, 5),
        reason="network",
        use_id="use-4",
    )
    append_chat_event(
        "reflection_ready",
        ts=_ms(2099, 1, 2, 9, 6),
        day="20981228",
        url="/app/reflections/20981228",
    )

    response = env.client.get(f"/app/chat/{day}")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "owner hello" in html
    assert "sol reply" in html
    assert 'title="full note"' in html
    assert 'data-talent-use-id="use-2"' in html
    assert 'data-talent-use-id="use-3"' in html
    assert "weekly reflection ready" in html
    assert 'href="/app/reflections/20981228"' in html
    assert "chat had trouble" in html


def test_chat_day_emits_raw_talent_markdown_source_for_bootstrap(
    journal_copy, monkeypatch
):
    day = "20990102"
    _set_today(monkeypatch, "20990103")
    env = _make_env(journal_copy, monkeypatch)
    append_chat_event(
        "talent_finished",
        ts=_ms(2099, 1, 2, 9, 3),
        use_id="use-md-1",
        name="search",
        summary="**done**",
    )
    append_chat_event(
        "talent_errored",
        ts=_ms(2099, 1, 2, 9, 4),
        use_id="use-md-2",
        name="exec",
        reason="**bad args**",
    )

    response = env.client.get(f"/app/chat/{day}")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert (
        html.count(
            '<div class="chat-talent-card-detail '
            'chat-talent-card-detail--markdown" data-markdown="1">'
        )
        == 2
    )
    assert (
        '<div class="chat-talent-card-detail '
        'chat-talent-card-detail--markdown" data-markdown="1">**done**</div>'
    ) in html
    assert (
        '<div class="chat-talent-card-detail '
        'chat-talent-card-detail--markdown" data-markdown="1">**bad args**</div>'
    ) in html
    assert "<strong>done</strong>" not in html
    assert "<strong>bad args</strong>" not in html


def test_chat_event_anchor_ids_are_stable(journal_copy, monkeypatch):
    day = "20990102"
    _set_today(monkeypatch, "20990103")
    env = _make_env(journal_copy, monkeypatch)
    append_chat_event(
        "owner_message",
        ts=_ms(2099, 1, 2, 10, 0),
        text="first",
        app="chat",
        path=f"/app/chat/{day}",
        facet="work",
    )
    append_chat_event(
        "sol_message",
        ts=_ms(2099, 1, 2, 10, 1),
        use_id="use-5",
        text="second",
        notes="",
        requested_target=None,
        requested_task=None,
    )

    first = env.client.get(f"/app/chat/{day}").get_data(as_text=True)
    second = env.client.get(f"/app/chat/{day}").get_data(as_text=True)

    assert first.count('id="event-0"') == 1
    assert first.count('id="event-1"') == 1
    assert second.count('id="event-0"') == 1
    assert second.count('id="event-1"') == 1


def test_chat_time_separator_is_inserted_client_side(journal_copy, monkeypatch):
    day = "20990102"
    _set_today(monkeypatch, "20990103")
    env = _make_env(journal_copy, monkeypatch)
    append_chat_event(
        "owner_message",
        ts=_ms(2099, 1, 2, 8, 0),
        text="early",
        app="chat",
        path=f"/app/chat/{day}",
        facet="work",
    )
    append_chat_event(
        "sol_message",
        ts=_ms(2099, 1, 2, 8, 25),
        use_id="use-6",
        text="later",
        notes="",
        requested_target=None,
        requested_task=None,
    )

    html = env.client.get(f"/app/chat/{day}").get_data(as_text=True)

    assert "early" in html
    assert "later" in html
    assert "insertTimeSeparators(transcript);" in html


def test_chat_invalid_days_return_404(journal_copy, monkeypatch):
    _set_today(monkeypatch, "20990101")
    env = _make_env(journal_copy, monkeypatch)

    assert env.client.get("/app/chat/abcd1234").status_code == 404
    assert env.client.get("/app/chat/20260101extra").status_code == 404


def test_universal_chat_bar_renders_on_today_and_past_day(journal_copy, monkeypatch):
    today = "20990102"
    past_day = "20990101"
    _set_today(monkeypatch, today)
    env = _make_env(journal_copy, monkeypatch)

    today_html = env.client.get(f"/app/chat/{today}").get_data(as_text=True)
    past_html = env.client.get(f"/app/chat/{past_day}").get_data(as_text=True)

    for html in (today_html, past_html):
        assert 'id="chatBarForm"' in html
        assert "past-day view" not in html
        assert html.count('id="chatBarForm"') == 1


def test_chat_today_page_records_owner_chat_open_for_unresolved_request(
    journal_copy,
    monkeypatch,
):
    today = "20990102"
    _set_today(monkeypatch, today)
    _set_chat_stream_now(monkeypatch, today)
    env = _make_env(journal_copy, monkeypatch)
    _append_sol_request(today, "req")

    response = env.client.get(f"/app/chat/{today}")

    assert response.status_code == 200
    events = read_chat_events(today)
    assert events[-1]["kind"] == KIND_OWNER_CHAT_OPEN
    assert events[-1]["request_id"] == "req"
    assert events[-1]["surface"] == SURFACE_CONVEY


def test_chat_today_page_without_unresolved_request_writes_no_open(
    journal_copy,
    monkeypatch,
):
    today = "20990102"
    _set_today(monkeypatch, today)
    _set_chat_stream_now(monkeypatch, today)
    env = _make_env(journal_copy, monkeypatch)

    response = env.client.get(f"/app/chat/{today}")

    assert response.status_code == 200
    assert [
        event
        for event in read_chat_events(today)
        if event.get("kind") == KIND_OWNER_CHAT_OPEN
    ] == []


def test_chat_past_day_request_does_not_record_owner_chat_open(
    journal_copy,
    monkeypatch,
):
    today = "20990103"
    past_day = "20990102"
    _set_today(monkeypatch, today)
    _set_chat_stream_now(monkeypatch, today)
    env = _make_env(journal_copy, monkeypatch)
    _append_sol_request(past_day, "req")

    response = env.client.get(f"/app/chat/{past_day}")

    assert response.status_code == 200
    assert [
        event
        for event in read_chat_events(past_day)
        if event.get("kind") == KIND_OWNER_CHAT_OPEN
    ] == []


def test_chat_today_page_records_repeated_owner_chat_open(
    journal_copy,
    monkeypatch,
):
    today = "20990102"
    _set_today(monkeypatch, today)
    _set_chat_stream_now(monkeypatch, today)
    env = _make_env(journal_copy, monkeypatch)
    _append_sol_request(today, "req")

    first = env.client.get(f"/app/chat/{today}")
    second = env.client.get(f"/app/chat/{today}")

    assert first.status_code == 200
    assert second.status_code == 200
    opens = [
        event
        for event in read_chat_events(today)
        if event.get("kind") == KIND_OWNER_CHAT_OPEN
    ]
    assert len(opens) == 2
    assert {event["request_id"] for event in opens} == {"req"}


def test_chat_today_initial_render_excludes_newly_written_open(
    journal_copy,
    monkeypatch,
):
    today = "20990102"
    _set_today(monkeypatch, today)
    _set_chat_stream_now(monkeypatch, today)
    env = _make_env(journal_copy, monkeypatch)
    _append_sol_request(today, "req")

    response = env.client.get(f"/app/chat/{today}")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="event-0"' in html
    assert 'id="event-1"' not in html
    assert len(read_chat_events(today)) == 2
