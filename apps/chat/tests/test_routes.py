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

from convey import create_app
from convey.chat_stream import append_chat_event


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
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(dst.resolve()))
    return dst


def _make_env(journal, monkeypatch) -> ChatTestEnv:
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))
    app = create_app(str(journal))
    app.config["TESTING"] = True
    client = app.test_client()
    with client.session_transaction() as session:
        session["logged_in"] = True
        session.permanent = True
    return ChatTestEnv(client=client, journal=journal)


def _set_today(monkeypatch, day: str) -> None:
    import apps.chat.routes as chat_routes

    class FixedDate(date):
        @classmethod
        def today(cls) -> date:
            return cls(int(day[:4]), int(day[4:6]), int(day[6:8]))

    monkeypatch.setattr(chat_routes, "date", FixedDate)


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
    assert 'id="chatComposerForm"' in html


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


def test_past_day_hides_composer(journal_copy, monkeypatch):
    today = "20990102"
    past_day = "20990101"
    _set_today(monkeypatch, today)
    env = _make_env(journal_copy, monkeypatch)

    html = env.client.get(f"/app/chat/{past_day}").get_data(as_text=True)

    assert 'id="chatComposerForm"' not in html
    assert "past-day view" in html


def test_today_shows_composer(journal_copy, monkeypatch):
    today = "20990102"
    _set_today(monkeypatch, today)
    env = _make_env(journal_copy, monkeypatch)

    html = env.client.get(f"/app/chat/{today}").get_data(as_text=True)

    assert 'id="chatComposerForm"' in html
