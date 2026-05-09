# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json

import pytest

from solstone.convey import create_app
from solstone.convey.sol_initiated.copy import (
    KIND_OWNER_CHAT_DISMISSED,
    KIND_OWNER_CHAT_OPEN,
    KIND_SOL_CHAT_REQUEST,
    KIND_SOL_CHAT_REQUEST_SUPERSEDED,
)


@pytest.fixture
def chat_client(tmp_path, monkeypatch):
    journal = tmp_path / "journal"
    config_dir = journal / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "journal.json").write_text(
        json.dumps(
            {
                "setup": {"completed_at": "2026-05-09T00:00:00Z"},
                "convey": {"trust_localhost": True},
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
    return client


def test_live_append_origin_tag_script_handles_sol_initiated_events(chat_client):
    response = chat_client.get("/app/chat/20990109")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "let pendingSolChatRequest = null;" in html
    assert "renderOriginTag" in html
    assert "origin: origin" in html
    assert "item.dataset.requestId = event.origin.request_id;" in html
    assert "const supersededRequestId = msg.request_id || '';" in html
    assert KIND_SOL_CHAT_REQUEST in html
    assert KIND_SOL_CHAT_REQUEST_SUPERSEDED in html
    assert KIND_OWNER_CHAT_OPEN in html
    assert KIND_OWNER_CHAT_DISMISSED in html
