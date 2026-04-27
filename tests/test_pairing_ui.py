# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import pytest

from convey import create_app


@pytest.fixture
def pairing_ui_client(journal_copy):
    app = create_app(str(journal_copy))
    app.config["TESTING"] = True
    client = app.test_client()
    with client.session_transaction() as session:
        session["logged_in"] = True
        session.permanent = True
    return client


def test_pairing_ui_smoke(pairing_ui_client):
    response = pairing_ui_client.get("/app/pairing/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "pair a phone" in body
    assert "/static/pairing-qr.js" in body
    assert "/static/pairing.js" in body
    assert "/static/pairing.css" in body
    for forbidden in ("keeper", "assistant", "record", "capture", "Capture"):
        assert forbidden not in body
