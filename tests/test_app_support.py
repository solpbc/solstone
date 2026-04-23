# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for support app routes."""

import pytest


@pytest.fixture
def support_client():
    """Create a Flask test client with support blueprint."""
    from flask import Flask

    from apps.support.routes import support_bp

    app = Flask(__name__)
    app.register_blueprint(support_bp)
    yield app.test_client()


class _TicketsClient:
    def __init__(self, tickets=None, error: Exception | None = None):
        self.tickets = tickets or []
        self.error = error

    def list_tickets(self, *, status=None):
        if self.error:
            raise self.error
        return self.tickets


def test_badge_count_enabled_empty(support_client, monkeypatch):
    monkeypatch.setattr("apps.support.routes._enabled", lambda: True)
    monkeypatch.setattr("apps.support.routes._get_client", lambda: _TicketsClient())

    resp = support_client.get("/app/support/api/badge-count")

    assert resp.status_code == 200
    assert resp.get_json() == {"count": 0}


def test_badge_count_disabled_returns_403(support_client, monkeypatch):
    monkeypatch.setattr("apps.support.routes._enabled", lambda: False)

    resp = support_client.get("/app/support/api/badge-count")

    assert resp.status_code == 403
    assert resp.get_json() == {"error": "Support is not enabled"}


def test_badge_count_error_returns_500(support_client, monkeypatch):
    monkeypatch.setattr("apps.support.routes._enabled", lambda: True)
    monkeypatch.setattr(
        "apps.support.routes._get_client",
        lambda: _TicketsClient(error=RuntimeError("simulated")),
    )

    resp = support_client.get("/app/support/api/badge-count")

    assert resp.status_code == 500
    assert "error" in resp.get_json()
