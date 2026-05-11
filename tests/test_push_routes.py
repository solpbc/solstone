# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import pytest

from solstone.convey import create_app
from solstone.think.push.runtime import stop_all_push_runtime


@pytest.fixture
def push_app(journal_copy):
    app = create_app(str(journal_copy))
    app.config["TESTING"] = True
    yield app
    stop_all_push_runtime()


@pytest.fixture
def push_client(push_app):
    return push_app.test_client()


def test_register_push_device_happy_path(push_client, monkeypatch):
    monkeypatch.setattr("solstone.convey.push.register_device", lambda **kwargs: 2)

    response = push_client.post(
        "/api/push/register",
        json={
            "device_token": "A" * 64,
            "bundle_id": "org.solpbc.solstone-swift",
            "environment": "development",
            "platform": "ios",
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {"registered": True, "device_count": 2}


def test_register_push_device_rejects_non_object(push_client):
    response = push_client.post("/api/push/register", json=["bad"])

    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "I couldn't read that JSON request."
    assert data["reason_code"] == "invalid_json_request"
    assert data["detail"] == "request body must be a JSON object"


def test_register_push_device_validates_fields(push_client):
    response = push_client.post("/api/push/register", json={"device_token": "x"})

    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "I couldn't use that push request."
    assert data["reason_code"] == "push_request_invalid"
    assert data["detail"] == "bundle_id is required"


def test_delete_push_device_happy_path(push_client, monkeypatch):
    monkeypatch.setattr("solstone.convey.push.remove_device", lambda token: True)
    monkeypatch.setattr("solstone.convey.push.load_devices", lambda: [{"token": "a"}])

    response = push_client.delete("/api/push/register", json={"device_token": "a" * 64})

    assert response.status_code == 200
    assert response.get_json() == {"removed": True, "device_count": 1}


def test_status_masks_tokens(push_client, monkeypatch):
    monkeypatch.setattr("solstone.convey.push.is_configured", lambda: True)
    monkeypatch.setattr(
        "solstone.convey.push.load_devices",
        lambda: [
            {
                "token": "a" * 64,
                "bundle_id": "org.solpbc.solstone-swift",
                "environment": "development",
                "platform": "ios",
                "registered_at": 2,
            }
        ],
    )
    monkeypatch.setattr(
        "solstone.convey.push.status_view",
        lambda device: {
            "token_suffix": "...aaaa",
            "bundle_id": device["bundle_id"],
            "environment": device["environment"],
            "platform": device["platform"],
            "registered_at": "2024-04-19T12:00:00Z",
        },
    )

    response = push_client.get("/api/push/status")

    assert response.status_code == 200
    assert response.get_json() == {
        "configured": True,
        "device_count": 1,
        "devices": [
            {
                "token_suffix": "...aaaa",
                "bundle_id": "org.solpbc.solstone-swift",
                "environment": "development",
                "platform": "ios",
                "registered_at": "2024-04-19T12:00:00Z",
            }
        ],
    }


def test_push_test_requires_configuration(push_client, monkeypatch):
    monkeypatch.setattr("solstone.convey.push.is_configured", lambda: False)

    response = push_client.post("/api/push/test")

    assert response.status_code == 503
    data = response.get_json()
    assert data["error"] == "I couldn't use that feature because it isn't enabled."
    assert data["reason_code"] == "feature_unavailable"
    assert data["detail"] == "push not configured"


def test_push_test_validates_category(push_client, monkeypatch):
    monkeypatch.setattr("solstone.convey.push.is_configured", lambda: True)

    response = push_client.post("/api/push/test", json={"category": "BAD"})

    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "I couldn't use that push request."
    assert data["reason_code"] == "push_request_invalid"
    assert data["detail"] == "category must be a known push category"


def test_push_test_happy_path(push_client, monkeypatch):
    monkeypatch.setattr("solstone.convey.push.is_configured", lambda: True)
    monkeypatch.setattr(
        "solstone.convey.push.triggers.send_agent_alert",
        lambda *, title, body, context_id, route: (1, 0),
    )

    response = push_client.post(
        "/api/push/test", json={"title": "Alert", "body": "Body"}
    )

    assert response.status_code == 200
    assert response.get_json() == {"sent": 1, "failed": 0}
