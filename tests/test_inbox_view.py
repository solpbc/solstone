"""Tests for dream inbox view integration with think.messages module."""

import json
from unittest.mock import patch

import pytest
from flask import Flask

from dream.views.inbox import bp


@pytest.fixture
def app():
    """Create test Flask app with inbox blueprint."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(bp)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@patch("dream.views.inbox.render_template")
def test_inbox_page(mock_render, client):
    """Test that inbox page route exists."""
    mock_render.return_value = "mocked template"
    response = client.get("/inbox")
    assert response.status_code == 200
    mock_render.assert_called_once_with("inbox.html", active="inbox")


@patch("dream.views.inbox.messages.list_messages")
def test_get_messages_uses_think_module(mock_list_messages, client):
    """Test that get_messages uses think.messages.list_messages."""
    mock_list_messages.return_value = [
        {
            "id": "msg_123",
            "timestamp": 1736000000000,
            "from": {"type": "agent", "id": "test"},
            "body": "Test message",
            "status": "unread",
        }
    ]

    response = client.get("/inbox/api/messages?status=active")
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data["total"] == 1
    assert data["unread_count"] == 1
    assert len(data["messages"]) == 1

    mock_list_messages.assert_called_once_with("active")


@patch("dream.views.inbox.messages.get_message")
def test_get_message_uses_think_module(mock_get_message, client):
    """Test that get_message uses think.messages.get_message."""
    mock_get_message.return_value = {
        "id": "msg_123",
        "timestamp": 1736000000000,
        "from": {"type": "agent", "id": "test"},
        "body": "Test message",
        "status": "unread",
    }

    response = client.get("/inbox/api/message/msg_123")
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data["id"] == "msg_123"
    assert "body_html" in data

    mock_get_message.assert_called_once_with("msg_123")


@patch("dream.views.inbox.messages.mark_read")
def test_mark_read_uses_think_module(mock_mark_read, client):
    """Test that mark_read uses think.messages.mark_read."""
    mock_mark_read.return_value = True

    response = client.post("/inbox/api/message/msg_123/read")
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data["success"] is True

    mock_mark_read.assert_called_once_with("msg_123")


@patch("dream.views.inbox.messages.archive_message")
def test_archive_uses_think_module(mock_archive, client):
    """Test that archive_message uses think.messages.archive_message."""
    mock_archive.return_value = True

    response = client.post("/inbox/api/message/msg_123/archive")
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data["success"] is True

    mock_archive.assert_called_once_with("msg_123")


@patch("dream.views.inbox.messages.unarchive_message")
def test_unarchive_uses_think_module(mock_unarchive, client):
    """Test that unarchive_message uses think.messages.unarchive_message."""
    mock_unarchive.return_value = True

    response = client.post("/inbox/api/message/msg_123/unarchive")
    assert response.status_code == 200

    data = json.loads(response.data)
    assert data["success"] is True

    mock_unarchive.assert_called_once_with("msg_123")


@patch("dream.views.inbox.messages.list_messages")
def test_handles_runtime_error(mock_list_messages, client):
    """Test that view properly handles RuntimeError from think.messages."""
    mock_list_messages.side_effect = RuntimeError("JOURNAL_PATH not set")

    response = client.get("/inbox/api/messages")
    assert response.status_code == 500

    data = json.loads(response.data)
    assert "JOURNAL_PATH not set" in data["error"]
