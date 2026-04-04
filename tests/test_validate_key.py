# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import think.providers
import think.providers.anthropic
import think.providers.google
import think.providers.openai
from convey import create_app
from think.providers import validate_key


@pytest.fixture
def settings_client(tmp_path, monkeypatch):
    src = Path(__file__).resolve().parent / "fixtures" / "journal"
    journal = tmp_path / "journal"
    shutil.copytree(src, journal, symlinks=True)
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(journal))

    app = create_app(str(journal))
    app.config["TESTING"] = True
    return app.test_client(), journal


def test_validate_key_anthropic_success():
    client = Mock()
    client.models.list.return_value = [Mock()]

    with patch("anthropic.Anthropic", return_value=client) as mock_cls:
        result = think.providers.anthropic.validate_key("test-key")

    assert result == {"valid": True}
    mock_cls.assert_called_once_with(api_key="test-key", timeout=10)


def test_validate_key_anthropic_auth_error():
    client = Mock()
    client.models.list.side_effect = Exception("invalid x-api-key")

    with patch("anthropic.Anthropic", return_value=client):
        result = think.providers.anthropic.validate_key("bad-key")

    assert result["valid"] is False
    assert "invalid x-api-key" in result["error"]


def test_validate_key_openai_success():
    client = Mock()
    client.models.list.return_value = [Mock()]

    with patch("openai.OpenAI", return_value=client) as mock_cls:
        result = think.providers.openai.validate_key("test-key")

    assert result == {"valid": True}
    mock_cls.assert_called_once_with(api_key="test-key", timeout=10)


def test_validate_key_openai_auth_error():
    client = Mock()
    client.models.list.side_effect = Exception("Incorrect API key")

    with patch("openai.OpenAI", return_value=client):
        result = think.providers.openai.validate_key("bad-key")

    assert result["valid"] is False
    assert "Incorrect API key" in result["error"]


def test_validate_key_google_success():
    client = Mock()
    client.models.list.return_value = [Mock()]

    with patch("think.providers.google.genai.Client", return_value=client) as mock_cls:
        result = think.providers.google.validate_key("test-key")

    assert result == {"valid": True}
    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["api_key"] == "test-key"


def test_validate_key_google_auth_error():
    client = Mock()
    client.models.list.side_effect = Exception("API key not valid")

    with patch("think.providers.google.genai.Client", return_value=client):
        result = think.providers.google.validate_key("bad-key")

    assert result["valid"] is False
    assert "API key not valid" in result["error"]


def test_validate_key_dispatcher_success():
    with patch("think.providers.google.validate_key", return_value={"valid": True}):
        result = validate_key("google", "test-key")

    assert result == {"valid": True}


def test_validate_key_dispatcher_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        validate_key("bogus", "test-key")


def test_validate_key_timeout():
    """Validate that timeout exceptions are caught and reported."""
    client = Mock()
    client.models.list.side_effect = TimeoutError("Connection timed out")

    with patch("openai.OpenAI", return_value=client):
        result = think.providers.openai.validate_key("test-key")

    assert result["valid"] is False
    assert "timed out" in result["error"]


def test_update_config_saves_key_validation(settings_client):
    client, journal = settings_client

    with patch("think.providers.validate_key", return_value={"valid": False, "error": "bad key"}):
        response = client.put(
            "/app/settings/api/config",
            json={"section": "env", "data": {"GOOGLE_API_KEY": "bad-key"}},
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["key_validation"]["google"]["valid"] is False
    assert payload["key_validation"]["google"]["error"] == "bad key"
    assert "timestamp" in payload["key_validation"]["google"]

    config = json.loads((journal / "config" / "journal.json").read_text())
    assert config["providers"]["auth"]["google"] == "api_key"
    assert config["providers"]["key_validation"]["google"]["valid"] is False


def test_update_config_clears_key_validation(settings_client):
    client, journal = settings_client
    config_path = journal / "config" / "journal.json"
    config = json.loads(config_path.read_text())
    config.setdefault("env", {})["GOOGLE_API_KEY"] = "existing-key"
    config.setdefault("providers", {}).setdefault("auth", {})["google"] = "api_key"
    config["providers"]["key_validation"] = {
        "google": {"valid": True, "timestamp": "2026-01-01T00:00:00+00:00"}
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    response = client.put(
        "/app/settings/api/config",
        json={"section": "env", "data": {"GOOGLE_API_KEY": ""}},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert "google" not in payload["key_validation"]

    saved = json.loads(config_path.read_text())
    assert saved["providers"]["auth"]["google"] == "platform"
    assert "google" not in saved["providers"]["key_validation"]


def test_get_providers_includes_key_validation(settings_client):
    client, journal = settings_client
    config_path = journal / "config" / "journal.json"
    config = json.loads(config_path.read_text())
    config.setdefault("providers", {})["key_validation"] = {
        "openai": {"valid": True, "timestamp": "2026-01-01T00:00:00+00:00"}
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    response = client.get("/app/settings/api/providers")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["key_validation"]["openai"]["valid"] is True


def test_validate_all_keys_endpoint(settings_client):
    client, journal = settings_client
    config_path = journal / "config" / "journal.json"
    config = json.loads(config_path.read_text())
    config.setdefault("env", {})["GOOGLE_API_KEY"] = "google-key"
    config["env"]["OPENAI_API_KEY"] = "openai-key"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def fake_validate(provider: str, api_key: str) -> dict:
        return {"valid": provider == "google", "error": "" if provider == "google" else "bad key"}

    with patch("think.providers.validate_key", side_effect=fake_validate):
        response = client.post("/app/settings/api/validate-keys")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["key_validation"]["google"]["valid"] is True
    assert payload["key_validation"]["openai"]["valid"] is False
    assert "timestamp" in payload["key_validation"]["google"]

    saved = json.loads(config_path.read_text())
    assert set(saved["providers"]["key_validation"]) == {"google", "openai"}
