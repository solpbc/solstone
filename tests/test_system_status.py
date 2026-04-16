# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the /api/system/status endpoint."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import convey

from convey import create_app

system_mod = convey.system


@pytest.fixture(autouse=True)
def _temp_journal(monkeypatch, tmp_path):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(tmp_path))
    journal = tmp_path
    (journal / "config").mkdir(parents=True, exist_ok=True)
    config = {
        "convey": {"password_hash": "", "trust_localhost": True},
        "setup": {"completed_at": 1},
    }
    (journal / "config" / "journal.json").write_text(json.dumps(config))
    app = create_app(str(journal))
    return app.test_client()


class TestSystemStatusEndpoint:
    def test_returns_valid_json_shape(self, client):
        with patch.object(system_mod, "list_observers", return_value=[]):
            resp = client.get("/api/system/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "version" in data
        assert "capture" in data
        assert "ok" in data
        assert "current" in data["version"]
        assert "status" in data["capture"]
        assert "observers" in data["capture"]

    def test_no_observers(self, client):
        with patch.object(system_mod, "list_observers", return_value=[]):
            data = client.get("/api/system/status").get_json()
        assert data["capture"]["status"] == "no_observers"
        assert data["capture"]["observers"] == []
        assert data["ok"] is True

    def test_active_observer(self, client):
        now = int(time.time() * 1000)
        observers = [{"name": "phone", "last_seen": now - 5000, "enabled": True}]
        with patch.object(system_mod, "list_observers", return_value=observers):
            data = client.get("/api/system/status").get_json()
        assert data["capture"]["status"] == "active"
        assert data["ok"] is True

    def test_stale_observer(self, client):
        now = int(time.time() * 1000)
        observers = [{"name": "phone", "last_seen": now - 60000, "enabled": True}]
        with patch.object(system_mod, "list_observers", return_value=observers):
            data = client.get("/api/system/status").get_json()
        assert data["capture"]["status"] == "stale"
        assert data["ok"] is False

    def test_offline_observer(self, client):
        now = int(time.time() * 1000)
        observers = [{"name": "phone", "last_seen": now - 300000, "enabled": True}]
        with patch.object(system_mod, "list_observers", return_value=observers):
            data = client.get("/api/system/status").get_json()
        assert data["capture"]["status"] == "offline"
        assert data["ok"] is False

    def test_revoked_observers_excluded(self, client):
        now = int(time.time() * 1000)
        observers = [
            {
                "name": "phone",
                "last_seen": now - 5000,
                "enabled": True,
                "revoked": True,
            },
        ]
        with patch.object(system_mod, "list_observers", return_value=observers):
            data = client.get("/api/system/status").get_json()
        assert data["capture"]["status"] == "no_observers"

    def test_worst_of_multiple_observers(self, client):
        now = int(time.time() * 1000)
        observers = [
            {"name": "phone", "last_seen": now - 5000, "enabled": True},
            {"name": "laptop", "last_seen": now - 60000, "enabled": True},
        ]
        with patch.object(system_mod, "list_observers", return_value=observers):
            data = client.get("/api/system/status").get_json()
        # At least one is active, so overall is active
        assert data["capture"]["status"] == "active"

    def test_version_github_failure_graceful(self, client):
        with (
            patch.object(system_mod, "list_observers", return_value=[]),
            patch.object(system_mod, "_check_latest_version", return_value=None),
        ):
            data = client.get("/api/system/status").get_json()
        assert "current" in data["version"]
        # No "latest" or "update_available" when GitHub fails and no cache
        assert (
            data["version"].get("update_available") is None
            or "latest" not in data["version"]
        )

    def test_version_with_update_available(self, client):
        with (
            patch.object(system_mod, "list_observers", return_value=[]),
            patch.object(
                system_mod, "_check_latest_version", return_value={"latest": "99.0.0"}
            ),
            patch.object(system_mod, "collect_version", return_value="0.1.0"),
        ):
            data = client.get("/api/system/status").get_json()
        assert data["version"]["current"] == "0.1.0"
        assert data["version"]["latest"] == "99.0.0"
        assert data["version"]["update_available"] is True


class TestCaptureHealthDerivation:
    """Unit tests for _get_capture_health logic."""

    def test_no_last_seen_is_offline(self):
        with patch.object(
            system_mod, "list_observers", return_value=[{"name": "x", "enabled": True}]
        ):
            result = system_mod._get_capture_health()
        assert result["observers"][0]["status"] == "offline"

    def test_disabled_observers_excluded(self):
        now = int(time.time() * 1000)
        with patch.object(
            system_mod,
            "list_observers",
            return_value=[{"name": "x", "last_seen": now, "enabled": False}],
        ):
            result = system_mod._get_capture_health()
        assert result["status"] == "no_observers"
