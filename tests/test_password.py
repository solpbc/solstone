# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for password hashing: login, migration, and settings API."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from werkzeug.security import check_password_hash

from convey import create_app


@pytest.fixture
def journal_dir(tmp_path, monkeypatch):
    """Copy test fixture to temp dir for mutation tests."""
    src = Path(__file__).resolve().parent / "fixtures" / "journal"
    dst = tmp_path / "journal"
    shutil.copytree(src, dst)
    monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(dst))
    return dst


@pytest.fixture
def client(journal_dir):
    app = create_app(str(journal_dir))
    app.config["TESTING"] = True
    return app.test_client()


def _read_config(journal_dir):
    return json.loads((journal_dir / "config" / "journal.json").read_text())


class TestLogin:
    def test_correct_password(self, client):
        resp = client.post("/login", data={"password": "test123"})
        assert resp.status_code == 302

    def test_wrong_password(self, client):
        resp = client.post("/login", data={"password": "wrong"})
        assert resp.status_code == 200
        assert b"Invalid password" in resp.data

    def test_no_password_configured(self, journal_dir, monkeypatch):
        config = _read_config(journal_dir)
        config["convey"].pop("password_hash", None)
        config["convey"].pop("password", None)
        (journal_dir / "config" / "journal.json").write_text(
            json.dumps(config, indent=2)
        )
        app = create_app(str(journal_dir))
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/login")
        assert b"sol password set" in resp.data


class TestMigration:
    def test_plaintext_migrated_to_hash(self, tmp_path, monkeypatch):
        """Plaintext password is hashed and old key removed on app creation."""
        src = Path(__file__).resolve().parent / "fixtures" / "journal"
        dst = tmp_path / "journal"
        shutil.copytree(src, dst)
        config_path = dst / "config" / "journal.json"
        config = json.loads(config_path.read_text())
        config["convey"].pop("password_hash", None)
        config["convey"]["password"] = "migrate-me"
        config_path.write_text(json.dumps(config, indent=2))
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(dst))

        create_app(str(dst))

        config = json.loads(config_path.read_text())
        assert "password" not in config["convey"]
        assert "password_hash" in config["convey"]
        assert check_password_hash(config["convey"]["password_hash"], "migrate-me")

    def test_empty_password_removed(self, tmp_path, monkeypatch):
        """Empty plaintext password is removed, not hashed."""
        src = Path(__file__).resolve().parent / "fixtures" / "journal"
        dst = tmp_path / "journal"
        shutil.copytree(src, dst)
        config_path = dst / "config" / "journal.json"
        config = json.loads(config_path.read_text())
        config["convey"].pop("password_hash", None)
        config["convey"]["password"] = ""
        config_path.write_text(json.dumps(config, indent=2))
        monkeypatch.setenv("_SOLSTONE_JOURNAL_OVERRIDE", str(dst))

        create_app(str(dst))

        config = json.loads(config_path.read_text())
        assert "password" not in config["convey"]
        assert "password_hash" not in config["convey"]

    def test_already_migrated_skipped(self, journal_dir):
        """If password_hash exists, migration is a no-op."""
        config_before = _read_config(journal_dir)
        hash_before = config_before["convey"]["password_hash"]

        create_app(str(journal_dir))

        config_after = _read_config(journal_dir)
        assert config_after["convey"]["password_hash"] == hash_before


class TestSettingsAPI:
    def test_get_config_strips_password(self, client):
        """GET /app/settings/api/config must not return password or password_hash."""
        resp = client.get("/app/settings/api/config")
        data = resp.get_json()
        convey = data.get("convey", {})
        assert "password" not in convey
        assert "password_hash" not in convey
        assert convey.get("has_password") is True

    def test_put_hashes_password(self, client, journal_dir):
        """PUT with convey.password hashes before writing to disk."""
        resp = client.put(
            "/app/settings/api/config",
            json={"section": "convey", "data": {"password": "new-secret"}},
            content_type="application/json",
        )
        assert resp.status_code == 200
        config = _read_config(journal_dir)
        assert "password" not in config["convey"]
        assert check_password_hash(config["convey"]["password_hash"], "new-secret")

    def test_put_empty_password_skipped(self, client, journal_dir):
        """PUT with empty password does not overwrite existing hash."""
        config_before = _read_config(journal_dir)
        hash_before = config_before["convey"]["password_hash"]

        resp = client.put(
            "/app/settings/api/config",
            json={"section": "convey", "data": {"password": ""}},
            content_type="application/json",
        )
        assert resp.status_code == 200
        config_after = _read_config(journal_dir)
        assert config_after["convey"]["password_hash"] == hash_before
