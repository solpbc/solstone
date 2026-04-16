import base64
import json

import pytest

from convey import create_app


def _read_config(journal_dir):
    return json.loads((journal_dir / "config" / "journal.json").read_text())


def _remove_password(journal_dir):
    config = _read_config(journal_dir)
    config["convey"].pop("password_hash", None)
    config["convey"].pop("password", None)
    config["convey"].pop("trust_localhost", None)
    config.pop("setup", None)
    (journal_dir / "config" / "journal.json").write_text(json.dumps(config, indent=2))


@pytest.fixture
def fresh_client(journal_copy):
    _remove_password(journal_copy)
    app = create_app(str(journal_copy))
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def configured_client(journal_copy):
    app = create_app(str(journal_copy))
    app.config["TESTING"] = True
    return app.test_client()


class TestInitDetection:
    def test_redirects_to_init_when_no_password(self, fresh_client):
        resp = fresh_client.get("/", headers={"X-Forwarded-For": "1.2.3.4"})
        assert resp.status_code == 302
        assert "/init" in resp.headers["Location"]

    def test_redirects_to_login_when_password_exists(self, configured_client):
        resp = configured_client.get("/", headers={"X-Forwarded-For": "1.2.3.4"})
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_init_page_renders(self, fresh_client):
        resp = fresh_client.get("/init")
        assert resp.status_code == 200
        assert b"set up solstone" in resp.data

    def test_init_redirects_when_configured(self, configured_client):
        resp = configured_client.get("/init")
        assert resp.status_code == 302


class TestInitValidateProvider:
    """Tests for the validate-only provider endpoint."""

    def test_validate_provider_valid_key(self, fresh_client, monkeypatch):
        monkeypatch.setattr(
            "think.providers.validate_key",
            lambda provider, key: {"valid": True},
        )
        resp = fresh_client.post(
            "/init/validate-provider",
            json={"key": "test-api-key-123"},
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["valid"] is True

    def test_validate_provider_invalid_key(self, fresh_client, monkeypatch):
        monkeypatch.setattr(
            "think.providers.validate_key",
            lambda provider, key: {"valid": False, "error": "Invalid key"},
        )
        resp = fresh_client.post(
            "/init/validate-provider",
            json={"key": "bad-key"},
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["valid"] is False
        assert data["error"] == "Invalid key"

    def test_validate_provider_no_config_write(
        self, fresh_client, journal_copy, monkeypatch
    ):
        """Validate endpoint must not write to config."""
        monkeypatch.setattr(
            "think.providers.validate_key",
            lambda provider, key: {"valid": True},
        )
        config_before = _read_config(journal_copy)
        fresh_client.post(
            "/init/validate-provider",
            json={"key": "test-api-key-123"},
            content_type="application/json",
        )
        config_after = _read_config(journal_copy)
        assert config_before == config_after


class TestInitObservers:
    """Tests for the observer list endpoint during onboarding."""

    def test_observers_no_password_required(self, fresh_client, monkeypatch):
        """Observers endpoint works without password_hash set."""
        monkeypatch.setattr(
            "apps.observer.utils.list_observers",
            lambda: [],
        )
        resp = fresh_client.get("/init/observers")
        assert resp.status_code == 200

    def test_observers_returns_list(self, fresh_client, monkeypatch):
        monkeypatch.setattr(
            "apps.observer.utils.list_observers",
            lambda: [
                {
                    "key": "abcd1234xxxx",
                    "name": "my-phone",
                    "created_at": 100,
                    "last_seen": None,
                    "last_segment": None,
                    "enabled": True,
                    "revoked": False,
                    "revoked_at": None,
                    "stats": {},
                },
                {
                    "key": "revoked1xxxx",
                    "name": "old-device",
                    "created_at": 50,
                    "last_seen": None,
                    "last_segment": None,
                    "enabled": False,
                    "revoked": True,
                    "revoked_at": 90,
                    "stats": {},
                },
            ],
        )
        resp = fresh_client.get("/init/observers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 1
        assert data[0]["name"] == "my-phone"
        assert data[0]["key_prefix"] == "abcd1234"


class TestInitFinalize:
    """Tests for the atomic finalize endpoint."""

    def test_finalize_saves_all_config(self, fresh_client, journal_copy):
        resp = fresh_client.post(
            "/init/finalize",
            json={
                "password": "securepass123",
                "name": "Jane Doe",
                "preferred": "Jane",
                "timezone": "America/Denver",
                "gemini_key": "test-api-key-123",
            },
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["redirect"] == "/"

        config = _read_config(journal_copy)
        # Password
        from werkzeug.security import check_password_hash

        assert check_password_hash(config["convey"]["password_hash"], "securepass123")
        assert config["convey"]["trust_localhost"] is True
        # Identity
        assert config["identity"]["name"] == "Jane Doe"
        assert config["identity"]["preferred"] == "Jane"
        assert config["identity"]["timezone"] == "America/Denver"
        # Provider
        assert config["env"]["GOOGLE_API_KEY"] == "test-api-key-123"
        # Setup
        assert "completed_at" in config["setup"]

    def test_finalize_password_required(self, fresh_client):
        resp = fresh_client.post(
            "/init/finalize",
            json={"name": "Jane"},
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "8 characters" in resp.get_json()["error"]

    def test_finalize_password_too_short(self, fresh_client):
        resp = fresh_client.post(
            "/init/finalize",
            json={"password": "short"},
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_finalize_minimal(self, fresh_client, journal_copy):
        """Finalize with only password — optional fields omitted."""
        resp = fresh_client.post(
            "/init/finalize",
            json={"password": "securepass123"},
            content_type="application/json",
        )
        assert resp.status_code == 200
        config = _read_config(journal_copy)
        assert "password_hash" in config["convey"]
        assert "completed_at" in config["setup"]
        # No gemini key written
        assert "GOOGLE_API_KEY" not in config.get("env", {})

    def test_finalize_auto_login(self, fresh_client, journal_copy):
        fresh_client.post(
            "/init/finalize",
            json={"password": "securepass123"},
            content_type="application/json",
        )
        resp = fresh_client.get("/", headers={"X-Forwarded-For": "1.2.3.4"})
        assert resp.status_code == 302
        location = resp.headers["Location"]
        assert "/login" not in location
        assert "/init" not in location

    def test_finalize_no_early_config_write(self, fresh_client, journal_copy):
        """Before finalize, config should have no password_hash or setup."""
        config = _read_config(journal_copy)
        assert "password_hash" not in config.get("convey", {})
        assert "setup" not in config or "completed_at" not in config.get("setup", {})

    def test_post_init_redirect(self, fresh_client, journal_copy):
        """After finalize, /init redirects away."""
        fresh_client.post(
            "/init/finalize",
            json={"password": "securepass123"},
            content_type="application/json",
        )
        resp = fresh_client.get("/init")
        assert resp.status_code == 302

    def test_finalize_with_retention_config(self, fresh_client, journal_copy):
        """Finalize with explicit retention config writes correct values."""
        resp = fresh_client.post(
            "/init/finalize",
            json={
                "password": "securepass123",
                "retention_mode": "processed",
                "retention_days": 30,
            },
            content_type="application/json",
        )
        assert resp.status_code == 200
        config = _read_config(journal_copy)
        assert config["retention"]["raw_media"] == "processed"
        assert config["retention"]["raw_media_days"] is None

    def test_finalize_default_retention(self, fresh_client, journal_copy):
        """Finalize without retention fields writes default (days/7)."""
        resp = fresh_client.post(
            "/init/finalize",
            json={"password": "securepass123"},
            content_type="application/json",
        )
        assert resp.status_code == 200
        config = _read_config(journal_copy)
        assert config["retention"]["raw_media"] == "days"
        assert config["retention"]["raw_media_days"] == 7


class TestRemovedEndpoints:
    """Verify old endpoints no longer exist."""

    def test_init_password_gone(self, fresh_client):
        resp = fresh_client.post(
            "/init/password",
            json={"password": "securepass123"},
            content_type="application/json",
        )
        assert resp.status_code in (404, 405)

    def test_init_identity_gone(self, fresh_client):
        resp = fresh_client.post(
            "/init/identity",
            json={"name": "Jane"},
            content_type="application/json",
        )
        assert resp.status_code in (404, 405)

    def test_init_provider_gone(self, fresh_client):
        resp = fresh_client.post(
            "/init/provider",
            json={"key": "some-key"},
            content_type="application/json",
        )
        assert resp.status_code in (404, 405)


class TestLocalhostBypass:
    """Tests for the opt-in trust_localhost bypass."""

    def test_localhost_fresh_install_redirects_to_init(self, fresh_client):
        """Plain localhost with no config → redirect to /init."""
        resp = fresh_client.get("/")
        assert resp.status_code == 302
        assert "/init" in resp.headers["Location"]

    def test_localhost_trust_bypass(self, journal_copy):
        """Localhost + trust_localhost + setup.completed_at → pass through."""
        config = _read_config(journal_copy)
        config["convey"]["trust_localhost"] = True
        config["setup"] = {"completed_at": 1700000000000}
        (journal_copy / "config" / "journal.json").write_text(
            json.dumps(config, indent=2)
        )
        app = create_app(str(journal_copy))
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/")
        assert resp.status_code == 302
        # Should redirect to home app, not login or init
        assert "/login" not in resp.headers["Location"]
        assert "/init" not in resp.headers["Location"]

    def test_localhost_trust_without_setup_redirects_to_init(self, journal_copy):
        """trust_localhost set but no setup.completed_at → redirect to /init."""
        config = _read_config(journal_copy)
        config["convey"]["trust_localhost"] = True
        config.pop("setup", None)
        config["convey"].pop("password_hash", None)
        (journal_copy / "config" / "journal.json").write_text(
            json.dumps(config, indent=2)
        )
        app = create_app(str(journal_copy))
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/")
        assert resp.status_code == 302
        assert "/init" in resp.headers["Location"]

    def test_localhost_no_trust_redirects_to_login(self, journal_copy):
        """Localhost + setup.completed_at but no trust_localhost → redirect to /login."""
        config = _read_config(journal_copy)
        config["convey"].pop("trust_localhost", None)
        config["setup"] = {"completed_at": 1700000000000}
        (journal_copy / "config" / "journal.json").write_text(
            json.dumps(config, indent=2)
        )
        app = create_app(str(journal_copy))
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_proxy_header_defeats_trust_localhost(self, configured_client):
        """Proxy headers prevent trust_localhost bypass."""
        resp = configured_client.get("/", headers={"X-Forwarded-For": "1.2.3.4"})
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]


class TestBasicAuth:
    """Tests for Basic Auth support."""

    def test_basic_auth_correct_password(self, configured_client):
        """Basic Auth with correct password → authenticated."""
        creds = base64.b64encode(b":test123").decode()
        resp = configured_client.get(
            "/",
            headers={
                "Authorization": f"Basic {creds}",
                "X-Forwarded-For": "1.2.3.4",
            },
        )
        assert resp.status_code == 302
        # Should redirect to home app, not login or init
        assert "/login" not in resp.headers["Location"]
        assert "/init" not in resp.headers["Location"]

    def test_basic_auth_wrong_password(self, configured_client):
        """Basic Auth with wrong password → redirect to /login."""
        creds = base64.b64encode(b":wrongpassword").decode()
        resp = configured_client.get(
            "/",
            headers={
                "Authorization": f"Basic {creds}",
                "X-Forwarded-For": "1.2.3.4",
            },
        )
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_basic_auth_no_session(self, configured_client):
        """Basic Auth does not create a session — next request without header fails."""
        creds = base64.b64encode(b":test123").decode()
        # First request with Basic Auth succeeds
        resp1 = configured_client.get(
            "/",
            headers={
                "Authorization": f"Basic {creds}",
                "X-Forwarded-For": "1.2.3.4",
            },
        )
        assert "/login" not in resp1.headers["Location"]

        # Second request without Basic Auth → should redirect to login
        resp2 = configured_client.get("/", headers={"X-Forwarded-For": "1.2.3.4"})
        assert resp2.status_code == 302
        assert "/login" in resp2.headers["Location"]


class TestSetupMigration:
    """Tests for the _migrate_setup_completed migration.

    Legacy-only: handles journals where password was set via CLI before
    web onboarding existed. New onboarding writes all config atomically.
    """

    def test_migration_writes_setup_and_trust(self, journal_copy):
        """App startup with password_hash but no setup.completed_at writes both."""
        config = _read_config(journal_copy)
        config.pop("setup", None)
        config["convey"].pop("trust_localhost", None)
        (journal_copy / "config" / "journal.json").write_text(
            json.dumps(config, indent=2)
        )

        # create_app triggers the migration
        create_app(str(journal_copy))

        config = _read_config(journal_copy)
        assert "completed_at" in config.get("setup", {})
        assert config["convey"].get("trust_localhost") is True

    def test_migration_idempotent(self, journal_copy):
        """Running migration twice is a no-op."""
        config = _read_config(journal_copy)
        config.pop("setup", None)
        config["convey"].pop("trust_localhost", None)
        (journal_copy / "config" / "journal.json").write_text(
            json.dumps(config, indent=2)
        )

        # First run triggers migration
        create_app(str(journal_copy))
        config1 = _read_config(journal_copy)
        ts1 = config1["setup"]["completed_at"]

        # Second run should be a no-op
        create_app(str(journal_copy))
        config2 = _read_config(journal_copy)
        assert config2["setup"]["completed_at"] == ts1
        assert config2["convey"]["trust_localhost"] is True
