# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for stale facet cookie validation in convey."""

import json

import pytest

from convey import create_app


@pytest.fixture
def client(journal_copy):
    """Create Flask test client with isolated journal copy."""
    app = create_app(str(journal_copy))
    return app.test_client()


def _cookie_deleted(response) -> bool:
    """Check if selectedFacet cookie was deleted in response."""
    for header in response.headers.getlist("Set-Cookie"):
        if header.startswith("selectedFacet=") and "Max-Age=0" in header:
            return True
    return False


class TestFacetCookieValidation:
    """Tests for _get_selected_facet() stale cookie validation."""

    def test_valid_cookie_returned(self, client):
        """Valid cookie for active facet -> returned as-is, no cookie cleared."""
        client.set_cookie("selectedFacet", "montague")
        response = client.get("/app/home/")
        assert response.status_code == 200
        assert not _cookie_deleted(response)

    def test_stale_cookie_cleared(self, client, journal_copy):
        """Stale cookie (nonexistent facet) -> cookie expired, config cleared."""
        client.set_cookie("selectedFacet", "nonexistent-facet")
        response = client.get("/app/home/")
        assert response.status_code == 200
        assert _cookie_deleted(response)

        config = json.loads((journal_copy / "config" / "convey.json").read_text())
        assert config["facets"]["selected"] is None

    def test_no_cookie_uses_config(self, client, journal_copy):
        """No cookie -> returns config default, no cookie modification."""
        config_path = journal_copy / "config" / "convey.json"
        config = json.loads(config_path.read_text())
        config["facets"]["selected"] = "montague"
        config_path.write_text(json.dumps(config, indent=2))

        response = client.get("/app/home/")
        assert response.status_code == 200
        assert not _cookie_deleted(response)

    def test_muted_facet_cookie_stale(self, client, journal_copy):
        """Muted facet in cookie -> treated as stale, cookie expired."""
        # muted-test fixture already exists in journal_copy (copied from fixtures)
        client.set_cookie("selectedFacet", "muted-test")
        response = client.get("/app/home/")
        assert response.status_code == 200
        assert _cookie_deleted(response)

        config = json.loads((journal_copy / "config" / "convey.json").read_text())
        assert config["facets"]["selected"] is None

    def test_empty_cookie_cleared(self, client, journal_copy):
        """Empty string cookie -> treated as stale, cookie expired, config cleared."""
        client.set_cookie("selectedFacet", "")
        response = client.get("/app/home/")
        assert response.status_code == 200
        assert _cookie_deleted(response)

        config = json.loads((journal_copy / "config" / "convey.json").read_text())
        assert config["facets"]["selected"] is None

    def test_stale_config_no_cookie(self, client, journal_copy):
        """Stale config value with no cookie -> config cleared."""
        config_path = journal_copy / "config" / "convey.json"
        config = json.loads(config_path.read_text())
        config["facets"]["selected"] = "nonexistent-facet"
        config_path.write_text(json.dumps(config, indent=2))

        response = client.get("/app/home/")
        assert response.status_code == 200

        config = json.loads(config_path.read_text())
        assert config["facets"]["selected"] is None
