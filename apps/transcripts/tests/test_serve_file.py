# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path

import pytest

from convey import create_app


@pytest.fixture
def client():
    journal = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "journal"
    app = create_app(str(journal))
    return app.test_client()


def test_serve_file_path_traversal_returns_non_200(client):
    response = client.get(
        "/app/transcripts/api/serve_file/20240101/../../../etc/passwd"
    )

    assert response.status_code != 200


def test_serve_file_malformed_day_returns_404(client):
    response = client.get("/app/transcripts/api/serve_file/notadate/foo")

    assert response.status_code == 404
