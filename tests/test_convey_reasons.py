# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from flask import Flask

from solstone.convey.reasons import AUTH_REQUIRED, Reason
from solstone.convey.utils import (
    error_response,
    error_response_with_reason,
)


def _app_context():
    app = Flask(__name__)
    return app.app_context()


def test_reason_status_defaults_to_400():
    reason = Reason("test_reason", "Test message.")

    assert reason.status == 400


def test_error_response_reason_uses_reason_status():
    with _app_context():
        response, status = error_response(AUTH_REQUIRED)

    assert status == AUTH_REQUIRED.status
    assert response.get_json() == {
        "error": AUTH_REQUIRED.message,
        "reason_code": AUTH_REQUIRED.code,
        "detail": "",
    }


def test_error_response_reason_allows_status_override():
    with _app_context():
        response, status = error_response(AUTH_REQUIRED, status=500)

    assert status == 500
    assert response.get_json() == {
        "error": AUTH_REQUIRED.message,
        "reason_code": AUTH_REQUIRED.code,
        "detail": "",
    }


def test_error_response_reason_includes_detail():
    with _app_context():
        response, status = error_response(AUTH_REQUIRED, detail="some detail")

    assert status == AUTH_REQUIRED.status
    assert response.get_json()["detail"] == "some detail"


def test_error_response_with_reason_emits_legacy_reason_key():
    with _app_context():
        response, status = error_response_with_reason(AUTH_REQUIRED, detail="x")

    assert status == AUTH_REQUIRED.status
    assert response.get_json() == {
        "error": AUTH_REQUIRED.message,
        "reason": AUTH_REQUIRED.code,
        "reason_code": AUTH_REQUIRED.code,
        "detail": "x",
    }
