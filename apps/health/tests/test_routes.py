# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for health app routes."""


class TestLogRoute:
    """Tests for GET /app/health/api/log."""

    def test_valid_log_path(self, health_env):
        env = health_env()
        resp = env.client.get(
            "/app/health/api/log?path=20260322/health/1774196508583_transcribe.log"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["path"] == "20260322/health/1774196508583_transcribe.log"
        assert "test log content" in data["content"]

    def test_path_traversal_rejected(self, health_env):
        env = health_env()
        resp = env.client.get("/app/health/api/log?path=../../../etc/passwd")
        assert resp.status_code == 400

    def test_non_log_extension_rejected(self, health_env):
        env = health_env()
        resp = env.client.get("/app/health/api/log?path=20260322/health/foo.txt")
        assert resp.status_code == 400

    def test_path_outside_health_dir_rejected(self, health_env):
        env = health_env()
        resp = env.client.get("/app/health/api/log?path=20260322/agents/something.log")
        assert resp.status_code == 400

    def test_missing_file_returns_404(self, health_env):
        env = health_env()
        resp = env.client.get(
            "/app/health/api/log?path=20260322/health/nonexistent.log"
        )
        assert resp.status_code == 404

    def test_missing_path_param_returns_400(self, health_env):
        env = health_env()
        resp = env.client.get("/app/health/api/log")
        assert resp.status_code == 400

    def test_encoded_traversal_rejected(self, health_env):
        env = health_env()
        resp = env.client.get(
            "/app/health/api/log?path=20260322/health/..%2F..%2Fetc%2Fpasswd.log"
        )
        assert resp.status_code == 400

    def test_null_byte_rejected(self, health_env):
        env = health_env()
        resp = env.client.get("/app/health/api/log?path=20260322/health/foo%00.log")
        assert resp.status_code == 400
