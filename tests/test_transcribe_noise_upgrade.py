# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for noise upgrade feature in transcription."""

from unittest.mock import patch

from observe.vad import VadResult


class TestHasToken:
    """Tests for revai.has_token() function."""

    def test_has_token_when_present(self):
        """has_token() returns True when token is configured."""
        from observe.transcribe.revai import has_token

        with patch.dict("os.environ", {"REVAI_ACCESS_TOKEN": "test-token"}):
            assert has_token() is True

    def test_has_token_when_missing(self):
        """has_token() returns False when token is not configured."""
        from observe.transcribe.revai import has_token

        with patch.dict("os.environ", {}, clear=True):
            # Also need to patch dotenv to not load any .env file
            with patch("observe.transcribe.revai.load_dotenv"):
                assert has_token() is False

    def test_has_token_with_alternate_env_var(self):
        """has_token() returns True with REV_ACCESS_TOKEN."""
        from observe.transcribe.revai import has_token

        with patch.dict("os.environ", {"REV_ACCESS_TOKEN": "test-token"}):
            assert has_token() is True


class TestNoiseUpgradeLogic:
    """Tests for noise upgrade decision logic."""

    def _make_vad_result(self, noisy_rms: float | None = None) -> VadResult:
        """Create a VadResult with specified noise level."""
        return VadResult(
            duration=10.0,
            speech_duration=5.0,
            has_speech=True,
            speech_segments=[(1.0, 6.0)],
            noisy_rms=noisy_rms,
            noisy_s=3.0 if noisy_rms else 0.0,
        )

    def test_is_noisy_threshold(self):
        """VadResult.is_noisy() uses 0.01 RMS threshold."""
        # Below threshold
        vad_quiet = self._make_vad_result(noisy_rms=0.005)
        assert vad_quiet.is_noisy() is False

        # Above threshold
        vad_noisy = self._make_vad_result(noisy_rms=0.015)
        assert vad_noisy.is_noisy() is True

        # Exactly at threshold (should be False, > not >=)
        vad_edge = self._make_vad_result(noisy_rms=0.01)
        assert vad_edge.is_noisy() is False

    def test_is_noisy_none_rms(self):
        """VadResult.is_noisy() returns False when RMS is None."""
        vad = self._make_vad_result(noisy_rms=None)
        assert vad.is_noisy() is False


class TestNoiseUpgradeConfig:
    """Tests for noise_upgrade config handling."""

    def test_noise_upgrade_defaults_to_true(self):
        """noise_upgrade should default to True when not in config."""
        config = {}
        noise_upgrade = config.get("noise_upgrade", True)
        assert noise_upgrade is True

    def test_noise_upgrade_explicit_true(self):
        """noise_upgrade can be explicitly set to True."""
        config = {"noise_upgrade": True}
        noise_upgrade = config.get("noise_upgrade", True)
        assert noise_upgrade is True

    def test_noise_upgrade_explicit_false(self):
        """noise_upgrade can be disabled."""
        config = {"noise_upgrade": False}
        noise_upgrade = config.get("noise_upgrade", True)
        assert noise_upgrade is False


class TestBackendMetadata:
    """Tests for backend field in JSONL metadata."""

    def test_backend_field_in_metadata(self):
        """_statements_to_jsonl includes backend field in metadata."""
        import datetime
        import json

        from observe.transcribe.main import _statements_to_jsonl

        statements = [{"id": 1, "start": 0.0, "end": 1.0, "text": "Hello"}]
        model_info = {"model": "medium.en", "device": "cpu", "compute_type": "int8"}
        base_dt = datetime.datetime(2025, 1, 15, 14, 30, 0)

        # Test with whisper backend
        lines = _statements_to_jsonl(
            statements, "audio.flac", base_dt, model_info, backend="whisper"
        )
        metadata = json.loads(lines[0])
        assert metadata["backend"] == "whisper"

        # Test with revai backend
        lines = _statements_to_jsonl(
            statements, "audio.flac", base_dt, model_info, backend="revai"
        )
        metadata = json.loads(lines[0])
        assert metadata["backend"] == "revai"

    def test_backend_field_defaults_to_unknown(self):
        """backend field defaults to 'unknown' when not provided."""
        import datetime
        import json

        from observe.transcribe.main import _statements_to_jsonl

        statements = [{"id": 1, "start": 0.0, "end": 1.0, "text": "Hello"}]
        model_info = {"model": "medium.en", "device": "cpu", "compute_type": "int8"}
        base_dt = datetime.datetime(2025, 1, 15, 14, 30, 0)

        lines = _statements_to_jsonl(statements, "audio.flac", base_dt, model_info)
        metadata = json.loads(lines[0])
        assert metadata["backend"] == "unknown"
