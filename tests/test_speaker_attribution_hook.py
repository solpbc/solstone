# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unit tests for talent.speaker_attribution pre_process stub-writing behavior."""

import json
from unittest.mock import patch


def _run_pre_process(context, seg_dir, attribute_result):
    """Helper: run pre_process with mocked dependencies."""
    with (
        patch(
            "apps.speakers.attribution.attribute_segment",
            return_value=attribute_result,
        ),
        patch(
            "think.utils.segment_path",
            return_value=seg_dir,
        ),
    ):
        from talent.speaker_attribution import pre_process

        return pre_process(context)


CONTEXT = {"day": "20260318", "segment": "100000_300", "stream": "default"}


class TestPreProcessStub:
    def test_error_with_npz_writes_stub(self, tmp_path):
        """no_owner_centroid error + .npz present -> stub written."""
        (tmp_path / "audio.npz").write_bytes(b"x")
        result = _run_pre_process(
            CONTEXT,
            tmp_path,
            {"error": "no_owner_centroid"},
        )
        stub_path = tmp_path / "agents" / "speaker_labels.json"
        assert stub_path.exists()
        data = json.loads(stub_path.read_text())
        assert data == {"labels": [], "skipped": True, "reason": "no_owner_centroid"}
        assert result == {"skip_reason": "no_owner_centroid"}

    def test_error_without_npz_no_stub(self, tmp_path):
        """no_owner_centroid error + no .npz -> no stub written."""
        result = _run_pre_process(
            CONTEXT,
            tmp_path,
            {"error": "no_owner_centroid"},
        )
        stub_path = tmp_path / "agents" / "speaker_labels.json"
        assert not stub_path.exists()
        assert result == {"skip_reason": "no_owner_centroid"}

    def test_empty_labels_with_npz_writes_stub(self, tmp_path):
        """Empty labels (loaded-but-empty .npz) + .npz present -> stub written."""
        (tmp_path / "audio.npz").write_bytes(b"x")
        result = _run_pre_process(
            CONTEXT,
            tmp_path,
            {"labels": []},
        )
        stub_path = tmp_path / "agents" / "speaker_labels.json"
        assert stub_path.exists()
        data = json.loads(stub_path.read_text())
        assert data == {"labels": [], "skipped": True, "reason": "no_embeddings"}
        assert result == {"skip_reason": "no_embeddings"}

    def test_empty_labels_without_npz_no_stub(self, tmp_path):
        """Empty labels + no .npz -> no stub written."""
        result = _run_pre_process(
            CONTEXT,
            tmp_path,
            {"labels": []},
        )
        stub_path = tmp_path / "agents" / "speaker_labels.json"
        assert not stub_path.exists()
        assert result == {"skip_reason": "no_embeddings"}

    def test_no_segment_context_no_stub(self, tmp_path):
        """Missing day/segment -> returns early before any stub logic."""
        (tmp_path / "audio.npz").write_bytes(b"x")
        with patch("think.utils.segment_path", return_value=tmp_path):
            from talent.speaker_attribution import pre_process

            result = pre_process({"stream": "default"})
        stub_path = tmp_path / "agents" / "speaker_labels.json"
        assert not stub_path.exists()
        assert result == {"skip_reason": "no_segment_context"}
