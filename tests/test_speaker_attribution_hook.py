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

    def test_layer4_returns_template_vars(self):
        """Unmatched sentences return template vars without changing transcript."""
        result = _run_pre_process(
            {
                **CONTEXT,
                "transcript": "some transcript",
                "meta": {},
            },
            None,
            {
                "labels": [
                    {
                        "sentence_id": 0,
                        "speaker": "owner",
                        "confidence": "high",
                        "method": "embedding",
                    },
                    {
                        "sentence_id": 1,
                        "speaker": None,
                        "confidence": None,
                        "method": None,
                    },
                    {
                        "sentence_id": 2,
                        "speaker": None,
                        "confidence": None,
                        "method": None,
                    },
                ],
                "unmatched": [1, 2],
                "unmatched_texts": {1: "Hello everyone", 2: "Let me explain"},
                "candidates": ["Alice Johnson", "Bob Smith"],
                "metadata": {"total": 3, "resolved": 1},
                "source": "audio",
            },
        )

        assert "meta" in result
        assert "attribution_result" in result["meta"]
        assert "template_vars" in result
        assert "unmatched_context" in result["template_vars"]
        assert "transcript" not in result

        unmatched_context = result["template_vars"]["unmatched_context"]
        assert "Alice Johnson" in unmatched_context
        assert "Bob Smith" in unmatched_context
        assert "Hello everyone" in unmatched_context
        assert "Let me explain" in unmatched_context
        assert "Sentence 1" in unmatched_context
        assert "Sentence 2" in unmatched_context
