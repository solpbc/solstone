# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for the bundled WeSpeaker embedder."""

import platform

import numpy as np
import pytest

from observe.transcribe.main import (
    EMBEDDER_NAME,
    _embed_statements,
    _get_embedder_session,
    _select_onnx_providers,
)


def test_session_loads() -> None:
    session = _get_embedder_session()
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    assert inputs[0].name == "feats"
    assert inputs[0].shape == ["B", "T", 80]
    assert outputs[0].name == "embs"
    assert outputs[0].shape == ["B", 256]


def test_embed_synthetic_shape_and_provenance() -> None:
    audio = np.zeros(3 * 16000, dtype=np.float32)
    result = _embed_statements(
        audio,
        [{"id": 1, "start": 0.0, "end": 3.0, "text": "x"}],
        16000,
    )

    assert result is not None
    assert result["embeddings"].shape == (1, 256)
    assert result["embeddings"].dtype == np.float32
    assert result["statement_ids"].tolist() == [1]
    assert result["encoder"].item() == EMBEDDER_NAME


def test_embed_determinism() -> None:
    rng = np.random.default_rng(42)
    audio = (rng.normal(0.0, 0.01, 3 * 16000)).astype(np.float32)
    statements = [{"id": 1, "start": 0.0, "end": 3.0, "text": "x"}]

    first = _embed_statements(audio, statements, 16000)
    second = _embed_statements(audio, statements, 16000)

    assert first is not None
    assert second is not None

    first_embedding = first["embeddings"][0]
    second_embedding = second["embeddings"][0]
    cosine = float(
        np.dot(first_embedding, second_embedding)
        / (np.linalg.norm(first_embedding) * np.linalg.norm(second_embedding))
    )
    assert cosine >= 0.99


@pytest.mark.parametrize(
    ("system_name", "expected"),
    [
        ("Darwin", ["CoreMLExecutionProvider", "CPUExecutionProvider"]),
        ("Linux", ["CPUExecutionProvider"]),
    ],
)
def test_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    system_name: str,
    expected: list[str],
) -> None:
    monkeypatch.setattr(platform, "system", lambda: system_name)
    assert _select_onnx_providers() == expected
