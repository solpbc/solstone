# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for merge-names CLI command."""

from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from apps.speakers.call import app as speakers_app

_runner = CliRunner()


def test_merge_names_cli_error_missing_entity(speakers_env):
    """CLI merge-names outputs error JSON and exits 1 for unknown entity."""
    speakers_env()
    result = _runner.invoke(speakers_app, ["merge-names", "Nobody", "Also Nobody"])
    assert result.exit_code == 1


def test_merge_names_cli_success(speakers_env):
    """CLI merge-names outputs JSON with merged=True on success."""
    env = speakers_env()
    entity_a = env.create_entity("Alice Alias")
    entity_b = env.create_entity("Alice Canonical")

    emb_a = np.random.default_rng(42).standard_normal((3, 256)).astype(np.float32)
    emb_b = np.random.default_rng(99).standard_normal((3, 256)).astype(np.float32)
    meta_a = np.array([json.dumps({"key": f"a_{i}"}) for i in range(3)], dtype=str)
    meta_b = np.array([json.dumps({"key": f"b_{i}"}) for i in range(3)], dtype=str)
    np.savez_compressed(entity_a / "voiceprints.npz", embeddings=emb_a, metadata=meta_a)
    np.savez_compressed(entity_b / "voiceprints.npz", embeddings=emb_b, metadata=meta_b)

    result = _runner.invoke(
        speakers_app, ["merge-names", "Alice Alias", "Alice Canonical"]
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["merged"] is True
    assert data["canonical_name"] == "Alice Canonical"
