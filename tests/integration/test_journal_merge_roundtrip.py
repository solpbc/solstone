# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Integration coverage for export -> validate -> unpack -> merge."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from solstone.think.importers.journal_archive import validate_journal_archive
from tests._baseline_harness import copytree_tracked


@pytest.mark.integration
def test_export_validate_merge_rescan_round_trip(tmp_path, integration_journal_path):
    sol_binary = shutil.which("sol")
    if sol_binary is None:
        pytest.skip("sol binary is not on PATH for integration round-trip test")

    source_journal = tmp_path / "source-journal"
    copytree_tracked(Path("tests/fixtures/journal").resolve(), source_journal)
    archive_path = tmp_path / "journal-export.zip"

    export_env = os.environ.copy()
    export_env["SOLSTONE_JOURNAL"] = str(source_journal.resolve())

    export_result = subprocess.run(
        [sol_binary, "call", "journal", "export", "--out", str(archive_path)],
        capture_output=True,
        text=True,
        check=False,
        env=export_env,
    )
    assert export_result.returncode == 0, export_result.stderr
    assert archive_path.exists()

    validation = validate_journal_archive(archive_path)
    assert validation.ok is True

    unpack_dir = tmp_path / "unpacked"
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(unpack_dir)

    merge_env = os.environ.copy()
    merge_env["SOLSTONE_JOURNAL"] = str(integration_journal_path.resolve())
    merge_env["SOL_SKIP_SUPERVISOR_CHECK"] = "1"

    merge_result = subprocess.run(
        [sol_binary, "call", "journal", "merge", str(unpack_dir), "--json"],
        capture_output=True,
        text=True,
        check=False,
        env=merge_env,
    )
    assert merge_result.returncode == 0, merge_result.stderr
    payload = json.loads(merge_result.stdout)
    assert payload["ok"] is True
    assert payload["indexer_returncode"] == 0

    assert (
        integration_journal_path / "chronicle" / "20260306" / "default" / "143000_300"
    ).exists()
    assert (
        integration_journal_path / "entities" / "romeo_montague" / "entity.json"
    ).exists()
