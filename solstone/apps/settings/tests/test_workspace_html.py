# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from __future__ import annotations

from pathlib import Path


def test_workspace_has_diagnostic_reports_toggle():
    workspace = Path(__file__).resolve().parents[1] / "workspace.html"
    text = workspace.read_text(encoding="utf-8")

    assert 'id="field-reporting-enabled"' in text
    assert "diagnostic reports" in text
