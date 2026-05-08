# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import dataclasses
import subprocess
import sys

import frontmatter
import pytest

from solstone.think import features as features_module
from solstone.think.features import Feature, MissingExtraError

PROBE = """
import sys
import solstone.apps.reflections.routes  # noqa: F401
import solstone.think.importers.documents  # noqa: F401
import solstone.think.importers.text  # noqa: F401
print("|".join(sorted(sys.modules)))
"""


def test_pdf_modules_are_not_loaded_by_static_imports():
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        check=True,
        text=True,
    )
    modules = set(result.stdout.strip().split("|"))

    assert "solstone.apps.reflections.routes" in modules
    assert "solstone.think.importers.documents" in modules
    assert "solstone.think.importers.text" in modules
    assert "weasyprint" not in modules
    assert "pypdf" not in modules
    assert "pdf2image" not in modules


def _force_missing_pdf(monkeypatch):
    real = features_module.FEATURES["pdf"]
    fake: Feature = dataclasses.replace(
        real,
        pip_modules=("definitely_not_installed_xyz",),
    )
    monkeypatch.setitem(features_module.FEATURES, "pdf", fake)


def test_render_reflection_pdf_missing_extra(monkeypatch, tmp_path):
    _force_missing_pdf(monkeypatch)
    from solstone.apps.reflections.routes import _render_reflection_pdf

    with pytest.raises(MissingExtraError) as exc:
        _render_reflection_pdf(tmp_path / "20260308.md", frontmatter.Post("# hi"))

    assert "pip install 'solstone[pdf]'" in str(exc.value)


def test_document_importer_process_pdf_missing_extra(monkeypatch, tmp_path):
    _force_missing_pdf(monkeypatch)
    from solstone.think.importers.documents import DocumentImporter

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")

    with pytest.raises(MissingExtraError) as exc:
        DocumentImporter().process(pdf, tmp_path)

    assert "pip install 'solstone[pdf]'" in str(exc.value)


def test_read_transcript_pdf_missing_extra(monkeypatch, tmp_path):
    _force_missing_pdf(monkeypatch)
    from solstone.think.importers.text import _read_transcript

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"")

    with pytest.raises(MissingExtraError) as exc:
        _read_transcript(str(pdf))

    assert "pip install 'solstone[pdf]'" in str(exc.value)
