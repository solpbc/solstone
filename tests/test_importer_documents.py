# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import datetime as dt
import importlib

from think.importers.file_importer import FILE_IMPORTER_REGISTRY


class MockPage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self):
        return self._text


class MockPdfReader:
    def __init__(self, path):
        self.path = str(path)
        self.pages = [
            MockPage(
                "Page text content here with enough characters to pass threshold."
            ),
            MockPage(
                "Second page text content here with enough characters to pass threshold."
            ),
        ]
        self.metadata = {"/CreationDate": "D:20260115120000"}


def test_detect_pdf_file(tmp_path):
    mod = importlib.import_module("think.importers.documents")
    pdf = tmp_path / "file.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert mod.importer.detect(pdf) is True


def test_detect_non_pdf(tmp_path):
    mod = importlib.import_module("think.importers.documents")
    txt = tmp_path / "file.txt"
    txt.write_text("hello", encoding="utf-8")
    assert mod.importer.detect(txt) is False


def test_detect_directory_with_pdfs(tmp_path):
    mod = importlib.import_module("think.importers.documents")
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    assert mod.importer.detect(tmp_path) is True


def test_detect_empty_directory(tmp_path):
    mod = importlib.import_module("think.importers.documents")
    assert mod.importer.detect(tmp_path) is False


def test_preview_single_pdf(tmp_path, monkeypatch):
    mod = importlib.import_module("think.importers.documents")
    pdf = tmp_path / "file.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(mod, "PdfReader", MockPdfReader)

    preview = mod.importer.preview(pdf)

    assert preview.date_range == ("20260115", "20260115")
    assert preview.item_count == 1
    assert preview.entity_count == 0
    assert preview.summary == "1 PDF documents, 2 total pages"


def test_process_text_pdf(tmp_path, monkeypatch):
    mod = importlib.import_module("think.importers.documents")
    pdf = tmp_path / "contract.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    monkeypatch.setattr(mod, "PdfReader", MockPdfReader)
    monkeypatch.setattr(mod, "day_path", lambda day: tmp_path / "chronicle" / day)
    monkeypatch.setattr(
        mod,
        "write_content_manifest",
        lambda import_id, entries: tmp_path / "manifest.jsonl",
    )
    monkeypatch.setattr(mod, "seed_entities", lambda facet, day, entities: entities)

    result = mod.importer.process(
        pdf, tmp_path, facet="work", import_id="20260115_120000"
    )

    seg_dir = tmp_path / "chronicle" / "20260115" / "import.document" / "120000_0"
    md_path = seg_dir / "document_transcript.md"

    assert result.entries_written == 1
    assert result.entities_seeded >= 0
    assert result.segments == [("20260115", "120000_0")]
    assert result.files_created == [str(md_path)]
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert content.startswith("# contract")
    assert "**Pages:** 2" in content
    assert "Page text content here" in content


def test_process_creates_original_pdf(tmp_path, monkeypatch):
    mod = importlib.import_module("think.importers.documents")
    pdf = tmp_path / "original.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    monkeypatch.setattr(mod, "PdfReader", MockPdfReader)
    monkeypatch.setattr(mod, "day_path", lambda day: tmp_path / "chronicle" / day)
    monkeypatch.setattr(
        mod,
        "write_content_manifest",
        lambda import_id, entries: tmp_path / "manifest.jsonl",
    )
    monkeypatch.setattr(mod, "seed_entities", lambda facet, day, entities: entities)

    result = mod.importer.process(pdf, tmp_path, import_id="20260115_120000")

    copied = (
        tmp_path
        / "chronicle"
        / "20260115"
        / "import.document"
        / "120000_0"
        / "original.pdf"
    )
    assert copied.exists()
    assert copied.read_bytes() == b"%PDF-1.4 fake"
    assert str(copied) not in result.files_created


def test_process_scanned_detection(tmp_path, monkeypatch):
    mod = importlib.import_module("think.importers.documents")

    class ScannedReader(MockPdfReader):
        def __init__(self, path):
            self.path = str(path)
            self.pages = [MockPage("x"), MockPage("y")]
            self.metadata = {"/CreationDate": "D:20260115120000"}

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    monkeypatch.setattr(mod, "PdfReader", ScannedReader)
    monkeypatch.setattr(mod, "day_path", lambda day: tmp_path / "chronicle" / day)
    monkeypatch.setattr(
        mod,
        "write_content_manifest",
        lambda import_id, entries: tmp_path / "manifest.jsonl",
    )
    monkeypatch.setattr(mod, "seed_entities", lambda facet, day, entities: entities)
    calls = []

    def fake_vision(pdf_path, page_count):
        calls.append((pdf_path.name, page_count))
        return "Vision extracted text"

    monkeypatch.setattr(mod, "_extract_text_vision", fake_vision)

    result = mod.importer.process(pdf, tmp_path, import_id="20260115_120000")

    assert calls == [("scan.pdf", 2)]
    assert result.errors == []
    md_path = (
        tmp_path
        / "chronicle"
        / "20260115"
        / "import.document"
        / "120000_0"
        / "document_transcript.md"
    )
    assert "Vision extracted text" in md_path.read_text(encoding="utf-8")


def test_process_scanned_all_fallback(tmp_path, monkeypatch):
    mod = importlib.import_module("think.importers.documents")

    class ScannedReader(MockPdfReader):
        def __init__(self, path):
            self.path = str(path)
            self.pages = [MockPage("x")]
            self.metadata = {"/CreationDate": "D:20260115120000"}

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    monkeypatch.setattr(mod, "PdfReader", ScannedReader)
    monkeypatch.setattr(mod, "day_path", lambda day: tmp_path / "chronicle" / day)
    monkeypatch.setattr(
        mod,
        "write_content_manifest",
        lambda import_id, entries: tmp_path / "manifest.jsonl",
    )
    monkeypatch.setattr(mod, "seed_entities", lambda facet, day, entities: entities)
    monkeypatch.setattr(
        mod,
        "_extract_text_vision",
        lambda pdf_path, page_count: (_ for _ in ()).throw(
            RuntimeError("vision failed")
        ),
    )

    result = mod.importer.process(pdf, tmp_path, import_id="20260115_120000")

    assert result.errors == [
        "scan.pdf: scanned PDF — vision failed (vision failed); using sparse pypdf text"
    ]
    md_path = (
        tmp_path
        / "chronicle"
        / "20260115"
        / "import.document"
        / "120000_0"
        / "document_transcript.md"
    )
    assert "\nx\n" in md_path.read_text(encoding="utf-8")


def test_process_multi_file(tmp_path, monkeypatch):
    mod = importlib.import_module("think.importers.documents")
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.write_bytes(b"%PDF-1.4")
    pdf_b.write_bytes(b"%PDF-1.4")
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    monkeypatch.setattr(mod, "PdfReader", MockPdfReader)
    monkeypatch.setattr(mod, "day_path", lambda day: tmp_path / "chronicle" / day)
    monkeypatch.setattr(
        mod,
        "write_content_manifest",
        lambda import_id, entries: tmp_path / "manifest.jsonl",
    )
    monkeypatch.setattr(mod, "seed_entities", lambda facet, day, entities: entities)

    result = mod.importer.process(tmp_path, tmp_path, import_id="20260115_120000")

    assert result.entries_written == 2
    assert result.segments == [("20260115", "120000_0"), ("20260115", "120001_0")]


def test_process_entity_seeding(tmp_path, monkeypatch):
    mod = importlib.import_module("think.importers.documents")

    class EntityReader(MockPdfReader):
        def __init__(self, path):
            self.path = str(path)
            self.pages = [
                MockPage(
                    "Signed by Jane Doe on behalf of Example Corp Inc for the purchase agreement and related closing documents."
                )
            ]
            self.metadata = {"/CreationDate": "D:20260115120000"}

    pdf = tmp_path / "parties.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setenv("SOLSTONE_JOURNAL", str(tmp_path))
    monkeypatch.setattr(mod, "PdfReader", EntityReader)
    monkeypatch.setattr(mod, "day_path", lambda day: tmp_path / "chronicle" / day)
    monkeypatch.setattr(
        mod,
        "write_content_manifest",
        lambda import_id, entries: tmp_path / "manifest.jsonl",
    )
    calls = []

    def fake_seed_entities(facet, day, entities):
        calls.append((facet, day, entities))
        return entities

    monkeypatch.setattr(mod, "seed_entities", fake_seed_entities)

    result = mod.importer.process(
        pdf, tmp_path, facet="legal", import_id="20260115_120000"
    )

    assert result.entities_seeded == len(calls[0][2])
    assert calls[0][0] == "legal"
    assert calls[0][1] == "20260115"
    assert any(entity["type"] == "Person" for entity in calls[0][2])
    assert any(entity["type"] == "Organization" for entity in calls[0][2])
    assert all(entity["observations"] == ["Named in parties"] for entity in calls[0][2])


def test_timestamp_from_metadata(tmp_path):
    mod = importlib.import_module("think.importers.documents")
    pdf = tmp_path / "file.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    reader = MockPdfReader(pdf)

    timestamp = mod._get_pdf_timestamp(reader, pdf)

    assert (
        dt.datetime.fromtimestamp(timestamp).strftime("%Y%m%d%H%M%S")
        == "20260115120000"
    )


def test_registry_entry():
    assert FILE_IMPORTER_REGISTRY["document"] == "think.importers.documents"
