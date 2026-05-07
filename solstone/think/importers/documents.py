# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""PDF document importer."""

from __future__ import annotations

import datetime as dt
import logging
import re
import shutil
from pathlib import Path
from typing import Callable

from pdf2image import convert_from_path
from pypdf import PdfReader

from solstone.think.entities.seeding import seed_entities
from solstone.think.importers.file_importer import ImportPreview, ImportResult
from solstone.think.importers.shared import write_content_manifest
from solstone.think.models import generate
from solstone.think.utils import day_path

logger = logging.getLogger(__name__)


def _find_pdfs(path: Path) -> list[Path]:
    """Return matching PDF files for a file or directory path."""
    if path.is_file() and path.suffix.lower() == ".pdf":
        return [path]
    if path.is_dir():
        return sorted(
            child
            for child in path.iterdir()
            if child.is_file() and child.suffix.lower() == ".pdf"
        )
    return []


def _get_pdf_timestamp(reader: PdfReader, pdf_path: Path) -> float:
    """Return a best-effort timestamp for a PDF."""
    metadata = reader.metadata or {}
    for key in ("/ModDate", "/CreationDate"):
        value = metadata.get(key)
        if not value:
            continue
        match = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", str(value))
        if not match:
            continue
        try:
            parsed = dt.datetime(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
                int(match.group(5)),
                int(match.group(6)),
            )
            return parsed.timestamp()
        except ValueError:
            continue
    try:
        return pdf_path.stat().st_mtime
    except OSError:
        return dt.datetime.now().timestamp()


def _extract_text_pypdf(reader: PdfReader) -> tuple[str, int, bool]:
    """Extract text and classify whether the PDF appears scanned."""
    parts: list[str] = []
    total_chars = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
        total_chars += len(text)
    page_count = len(reader.pages)
    is_scanned = (total_chars / max(page_count, 1)) < 50
    return "\n\n".join(parts).strip(), page_count, is_scanned


def _extract_text_vision(pdf_path: Path, page_count: int) -> str:
    """Extract text from scanned PDFs using vision models."""
    prompt = (
        "Extract all text content from this document. Preserve the document "
        "structure including headings, paragraphs, lists, and tables. Return "
        "the content as clean markdown."
    )
    images = convert_from_path(str(pdf_path), dpi=200)
    if page_count <= 10:
        return generate(
            contents=[prompt, *images], context="import.document.vision"
        ).strip()

    pages: list[str] = []
    for image in images:
        page_text = generate(
            contents=[prompt, image], context="import.document.vision"
        ).strip()
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages).strip()


def _render_document_markdown(title: str, text: str, metadata: dict) -> str:
    """Render extracted document text as markdown."""
    lines = [f"# {title}", "", "**Type:** Document"]
    if metadata.get("page_count") is not None:
        lines.append(f"**Pages:** {metadata['page_count']}")
    if metadata.get("date"):
        lines.append(f"**Date:** {metadata['date']}")
    lines.extend(["", "---", "", text.strip()])
    return "\n".join(lines).rstrip() + "\n"


def _extract_entities(text: str, title: str) -> list[dict]:
    """Extract simple named people and organizations from document text."""
    names = set(
        m.group(1).strip(" ,.")
        for m in re.finditer(
            r"(?i)\b(?:by|from|to|between|signed by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
            text,
        )
    )
    orgs = set(
        m.group(1).strip(" ,.")
        for m in re.finditer(
            r"\b([A-Z][A-Za-z0-9&.,' -]{1,80}\s+(?:LLC|Inc|Corp|Corporation|Trust|Ltd|Company)(?:\s+(?:LLC|Inc|Corp|Corporation|Trust|Ltd|Company))*)\b",
            text,
        )
    )
    observation = f"Named in {title}"
    entities = [
        {"name": name, "type": "Person", "observations": [observation]}
        for name in sorted(names)
    ]
    entities.extend(
        {"name": name, "type": "Organization", "observations": [observation]}
        for name in sorted(orgs)
    )
    return entities


class DocumentImporter:
    name = "document"
    display_name = "Documents"
    file_patterns = ["*.pdf"]
    description = (
        "Import PDF documents with text extraction and vision fallback for scanned PDFs"
    )

    def detect(self, path: Path) -> bool:
        return bool(_find_pdfs(path))

    def preview(self, path: Path) -> ImportPreview:
        pdfs = _find_pdfs(path)
        if not pdfs:
            return ImportPreview(
                date_range=("", ""),
                item_count=0,
                entity_count=0,
                summary="No PDF documents found",
            )

        timestamps: list[float] = []
        total_pages = 0
        for pdf_path in pdfs:
            reader = PdfReader(str(pdf_path))
            timestamps.append(_get_pdf_timestamp(reader, pdf_path))
            total_pages += len(reader.pages)

        dates = sorted(
            dt.datetime.fromtimestamp(ts).strftime("%Y%m%d") for ts in timestamps
        )
        return ImportPreview(
            date_range=(dates[0], dates[-1]),
            item_count=len(pdfs),
            entity_count=0,
            summary=f"{len(pdfs)} PDF documents, {total_pages} total pages",
        )

    def process(
        self,
        path: Path,
        journal_root: Path,
        *,
        facet: str | None = None,
        import_id: str | None = None,
        progress_callback: Callable | None = None,
        dry_run: bool = False,
    ) -> ImportResult:
        pdfs = _find_pdfs(path)
        import_id = import_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not pdfs:
            return ImportResult(
                entries_written=0,
                entities_seeded=0,
                files_created=[],
                errors=[],
                summary="No PDF documents found to import",
            )

        created_files: list[str] = []
        errors: list[str] = []
        segments: list[tuple[str, str]] = []
        manifest_entries: list[dict] = []
        entities_seeded = 0
        timestamps: list[float] = []
        used_keys: dict[str, set[str]] = {}

        for index, pdf_path in enumerate(pdfs):
            try:
                reader = PdfReader(str(pdf_path))
                ts = _get_pdf_timestamp(reader, pdf_path)
                text, page_count, is_scanned = _extract_text_pypdf(reader)
                extraction_method = "pypdf"

                if is_scanned:
                    try:
                        text = _extract_text_vision(pdf_path, page_count)
                        extraction_method = "vision"
                    except Exception as vision_exc:
                        logger.warning(
                            "Vision extraction failed for %s: %s", pdf_path, vision_exc
                        )
                        errors.append(
                            f"{pdf_path.name}: scanned PDF — vision failed ({vision_exc}); using sparse pypdf text"
                        )

                seg_dt = dt.datetime.fromtimestamp(ts)
                day = seg_dt.strftime("%Y%m%d")
                seg_key = f"{seg_dt.strftime('%H%M%S')}_0"
                day_used = used_keys.setdefault(day, set())
                while seg_key in day_used:
                    ts += 1
                    seg_dt = dt.datetime.fromtimestamp(ts)
                    day = seg_dt.strftime("%Y%m%d")
                    seg_key = f"{seg_dt.strftime('%H%M%S')}_0"
                    day_used = used_keys.setdefault(day, set())
                day_used.add(seg_key)
                timestamps.append(ts)

                title = pdf_path.stem
                date_str = seg_dt.strftime("%Y-%m-%d")
                segment_dir = day_path(day) / "import.document" / seg_key
                segment_dir.mkdir(parents=True, exist_ok=True)

                original_path = segment_dir / "original.pdf"
                shutil.copy2(pdf_path, original_path)
                md_path = segment_dir / "document_transcript.md"
                md_text = _render_document_markdown(
                    title,
                    text,
                    {"page_count": page_count, "date": date_str},
                )
                md_path.write_text(md_text, encoding="utf-8")

                created_files.append(str(md_path))
                segments.append((day, seg_key))
                manifest_entries.append(
                    {
                        "id": f"document-{index}",
                        "title": title,
                        "date": day,
                        "type": "document",
                        "preview": text[:200],
                        "meta": {
                            "page_count": page_count,
                            "extraction_method": extraction_method,
                        },
                        "segments": [{"day": day, "key": seg_key}],
                    }
                )

                if facet:
                    try:
                        resolved = seed_entities(
                            facet, day, _extract_entities(text, title)
                        )
                        entities_seeded += len(resolved)
                    except Exception as exc:
                        errors.append(
                            f"Failed to seed entities for {pdf_path.name}: {exc}"
                        )
            except Exception as exc:
                errors.append(f"Failed to process {pdf_path.name}: {exc}")

            if progress_callback:
                earliest = None
                latest = None
                if timestamps:
                    earliest = dt.datetime.fromtimestamp(min(timestamps)).strftime(
                        "%Y%m%d"
                    )
                    latest = dt.datetime.fromtimestamp(max(timestamps)).strftime(
                        "%Y%m%d"
                    )
                progress_callback(
                    index + 1,
                    len(pdfs),
                    earliest_date=earliest,
                    latest_date=latest,
                    entities_found=entities_seeded,
                )

        write_content_manifest(import_id, manifest_entries)

        if timestamps:
            earliest = dt.datetime.fromtimestamp(min(timestamps)).strftime("%Y%m%d")
            latest = dt.datetime.fromtimestamp(max(timestamps)).strftime("%Y%m%d")
            date_range: tuple[str, str] | None = (earliest, latest)
        else:
            date_range = None

        return ImportResult(
            entries_written=len(segments),
            entities_seeded=entities_seeded,
            files_created=created_files,
            errors=errors,
            summary=(
                f"Imported {len(segments)} PDF documents across "
                f"{len({day for day, _ in segments})} days into {len(segments)} segments"
            ),
            segments=segments,
            date_range=date_range,
        )


importer = DocumentImporter()
