# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""File importer framework for structured file/directory imports."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class ImportPreview:
    """Preview of what a file import would produce."""

    date_range: tuple[str, str]  # (earliest YYYYMMDD, latest YYYYMMDD)
    item_count: int
    entity_count: int
    summary: str


@dataclass
class ImportResult:
    """Result of a completed file import."""

    entries_written: int
    entities_seeded: int
    files_created: list[str]
    errors: list[str]
    summary: str


@runtime_checkable
class FileImporter(Protocol):
    """Protocol for file/directory importers."""

    name: str
    display_name: str
    file_patterns: list[str]
    description: str

    def detect(self, path: Path) -> bool: ...
    def preview(self, path: Path) -> ImportPreview: ...
    def process(
        self,
        path: Path,
        journal_root: Path,
        *,
        facet: str | None = None,
        progress_callback: Callable | None = None,
    ) -> ImportResult: ...


FILE_IMPORTER_REGISTRY: dict[str, str] = {
    "ics": "think.importers.ics",
    "obsidian": "think.importers.obsidian",
    "claude": "think.importers.claude_chat",
    "chatgpt": "think.importers.chatgpt",
    "kindle": "think.importers.kindle",
    "gemini": "think.importers.gemini",
}


def get_file_importers() -> list[FileImporter]:
    """Discover and instantiate all registered file importers."""
    import importlib

    importers: list[FileImporter] = []
    for name, module_path in FILE_IMPORTER_REGISTRY.items():
        try:
            module = importlib.import_module(module_path)
            imp = getattr(module, "importer", None)
            if isinstance(imp, FileImporter):
                importers.append(imp)
            else:
                logger.warning(
                    "Importer %s from %s does not conform to FileImporter",
                    name,
                    module_path,
                )
        except Exception as exc:
            logger.warning("Failed to load file importer %s: %s", name, exc)
    return importers


def get_file_importer(name: str) -> FileImporter | None:
    """Get a specific file importer by registry name."""
    import importlib

    module_path = FILE_IMPORTER_REGISTRY.get(name)
    if module_path is None:
        return None
    try:
        module = importlib.import_module(module_path)
        imp = getattr(module, "importer", None)
        if isinstance(imp, FileImporter):
            return imp
        logger.warning(
            "Importer %s from %s does not conform to FileImporter",
            name,
            module_path,
        )
    except Exception as exc:
        logger.warning("Failed to load file importer %s: %s", name, exc)
    return None


def detect_file_importer(path: Path) -> FileImporter | None:
    """Try each registered importer's detect() to find one that matches the path."""
    for imp in get_file_importers():
        try:
            if imp.detect(path):
                return imp
        except Exception as exc:
            logger.debug("Importer %s detection failed for %s: %s", imp.name, path, exc)
    return None
