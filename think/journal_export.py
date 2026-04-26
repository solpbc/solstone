# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Helpers for exporting a journal as a portable ZIP archive."""

from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Iterator

EXPORT_TOP_LEVEL_DIRS = ("chronicle", "entities", "facets", "imports")
DATE_RE = re.compile(r"^\d{8}$")


def _default_export_path(journal_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    export_dir = journal_root.parent / f"{journal_root.name}.exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir / f"{timestamp}.zip"


def _count_journal_export_stats(journal_root: Path) -> tuple[int, int, int]:
    chronicle_dir = journal_root / "chronicle"
    if chronicle_dir.is_dir():
        day_count = sum(
            1
            for entry in chronicle_dir.iterdir()
            if entry.is_dir() and DATE_RE.match(entry.name)
        )
    else:
        day_count = 0

    entities_dir = journal_root / "entities"
    if entities_dir.is_dir():
        entity_count = sum(
            1
            for entry in entities_dir.iterdir()
            if entry.is_dir() and (entry / "entity.json").is_file()
        )
    else:
        entity_count = 0

    facets_dir = journal_root / "facets"
    if facets_dir.is_dir():
        facet_count = sum(
            1
            for entry in facets_dir.iterdir()
            if entry.is_dir() and (entry / "facet.json").is_file()
        )
    else:
        facet_count = 0

    return day_count, entity_count, facet_count


def get_skipped_export_entries(journal_root: Path) -> list[str]:
    if not journal_root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in journal_root.iterdir()
        if entry.name not in EXPORT_TOP_LEVEL_DIRS
    )


def _iter_export_members(journal_root: Path) -> Iterator[tuple[Path, str]]:
    for root_name in EXPORT_TOP_LEVEL_DIRS:
        root_path = journal_root / root_name
        if not root_path.exists():
            continue
        if not root_path.is_dir():
            raise NotADirectoryError(f"expected directory at {root_path}")
        for path in sorted(root_path.rglob("*")):
            if path.is_file():
                yield path, path.relative_to(journal_root).as_posix()


def _build_export_manifest(journal_root: Path) -> dict[str, str | int]:
    day_count, entity_count, facet_count = _count_journal_export_stats(journal_root)
    exported_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "solstone_version": version("solstone"),
        "exported_at": exported_at,
        "source_journal": str(journal_root.resolve()),
        "day_count": day_count,
        "entity_count": entity_count,
        "facet_count": facet_count,
    }


def export_journal_archive(journal_root: Path, out_path: Path | None = None) -> Path:
    if not journal_root.is_dir():
        raise FileNotFoundError(f"journal root is not a directory: {journal_root}")

    final_path = (out_path or _default_export_path(journal_root)).expanduser().resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = final_path.parent / f"{final_path.name}.partial"
    manifest = _build_export_manifest(journal_root)

    fd = os.open(partial_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as raw_handle:
            with zipfile.ZipFile(
                raw_handle,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                for root_name in EXPORT_TOP_LEVEL_DIRS:
                    archive.writestr(f"{root_name}/", b"")
                for source_path, arcname in _iter_export_members(journal_root):
                    archive.write(source_path, arcname=arcname)
                archive.writestr("_export.json", json.dumps(manifest, indent=2))
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        os.replace(partial_path, final_path)
    except Exception:
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return final_path
