# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Read-only validation for exported journal archives."""

# L1 read-only validator. L2 will add a JournalArchiveImporter class to this
# module; the validator must remain free of writes (scope §4 / AGENTS.md §7 L7).

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATE_RE = re.compile(r"^\d{8}$")
JOURNAL_ROOT_ENTRIES = {"chronicle", "entities", "facets", "imports", "_export.json"}
MANIFEST_FIELDS = (
    "solstone_version",
    "exported_at",
    "source_journal",
    "day_count",
    "entity_count",
    "facet_count",
)


@dataclass
class ArchiveWarning:
    code: str
    message: str


@dataclass
class ArchiveValidation:
    ok: bool
    archive_path: Path
    root_prefix: str
    manifest: dict[str, Any] | None
    warnings: list[ArchiveWarning] = field(default_factory=list)
    day_count: int = 0
    entity_count: int = 0
    facet_count: int = 0


def _visible_name(name: str) -> str | None:
    parts = [part for part in name.split("/") if part]
    if not parts:
        return None
    if parts[0] == "__MACOSX":
        return None
    if parts[-1] == ".DS_Store":
        return None
    return "/".join(parts)


def _top_level_names(names: list[str]) -> set[str]:
    return {name.split("/", 1)[0] for name in names if name}


def _resolve_root_prefix(names: list[str]) -> str | None:
    top_level = _top_level_names(names)
    if top_level & JOURNAL_ROOT_ENTRIES:
        return ""

    top_level_dirs = {
        name for name in top_level if any(item.startswith(f"{name}/") for item in names)
    }
    if len(top_level_dirs) != 1:
        return None

    wrapper = next(iter(top_level_dirs))
    nested = []
    prefix = f"{wrapper}/"
    for name in names:
        if name.startswith(prefix):
            stripped = name[len(prefix) :]
            if stripped:
                nested.append(stripped)
    if _top_level_names(nested) & JOURNAL_ROOT_ENTRIES:
        return prefix
    return None


def _scan_counts(names: list[str], root_prefix: str) -> tuple[int, int, int]:
    day_dirs: set[str] = set()
    entity_slugs: set[str] = set()
    facet_slugs: set[str] = set()

    for name in names:
        if root_prefix and not name.startswith(root_prefix):
            continue
        relative_name = name[len(root_prefix) :] if root_prefix else name
        parts = relative_name.split("/")
        if len(parts) >= 2 and parts[0] == "chronicle" and DATE_RE.match(parts[1]):
            day_dirs.add(parts[1])
        if len(parts) == 3 and parts[0] == "entities" and parts[2] == "entity.json":
            entity_slugs.add(parts[1])
        if len(parts) == 3 and parts[0] == "facets" and parts[2] == "facet.json":
            facet_slugs.add(parts[1])

    return len(day_dirs), len(entity_slugs), len(facet_slugs)


def _build_fatal(
    archive_path: Path,
    code: str,
    message: str,
    *,
    warnings: list[ArchiveWarning] | None = None,
) -> ArchiveValidation:
    all_warnings = list(warnings or [])
    all_warnings.append(ArchiveWarning(code=code, message=message))
    return ArchiveValidation(
        ok=False,
        archive_path=archive_path,
        root_prefix="",
        manifest=None,
        warnings=all_warnings,
    )


def validate_journal_archive(
    path: Path,
    *,
    max_size_bytes: int = 50 * 1024**3,
) -> ArchiveValidation:
    archive_path = path.expanduser().resolve()
    warnings: list[ArchiveWarning] = []

    if not archive_path.exists():
        return _build_fatal(
            archive_path,
            "archive-not-found",
            "Archive file does not exist.",
        )

    if archive_path.stat().st_size > max_size_bytes:
        return _build_fatal(
            archive_path,
            "archive-too-large",
            "Archive exceeds 50 GiB safety limit.",
        )

    if not zipfile.is_zipfile(archive_path):
        return _build_fatal(
            archive_path,
            "archive-invalid-zip",
            "Archive is not a readable ZIP file.",
        )

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if any(info.flag_bits & 0x1 for info in infos):
                return _build_fatal(
                    archive_path,
                    "archive-encrypted",
                    "Encrypted ZIP entries are not supported.",
                )

            visible_names = [
                name
                for info in infos
                if (name := _visible_name(info.filename)) is not None
            ]
            root_prefix = _resolve_root_prefix(visible_names)
            if root_prefix is None:
                return _build_fatal(
                    archive_path,
                    "archive-structure-invalid",
                    "Archive does not contain a recognizable journal root.",
                    warnings=warnings,
                )

            day_count, entity_count, facet_count = _scan_counts(
                visible_names, root_prefix
            )
            manifest: dict[str, Any] | None = None
            manifest_name = f"{root_prefix}_export.json"
            try:
                manifest_bytes = archive.read(manifest_name)
            except KeyError:
                warnings.append(
                    ArchiveWarning(
                        code="manifest-missing",
                        message="Manifest is missing optional export metadata.",
                    )
                )
            else:
                try:
                    manifest = json.loads(manifest_bytes.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    warnings.append(
                        ArchiveWarning(
                            code="manifest-unparseable",
                            message="Manifest could not be parsed as JSON.",
                        )
                    )
                    manifest = None

            if manifest is not None:
                missing_fields = [
                    field for field in MANIFEST_FIELDS if field not in manifest
                ]
                if missing_fields:
                    warnings.append(
                        ArchiveWarning(
                            code="manifest-fields-missing",
                            message=(
                                "Manifest is missing required export metadata fields: "
                                + ", ".join(missing_fields)
                            ),
                        )
                    )
                for field_name, actual_value in (
                    ("day_count", day_count),
                    ("entity_count", entity_count),
                    ("facet_count", facet_count),
                ):
                    manifest_value = manifest.get(field_name)
                    if (
                        isinstance(manifest_value, int)
                        and manifest_value != actual_value
                    ):
                        warnings.append(
                            ArchiveWarning(
                                code="manifest-count-mismatch",
                                message=(
                                    f"Manifest {field_name}={manifest_value} does not match "
                                    f"archive contents ({actual_value})."
                                ),
                            )
                        )

            has_chronicle = any(
                name == f"{root_prefix}chronicle"
                or name.startswith(f"{root_prefix}chronicle/")
                for name in visible_names
            )
            if not has_chronicle:
                warnings.append(
                    ArchiveWarning(
                        code="chronicle-missing",
                        message="Archive has no chronicle/ directory; treating as partial journal.",
                    )
                )

            return ArchiveValidation(
                ok=True,
                archive_path=archive_path,
                root_prefix=root_prefix,
                manifest=manifest,
                warnings=warnings,
                day_count=day_count,
                entity_count=entity_count,
                facet_count=facet_count,
            )
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return _build_fatal(
            archive_path,
            "archive-invalid-zip",
            "Archive is not a readable ZIP file.",
            warnings=warnings,
        )
