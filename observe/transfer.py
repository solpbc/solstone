# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Transfer observed segments between solstone instances.

Provides export and import commands for creating and unpacking day archives
containing fully-processed observation segments.

Usage:
    sol transfer export --day YYYYMMDD [--output PATH]
    sol transfer import --archive PATH [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import tarfile
import time
from pathlib import Path
from typing import Any

from think.callosum import callosum_send
from think.utils import get_journal, now_ms, segment_key, setup_cli

from .utils import compute_file_sha256, find_available_segment

logger = logging.getLogger(__name__)

# Archive manifest version
MANIFEST_VERSION = 1


def _get_hostname() -> str:
    """Get hostname for archive naming."""
    return platform.node() or "unknown"


def _list_segment_dirs(day_dir: Path) -> list[str]:
    """List all valid segment directories in a day directory.

    Args:
        day_dir: Path to day directory

    Returns:
        List of segment keys (HHMMSS_LEN format)
    """
    segments = []
    if not day_dir.exists():
        return segments

    for entry in day_dir.iterdir():
        if entry.is_dir() and segment_key(entry.name):
            segments.append(entry.name)

    return sorted(segments)


def _build_segment_manifest(segment_dir: Path) -> dict[str, Any]:
    """Build manifest entry for a segment directory.

    Args:
        segment_dir: Path to segment directory

    Returns:
        Dict with file list and SHA256 hashes
    """
    files = []
    for file_path in sorted(segment_dir.iterdir()):
        if file_path.is_file():
            files.append(
                {
                    "name": file_path.name,
                    "sha256": compute_file_sha256(file_path),
                    "size": file_path.stat().st_size,
                }
            )
    return {"files": files}


def create_archive(day: str, output_path: Path | None = None) -> Path:
    """Create a day archive with all segments.

    Args:
        day: Day in YYYYMMDD format
        output_path: Optional output path (default: scratch/{day}_{hostname}.tgz)

    Returns:
        Path to created archive

    Raises:
        ValueError: If day directory doesn't exist or has no segments
    """
    journal = get_journal()
    day_dir = Path(journal) / day

    if not day_dir.exists():
        raise ValueError(f"Day directory does not exist: {day_dir}")

    segments = _list_segment_dirs(day_dir)
    if not segments:
        raise ValueError(f"No segments found in {day_dir}")

    # Build manifest
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "day": day,
        "created_at": now_ms(),
        "host": _get_hostname(),
        "segments": {},
    }

    for segment in segments:
        segment_dir = day_dir / segment
        manifest["segments"][segment] = _build_segment_manifest(segment_dir)

    # Determine output path (default: scratch/ in project root)
    if output_path is None:
        scratch_dir = Path(__file__).parent.parent / "scratch"
        scratch_dir.mkdir(exist_ok=True)
        output_path = scratch_dir / f"{day}_{_get_hostname()}.tgz"

    # Create archive
    logger.info(f"Creating archive: {output_path}")
    logger.info(f"  Day: {day}")
    logger.info(f"  Segments: {len(segments)}")

    with tarfile.open(output_path, "w:gz") as tar:
        # Add manifest
        manifest_json = json.dumps(manifest, indent=2).encode("utf-8")
        import io

        manifest_info = tarfile.TarInfo(name="manifest.json")
        manifest_info.size = len(manifest_json)
        manifest_info.mtime = int(time.time())
        tar.addfile(manifest_info, io.BytesIO(manifest_json))

        # Add segment directories
        for segment in segments:
            segment_dir = day_dir / segment
            for file_path in segment_dir.iterdir():
                if file_path.is_file():
                    arcname = f"{segment}/{file_path.name}"
                    tar.add(file_path, arcname=arcname)
                    logger.debug(f"  Added: {arcname}")

    total_size = output_path.stat().st_size
    logger.info(f"  Archive size: {total_size / (1024 * 1024):.1f} MB")

    return output_path


def _read_manifest(archive_path: Path) -> dict[str, Any]:
    """Read and validate manifest from archive.

    Args:
        archive_path: Path to archive file

    Returns:
        Manifest dict

    Raises:
        ValueError: If manifest is missing or invalid
    """
    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            manifest_file = tar.extractfile("manifest.json")
            if manifest_file is None:
                raise ValueError("manifest.json not found in archive")
            manifest = json.load(manifest_file)
        except KeyError:
            raise ValueError("manifest.json not found in archive")

    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported manifest version: {manifest.get('version')} "
            f"(expected {MANIFEST_VERSION})"
        )

    if "day" not in manifest or "segments" not in manifest:
        raise ValueError("Invalid manifest: missing required fields")

    return manifest


def _check_segment_match(
    day_dir: Path, segment: str, manifest_files: list[dict]
) -> bool:
    """Check if local segment matches manifest exactly.

    Args:
        day_dir: Path to day directory
        segment: Segment key
        manifest_files: List of file dicts from manifest

    Returns:
        True if all files exist with matching SHA256
    """
    segment_dir = day_dir / segment
    if not segment_dir.exists():
        return False

    manifest_by_name = {f["name"]: f["sha256"] for f in manifest_files}

    # Check all manifest files exist with correct hash
    for name, expected_sha256 in manifest_by_name.items():
        file_path = segment_dir / name
        if not file_path.exists():
            return False
        if compute_file_sha256(file_path) != expected_sha256:
            return False

    return True


def validate_archive(archive_path: Path) -> dict[str, Any]:
    """Validate archive and check for conflicts.

    Args:
        archive_path: Path to archive file

    Returns:
        Dict with validation results:
        - manifest: The parsed manifest
        - skip: List of segments to skip (already synced)
        - import_as: Dict mapping original segment -> target segment
        - deconflicted: List of segments that needed key adjustment
    """
    manifest = _read_manifest(archive_path)
    day = manifest["day"]

    journal = get_journal()
    day_dir = Path(journal) / day

    result = {
        "manifest": manifest,
        "skip": [],
        "import_as": {},
        "deconflicted": [],
    }

    for segment, segment_data in manifest["segments"].items():
        files = segment_data.get("files", [])

        if _check_segment_match(day_dir, segment, files):
            # Full match - skip
            result["skip"].append(segment)
            continue

        # Check if segment exists but doesn't match
        if (day_dir / segment).exists():
            # Need deconfliction
            new_segment = find_available_segment(day_dir, segment)
            if new_segment is None:
                raise ValueError(f"Cannot find available slot for segment {segment}")
            result["import_as"][segment] = new_segment
            result["deconflicted"].append(segment)
        else:
            # Original slot available
            result["import_as"][segment] = segment

    return result


def import_archive(
    archive_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import archive into journal.

    Args:
        archive_path: Path to archive file
        dry_run: If True, validate only without extracting

    Returns:
        Dict with import results
    """
    validation = validate_archive(archive_path)
    manifest = validation["manifest"]
    day = manifest["day"]

    logger.info(f"Importing archive: {archive_path}")
    logger.info(f"  Day: {day}")
    logger.info(f"  Source host: {manifest.get('host', 'unknown')}")
    logger.info(f"  Total segments: {len(manifest['segments'])}")
    logger.info(f"  Skip (already synced): {len(validation['skip'])}")
    logger.info(f"  Import: {len(validation['import_as'])}")
    if validation["deconflicted"]:
        logger.info(f"  Deconflicted: {len(validation['deconflicted'])}")

    if dry_run:
        logger.info("Dry run - no changes made")
        return {
            "status": "dry_run",
            "validation": validation,
        }

    if not validation["import_as"]:
        logger.info("Nothing to import - all segments already synced")
        return {
            "status": "nothing_to_import",
            "validation": validation,
        }

    # Ensure day directory exists
    journal = get_journal()
    day_dir = Path(journal) / day
    day_dir.mkdir(parents=True, exist_ok=True)

    # Extract segments
    imported = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for original_segment, target_segment in validation["import_as"].items():
            target_dir = day_dir / target_segment
            target_dir.mkdir(exist_ok=True)

            # Extract files for this segment
            prefix = f"{original_segment}/"
            for member in tar.getmembers():
                if member.name.startswith(prefix) and member.isfile():
                    # Extract to target segment directory
                    filename = member.name[len(prefix) :]
                    target_path = target_dir / filename

                    # Extract file content
                    source = tar.extractfile(member)
                    if source:
                        with open(target_path, "wb") as f:
                            f.write(source.read())

                        # Preserve modification time
                        os.utime(target_path, (member.mtime, member.mtime))

            if original_segment != target_segment:
                logger.info(f"  Imported: {original_segment} -> {target_segment}")
            else:
                logger.info(f"  Imported: {original_segment}")

            imported.append(target_segment)

    # Trigger indexer rescan via supervisor queue (fire-and-forget)
    # Supervisor serializes indexer runs to prevent concurrent writes
    logger.info(f"Requesting indexer rescan for {day}...")
    sent = callosum_send(
        "supervisor",
        "request",
        cmd=["sol", "indexer", "--rescan"],
    )
    if sent:
        logger.info("  Indexer rescan queued")
    else:
        logger.warning("  Failed to queue indexer rescan (supervisor not running?)")

    return {
        "status": "imported",
        "day": day,
        "imported": imported,
        "skipped": validation["skip"],
        "deconflicted": validation["deconflicted"],
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transfer observed segments between solstone instances"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export subcommand
    export_parser = subparsers.add_parser(
        "export", help="Create archive from day's segments"
    )
    export_parser.add_argument(
        "--day",
        required=True,
        help="Day to export (YYYYMMDD format)",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output archive path (default: scratch/{day}_{hostname}.tgz)",
    )

    # Import subcommand
    import_parser = subparsers.add_parser("import", help="Import archive into journal")
    import_parser.add_argument(
        "--archive",
        "-a",
        required=True,
        type=Path,
        help="Archive file to import",
    )
    import_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate archive without extracting",
    )

    args = setup_cli(parser)

    if args.command == "export":
        try:
            output = create_archive(args.day, args.output)
            print(f"Created archive: {output}")
        except ValueError as e:
            parser.error(str(e))

    elif args.command == "import":
        if not args.archive.exists():
            parser.error(f"Archive not found: {args.archive}")

        try:
            result = import_archive(args.archive, dry_run=args.dry_run)
            if result["status"] == "imported":
                print(f"Imported {len(result['imported'])} segments to {result['day']}")
                if result["skipped"]:
                    print(f"Skipped {len(result['skipped'])} already-synced segments")
                if result["deconflicted"]:
                    print(f"Deconflicted {len(result['deconflicted'])} segments")
            elif result["status"] == "nothing_to_import":
                print("Nothing to import - all segments already synced")
            elif result["status"] == "dry_run":
                v = result["validation"]
                print("Dry run validation:")
                print(f"  Would skip: {len(v['skip'])} segments")
                print(f"  Would import: {len(v['import_as'])} segments")
                if v["deconflicted"]:
                    print(f"  Would deconflict: {len(v['deconflicted'])} segments")
        except ValueError as e:
            parser.error(str(e))
