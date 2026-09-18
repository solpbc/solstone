# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Platform manifest generation engine."""

from pathlib import Path
from typing import Any, Optional

from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.ingest import (
    IngestedComponent,
    ingest_desktop,
    ingest_journal,
    ingest_tmux,
)
from solstone_platform.pins import PinSet, embedded_pins
from solstone_platform.refusals import (
    RELEASE_COHERENCE,
    SCHEMA_INVALID,
    VERSION_INVALID,
    Refusal,
)
from solstone_platform.schema import is_valid_semver, validate_platform_manifest


def load_minimum_installer_revision(repo_root: Path) -> int:
    compat_path = repo_root / "compat" / "minimum_installer_revision"
    if not compat_path.is_file():
        raise Refusal(SCHEMA_INVALID, "compat/minimum_installer_revision file not found")
    content = compat_path.read_text(encoding="utf-8").strip()
    try:
        rev = int(content)
        if rev < 1:
            raise ValueError()
        return rev
    except ValueError as err:
        raise Refusal(SCHEMA_INVALID, f"invalid minimum_installer_revision: '{content}'") from err


def load_handler_contract(repo_root: Path, comp_name: str) -> dict[str, Any]:
    contract_path = repo_root / "contracts" / f"{comp_name}.v1.json"
    if not contract_path.is_file():
        raise Refusal(SCHEMA_INVALID, f"handler contract not found: {contract_path}")
    data = parse_json_strict(contract_path.read_bytes())
    if not isinstance(data, dict):
        raise Refusal(SCHEMA_INVALID, f"invalid contract format for {comp_name}")
    return data


def build_component_manifest(
    ingested: IngestedComponent,
    contract: dict[str, Any],
) -> dict[str, Any]:
    comp_obj: dict[str, Any] = {
        "version": ingested.version,
        "handler_contract_version": contract["contract_version"],
        "install_entrypoint": contract["install_entrypoint"],
        "uninstall_service_entrypoint": contract["uninstall_service_entrypoint"],
    }
    if ingested.provenance is not None:
        comp_obj["provenance"] = ingested.provenance

    comp_obj["arches"] = ingested.arches
    return comp_obj


def generate_platform_manifest(
    version: str,
    lane: str,
    created_unix: int,
    source_commit: str,
    platform_key_id: str,
    repo_root: Path,
    journal_dir: Path,
    desktop_dir: Path,
    tmux_dir: Path,
    journal_origin: str,
    bootstrap_file: Optional[Path] = None,
    pins: Optional[PinSet] = None,
) -> bytes:
    """Generate a canonical, validated platform.json manifest."""
    if not is_valid_semver(version):
        raise Refusal(VERSION_INVALID, f"invalid platform version: '{version}'")

    if pins is None:
        pins = embedded_pins()

    min_installer_rev = load_minimum_installer_revision(repo_root)

    # Ingest components
    journal_comp = ingest_journal(
        native_dir=journal_dir,
        lane=lane,
        origin=journal_origin,
        bootstrap_file=bootstrap_file,
        pin=pins.journal,
    )
    desktop_comp = ingest_desktop(
        native_dir=desktop_dir,
        pin=pins.desktop,
    )
    tmux_comp = ingest_tmux(
        native_dir=tmux_dir,
        pin=pins.tmux,
    )

    journal_contract = load_handler_contract(repo_root, "journal")
    desktop_contract = load_handler_contract(repo_root, "desktop")
    tmux_contract = load_handler_contract(repo_root, "tmux")

    components = {
        "journal": build_component_manifest(journal_comp, journal_contract),
        "desktop": build_component_manifest(desktop_comp, desktop_contract),
        "tmux": build_component_manifest(tmux_comp, tmux_contract),
    }

    manifest = {
        "schema_version": 1,
        "protocol_version": 1,
        "version": version,
        "lane": lane,
        "created_unix": created_unix,
        "platform_key_id": platform_key_id,
        "minimum_installer_revision": min_installer_rev,
        "source_commit": source_commit,
        "components": components,
    }

    # Validate against full schema rules
    validate_platform_manifest(manifest)

    # Serialize to RFC 8785 canonical bytes
    return canonical_json_bytes(manifest)
