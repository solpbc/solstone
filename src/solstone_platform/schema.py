# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Self-contained validator for platform.v1.json metadata."""

import re
from typing import Any

from solstone_platform.refusals import (
    DESKTOP_AARCH64,
    INCOMPLETE_VARIANTS,
    LANE_INVALID,
    SCHEMA_INVALID,
    VERSION_INVALID,
    Refusal,
)

SEMVER_REGEX = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
SHA256_REGEX = re.compile(r"^[0-9a-f]{64}$")
COMMIT_REGEX = re.compile(r"^[0-9a-f]{40}$")
KEYID_REGEX = re.compile(r"^[0-9A-F]{16}$")
VERIFIER_ID_REGEX = re.compile(r"^minisign:[0-9A-F]{16}$")
VALID_LANES = {"release", "staging", "dev"}


def is_valid_semver(val: str) -> bool:
    return bool(isinstance(val, str) and SEMVER_REGEX.match(val))


def is_valid_sha256(val: str) -> bool:
    return bool(isinstance(val, str) and SHA256_REGEX.match(val))


def validate_platform_manifest(manifest: Any) -> None:
    """Validate a platform.json manifest dictionary strictly against v1 schema."""
    if not isinstance(manifest, dict):
        raise Refusal(SCHEMA_INVALID, "manifest must be a JSON object")

    expected_top_keys = {
        "schema_version",
        "protocol_version",
        "version",
        "lane",
        "created_unix",
        "platform_key_id",
        "minimum_installer_revision",
        "source_commit",
        "components",
    }
    top_keys = set(manifest.keys())
    if top_keys != expected_top_keys:
        extra = top_keys - expected_top_keys
        missing = expected_top_keys - top_keys
        if extra:
            raise Refusal(SCHEMA_INVALID, f"unexpected top-level keys: {sorted(extra)}")
        if missing:
            raise Refusal(SCHEMA_INVALID, f"missing top-level keys: {sorted(missing)}")

    if manifest["schema_version"] != 1:
        raise Refusal(SCHEMA_INVALID, f"unsupported schema_version: {manifest['schema_version']}")
    if manifest["protocol_version"] != 1:
        raise Refusal(SCHEMA_INVALID, f"unsupported protocol_version: {manifest['protocol_version']}")

    version = manifest["version"]
    if not is_valid_semver(version):
        raise Refusal(VERSION_INVALID, f"invalid platform version format '{version}'")

    lane = manifest["lane"]
    if lane not in VALID_LANES:
        raise Refusal(LANE_INVALID, f"invalid lane '{lane}', must be one of {sorted(VALID_LANES)}")

    created_unix = manifest["created_unix"]
    if not isinstance(created_unix, int) or isinstance(created_unix, bool) or created_unix < 0:
        raise Refusal(SCHEMA_INVALID, f"created_unix must be non-negative integer, got {created_unix}")

    platform_key_id = manifest["platform_key_id"]
    if not isinstance(platform_key_id, str) or not KEYID_REGEX.match(platform_key_id):
        raise Refusal(SCHEMA_INVALID, f"invalid platform_key_id: '{platform_key_id}'")

    min_rev = manifest["minimum_installer_revision"]
    if not isinstance(min_rev, int) or isinstance(min_rev, bool) or min_rev < 1:
        raise Refusal(SCHEMA_INVALID, f"minimum_installer_revision must be positive integer, got {min_rev}")

    source_commit = manifest["source_commit"]
    if not isinstance(source_commit, str) or not COMMIT_REGEX.match(source_commit):
        raise Refusal(SCHEMA_INVALID, f"source_commit must be 40-char lowercase hex, got '{source_commit}'")

    components = manifest["components"]
    if not isinstance(components, dict):
        raise Refusal(SCHEMA_INVALID, "components must be an object")

    comp_keys = set(components.keys())
    expected_components = {"journal", "desktop", "tmux"}
    if comp_keys != expected_components:
        extra = comp_keys - expected_components
        missing = expected_components - comp_keys
        if extra:
            raise Refusal(SCHEMA_INVALID, f"unexpected component keys: {sorted(extra)}")
        if missing:
            raise Refusal(SCHEMA_INVALID, f"missing component keys: {sorted(missing)}")

    for comp_name in ["journal", "desktop", "tmux"]:
        _validate_component(comp_name, components[comp_name])


def _validate_component(name: str, comp: Any) -> None:
    if not isinstance(comp, dict):
        raise Refusal(SCHEMA_INVALID, f"component '{name}' must be an object")

    base_keys = {
        "version",
        "handler_contract_version",
        "install_entrypoint",
        "uninstall_service_entrypoint",
        "arches",
    }
    if name == "journal":
        expected_keys = base_keys | {"provenance"}
    else:
        expected_keys = base_keys

    actual_keys = set(comp.keys())
    if actual_keys != expected_keys:
        extra = actual_keys - expected_keys
        missing = expected_keys - actual_keys
        if extra:
            raise Refusal(SCHEMA_INVALID, f"component '{name}' has unexpected keys: {sorted(extra)}")
        if missing:
            raise Refusal(SCHEMA_INVALID, f"component '{name}' missing required keys: {sorted(missing)}")

    if not is_valid_semver(comp["version"]):
        raise Refusal(VERSION_INVALID, f"component '{name}' invalid version format '{comp['version']}'")

    hcv = comp["handler_contract_version"]
    if not isinstance(hcv, int) or isinstance(hcv, bool) or hcv < 1:
        raise Refusal(SCHEMA_INVALID, f"component '{name}' handler_contract_version must be positive integer")

    if not isinstance(comp["install_entrypoint"], str) or not comp["install_entrypoint"]:
        raise Refusal(SCHEMA_INVALID, f"component '{name}' invalid install_entrypoint")
    if not isinstance(comp["uninstall_service_entrypoint"], str) or not comp["uninstall_service_entrypoint"]:
        raise Refusal(SCHEMA_INVALID, f"component '{name}' invalid uninstall_service_entrypoint")

    if name == "journal":
        _validate_journal_provenance(comp["provenance"])

    arches = comp["arches"]
    if not isinstance(arches, dict):
        raise Refusal(SCHEMA_INVALID, f"component '{name}' arches must be an object")

    if name == "desktop":
        if "aarch64" in arches:
            raise Refusal(DESKTOP_AARCH64, "desktop component must not contain aarch64 architecture")
        if set(arches.keys()) != {"x86_64"}:
            raise Refusal(SCHEMA_INVALID, f"desktop arches must only contain x86_64, got {sorted(arches.keys())}")
    else:
        if set(arches.keys()) != {"x86_64", "aarch64"}:
            raise Refusal(SCHEMA_INVALID, f"component '{name}' arches must contain x86_64 and aarch64")

    for arch_name, variants in arches.items():
        _validate_arch_variants(name, arch_name, variants)


def _validate_journal_provenance(prov: Any) -> None:
    if not isinstance(prov, dict):
        raise Refusal(SCHEMA_INVALID, "journal provenance must be an object")

    expected_prov_keys = {
        "bootstrap",
        "upgrade_epoch",
        "state_reader_min",
        "state_reader_max",
        "retention_window",
    }
    if set(prov.keys()) != expected_prov_keys:
        raise Refusal(SCHEMA_INVALID, f"journal provenance keys mismatch: expected {sorted(expected_prov_keys)}, got {sorted(prov.keys())}")

    bootstrap = prov["bootstrap"]
    if not isinstance(bootstrap, dict):
        raise Refusal(SCHEMA_INVALID, "journal bootstrap provenance must be an object")

    expected_boot_keys = {"url", "sha256", "contract_version"}
    if set(bootstrap.keys()) != expected_boot_keys:
        raise Refusal(SCHEMA_INVALID, f"bootstrap provenance keys mismatch: {sorted(bootstrap.keys())}")

    if not isinstance(bootstrap["url"], str) or not bootstrap["url"].startswith(("http://", "https://")):
        raise Refusal(SCHEMA_INVALID, f"invalid bootstrap url: {bootstrap.get('url')}")
    if not is_valid_sha256(bootstrap["sha256"]):
        raise Refusal(SCHEMA_INVALID, f"invalid bootstrap sha256: {bootstrap.get('sha256')}")
    bcv = bootstrap["contract_version"]
    if not isinstance(bcv, int) or isinstance(bcv, bool) or bcv < 1:
        raise Refusal(SCHEMA_INVALID, "bootstrap contract_version must be positive integer")

    if not isinstance(prov["upgrade_epoch"], str) or not prov["upgrade_epoch"]:
        raise Refusal(SCHEMA_INVALID, "invalid upgrade_epoch in journal provenance")
    if not is_valid_semver(prov["state_reader_min"]):
        raise Refusal(SCHEMA_INVALID, "invalid state_reader_min format in journal provenance")
    if not is_valid_semver(prov["state_reader_max"]):
        raise Refusal(SCHEMA_INVALID, "invalid state_reader_max format in journal provenance")
    rw = prov["retention_window"]
    if not isinstance(rw, int) or isinstance(rw, bool) or rw < 1:
        raise Refusal(SCHEMA_INVALID, "retention_window must be positive integer")


def _validate_arch_variants(comp_name: str, arch: str, variants: Any) -> None:
    if not isinstance(variants, dict):
        raise Refusal(SCHEMA_INVALID, f"variants for {comp_name}/{arch} must be an object")

    expected_variants = {"tree", "deb", "rpm"}
    if set(variants.keys()) != expected_variants:
        raise Refusal(INCOMPLETE_VARIANTS, f"{comp_name}/{arch} must contain tree, deb, and rpm; got {sorted(variants.keys())}")

    for variant_type in ["tree", "deb", "rpm"]:
        _validate_variant_object(comp_name, arch, variant_type, variants[variant_type])


def _validate_variant_object(comp_name: str, arch: str, variant_type: str, var_obj: Any) -> None:
    if not isinstance(var_obj, dict):
        raise Refusal(SCHEMA_INVALID, f"variant {comp_name}/{arch}/{variant_type} must be an object")

    expected_keys = {
        "filename",
        "sha256",
        "bytes",
        "native_target",
        "executable",
        "payload_build_id",
        "archive_inventory",
        "authority",
    }
    # Optional package_identity for deb/rpm
    actual_keys = set(var_obj.keys())
    if "package_identity" in actual_keys:
        actual_keys_check = actual_keys - {"package_identity"}
    else:
        actual_keys_check = actual_keys

    if actual_keys_check != expected_keys:
        raise Refusal(SCHEMA_INVALID, f"variant {comp_name}/{arch}/{variant_type} keys mismatch: {sorted(actual_keys)}")

    if not isinstance(var_obj["filename"], str) or not var_obj["filename"]:
        raise Refusal(SCHEMA_INVALID, f"invalid filename in {comp_name}/{arch}/{variant_type}")
    if not is_valid_sha256(var_obj["sha256"]):
        raise Refusal(SCHEMA_INVALID, f"invalid sha256 in {comp_name}/{arch}/{variant_type}")

    byte_len = var_obj["bytes"]
    if not isinstance(byte_len, int) or isinstance(byte_len, bool) or byte_len < 1:
        raise Refusal(SCHEMA_INVALID, f"invalid byte length in {comp_name}/{arch}/{variant_type}")

    target = var_obj["native_target"]
    if target not in {"linux-x86_64", "linux-aarch64"}:
        raise Refusal(SCHEMA_INVALID, f"invalid native_target '{target}' in {comp_name}/{arch}/{variant_type}")

    pkg_id = var_obj.get("package_identity")
    if variant_type in {"deb", "rpm"}:
        if not isinstance(pkg_id, dict):
            raise Refusal(SCHEMA_INVALID, f"package_identity must be an object for {variant_type}")
        if set(pkg_id.keys()) != {"name", "version", "arch"}:
            raise Refusal(SCHEMA_INVALID, f"package_identity keys mismatch in {variant_type}: {sorted(pkg_id.keys())}")
        for k in ["name", "version", "arch"]:
            if not isinstance(pkg_id[k], str) or not pkg_id[k]:
                raise Refusal(SCHEMA_INVALID, f"package_identity.{k} must be non-empty string in {variant_type}")
    elif pkg_id is not None:
        raise Refusal(SCHEMA_INVALID, f"package_identity must be null or omitted for tree variant")

    exe = var_obj["executable"]
    if not isinstance(exe, dict) or set(exe.keys()) != {"name", "sha256", "version_command"}:
        raise Refusal(SCHEMA_INVALID, f"invalid executable object in {comp_name}/{arch}/{variant_type}")
    if not isinstance(exe["name"], str) or not exe["name"]:
        raise Refusal(SCHEMA_INVALID, "executable.name must be non-empty string")
    if not is_valid_sha256(exe["sha256"]):
        raise Refusal(SCHEMA_INVALID, "executable.sha256 must be 64-char hex")
    if not isinstance(exe["version_command"], list) or not exe["version_command"] or not all(isinstance(x, str) and x for x in exe["version_command"]):
        raise Refusal(SCHEMA_INVALID, "executable.version_command must be non-empty list of non-empty strings")

    if not is_valid_sha256(var_obj["payload_build_id"]):
        raise Refusal(SCHEMA_INVALID, "payload_build_id must be 64-char hex")

    inv = var_obj["archive_inventory"]
    if not isinstance(inv, list):
        raise Refusal(SCHEMA_INVALID, "archive_inventory must be a list")
    for item in inv:
        _validate_inventory_entry(item)

    auth = var_obj["authority"]
    _validate_authority(auth)


def _validate_inventory_entry(item: Any) -> None:
    if not isinstance(item, dict):
        raise Refusal(SCHEMA_INVALID, "inventory entry must be an object")

    kind = item.get("kind")
    if kind not in {"file", "dir", "symlink"}:
        raise Refusal(SCHEMA_INVALID, f"invalid inventory entry kind: {kind}")

    path = item.get("path")
    if not isinstance(path, str) or not path or path.startswith("/"):
        raise Refusal(SCHEMA_INVALID, f"invalid inventory entry path: {path}")

    size = item.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise Refusal(SCHEMA_INVALID, f"invalid inventory entry size: {size}")

    if kind == "file":
        if "sha256" not in item or not is_valid_sha256(item["sha256"]):
            raise Refusal(SCHEMA_INVALID, "file inventory entry must have valid sha256")
        expected_keys = {"path", "kind", "size", "sha256"}
    elif kind == "dir":
        if size != 0:
            raise Refusal(SCHEMA_INVALID, "directory inventory entry must have size 0")
        expected_keys = {"path", "kind", "size"}
    elif kind == "symlink":
        if size != 0:
            raise Refusal(SCHEMA_INVALID, "symlink inventory entry must have size 0")
        if "link_target" not in item or not isinstance(item["link_target"], str):
            raise Refusal(SCHEMA_INVALID, "symlink inventory entry must have link_target")
        expected_keys = {"path", "kind", "size", "link_target"}

    if set(item.keys()) != expected_keys:
        raise Refusal(SCHEMA_INVALID, f"inventory entry keys mismatch for kind {kind}: {sorted(item.keys())}")


def _validate_authority(auth: Any) -> None:
    if not isinstance(auth, dict):
        raise Refusal(SCHEMA_INVALID, "authority must be an object")

    auth_type = auth.get("type")
    verifier_id = auth.get("verifier_id")
    if not isinstance(verifier_id, str) or not VERIFIER_ID_REGEX.match(verifier_id):
        raise Refusal(SCHEMA_INVALID, f"invalid authority verifier_id '{verifier_id}'")

    if auth_type == "journal-manifest-release-bootstrap":
        expected = {"type", "verifier_id", "manifest_sha256", "signature_sha256", "release_sha256", "bootstrap_sha256"}
    elif auth_type == "desktop-rust-release-manifest":
        expected = {"type", "verifier_id", "manifest_sha256", "signature_sha256"}
    elif auth_type == "tmux-sha256sums-target":
        expected = {"type", "verifier_id", "sums_sha256", "signature_sha256", "target_json_sha256"}
    else:
        raise Refusal(SCHEMA_INVALID, f"invalid authority type: {auth_type}")

    if set(auth.keys()) != expected:
        raise Refusal(SCHEMA_INVALID, f"authority keys mismatch for type {auth_type}: {sorted(auth.keys())}")

    for k in expected - {"type", "verifier_id"}:
        if not is_valid_sha256(auth[k]):
            raise Refusal(SCHEMA_INVALID, f"invalid sha256 for authority field {k}")
