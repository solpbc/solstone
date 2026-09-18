# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Self-contained validator for platform.v1.json metadata."""

import re
from typing import Any

from solstone_platform.archive import normalize_member_path, validate_symlink_target
from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.refusals import (
    DESKTOP_AARCH64,
    INCOMPLETE_VARIANTS,
    LANE_INVALID,
    SCHEMA_INVALID,
    VERSION_INVALID,
    Refusal,
)
from solstone_platform.targets import TARGET_MAPPINGS

MAX_PLATFORM_BYTES = 4 * 1024 * 1024
MAX_PLATFORM_DEPTH = 64
MAX_SEMVER_DIGITS = 20

SEMVER_REGEX = re.compile(
    rf"^(0|[1-9][0-9]{{0,{MAX_SEMVER_DIGITS - 1}}})\."
    rf"(0|[1-9][0-9]{{0,{MAX_SEMVER_DIGITS - 1}}})\."
    rf"(0|[1-9][0-9]{{0,{MAX_SEMVER_DIGITS - 1}}})$"
)
SHA256_REGEX = re.compile(r"^[0-9a-f]{64}$")
COMMIT_REGEX = re.compile(r"^[0-9a-f]{40}$")
KEYID_REGEX = re.compile(r"^[0-9A-F]{16}$")
VERIFIER_ID_REGEX = re.compile(r"^minisign:[0-9A-F]{16}$")
VALID_LANES = {"release", "staging", "dev"}

COMPONENT_CONTRACTS = {
    "journal": {
        "handler_contract_version": 1,
        "install_entrypoint": "install-journal",
        "uninstall_service_entrypoint": "uninstall-journal-service",
        "executable_name": "journal",
        "version_command": ["journal", "--version"],
        "authority_type": "journal-manifest-release-bootstrap",
        "package_name": "solstone-journal",
    },
    "desktop": {
        "handler_contract_version": 1,
        "install_entrypoint": "install-desktop",
        "uninstall_service_entrypoint": "uninstall-desktop-service",
        "executable_name": "solstone-linux",
        "version_command": ["solstone-linux", "--version"],
        "authority_type": "desktop-rust-release-manifest",
        "package_name": "solstone-linux",
    },
    "tmux": {
        "handler_contract_version": 1,
        "install_entrypoint": "install-tmux",
        "uninstall_service_entrypoint": "uninstall-tmux-service",
        "executable_name": "solstone-tmux",
        "version_command": ["solstone-tmux", "--version"],
        "authority_type": "tmux-sha256sums-target",
        "package_name": "solstone-tmux",
    },
}


def is_valid_semver(val: str) -> bool:
    return bool(isinstance(val, str) and SEMVER_REGEX.match(val))


def is_valid_sha256(val: str) -> bool:
    return bool(isinstance(val, str) and SHA256_REGEX.match(val))


def _semver_key(value: str) -> tuple[tuple[int, str], ...]:
    if not is_valid_semver(value):
        raise Refusal(VERSION_INVALID, f"invalid semver '{value}'")
    return tuple((len(part), part) for part in value.split("."))


def _scan_platform_json_limits(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > MAX_PLATFORM_DEPTH:
                raise Refusal(SCHEMA_INVALID, f"platform JSON nesting exceeds {MAX_PLATFORM_DEPTH}")
        elif char in "]}":
            depth -= 1
        elif "0" <= char <= "9":
            end = index + 1
            while end < len(text) and "0" <= text[end] <= "9":
                end += 1
            if end - index > MAX_SEMVER_DIGITS:
                raise Refusal(
                    SCHEMA_INVALID,
                    f"platform JSON integer token exceeds {MAX_SEMVER_DIGITS} digits",
                )
            index = end
            continue
        index += 1


def load_platform_manifest_bytes(
    data: bytes,
    *,
    canonical_refusal: str = SCHEMA_INVALID,
) -> dict[str, Any]:
    """Load exact canonical platform.v1 bytes through the total semantic validator."""
    if not isinstance(data, bytes):
        raise Refusal(SCHEMA_INVALID, "platform manifest input must be bytes")
    if len(data) > MAX_PLATFORM_BYTES:
        raise Refusal(SCHEMA_INVALID, f"platform manifest exceeds {MAX_PLATFORM_BYTES} bytes")
    if data.startswith(b"\xef\xbb\xbf"):
        raise Refusal(SCHEMA_INVALID, "platform manifest must not contain a UTF-8 BOM")
    if b"\x00" in data:
        raise Refusal(SCHEMA_INVALID, "platform manifest must not contain NUL bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as err:
        raise Refusal(SCHEMA_INVALID, f"invalid UTF-8: {err}") from err
    _scan_platform_json_limits(text)
    manifest = parse_json_strict(text)
    validate_platform_manifest(manifest)
    if canonical_json_bytes(manifest) != data:
        raise Refusal(canonical_refusal, "platform manifest must use canonical JSON bytes")
    return manifest


def validate_platform_manifest(manifest: Any) -> None:
    """Validate a platform object, translating all unsupported values to Refusal."""
    try:
        _validate_platform_manifest(manifest)
    except Refusal:
        raise
    except (TypeError, ValueError, KeyError, AttributeError, UnicodeError, RecursionError) as err:
        raise Refusal(SCHEMA_INVALID, f"invalid platform manifest value: {err}") from err


def _validate_platform_manifest(manifest: Any) -> None:
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

    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise Refusal(SCHEMA_INVALID, f"unsupported schema_version: {manifest['schema_version']}")
    if type(manifest["protocol_version"]) is not int or manifest["protocol_version"] != 1:
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

    seen_filenames: set[str] = set()
    for comp_name in ["journal", "desktop", "tmux"]:
        _validate_component(comp_name, components[comp_name], seen_filenames)


def _validate_component(name: str, comp: Any, seen_filenames: set[str]) -> None:
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

    contract = COMPONENT_CONTRACTS[name]
    hcv = comp["handler_contract_version"]
    if type(hcv) is not int or hcv != contract["handler_contract_version"]:
        raise Refusal(SCHEMA_INVALID, f"component '{name}' handler_contract_version must equal {contract['handler_contract_version']}")

    if comp["install_entrypoint"] != contract["install_entrypoint"]:
        raise Refusal(SCHEMA_INVALID, f"component '{name}' install_entrypoint mismatch")
    if comp["uninstall_service_entrypoint"] != contract["uninstall_service_entrypoint"]:
        raise Refusal(SCHEMA_INVALID, f"component '{name}' uninstall_service_entrypoint mismatch")

    if name == "journal":
        _validate_journal_provenance(comp["provenance"], comp["version"])

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
        _validate_arch_variants(name, arch_name, variants, seen_filenames)


def _validate_journal_provenance(prov: Any, component_version: str) -> None:
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
    if type(bcv) is not int or bcv != 2:
        raise Refusal(SCHEMA_INVALID, "bootstrap contract_version must equal 2")

    if not isinstance(prov["upgrade_epoch"], str) or not prov["upgrade_epoch"]:
        raise Refusal(SCHEMA_INVALID, "invalid upgrade_epoch in journal provenance")
    if not is_valid_semver(prov["state_reader_min"]):
        raise Refusal(SCHEMA_INVALID, "invalid state_reader_min format in journal provenance")
    if not is_valid_semver(prov["state_reader_max"]):
        raise Refusal(SCHEMA_INVALID, "invalid state_reader_max format in journal provenance")
    reader_min = _semver_key(prov["state_reader_min"])
    reader_max = _semver_key(prov["state_reader_max"])
    journal_version = _semver_key(component_version)
    if reader_min > reader_max:
        raise Refusal(SCHEMA_INVALID, "state reader range minimum exceeds maximum")
    if journal_version < reader_min or journal_version > reader_max:
        raise Refusal(SCHEMA_INVALID, "journal version lies outside state reader range")
    rw = prov["retention_window"]
    if not isinstance(rw, int) or isinstance(rw, bool) or rw < 1:
        raise Refusal(SCHEMA_INVALID, "retention_window must be positive integer")


def _validate_arch_variants(
    comp_name: str,
    arch: str,
    variants: Any,
    seen_filenames: set[str],
) -> None:
    if not isinstance(variants, dict):
        raise Refusal(SCHEMA_INVALID, f"variants for {comp_name}/{arch} must be an object")

    expected_variants = {"tree", "deb", "rpm"}
    if set(variants.keys()) != expected_variants:
        raise Refusal(INCOMPLETE_VARIANTS, f"{comp_name}/{arch} must contain tree, deb, and rpm; got {sorted(variants.keys())}")

    for variant_type in ["tree", "deb", "rpm"]:
        _validate_variant_object(
            comp_name,
            arch,
            variant_type,
            variants[variant_type],
            seen_filenames,
        )


def _validate_variant_object(
    comp_name: str,
    arch: str,
    variant_type: str,
    var_obj: Any,
    seen_filenames: set[str],
) -> None:
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

    filename = var_obj["filename"]
    if not isinstance(filename, str) or not filename:
        raise Refusal(SCHEMA_INVALID, f"invalid filename in {comp_name}/{arch}/{variant_type}")
    if filename in {".", ".."} or "/" in filename or "\\" in filename or any(ord(char) < 32 or ord(char) == 127 for char in filename):
        raise Refusal(SCHEMA_INVALID, f"unsafe filename in {comp_name}/{arch}/{variant_type}: '{filename}'")
    if filename in seen_filenames:
        raise Refusal(SCHEMA_INVALID, f"duplicate variant filename: '{filename}'")
    seen_filenames.add(filename)
    if not is_valid_sha256(var_obj["sha256"]):
        raise Refusal(SCHEMA_INVALID, f"invalid sha256 in {comp_name}/{arch}/{variant_type}")

    byte_len = var_obj["bytes"]
    if not isinstance(byte_len, int) or isinstance(byte_len, bool) or byte_len < 1:
        raise Refusal(SCHEMA_INVALID, f"invalid byte length in {comp_name}/{arch}/{variant_type}")

    target = var_obj["native_target"]
    expected_target = TARGET_MAPPINGS[arch].native_target
    if target != expected_target:
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
        expected_arch = TARGET_MAPPINGS[arch].deb_arch if variant_type == "deb" else TARGET_MAPPINGS[arch].rpm_arch
        contract = COMPONENT_CONTRACTS[comp_name]
        if pkg_id["name"] != contract["package_name"] or pkg_id["arch"] != expected_arch:
            raise Refusal(SCHEMA_INVALID, f"package identity mismatch in {comp_name}/{arch}/{variant_type}")
    elif pkg_id is not None:
        raise Refusal(SCHEMA_INVALID, f"package_identity must be null or omitted for tree variant")

    exe = var_obj["executable"]
    if not isinstance(exe, dict) or set(exe.keys()) != {"name", "sha256", "version_command"}:
        raise Refusal(SCHEMA_INVALID, f"invalid executable object in {comp_name}/{arch}/{variant_type}")
    contract = COMPONENT_CONTRACTS[comp_name]
    if exe["name"] != contract["executable_name"]:
        raise Refusal(SCHEMA_INVALID, f"executable.name mismatch for {comp_name}")
    if not is_valid_sha256(exe["sha256"]):
        raise Refusal(SCHEMA_INVALID, "executable.sha256 must be 64-char hex")
    if not isinstance(exe["version_command"], list) or not exe["version_command"] or not all(isinstance(x, str) and x for x in exe["version_command"]):
        raise Refusal(SCHEMA_INVALID, "executable.version_command must be non-empty list of non-empty strings")
    if exe["version_command"] != contract["version_command"]:
        raise Refusal(SCHEMA_INVALID, f"executable.version_command mismatch for {comp_name}")

    if not is_valid_sha256(var_obj["payload_build_id"]):
        raise Refusal(SCHEMA_INVALID, "payload_build_id must be 64-char hex")

    inv = var_obj["archive_inventory"]
    if not isinstance(inv, list):
        raise Refusal(SCHEMA_INVALID, "archive_inventory must be a list")
    inventory_paths: list[str] = []
    for item in inv:
        _validate_inventory_entry(item)
        inventory_paths.append(item["path"])
    if inventory_paths != sorted(inventory_paths):
        raise Refusal(SCHEMA_INVALID, "archive_inventory paths must be sorted lexicographically")
    if len(inventory_paths) != len(set(inventory_paths)):
        raise Refusal(SCHEMA_INVALID, "archive_inventory paths must be unique")
    executable_entries = [
        item
        for item in inv
        if item["kind"] == "file" and item["path"].rsplit("/", 1)[-1] == exe["name"]
    ]
    if len(executable_entries) != 1:
        raise Refusal(SCHEMA_INVALID, f"archive_inventory must contain exactly one executable '{exe['name']}'")
    if executable_entries[0]["sha256"] != exe["sha256"]:
        raise Refusal(SCHEMA_INVALID, "executable.sha256 must match archive inventory")

    auth = var_obj["authority"]
    _validate_authority(auth, comp_name)


def _validate_inventory_entry(item: Any) -> None:
    if not isinstance(item, dict):
        raise Refusal(SCHEMA_INVALID, "inventory entry must be an object")

    kind = item.get("kind")
    if kind not in {"file", "dir", "symlink"}:
        raise Refusal(SCHEMA_INVALID, f"invalid inventory entry kind: {kind}")

    path = item.get("path")
    if not isinstance(path, str) or not path:
        raise Refusal(SCHEMA_INVALID, f"invalid inventory entry path: {path}")
    try:
        normalized_path = normalize_member_path(path)
    except Refusal as err:
        raise Refusal(SCHEMA_INVALID, f"invalid inventory entry path '{path}': {err.detail or err.name}") from err
    if normalized_path != path:
        raise Refusal(SCHEMA_INVALID, f"inventory entry path is not normalized: '{path}'")

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
        try:
            validate_symlink_target(path, item["link_target"])
        except Refusal as err:
            raise Refusal(SCHEMA_INVALID, f"unsafe symlink target: {err.detail or err.name}") from err
        expected_keys = {"path", "kind", "size", "link_target"}

    if set(item.keys()) != expected_keys:
        raise Refusal(SCHEMA_INVALID, f"inventory entry keys mismatch for kind {kind}: {sorted(item.keys())}")


def _validate_authority(auth: Any, comp_name: str) -> None:
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

    expected_type = COMPONENT_CONTRACTS[comp_name]["authority_type"]
    if auth_type != expected_type:
        raise Refusal(SCHEMA_INVALID, f"authority type mismatch for component {comp_name}")

    if set(auth.keys()) != expected:
        raise Refusal(SCHEMA_INVALID, f"authority keys mismatch for type {auth_type}: {sorted(auth.keys())}")

    for k in expected - {"type", "verifier_id"}:
        if not is_valid_sha256(auth[k]):
            raise Refusal(SCHEMA_INVALID, f"invalid sha256 for authority field {k}")
