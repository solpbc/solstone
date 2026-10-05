# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Self-contained validator for platform release metadata and catalogue identity."""

import json
import re
from typing import Any

from solstone_platform.archive import normalize_member_path, validate_symlink_target
from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.refusals import (
    DESKTOP_AARCH64,
    INCOMPLETE_VARIANTS,
    LANE_INVALID,
    RELEASE_COHERENCE,
    SCHEMA_INVALID,
    VERSION_INVALID,
    Refusal,
)
from solstone_platform.targets import TARGET_MAPPINGS

MAX_PLATFORM_BYTES = 4 * 1024 * 1024
MAX_PLATFORM_DEPTH = 64
MAX_SEMVER_DIGITS = 20
SCHEMA_2_INSTALLER_FLOOR = 8
MAX_CATALOGUE_REVISION = 10**MAX_SEMVER_DIGITS - 1
CATALOGUE_REVISION_REGEX = re.compile(rf"[1-9][0-9]{{0,{MAX_SEMVER_DIGITS - 1}}}")

SEMVER_REGEX = re.compile(
    rf"^(0|[1-9][0-9]{{0,{MAX_SEMVER_DIGITS - 1}}})\."
    rf"(0|[1-9][0-9]{{0,{MAX_SEMVER_DIGITS - 1}}})\."
    rf"(0|[1-9][0-9]{{0,{MAX_SEMVER_DIGITS - 1}}})$"
)
CATALOGUE_REVISED_REGEX = re.compile(
    rf"{SEMVER_REGEX.pattern[:-1]}-r({CATALOGUE_REVISION_REGEX.pattern})$"
)
MAX_LATEST_BYTES = 4 * MAX_SEMVER_DIGITS + 6
SHA256_REGEX = re.compile(r"^[0-9a-f]{64}$")
COMMIT_REGEX = re.compile(r"^[0-9a-f]{40}$")
KEYID_REGEX = re.compile(r"^[0-9A-F]{16}$")
VERIFIER_ID_REGEX = re.compile(r"^minisign:[0-9A-F]{16}$")
VALID_LANES = {"release", "staging", "dev"}

SCHEMA_1_KEYS: tuple[str, ...] = (
    "schema_version",
    "protocol_version",
    "version",
    "lane",
    "created_unix",
    "platform_key_id",
    "minimum_installer_revision",
    "source_commit",
    "components",
)
SCHEMA_2_KEYS: tuple[str, ...] = SCHEMA_1_KEYS + ("catalogue_revision",)

COMPONENT_CONTRACTS = {
    "journal": {
        "handler_contract_version": 1,
        "install_entrypoint": "install-journal",
        "uninstall_service_entrypoint": "uninstall-journal-service",
        "executable_name": "solstone",
        "version_command": ["solstone", "journal", "--version"],
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
    return bool(isinstance(val, str) and SEMVER_REGEX.fullmatch(val))


def is_valid_sha256(val: str) -> bool:
    return bool(isinstance(val, str) and SHA256_REGEX.match(val))


def parse_catalogue_coordinate(text: str) -> tuple[str, tuple[int, int, int], int | None]:
    if not isinstance(text, str):
        raise Refusal(VERSION_INVALID, f"invalid catalogue coordinate '{text}'")
    m_rev = CATALOGUE_REVISED_REGEX.fullmatch(text)
    if m_rev:
        triple = (int(m_rev.group(1)), int(m_rev.group(2)), int(m_rev.group(3)))
        rev = int(m_rev.group(4))
        return "revised", triple, rev
    if is_valid_semver(text):
        parts = text.split(".")
        triple = (int(parts[0]), int(parts[1]), int(parts[2]))
        return "bare", triple, None
    raise Refusal(VERSION_INVALID, f"invalid catalogue coordinate format '{text}'")


def catalogue_coordinate(manifest: dict[str, Any]) -> str:
    schema_ver = manifest.get("schema_version")
    if schema_ver == 1:
        return str(manifest.get("version", ""))
    if schema_ver == 2:
        return f"{manifest.get('version', '')}-r{manifest.get('catalogue_revision', '')}"
    raise Refusal(SCHEMA_INVALID, f"unsupported schema_version: {schema_ver}")


def compare_catalogue_coordinates(a: str, b: str) -> int:
    kind_a, triple_a, rev_a = parse_catalogue_coordinate(a)
    kind_b, triple_b, rev_b = parse_catalogue_coordinate(b)

    if kind_a == "revised" and kind_b == "bare":
        return 1
    if kind_a == "bare" and kind_b == "revised":
        return -1
    if kind_a == "bare" and kind_b == "bare":
        if triple_a < triple_b:
            return -1
        if triple_a > triple_b:
            return 1
        return 0
    # Both revised
    if triple_a < triple_b:
        return -1
    if triple_a > triple_b:
        return 1
    assert rev_a is not None and rev_b is not None
    if rev_a < rev_b:
        return -1
    if rev_a > rev_b:
        return 1
    return 0


def is_valid_catalogue_revision(value: Any) -> bool:
    return type(value) is int and 1 <= value <= MAX_CATALOGUE_REVISION


def parse_catalogue_revision(text: str) -> int:
    if not isinstance(text, str) or not CATALOGUE_REVISION_REGEX.fullmatch(text):
        raise Refusal(VERSION_INVALID, f"invalid catalogue revision: {text!r}")
    return int(text)


def allocate_catalogue_revision(
    base_coordinate: str,
    candidate_journal_version: str,
    explicit_revision: int | None = None,
) -> int:
    if not is_valid_semver(candidate_journal_version):
        raise Refusal(VERSION_INVALID, f"invalid candidate journal version: {candidate_journal_version!r}")
    if explicit_revision is not None and not is_valid_catalogue_revision(explicit_revision):
        raise Refusal(VERSION_INVALID, f"invalid catalogue revision: {explicit_revision!r}")
    kind, base_core, base_revision = parse_catalogue_coordinate(base_coordinate)
    candidate_core = tuple(int(part) for part in candidate_journal_version.split("."))
    minimum = 1
    if kind == "revised":
        if candidate_core < base_core:
            raise Refusal(RELEASE_COHERENCE, "candidate journal version is older than the base")
        if candidate_core == base_core:
            minimum = base_revision + 1
    if minimum > MAX_CATALOGUE_REVISION:
        raise Refusal(RELEASE_COHERENCE, "catalogue revision decimal exhausted")
    revision = minimum if explicit_revision is None else explicit_revision
    if revision < minimum:
        raise Refusal(RELEASE_COHERENCE, "catalogue revision must be greater than the base")
    return revision


def validate_catalogue_transition(
    base: dict[str, Any],
    candidate: dict[str, Any],
    required_schema2_floor: int,
) -> None:
    base_schema = base.get("schema_version")
    cand_schema = candidate.get("schema_version")
    for field in ("protocol_version", "lane", "platform_key_id"):
        if candidate.get(field) != base.get(field):
            raise Refusal(RELEASE_COHERENCE, f"recut changed protected field {field}")

    if base_schema == 1 and cand_schema == 2:
        cand_floor = candidate.get("minimum_installer_revision")
        if cand_floor != required_schema2_floor:
            raise Refusal(
                RELEASE_COHERENCE,
                f"schema 1 to schema 2 transition requires minimum_installer_revision {required_schema2_floor}, got {cand_floor}",
            )
        return

    if base_schema == 2 and cand_schema == 2:
        for field in (
            "schema_version",
            "protocol_version",
            "lane",
            "platform_key_id",
            "minimum_installer_revision",
        ):
            if candidate.get(field) != base.get(field):
                raise Refusal(
                    RELEASE_COHERENCE,
                    f"schema 2 transition changed protected field {field}: {candidate.get(field)} vs {base.get(field)}",
                )
        return

    if base_schema == 1 and cand_schema == 1:
        if candidate.get("minimum_installer_revision") != base.get("minimum_installer_revision"):
            raise Refusal(
                RELEASE_COHERENCE,
                f"schema 1 continuity cannot change minimum_installer_revision: {candidate.get('minimum_installer_revision')} vs {base.get('minimum_installer_revision')}",
            )
        return

    if base_schema == 2 and cand_schema == 1:
        raise Refusal(RELEASE_COHERENCE, "schema 2 to schema 1 rollback refused")

    raise Refusal(RELEASE_COHERENCE, f"unsupported schema transition from {base_schema} to {cand_schema}")


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
    """Validate a platform.json manifest dictionary strictly against v1 or v2 schema."""
    if not isinstance(manifest, dict):
        raise Refusal(SCHEMA_INVALID, "manifest must be a JSON object")

    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise Refusal(SCHEMA_INVALID, f"unsupported schema_version: {schema_version}")

    if schema_version == 1:
        expected_top_keys = set(SCHEMA_1_KEYS)
    else:
        expected_top_keys = set(SCHEMA_2_KEYS)

    top_keys = set(manifest.keys())
    if top_keys != expected_top_keys:
        extra = top_keys - expected_top_keys
        missing = expected_top_keys - top_keys
        if extra:
            raise Refusal(SCHEMA_INVALID, f"unexpected top-level keys: {sorted(extra)}")
        if missing:
            raise Refusal(SCHEMA_INVALID, f"missing top-level keys: {sorted(missing)}")

    if type(manifest["protocol_version"]) is not int or manifest["protocol_version"] != 1:
        raise Refusal(SCHEMA_INVALID, f"unsupported protocol_version: {manifest['protocol_version']}")

    version = manifest["version"]
    if not is_valid_semver(version):
        raise Refusal(VERSION_INVALID, f"invalid platform version format '{version}'")

    if schema_version == 2:
        cat_rev = manifest["catalogue_revision"]
        if not is_valid_catalogue_revision(cat_rev):
            raise Refusal(SCHEMA_INVALID, f"invalid catalogue_revision: {cat_rev}")

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

    if schema_version == 2:
        journal_ver = components["journal"].get("version")
        if version != journal_ver:
            raise Refusal(
                SCHEMA_INVALID,
                f"schema 2 platform version '{version}' must equal components.journal.version '{journal_ver}'",
            )


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
    # Already published manifests bind the still-functional alias. Admit that
    # complete pair without allowing a mixed executable/argument identity.
    if comp_name == "journal" and exe["name"] == "journal":
        contract = {"executable_name": "journal", "version_command": ["journal", "--version"]}
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


def render_platform_schema() -> str:
    """Render the JSON Schema document for platform manifests."""
    common_props = {
        "protocol_version": {"type": "integer", "const": 1},
        "version": {
            "type": "string",
            "pattern": SEMVER_REGEX.pattern,
        },
        "lane": {"type": "string", "enum": sorted(VALID_LANES)},
        "created_unix": {"type": "integer", "minimum": 0},
        "platform_key_id": {"type": "string", "pattern": "^[0-9A-F]{16}$"},
        "minimum_installer_revision": {"type": "integer", "minimum": 1},
        "source_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "components": {
            "type": "object",
            "additionalProperties": False,
            "required": ["journal", "desktop", "tmux"],
            "properties": {
                "journal": {"$ref": "#/definitions/journal_component"},
                "desktop": {"$ref": "#/definitions/desktop_component"},
                "tmux": {"$ref": "#/definitions/tmux_component"},
            },
        },
    }

    schema_1_props = {"schema_version": {"type": "integer", "const": 1}, **common_props}
    schema_2_props = {
        "schema_version": {"type": "integer", "const": 2},
        **common_props,
        "catalogue_revision": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10**MAX_SEMVER_DIGITS - 1,
        },
    }

    definitions = {
        "journal_component": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "version",
                "handler_contract_version",
                "install_entrypoint",
                "uninstall_service_entrypoint",
                "provenance",
                "arches",
            ],
            "properties": {
                "version": {
                    "type": "string",
                    "pattern": SEMVER_REGEX.pattern,
                },
                "handler_contract_version": {
                    "type": "integer",
                    "const": COMPONENT_CONTRACTS["journal"]["handler_contract_version"],
                },
                "install_entrypoint": {
                    "type": "string",
                    "const": COMPONENT_CONTRACTS["journal"]["install_entrypoint"],
                },
                "uninstall_service_entrypoint": {
                    "type": "string",
                    "const": COMPONENT_CONTRACTS["journal"]["uninstall_service_entrypoint"],
                },
                "provenance": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "bootstrap",
                        "upgrade_epoch",
                        "state_reader_min",
                        "state_reader_max",
                        "retention_window",
                    ],
                    "properties": {
                        "bootstrap": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["url", "sha256", "contract_version"],
                            "properties": {
                                "url": {"type": "string", "format": "uri"},
                                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                                "contract_version": {"type": "integer", "const": 2},
                            },
                        },
                        "upgrade_epoch": {"type": "string", "minLength": 1},
                        "state_reader_min": {
                            "type": "string",
                            "pattern": SEMVER_REGEX.pattern,
                        },
                        "state_reader_max": {
                            "type": "string",
                            "pattern": SEMVER_REGEX.pattern,
                        },
                        "retention_window": {"type": "integer", "minimum": 1},
                    },
                },
                "arches": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x86_64", "aarch64"],
                    "properties": {
                        "x86_64": {"$ref": "#/definitions/arch_variants"},
                        "aarch64": {"$ref": "#/definitions/arch_variants"},
                    },
                },
            },
        },
        "desktop_component": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "version",
                "handler_contract_version",
                "install_entrypoint",
                "uninstall_service_entrypoint",
                "arches",
            ],
            "properties": {
                "version": {
                    "type": "string",
                    "pattern": SEMVER_REGEX.pattern,
                },
                "handler_contract_version": {
                    "type": "integer",
                    "const": COMPONENT_CONTRACTS["desktop"]["handler_contract_version"],
                },
                "install_entrypoint": {
                    "type": "string",
                    "const": COMPONENT_CONTRACTS["desktop"]["install_entrypoint"],
                },
                "uninstall_service_entrypoint": {
                    "type": "string",
                    "const": COMPONENT_CONTRACTS["desktop"]["uninstall_service_entrypoint"],
                },
                "arches": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x86_64"],
                    "properties": {
                        "x86_64": {"$ref": "#/definitions/arch_variants"},
                    },
                },
            },
        },
        "tmux_component": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "version",
                "handler_contract_version",
                "install_entrypoint",
                "uninstall_service_entrypoint",
                "arches",
            ],
            "properties": {
                "version": {
                    "type": "string",
                    "pattern": SEMVER_REGEX.pattern,
                },
                "handler_contract_version": {
                    "type": "integer",
                    "const": COMPONENT_CONTRACTS["tmux"]["handler_contract_version"],
                },
                "install_entrypoint": {
                    "type": "string",
                    "const": COMPONENT_CONTRACTS["tmux"]["install_entrypoint"],
                },
                "uninstall_service_entrypoint": {
                    "type": "string",
                    "const": COMPONENT_CONTRACTS["tmux"]["uninstall_service_entrypoint"],
                },
                "arches": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x86_64", "aarch64"],
                    "properties": {
                        "x86_64": {"$ref": "#/definitions/arch_variants"},
                        "aarch64": {"$ref": "#/definitions/arch_variants"},
                    },
                },
            },
        },
        "arch_variants": {
            "type": "object",
            "additionalProperties": False,
            "required": ["tree", "deb", "rpm"],
            "properties": {
                "tree": {"$ref": "#/definitions/variant_object"},
                "deb": {"$ref": "#/definitions/variant_object"},
                "rpm": {"$ref": "#/definitions/variant_object"},
            },
        },
        "variant_object": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "filename",
                "sha256",
                "bytes",
                "native_target",
                "executable",
                "payload_build_id",
                "archive_inventory",
                "authority",
            ],
            "properties": {
                "filename": {"type": "string", "minLength": 1},
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "bytes": {"type": "integer", "minimum": 1},
                "native_target": {
                    "type": "string",
                    "enum": ["linux-x86_64", "linux-aarch64"],
                },
                "package_identity": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "required": ["name", "version", "arch"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "version": {"type": "string", "minLength": 1},
                        "arch": {"type": "string", "minLength": 1},
                    },
                },
                "executable": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "sha256", "version_command"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "version_command": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                },
                "payload_build_id": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "archive_inventory": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/inventory_entry"},
                },
                "authority": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "verifier_id"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "journal-manifest-release-bootstrap",
                                "desktop-rust-release-manifest",
                                "tmux-sha256sums-target",
                            ],
                        },
                        "verifier_id": {"type": "string", "pattern": "^minisign:[0-9A-F]{16}$"},
                        "manifest_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "signature_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "release_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "bootstrap_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "sums_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "target_json_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    },
                },
            },
        },
        "inventory_entry": {
            "type": "object",
            "additionalProperties": False,
            "required": ["path", "kind", "size"],
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "kind": {"type": "string", "enum": ["file", "dir", "symlink"]},
                "size": {"type": "integer", "minimum": 0},
                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "link_target": {"type": "string"},
            },
        },
    }

    schema_doc = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://solstone.app/schemas/platform.v1.json",
        "title": "SolstonePlatformManifestV1",
        "$comment": (
            "Structural subset only. The canonical platform byte loader additionally enforces "
            "canonical bytes and in-object ordering, identity, digest, and cross-field invariants. "
            "Bootstrap origin/lane/version and native package/artifact coherence are contextual checks "
            "outside this JSON Schema."
        ),
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": list(SCHEMA_1_KEYS),
                "properties": schema_1_props,
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": list(SCHEMA_2_KEYS),
                "properties": schema_2_props,
            },
        ],
        "definitions": definitions,
    }
    return json.dumps(schema_doc, indent=2) + "\n"


def render_posix_identity_fragment() -> str:
    """Render POSIX shell functions for catalogue identity verification."""
    return """# BEGIN GENERATED CATALOGUE IDENTITY
CATALOGUE_MAX_DIGITS=@MAX_DIGITS@
CATALOGUE_LATEST_MAX_BYTES=@LATEST_BYTES@

is_bare_coordinate() {
    candidate_version="$1"
    is_canonical_version "$candidate_version"
}

is_revised_coordinate() {
    rc_raw="$1"
    case "$rc_raw" in
        *-r*) ;;
        *) return 1 ;;
    esac
    rc_core="${rc_raw%-r*}"
    rc_rev="${rc_raw##*-r}"
    is_canonical_version "$rc_core" || return 1
    case "$rc_rev" in
        ""|0|0*|*[!0123456789]*) return 1 ;;
    esac
    [ "${#rc_rev}" -le @MAX_DIGITS@ ] || return 1
    return 0
}

is_catalogue_coordinate() {
    is_bare_coordinate "$1" || is_revised_coordinate "$1"
}

validate_catalogue_identity() {
    vci_manifest_dir="$1"
    vci_schema="$2"
    vci_protocol="$3"
    vci_version="$4"
    vci_resolved="$5"
    if [ "$vci_protocol" != "1" ]; then
        report_exit "refusal" "schema-invalid" "manifest protocol version must be 1"
    fi
    if [ "$vci_schema" = "1" ]; then
        if [ -f "${vci_manifest_dir}/catalogue_revision" ]; then
            report_exit "refusal" "schema-invalid" "schema 1 manifest must not contain catalogue_revision"
        fi
        if ! is_bare_coordinate "$vci_version"; then
            report_exit "refusal" "schema-invalid" "manifest platform version is invalid"
        fi
        if [ "$vci_version" != "$vci_resolved" ]; then
            report_exit "refusal" "release-coherence" "manifest platform version does not match the requested coordinate"
        fi
    elif [ "$vci_schema" = "2" ]; then
        if [ ! -f "${vci_manifest_dir}/catalogue_revision" ]; then
            report_exit "refusal" "schema-invalid" "schema 2 manifest missing catalogue_revision"
        fi
        vci_rev=$(cat "${vci_manifest_dir}/catalogue_revision")
        case "$vci_rev" in
            ""|0|0*|*[!0123456789]*)
                report_exit "refusal" "schema-invalid" "manifest catalogue_revision is invalid"
                ;;
        esac
        if [ "${#vci_rev}" -gt @MAX_DIGITS@ ]; then
            report_exit "refusal" "schema-invalid" "manifest catalogue_revision exceeds maximum digits"
        fi
        if ! is_canonical_version "$vci_version"; then
            report_exit "refusal" "schema-invalid" "manifest platform version is invalid"
        fi
        vci_journal_ver=$(cat "${vci_manifest_dir}/components/journal/version" 2>/dev/null || true)
        if [ "$vci_version" != "$vci_journal_ver" ]; then
            report_exit "refusal" "schema-invalid" "schema 2 manifest version must equal components.journal.version"
        fi
        vci_derived="${vci_version}-r${vci_rev}"
        if [ "$vci_derived" != "$vci_resolved" ]; then
            report_exit "refusal" "release-coherence" "manifest catalogue coordinate does not match the requested coordinate"
        fi
    else
        report_exit "refusal" "schema-invalid" "unsupported schema_version: $vci_schema"
    fi
}
# END GENERATED CATALOGUE IDENTITY""".replace("@MAX_DIGITS@", str(MAX_SEMVER_DIGITS)).replace("@LATEST_BYTES@", str(MAX_LATEST_BYTES))


def render_posix_catalogue_keys() -> str:
    """Render AWK parser top-level keys alternation regex."""
    all_keys = sorted(set(SCHEMA_2_KEYS))
    keys_regex = "|".join(all_keys)
    return f"""    # BEGIN GENERATED CATALOGUE KEYS
    if (p ~ /^({keys_regex})$/) return 1
    # END GENERATED CATALOGUE KEYS"""
