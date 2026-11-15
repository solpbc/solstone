# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Build, sign, verify, and atomically publish APT and RPM package repositories."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable

from solstone_platform.archive import scan_deb_file, scan_rpm_file
from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.destination import Destination
from solstone_platform.gpg_sign import (
    isolated_gnupghome,
    verify_openpgp_signature,
    verify_signed_rpm,
)
from solstone_platform.ingest import verify_minisign_signature
from solstone_platform.metapkg import build_metapackage_for_arch
from solstone_platform.pins import (
    MinisignPin,
    OpenPGPPin,
    require_production_openpgp_pin,
    require_production_platform_pin,
)
from solstone_platform.publish import (
    _claim_immutable_object,
    _verify_immutable_object,
)
from solstone_platform.r2 import classify_destination_key
from solstone_platform.refusals import (
    PRODUCTION_UNAVAILABLE,
    PUBLISH_INDETERMINATE,
    RELEASE_COHERENCE,
    ROLLBACK_REFUSED,
    SAME_VERSION_DIFFERENT_BYTES,
    SCHEMA_INVALID,
    SIGNATURE_PIN_MISMATCH,
    UNSAFE_FILENAME,
    Refusal,
)
from solstone_platform.repo_apt import build_apt_repository
from solstone_platform.repo_rpm import build_rpm_repository
from solstone_platform.schema import (
    catalogue_coordinate,
    compare_catalogue_coordinates,
    load_platform_manifest_bytes,
)

IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"
POINTER_CACHE_CONTROL = "no-store, max-age=0"


@dataclass(frozen=True)
class RepoPublishReport:
    version: str
    lane: str
    immutable_count: int
    pointer_count: int
    pointers_promoted: int


def _content_type_for_repo_key(key: str) -> str:
    if key == "linux/keys/solstone-archive-keyring.asc":
        return "application/pgp-keys"
    if key.endswith("/Release") or key.endswith("/InRelease") or key.endswith("/Packages"):
        return "text/plain; charset=utf-8"
    if key.endswith("/Release.gpg") or key.endswith(".asc"):
        return "application/pgp-signature"
    if key.endswith("/repomd.xml"):
        return "application/xml"
    return "application/octet-stream"


def _locate_component_artifact(base_dir: Path, arch: str, filename: str) -> Path:
    arch_candidate = base_dir / f"linux-{arch}" / filename
    if arch_candidate.is_file():
        return arch_candidate
    flat_candidate = base_dir / filename
    if flat_candidate.is_file():
        return flat_candidate
    raise Refusal(
        RELEASE_COHERENCE,
        f"missing component package artifact '{filename}' for arch '{arch}' under {base_dir}",
    )


def _verify_component_package(
    path: Path,
    arch: str,
    variant_kind: str,
    manifest_entry: dict[str, Any],
) -> None:
    if not isinstance(manifest_entry, dict):
        raise Refusal(SCHEMA_INVALID, f"invalid manifest entry for {arch}/{variant_kind}")
    expected_filename = manifest_entry.get("filename")
    if path.name != expected_filename:
        raise Refusal(
            RELEASE_COHERENCE,
            f"filename mismatch for {arch}/{variant_kind}: {path.name} vs {expected_filename}",
        )
    data = path.read_bytes()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != manifest_entry.get("sha256") or len(data) != manifest_entry.get("bytes"):
        raise Refusal(
            SAME_VERSION_DIFFERENT_BYTES,
            f"digest or size mismatch on {path.name} against platform.json",
        )
    scanned = scan_deb_file(path) if variant_kind == "deb" else scan_rpm_file(path)
    actual_pkg_id = scanned.package_identity.to_dict() if scanned.package_identity else None
    if actual_pkg_id != manifest_entry.get("package_identity"):
        raise Refusal(
            RELEASE_COHERENCE,
            f"package_identity mismatch on {path.name}: {actual_pkg_id} vs {manifest_entry.get('package_identity')}",
        )
    if scanned.payload_build_id != manifest_entry.get("payload_build_id"):
        raise Refusal(
            RELEASE_COHERENCE,
            f"payload_build_id mismatch on {path.name}",
        )
    if scanned.inventory != manifest_entry.get("archive_inventory"):
        raise Refusal(
            RELEASE_COHERENCE,
            f"archive_inventory mismatch on {path.name}",
        )
    exe_info = manifest_entry.get("executable", {})
    exe_name = exe_info.get("name")
    if not exe_name or scanned.executable_sha256.get(exe_name) != exe_info.get("sha256"):
        raise Refusal(
            RELEASE_COHERENCE,
            f"executable digest mismatch on {path.name}",
        )


def build_repos(
    *,
    manifest_path: Path,
    signature_path: Path,
    journal_dir: Path,
    desktop_dir: Path,
    output_dir: Path,
    gpg_secret_key_path: Path,
    passphrase_callback: Callable[[], str] | None = None,
    repo_root: Path | None = None,
    selected_platform_pin: MinisignPin | None = None,
    selected_openpgp_pin: OpenPGPPin | None = None,
    is_production: bool = False,
    acknowledge_production: bool = False,
) -> dict[str, Any]:
    """Build signed APT and RPM repository trees and repo-manifest.json from a signed platform release."""
    resolved_repo_root = repo_root or Path(__file__).resolve().parents[2]
    prod_platform_pin = require_production_platform_pin(resolved_repo_root)
    prod_openpgp_pin = require_production_openpgp_pin(resolved_repo_root)

    platform_pin = selected_platform_pin or prod_platform_pin
    openpgp_pin = selected_openpgp_pin or prod_openpgp_pin

    if is_production:
        if not acknowledge_production:
            raise Refusal(
                PRODUCTION_UNAVAILABLE,
                "--acknowledge-production required for production repository build",
            )
        if os.environ.get("SOLSTONE_PLATFORM_PRODUCTION") != "ack":
            raise Refusal(
                PRODUCTION_UNAVAILABLE,
                "SOLSTONE_PLATFORM_PRODUCTION=ack environment variable required",
            )
        if platform_pin != prod_platform_pin or openpgp_pin != prod_openpgp_pin:
            raise Refusal(
                SIGNATURE_PIN_MISMATCH,
                "production build_repos requires production platform and OpenPGP pins",
            )

    if not manifest_path.is_file() or not signature_path.is_file():
        raise Refusal(SCHEMA_INVALID, "manifest_path and signature_path must be existing files")

    verify_minisign_signature(platform_pin, manifest_path, signature_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest_obj = load_platform_manifest_bytes(
        manifest_bytes,
        canonical_refusal=RELEASE_COHERENCE,
    )

    if manifest_obj.get("platform_key_id") != platform_pin.key_id:
        raise Refusal(
            SIGNATURE_PIN_MISMATCH,
            f"manifest platform_key_id '{manifest_obj.get('platform_key_id')}' does not match pin '{platform_pin.key_id}'",
        )

    version = catalogue_coordinate(manifest_obj)
    lane = manifest_obj["lane"]
    created_unix = int(manifest_obj["created_unix"])
    catalogue_revision = int(manifest_obj.get("catalogue_revision", 1))

    components = manifest_obj.get("components", {})
    j_comp = components.get("journal")
    d_comp = components.get("desktop")
    if not isinstance(j_comp, dict) or not isinstance(d_comp, dict):
        raise Refusal(RELEASE_COHERENCE, "platform.json missing journal or desktop component")

    journal_version = str(j_comp["version"])
    j_arches = j_comp.get("arches", {})
    d_arches = d_comp.get("arches", {})
    if not isinstance(j_arches, dict) or not isinstance(d_arches, dict) or not j_arches:
        raise Refusal(RELEASE_COHERENCE, "platform.json components must declare non-empty arches")
    if set(j_arches.keys()) != set(d_arches.keys()):
        raise Refusal(
            RELEASE_COHERENCE,
            f"journal and desktop arches do not match: {sorted(j_arches.keys())} vs {sorted(d_arches.keys())}",
        )

    verified_debs: dict[str, list[tuple[str, Path]]] = {}
    verified_rpms: dict[str, list[tuple[str, Path]]] = {}

    for arch in sorted(j_arches.keys()):
        j_arch_vars = j_arches[arch]
        d_arch_vars = d_arches[arch]
        if "deb" not in j_arch_vars or "rpm" not in j_arch_vars:
            raise Refusal(RELEASE_COHERENCE, f"journal arch '{arch}' missing deb or rpm variant")
        if "deb" not in d_arch_vars or "rpm" not in d_arch_vars:
            raise Refusal(RELEASE_COHERENCE, f"desktop arch '{arch}' missing deb or rpm variant")

        j_deb_path = _locate_component_artifact(journal_dir, arch, j_arch_vars["deb"]["filename"])
        j_rpm_path = _locate_component_artifact(journal_dir, arch, j_arch_vars["rpm"]["filename"])
        d_deb_path = _locate_component_artifact(desktop_dir, arch, d_arch_vars["deb"]["filename"])
        d_rpm_path = _locate_component_artifact(desktop_dir, arch, d_arch_vars["rpm"]["filename"])

        _verify_component_package(j_deb_path, arch, "deb", j_arch_vars["deb"])
        _verify_component_package(j_rpm_path, arch, "rpm", j_arch_vars["rpm"])
        _verify_component_package(d_deb_path, arch, "deb", d_arch_vars["deb"])
        _verify_component_package(d_rpm_path, arch, "rpm", d_arch_vars["rpm"])

        verified_debs[arch] = [("solstone-journal", j_deb_path), ("solstone-linux", d_deb_path)]
        verified_rpms[arch] = [("solstone-journal", j_rpm_path), ("solstone-linux", d_rpm_path)]

    passphrase = passphrase_callback() if passphrase_callback is not None else ""

    with tempfile.TemporaryDirectory(prefix="solstone-repo-stage-") as tmp_str:
        tmp_root = Path(tmp_str)
        meta_dir = tmp_root / "metapkg"
        stage_dir = tmp_root / "stage"
        stage_dir.mkdir(parents=True, exist_ok=True)

        for arch in sorted(j_arches.keys()):
            j_arch_vars = j_arches[arch]
            d_arch_vars = d_arches[arch]
            meta_pkg = build_metapackage_for_arch(
                output_dir=meta_dir / arch,
                arch=arch,
                journal_version=journal_version,
                catalogue_revision=catalogue_revision,
                journal_deb_version=str(j_arch_vars["deb"]["package_identity"]["version"]),
                desktop_deb_version=str(d_arch_vars["deb"]["package_identity"]["version"]),
                journal_rpm_version=str(j_arch_vars["rpm"]["package_identity"]["version"]),
                desktop_rpm_version=str(d_arch_vars["rpm"]["package_identity"]["version"]),
                repo_root=resolved_repo_root,
                created_unix=created_unix,
            )
            verified_debs[arch].append(("solstone", meta_pkg.deb_path))
            verified_rpms[arch].append(("solstone", meta_pkg.rpm_path))

        with isolated_gnupghome(
            secret_key_path=gpg_secret_key_path,
            passphrase=passphrase,
            selected_pin=openpgp_pin,
            is_production=is_production,
            acknowledge_production=acknowledge_production,
        ) as gnupghome:
            apt_imm_keys, apt_ptr_keys = build_apt_repository(
                repo_root_dir=stage_dir,
                lane=lane,
                coordinate=version,
                created_unix=created_unix,
                arch_deb_packages=verified_debs,
                gnupghome=gnupghome,
                openpgp_pin=openpgp_pin,
                passphrase=passphrase,
            )

            rpm_imm_keys, rpm_ptr_keys = build_rpm_repository(
                repo_root_dir=stage_dir,
                lane=lane,
                created_unix=created_unix,
                arch_rpm_packages=verified_rpms,
                gnupghome=gnupghome,
                openpgp_pin=openpgp_pin,
                passphrase=passphrase,
            )

        keys_dir = stage_dir / "linux" / "keys"
        keys_dir.mkdir(parents=True, exist_ok=True)
        gpg_key_file = keys_dir / "solstone-archive-keyring.gpg"
        asc_key_file = keys_dir / "solstone-archive-keyring.asc"
        minisig_key_file = keys_dir / "solstone-archive-keyring.asc.minisig"

        gpg_key_file.write_bytes(openpgp_pin.gpg_bytes)
        asc_key_file.write_bytes(openpgp_pin.asc_bytes)
        minisig_key_file.write_bytes(openpgp_pin.asc_minisig_bytes)

        verify_minisign_signature(platform_pin, asc_key_file, minisig_key_file)

        keyring_ptr_keys = [
            "linux/keys/solstone-archive-keyring.gpg",
            "linux/keys/solstone-archive-keyring.asc",
            "linux/keys/solstone-archive-keyring.asc.minisig",
        ]

        all_imm_keys = sorted([*apt_imm_keys, *rpm_imm_keys])
        all_ptr_keys = sorted([*keyring_ptr_keys, *apt_ptr_keys, *rpm_ptr_keys])

        seen_keys: set[str] = set()
        immutable_entries: list[dict[str, Any]] = []
        pointer_entries: list[dict[str, Any]] = []

        for key in all_imm_keys:
            if key in seen_keys:
                raise Refusal(RELEASE_COHERENCE, f"duplicate repository object key: {key}")
            seen_keys.add(key)
            if classify_destination_key(key) != "immutable":
                raise Refusal(UNSAFE_FILENAME, f"expected immutable key classification for {key}")
            fpath = stage_dir / key
            data = fpath.read_bytes()
            immutable_entries.append({
                "key": key,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "content_type": _content_type_for_repo_key(key),
                "cache_control": IMMUTABLE_CACHE_CONTROL,
            })

        for key in all_ptr_keys:
            if key in seen_keys:
                raise Refusal(RELEASE_COHERENCE, f"duplicate repository object key: {key}")
            seen_keys.add(key)
            if classify_destination_key(key) != "pointer":
                raise Refusal(UNSAFE_FILENAME, f"expected pointer key classification for {key}")
            fpath = stage_dir / key
            data = fpath.read_bytes()
            pointer_entries.append({
                "key": key,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "content_type": _content_type_for_repo_key(key),
                "cache_control": POINTER_CACHE_CONTROL,
            })

        repo_manifest: dict[str, Any] = {
            "schema_version": 1,
            "version": version,
            "lane": lane,
            "created_unix": created_unix,
            "platform_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "platform_key_id": platform_pin.key_id,
            "openpgp_fingerprint": openpgp_pin.fingerprint,
            "immutable_objects": immutable_entries,
            "pointer_objects": pointer_entries,
        }
        (stage_dir / "repo-manifest.json").write_bytes(canonical_json_bytes(repo_manifest))

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage_dir, output_dir)

    return repo_manifest


def _extract_inrelease_version(inrelease_bytes: bytes) -> str | None:
    text = inrelease_bytes.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return None


def _pointer_promotion_sort_key(entry: dict[str, Any]) -> tuple[int, str]:
    key = str(entry["key"])
    if key.startswith("linux/keys/"):
        return (0, key)
    if "/main/binary-" in key:
        return (1, key)
    if key.endswith("/Release"):
        return (2, key)
    if key.endswith("/Release.gpg"):
        return (3, key)
    if key.endswith("/InRelease"):
        return (4, key)
    if key.endswith(("/repomd.xml", "/repomd.xml.asc")):
        return (5, key)
    return (6, key)


def _validate_and_verify_staged_repos(
    repo_dir: Path,
    platform_pin: MinisignPin,
    openpgp_pin: OpenPGPPin,
) -> tuple[dict[str, Any], list[tuple[str, bytes, str, str]], list[tuple[str, bytes, str, str]]]:
    manifest_file = repo_dir / "repo-manifest.json"
    if not manifest_file.is_file():
        raise Refusal(SCHEMA_INVALID, f"repo-manifest.json not found in {repo_dir}")

    repo_manifest = parse_json_strict(manifest_file.read_bytes())
    if not isinstance(repo_manifest, dict) or repo_manifest.get("schema_version") != 1:
        raise Refusal(SCHEMA_INVALID, "invalid repo-manifest.json schema_version")

    version = repo_manifest.get("version")
    lane = repo_manifest.get("lane")
    platform_sha = repo_manifest.get("platform_manifest_sha256")
    if (
        not isinstance(version, str)
        or lane not in ("release", "staging", "dev")
        or not isinstance(platform_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", platform_sha)
    ):
        raise Refusal(SCHEMA_INVALID, "invalid version, lane, or platform_manifest_sha256 in repo-manifest.json")

    if repo_manifest.get("platform_key_id") != platform_pin.key_id:
        raise Refusal(
            SIGNATURE_PIN_MISMATCH,
            f"repo-manifest platform_key_id '{repo_manifest.get('platform_key_id')}' does not match pin '{platform_pin.key_id}'",
        )
    if repo_manifest.get("openpgp_fingerprint") != openpgp_pin.fingerprint:
        raise Refusal(
            SIGNATURE_PIN_MISMATCH,
            f"repo-manifest openpgp_fingerprint '{repo_manifest.get('openpgp_fingerprint')}' does not match pin '{openpgp_pin.fingerprint}'",
        )

    imm_entries = repo_manifest.get("immutable_objects")
    ptr_entries = repo_manifest.get("pointer_objects")
    if (
        not isinstance(imm_entries, list)
        or not imm_entries
        or not isinstance(ptr_entries, list)
        or not ptr_entries
    ):
        raise Refusal(SCHEMA_INVALID, "repo-manifest.json must contain non-empty immutable_objects and pointer_objects")

    expected_keys: set[str] = set()
    immutable_claims: list[tuple[str, bytes, str, str]] = []
    pointer_claims: list[tuple[str, bytes, str, str]] = []

    for entry in imm_entries:
        if not isinstance(entry, dict):
            raise Refusal(SCHEMA_INVALID, "invalid entry in immutable_objects")
        key = str(entry.get("key", ""))
        if classify_destination_key(key) != "immutable":
            raise Refusal(UNSAFE_FILENAME, f"key '{key}' in immutable_objects is not an immutable key")
        if key in expected_keys:
            raise Refusal(RELEASE_COHERENCE, f"duplicate key '{key}' in repo-manifest.json")
        expected_keys.add(key)

        fpath = repo_dir / key
        if not fpath.is_file():
            raise Refusal(RELEASE_COHERENCE, f"staged immutable file missing: {key}")
        data = fpath.read_bytes()
        if len(data) != entry.get("bytes") or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            raise Refusal(SAME_VERSION_DIFFERENT_BYTES, f"staged file digest or size mismatch: {key}")
        ct = str(entry.get("content_type", ""))
        cc = str(entry.get("cache_control", ""))
        if cc != IMMUTABLE_CACHE_CONTROL or not ct:
            raise Refusal(SCHEMA_INVALID, f"invalid metadata on immutable object {key}")
        immutable_claims.append((key, data, ct, cc))

    sorted_ptr_entries = sorted(ptr_entries, key=_pointer_promotion_sort_key)
    for entry in sorted_ptr_entries:
        if not isinstance(entry, dict):
            raise Refusal(SCHEMA_INVALID, "invalid entry in pointer_objects")
        key = str(entry.get("key", ""))
        if classify_destination_key(key) != "pointer":
            raise Refusal(UNSAFE_FILENAME, f"key '{key}' in pointer_objects is not a pointer key")
        if key in expected_keys:
            raise Refusal(RELEASE_COHERENCE, f"duplicate key '{key}' in repo-manifest.json")
        expected_keys.add(key)

        fpath = repo_dir / key
        if not fpath.is_file():
            raise Refusal(RELEASE_COHERENCE, f"staged pointer file missing: {key}")
        data = fpath.read_bytes()
        if len(data) != entry.get("bytes") or hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            raise Refusal(SAME_VERSION_DIFFERENT_BYTES, f"staged file digest or size mismatch: {key}")
        ct = str(entry.get("content_type", ""))
        cc = str(entry.get("cache_control", ""))
        if cc != POINTER_CACHE_CONTROL or not ct:
            raise Refusal(SCHEMA_INVALID, f"invalid metadata on pointer object {key}")
        pointer_claims.append((key, data, ct, cc))

    # Ensure no uninventoried files exist under repo_dir/linux
    linux_root = repo_dir / "linux"
    if not linux_root.is_dir():
        raise Refusal(RELEASE_COHERENCE, "staged repo_dir is missing linux/ directory")
    for child in sorted(linux_root.rglob("*")):
        if child.is_file():
            rel_key = child.relative_to(repo_dir).as_posix()
            if rel_key not in expected_keys:
                raise Refusal(RELEASE_COHERENCE, f"uninventoried file in staged repo_dir: {rel_key}")

    # Verify keyring files match pin and Minisign signature
    gpg_path = repo_dir / "linux/keys/solstone-archive-keyring.gpg"
    asc_path = repo_dir / "linux/keys/solstone-archive-keyring.asc"
    minisig_path = repo_dir / "linux/keys/solstone-archive-keyring.asc.minisig"
    if not gpg_path.is_file() or not asc_path.is_file() or not minisig_path.is_file():
        raise Refusal(RELEASE_COHERENCE, "staged repo_dir is missing linux/keys/solstone-archive-keyring files")
    if gpg_path.read_bytes() != openpgp_pin.gpg_bytes or asc_path.read_bytes() != openpgp_pin.asc_bytes:
        raise Refusal(SIGNATURE_PIN_MISMATCH, "staged archive keyring bytes do not match OpenPGP pin")
    verify_minisign_signature(platform_pin, asc_path, minisig_path)

    # Verify APT Release / InRelease / Release.gpg signatures and Version header
    release_path = repo_dir / f"linux/apt/dists/{lane}/Release"
    inrelease_path = repo_dir / f"linux/apt/dists/{lane}/InRelease"
    release_gpg_path = repo_dir / f"linux/apt/dists/{lane}/Release.gpg"
    if not release_path.is_file() or not inrelease_path.is_file() or not release_gpg_path.is_file():
        raise Refusal(RELEASE_COHERENCE, f"staged APT dists/{lane} metadata files missing")
    verify_openpgp_signature(
        gpg_bytes=openpgp_pin.gpg_bytes,
        expected_fingerprint=openpgp_pin.fingerprint,
        sig_or_inline_path=inrelease_path,
    )
    verify_openpgp_signature(
        gpg_bytes=openpgp_pin.gpg_bytes,
        expected_fingerprint=openpgp_pin.fingerprint,
        sig_or_inline_path=release_gpg_path,
        signed_data_path=release_path,
    )
    inrelease_ver = _extract_inrelease_version(inrelease_path.read_bytes())
    if inrelease_ver != version:
        raise Refusal(
            RELEASE_COHERENCE,
            f"staged InRelease Version '{inrelease_ver}' does not match repo-manifest version '{version}'",
        )

    # Verify RPM repomd.xml.asc and every RPM signature
    for key, _, _, _ in pointer_claims:
        if key.endswith("/repomd.xml.asc"):
            repomd_asc = repo_dir / key
            repomd_xml = repo_dir / key.removesuffix(".asc")
            verify_openpgp_signature(
                gpg_bytes=openpgp_pin.gpg_bytes,
                expected_fingerprint=openpgp_pin.fingerprint,
                sig_or_inline_path=repomd_asc,
                signed_data_path=repomd_xml,
            )

    for key, _, _, _ in immutable_claims:
        if key.endswith(".rpm"):
            verify_signed_rpm(repo_dir / key, openpgp_pin.asc_bytes)

    return repo_manifest, immutable_claims, pointer_claims


def publish_repos(
    *,
    repo_dir: Path,
    dest: Destination,
    repo_root: Path | None = None,
    selected_platform_pin: MinisignPin | None = None,
    selected_openpgp_pin: OpenPGPPin | None = None,
) -> RepoPublishReport:
    """Verify staged APT/RPM repositories and publish them atomically to Destination."""
    platform_pin = (
        selected_platform_pin
        or getattr(dest, "platform_pin", None)
        or require_production_platform_pin(repo_root)
    )
    openpgp_pin = (
        selected_openpgp_pin
        or getattr(dest, "openpgp_pin", None)
        or require_production_openpgp_pin(repo_root)
    )

    repo_manifest, immutable_claims, pointer_claims = _validate_and_verify_staged_repos(
        repo_dir=repo_dir,
        platform_pin=platform_pin,
        openpgp_pin=openpgp_pin,
    )

    version = str(repo_manifest["version"])
    lane = str(repo_manifest["lane"])
    expected_platform_sha = str(repo_manifest["platform_manifest_sha256"])

    # 1. Verify destination already holds the promoted platform release for this coordinate
    latest_key = f"solstone/{lane}/latest"
    latest_res = dest.get(latest_key)
    if not latest_res.is_ok() or latest_res.body is None:
        raise Refusal(
            RELEASE_COHERENCE,
            f"cannot publish repositories before {latest_key} is published (status: {latest_res.status})",
        )
    observed_latest = latest_res.body.decode("utf-8", errors="replace").strip()
    if observed_latest != version:
        try:
            cmp = compare_catalogue_coordinates(version, observed_latest)
        except Refusal:
            cmp = 0
        if cmp < 0:
            raise Refusal(
                ROLLBACK_REFUSED,
                f"repo-manifest version {version} is older than destination {latest_key} ({observed_latest})",
            )
        raise Refusal(
            RELEASE_COHERENCE,
            f"repo-manifest version {version} does not match destination {latest_key} ({observed_latest})",
        )

    platform_key = f"solstone/{lane}/{version}/platform.json"
    platform_res = dest.get(platform_key)
    if not platform_res.is_ok() or platform_res.body is None:
        raise Refusal(
            RELEASE_COHERENCE,
            f"destination missing {platform_key} required by repo-manifest.json",
        )
    if hashlib.sha256(platform_res.body).hexdigest() != expected_platform_sha:
        raise Refusal(
            SAME_VERSION_DIFFERENT_BYTES,
            f"destination {platform_key} digest does not match repo-manifest platform_manifest_sha256",
        )

    # 2. Check existing remote InRelease for monotonic version progression
    inrelease_key = f"linux/apt/dists/{lane}/InRelease"
    remote_inrelease = dest.get(inrelease_key)
    if remote_inrelease.is_ok() and remote_inrelease.body is not None:
        remote_ver = _extract_inrelease_version(remote_inrelease.body)
        if not remote_ver:
            raise Refusal(
                RELEASE_COHERENCE,
                f"existing {inrelease_key} is missing Version header",
            )
        if compare_catalogue_coordinates(version, remote_ver) < 0:
            raise Refusal(
                ROLLBACK_REFUSED,
                f"incoming repository version {version} is older than published InRelease version {remote_ver}",
            )
    elif not remote_inrelease.is_absent():
        raise Refusal(
            PUBLISH_INDETERMINATE,
            f"failed to read {inrelease_key}: {remote_inrelease.status}",
        )

    # 3. Phase 1: Claim all immutable objects (pool .debs, by-hash indices, RPM Packages, RPM hashed repodata)
    for key, body, ct, cc in immutable_claims:
        _claim_immutable_object(dest, key, body, ct, cc)

    # 4. Phase 2: Authoritatively re-read and verify all immutable objects before touching any pointer
    for key, body, ct, cc in immutable_claims:
        _verify_immutable_object(dest, key, body, ct, cc)

    # 5. Phase 3: Promote pointer objects via compare-and-swap in deterministic order
    promoted_count = 0
    for key, body, ct, cc in pointer_claims:
        curr = dest.get(key)
        if curr.is_ok():
            if curr.body == body and curr.content_type == ct and curr.cache_control == cc:
                _verify_immutable_object(dest, key, body, ct, cc)
                continue
            expected_etag = curr.etag or ""
        elif curr.is_absent():
            expected_etag = ""
        else:
            raise Refusal(
                PUBLISH_INDETERMINATE,
                f"failed to read pointer {key} prior to CAS: {curr.status}",
            )

        cas_res = dest.compare_and_swap(
            key=key,
            body=body,
            expected_etag=expected_etag,
            content_type=ct,
            cache_control=cc,
        )
        if not cas_res.is_ok():
            reread = dest.get(key)
            if reread.is_ok() and reread.body == body and reread.content_type == ct and reread.cache_control == cc:
                _verify_immutable_object(dest, key, body, ct, cc)
                continue
            raise Refusal(
                PUBLISH_INDETERMINATE,
                f"CAS failed for pointer {key}: {cas_res.status} ({cas_res.detail})",
            )
        _verify_immutable_object(dest, key, body, ct, cc)
        promoted_count += 1

    return RepoPublishReport(
        version=version,
        lane=lane,
        immutable_count=len(immutable_claims),
        pointer_count=len(pointer_claims),
        pointers_promoted=promoted_count,
    )
