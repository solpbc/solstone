# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Atomic platform release publisher and latest pointer promoter."""

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Callable, Optional

from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.destination import Destination, ResultStatus
from solstone_platform.ingest import verify_minisign_signature
from solstone_platform.pins import MinisignPin
from solstone_platform.refusals import (
    PUBLISH_INDETERMINATE,
    RELEASE_COHERENCE,
    ROLLBACK_REFUSED,
    SAME_VERSION_DIFFERENT_BYTES,
    SCHEMA_INVALID,
    SIGNATURE_PIN_MISMATCH,
    Refusal,
)
from solstone_platform.schema import validate_platform_manifest


def compare_semver(v1: str, v2: str) -> int:
    """Compare two strict X.Y.Z SemVer strings. Returns -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2."""
    p1 = [int(x) for x in v1.split(".")]
    p2 = [int(x) for x in v2.split(".")]
    if p1 < p2:
        return -1
    if p1 > p2:
        return 1
    return 0


@dataclass
class PublishReport:
    lane: str
    version: str
    manifest_key: str
    signature_key: str
    latest_key: str
    latest_promoted: bool


def publish_release(
    manifest_bytes: bytes,
    signature_bytes: bytes,
    selected_pin: MinisignPin,
    dest: Destination,
    key_prefix: str = "",
    verify_native_callback: Optional[Callable[[], None]] = None,
) -> PublishReport:
    """Publish platform release and atomically promote latest pointer."""
    # Step 1: Canonicalize, reparse, and verify signature locally before first destination operation
    parsed_manifest = parse_json_strict(manifest_bytes)
    validate_platform_manifest(parsed_manifest)
    if parsed_manifest["platform_key_id"] != selected_pin.key_id:
        raise Refusal(SIGNATURE_PIN_MISMATCH, "manifest platform_key_id does not match selected verification pin")
    canonical_bytes = canonical_json_bytes(parsed_manifest)
    if canonical_bytes != manifest_bytes:
        manifest_bytes = canonical_bytes

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_m = Path(tmp_dir) / "platform.json"
        tmp_m.write_bytes(manifest_bytes)
        tmp_s = Path(tmp_dir) / "platform.json.minisig"
        tmp_s.write_bytes(signature_bytes)
        verify_minisign_signature(selected_pin, tmp_m, tmp_s)

    lane = parsed_manifest["lane"]
    version = parsed_manifest["version"]

    prefix = key_prefix.strip("/")
    base_prefix = f"{prefix}/solstone" if prefix else "solstone"

    manifest_key = f"{base_prefix}/{lane}/{version}/platform.json"
    signature_key = f"{base_prefix}/{lane}/{version}/platform.json.minisig"
    latest_key = f"{base_prefix}/{lane}/latest"

    json_content_type = "application/json"
    sig_content_type = "application/octet-stream"
    immutable_cache = "public, max-age=31536000, immutable"
    latest_content_type = "text/plain; charset=utf-8"
    latest_cache = "no-store, max-age=0"

    # Step 2: Claim platform.json via put_if_absent
    put_m = dest.put_if_absent(manifest_key, manifest_bytes, json_content_type, immutable_cache)
    if put_m.is_precondition_failed():
        # Object exists; re-read authoritatively to check if identical
        get_m = dest.get(manifest_key)
        if not get_m.is_ok() or get_m.body != manifest_bytes:
            raise Refusal(
                SAME_VERSION_DIFFERENT_BYTES,
                f"platform.json already exists for {version} on lane '{lane}' with different content",
            )
    elif not put_m.is_ok():
        # Ambiguous error; re-read
        get_m = dest.get(manifest_key)
        if not get_m.is_ok() or get_m.body != manifest_bytes:
            raise Refusal(PUBLISH_INDETERMINATE, f"put platform.json failed: {put_m.detail}")

    # Step 3: Put platform.json.minisig via put_if_absent
    put_s = dest.put_if_absent(signature_key, signature_bytes, sig_content_type, immutable_cache)
    if put_s.is_precondition_failed():
        get_s = dest.get(signature_key)
        if not get_s.is_ok() or get_s.body != signature_bytes:
            raise Refusal(
                SAME_VERSION_DIFFERENT_BYTES,
                f"platform.json.minisig already exists for {version} on lane '{lane}' with different signature",
            )
    elif not put_s.is_ok():
        get_s = dest.get(signature_key)
        if not get_s.is_ok() or get_s.body != signature_bytes:
            raise Refusal(PUBLISH_INDETERMINATE, f"put platform.json.minisig failed: {put_s.detail}")

    # Step 4: Authoritative byte-for-byte and metadata read-back verification of both
    check_m = dest.get(manifest_key)
    check_s = dest.get(signature_key)
    if not check_m.is_ok() or check_m.body != manifest_bytes:
        raise Refusal(PUBLISH_INDETERMINATE, "authoritative read-back of platform.json failed")
    if check_m.content_type != json_content_type or check_m.cache_control != immutable_cache:
        raise Refusal(PUBLISH_INDETERMINATE, f"read-back metadata mismatch for platform.json: {check_m.content_type}, {check_m.cache_control}")

    if not check_s.is_ok() or check_s.body != signature_bytes:
        raise Refusal(PUBLISH_INDETERMINATE, "authoritative read-back of platform.json.minisig failed")
    if check_s.content_type != sig_content_type or check_s.cache_control != immutable_cache:
        raise Refusal(PUBLISH_INDETERMINATE, f"read-back metadata mismatch for platform.json.minisig: {check_s.content_type}, {check_s.cache_control}")

    # Step 5: Re-verify native dependencies if callback provided
    if verify_native_callback is not None:
        try:
            verify_native_callback()
        except Exception as err:
            raise Refusal(RELEASE_COHERENCE, f"pre-promotion native verification failed: {err}") from err

    # Step 6: Fetch current latest pointer and check SemVer monotonicity
    latest_get = dest.get(latest_key)
    latest_body = f"{version}\n".encode("utf-8")
    expected_etag: str = ""

    if latest_get.is_ok() and latest_get.body is not None:
        expected_etag = latest_get.etag or ""
        current_latest_str = latest_get.body.decode("utf-8", errors="replace").strip()
        cmp = compare_semver(version, current_latest_str)
        if cmp < 0:
            raise Refusal(
                ROLLBACK_REFUSED,
                f"cannot promote {version} because current latest is newer: {current_latest_str}",
            )
        if cmp == 0:
            # Same version already promoted
            return PublishReport(
                lane=lane,
                version=version,
                manifest_key=manifest_key,
                signature_key=signature_key,
                latest_key=latest_key,
                latest_promoted=True,
            )
    elif latest_get.is_absent():
        expected_etag = ""
    else:
        raise Refusal(PUBLISH_INDETERMINATE, f"failed to read current latest pointer: {latest_get.detail}")

    # Step 7: Atomic CAS on latest pointer with retry loop for N vs N+1 races
    max_attempts = 3
    for attempt in range(max_attempts):
        cas_res = dest.compare_and_swap(latest_key, latest_body, expected_etag, latest_content_type, latest_cache)
        if cas_res.is_ok():
            # Authoritative read-back verification of latest pointer
            check_latest = dest.get(latest_key)
            if not check_latest.is_ok() or check_latest.body != latest_body:
                raise Refusal(PUBLISH_INDETERMINATE, "authoritative read-back of latest failed")
            if check_latest.content_type != latest_content_type or check_latest.cache_control != latest_cache:
                raise Refusal(PUBLISH_INDETERMINATE, f"read-back metadata mismatch for latest: {check_latest.content_type}, {check_latest.cache_control}")
            return PublishReport(
                lane=lane,
                version=version,
                manifest_key=manifest_key,
                signature_key=signature_key,
                latest_key=latest_key,
                latest_promoted=True,
            )

        # Re-read authoritatively
        reread_latest = dest.get(latest_key)
        if not reread_latest.is_ok() or reread_latest.body is None:
            raise Refusal(PUBLISH_INDETERMINATE, f"CAS of latest pointer failed: {cas_res.detail}")

        observed = reread_latest.body.decode("utf-8", errors="replace").strip()
        cmp = compare_semver(version, observed)
        if cmp < 0:
            # A newer version won the race (e.g. N+2)
            return PublishReport(
                lane=lane,
                version=version,
                manifest_key=manifest_key,
                signature_key=signature_key,
                latest_key=latest_key,
                latest_promoted=False,
            )
        elif cmp == 0:
            # Already promoted to this version
            return PublishReport(
                lane=lane,
                version=version,
                manifest_key=manifest_key,
                signature_key=signature_key,
                latest_key=latest_key,
                latest_promoted=True,
            )
        else:
            # Observed version is older than incoming version (e.g. 2.0.3 was written while we were trying to promote 2.0.4)
            # Retry CAS against the newly observed ETag
            expected_etag = reread_latest.etag or ""

    raise Refusal(PUBLISH_INDETERMINATE, f"CAS of latest pointer failed after {max_attempts} attempts: {cas_res.detail}")
