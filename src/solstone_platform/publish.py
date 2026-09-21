# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Atomic platform publication rail with closed signature and complete dependency claim."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Optional

from solstone_platform.capture import capture_release_sources
from solstone_platform.canonical import parse_json_strict
from solstone_platform.destination import Destination, ResultStatus
from solstone_platform.ingest import (
    IngestedComponent,
    ingest_desktop,
    ingest_journal,
    ingest_tmux,
    verify_minisign_signature,
)
from solstone_platform.pins import embedded_pins, require_production_platform_pin
from solstone_platform.refusals import (
    PUBLISH_INDETERMINATE,
    RELEASE_COHERENCE,
    ROLLBACK_REFUSED,
    SAME_VERSION_DIFFERENT_BYTES,
    SCHEMA_INVALID,
    SIGNATURE_PIN_MISMATCH,
    UNSAFE_FILENAME,
    Refusal,
)
from solstone_platform.schema import load_platform_manifest_bytes
from solstone_platform.targets import get_target_mapping
from solstone_platform.testdest import FIXTURE_BUILD_TOKEN, FixtureDestination
from solstone_platform.witness import get_witness


@dataclass(frozen=True)
class PublishReport:
    version: str
    lane: str
    manifest_key: str
    signature_key: str
    latest_key: str
    latest_promoted: bool


def compare_semver(a: str, b: str) -> int:
    """Compare two valid SemVer strings. Return -1 if a < b, 0 if a == b, 1 if a > b."""
    match_a = re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", a)
    match_b = re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", b)
    if not match_a or not match_b:
        raise Refusal(RELEASE_COHERENCE, f"invalid semver comparison: '{a}' vs '{b}'")
    ta = tuple(int(x) for x in match_a.groups())
    tb = tuple(int(x) for x in match_b.groups())
    if ta < tb:
        return -1
    if ta > tb:
        return 1
    return 0


def _claim_immutable_object(
    dest: Destination,
    key: str,
    body: bytes,
    content_type: str,
    cache_control: str,
) -> None:
    """Reuse an exact immutable object or claim its absent key conditionally."""
    get_witness().record("dest_key_select", key)
    existing = dest.get(key)
    if existing.is_ok():
        if existing.body is None:
            raise Refusal(PUBLISH_INDETERMINATE, f"existing object has no readable body on {key}")
        if existing.body != body:
            raise Refusal(
                SAME_VERSION_DIFFERENT_BYTES,
                f"destination already contains different bytes for immutable key {key}",
            )
        if (existing.content_type and existing.content_type != content_type) or (
            existing.cache_control and existing.cache_control != cache_control
        ):
            raise Refusal(
                RELEASE_COHERENCE,
                f"metadata mismatch on existing object {key}: "
                f"content-type ({existing.content_type} vs {content_type}), "
                f"cache-control ({existing.cache_control} vs {cache_control})",
            )
        return
    if not existing.is_absent():
        raise Refusal(PUBLISH_INDETERMINATE, f"preflight read failed on immutable object {key}: {existing.status}")

    put_res = dest.put_if_absent(
        key=key,
        body=body,
        content_type=content_type,
        cache_control=cache_control,
    )
    if put_res.status == ResultStatus.OK:
        return
    if put_res.is_precondition_failed():
        # Re-read authoritatively
        get_res = dest.get(key)
        if not get_res.is_ok() or get_res.body is None:
            raise Refusal(PUBLISH_INDETERMINATE, f"re-read failed after 412 on {key}")
        if get_res.body != body:
            raise Refusal(
                SAME_VERSION_DIFFERENT_BYTES,
                f"destination already contains different bytes for immutable key {key}",
            )
        if (get_res.content_type and get_res.content_type != content_type) or (
            get_res.cache_control and get_res.cache_control != cache_control
        ):
            raise Refusal(
                RELEASE_COHERENCE,
                f"metadata mismatch on existing object {key}: "
                f"content-type ({get_res.content_type} vs {content_type}), "
                f"cache-control ({get_res.cache_control} vs {cache_control})",
            )
        return

    raise Refusal(PUBLISH_INDETERMINATE, f"failed to claim immutable object {key}: {put_res.status}")


def _verify_immutable_object(
    dest: Destination,
    key: str,
    expected_body: bytes,
    expected_content_type: str,
    expected_cache_control: str,
) -> None:
    """Authoritative re-read and byte/metadata assertion."""
    get_witness().record("dest_key_verify", key)
    get_res = dest.get(key)
    if not get_res.is_ok() or get_res.body is None:
        raise Refusal(PUBLISH_INDETERMINATE, f"authoritative re-read failed for {key}")
    if get_res.body != expected_body:
        raise Refusal(SAME_VERSION_DIFFERENT_BYTES, f"re-read bytes mismatch on {key}")
    if get_res.content_type != expected_content_type:
        raise Refusal(RELEASE_COHERENCE, f"re-read content-type mismatch on {key}: {get_res.content_type} vs {expected_content_type}")
    if get_res.cache_control != expected_cache_control:
        raise Refusal(RELEASE_COHERENCE, f"re-read cache-control mismatch on {key}: {get_res.cache_control} vs {expected_cache_control}")


def _add_claim(
    claims: dict[str, tuple[bytes, str, str]],
    key: str,
    claim: tuple[bytes, str, str],
) -> None:
    if key in claims:
        raise Refusal(RELEASE_COHERENCE, f"duplicate destination claim for {key}")
    claims[key] = claim


def _snapshot_prepared_inventory(snapshot) -> list[dict[str, object]]:
    """Return the receipt-defined prepared file set from captured bytes."""
    roots = (
        (snapshot.manifest_path, "platform.json"),
        (snapshot.journal_dir, "journal"),
        (snapshot.desktop_dir, "desktop"),
        (snapshot.tmux_dir, "tmux"),
    )
    files: list[tuple[Path, str]] = []
    for path, rel in roots:
        if path.is_file():
            files.append((path, rel))
        else:
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    files.append((child, f"{rel}/{child.relative_to(path)}"))
    if snapshot.bootstrap_file:
        files.append((snapshot.bootstrap_file, f"bootstrap/{snapshot.bootstrap_file.name}"))
    result: list[dict[str, object]] = []
    for path, rel in sorted(files, key=lambda item: item[1]):
        data = path.read_bytes()
        result.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return result


def _validate_recut_receipt(
    receipt: dict,
    snapshot,
    manifest_obj: dict,
    manifest_bytes: bytes,
    base_obj: dict,
) -> None:
    if receipt.get("lane") != manifest_obj.get("lane"):
        raise Refusal(RELEASE_COHERENCE, "recut receipt lane does not match manifest")
    candidate = receipt.get("candidate")
    base = receipt.get("base")
    if not isinstance(candidate, dict) or not isinstance(base, dict):
        raise Refusal(SCHEMA_INVALID, "recut receipt requires base and candidate objects")
    if candidate.get("version") != manifest_obj.get("version"):
        raise Refusal(RELEASE_COHERENCE, "recut receipt candidate version does not match manifest")
    if candidate.get("sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise Refusal(RELEASE_COHERENCE, "recut receipt candidate digest does not match manifest")
    if not isinstance(base.get("body"), str) or not isinstance(base.get("etag"), str):
        raise Refusal(SCHEMA_INVALID, "recut receipt base pointer is incomplete")
    if compare_semver(manifest_obj["version"], str(base.get("version"))) <= 0:
        raise Refusal(RELEASE_COHERENCE, "recut receipt base is not older than candidate")
    prepared_files = receipt.get("prepared_files")
    if not isinstance(prepared_files, list) or any(not isinstance(item, dict) for item in prepared_files):
        raise Refusal(SCHEMA_INVALID, "recut receipt prepared_files must be an object list")
    if prepared_files != _snapshot_prepared_inventory(snapshot):
        raise Refusal(RELEASE_COHERENCE, "captured prepared file inventory differs from recut receipt")
    prepared_by_path = {item.get("path"): item for item in prepared_files}
    if None in prepared_by_path or len(prepared_by_path) != len(prepared_files):
        raise Refusal(SCHEMA_INVALID, "recut receipt prepared file paths must be unique strings")
    sources = receipt.get("sources")
    if not isinstance(sources, list):
        raise Refusal(SCHEMA_INVALID, "recut receipt sources must be a list")
    source_by_path: dict[str, dict] = {}
    for source in sources:
        if not isinstance(source, dict) or source.get("path") not in prepared_by_path:
            raise Refusal(RELEASE_COHERENCE, "recut receipt source path is not inventoried")
        source_path = source["path"]
        if source_path in source_by_path:
            raise Refusal(RELEASE_COHERENCE, "recut receipt maps a prepared path more than once")
        source_by_path[source_path] = source
        prepared = prepared_by_path[source["path"]]
        if source.get("sha256") != prepared.get("sha256") or source.get("bytes") != prepared.get("bytes"):
            raise Refusal(RELEASE_COHERENCE, "recut receipt source fact differs from captured file")

    replacements = receipt.get("replacements")
    if not isinstance(replacements, dict) or not replacements or not set(replacements).issubset({"journal", "desktop", "tmux"}):
        raise Refusal(SCHEMA_INVALID, "recut receipt replacements are invalid")
    origin = "https://updates.solstone.app"
    base_version = base_obj["version"]
    for field in ("schema_version", "protocol_version", "lane", "platform_key_id", "minimum_installer_revision"):
        if manifest_obj.get(field) != base_obj.get(field):
            raise Refusal(RELEASE_COHERENCE, f"recut changed protected top-level field {field}")
    expected_source_paths = set(prepared_by_path) - {"platform.json"}
    if set(source_by_path) != expected_source_paths:
        raise Refusal(RELEASE_COHERENCE, "recut receipt source mapping is incomplete or has extras")

    for component in ("journal", "desktop", "tmux"):
        base_component = base_obj["components"][component]
        candidate_component = manifest_obj["components"][component]
        if component not in replacements:
            if candidate_component != base_component:
                raise Refusal(RELEASE_COHERENCE, f"recut changed unselected component {component}")
        else:
            if replacements[component] != candidate_component["version"]:
                raise Refusal(RELEASE_COHERENCE, f"recut replacement version mismatch for {component}")
            if compare_semver(candidate_component["version"], base_component["version"]) <= 0:
                raise Refusal(RELEASE_COHERENCE, f"recut replacement is not newer for {component}")
            for field in candidate_component:
                if field not in {"version", "arches", "provenance"} and candidate_component[field] != base_component.get(field):
                    raise Refusal(RELEASE_COHERENCE, f"recut changed protected {component} field {field}")

    component_prefixes = {
        "journal": "solstone-journal" if "journal" in replacements else "solstone",
        "desktop": "solstone-linux" if "desktop" in replacements else "solstone",
        "tmux": "solstone-tmux" if "tmux" in replacements else "solstone",
    }
    for path, source in source_by_path.items():
        if path.startswith("bootstrap/"):
            journal_version = manifest_obj["components"]["journal"]["version"]
            expected_url = f"{origin}/solstone-journal/release/{journal_version}/{Path(path).name}"
        else:
            component = path.split("/", 1)[0]
            version = manifest_obj["components"][component]["version"] if component in replacements else base_version
            expected_url = f"{origin}/{component_prefixes[component]}/release/{version}/{Path(path).name}"
        if source.get("url") != expected_url:
            raise Refusal(RELEASE_COHERENCE, f"recut receipt source coordinate mismatch for {path}")


def _verify_base_release(dest: Destination, receipt: dict, platform_pin, snapshot_root: Path) -> dict:
    base = receipt["base"]
    base_version = base["version"]
    manifest_res = dest.get(f"solstone/release/{base_version}/platform.json")
    signature_res = dest.get(f"solstone/release/{base_version}/platform.json.minisig")
    if not manifest_res.is_ok() or manifest_res.body is None or not signature_res.is_ok() or signature_res.body is None:
        raise Refusal(PUBLISH_INDETERMINATE, "could not independently read signed recut base")
    manifest_path = snapshot_root / "verified-base-platform.json"
    signature_path = snapshot_root / "verified-base-platform.json.minisig"
    manifest_path.write_bytes(manifest_res.body)
    signature_path.write_bytes(signature_res.body)
    verify_minisign_signature(platform_pin, manifest_path, signature_path)
    base_obj = load_platform_manifest_bytes(manifest_res.body, canonical_refusal=RELEASE_COHERENCE)
    if (
        base_obj.get("version") != base_version
        or base_obj.get("lane") != receipt.get("lane")
        or base_obj.get("platform_key_id") != platform_pin.key_id
    ):
        raise Refusal(RELEASE_COHERENCE, "signed recut base does not match receipt coordinate")
    return base_obj


def publish_release(*args, **kwargs) -> PublishReport:
    """Closed production publication entrypoint."""
    # Strict argument allowlist check BEFORE capture
    allowed_kw = {
        "manifest_path", "signature_path", "journal_dir", "desktop_dir", "tmux_dir",
        "dest", "bootstrap_file", "expected_latest_receipt", "expect_latest_absent",
    }
    if len(args) > 7 or any(k not in allowed_kw for k in kwargs):
        raise Refusal(UNSAFE_FILENAME, "publish_release received unexpected arguments outside closed allowlist")

    # Map parameters
    pos_names = ["manifest_path", "signature_path", "journal_dir", "desktop_dir", "tmux_dir", "dest", "bootstrap_file"]
    bound = {}
    for idx, arg in enumerate(args):
        bound[pos_names[idx]] = arg
    for k, v in kwargs.items():
        if k in bound:
            raise Refusal(SCHEMA_INVALID, f"duplicate argument: {k}")
        bound[k] = v

    for req in ["manifest_path", "signature_path", "journal_dir", "desktop_dir", "tmux_dir", "dest"]:
        if req not in bound:
            raise Refusal(SCHEMA_INVALID, f"missing required argument: {req}")

    manifest_path: Path = Path(bound["manifest_path"])
    signature_path: Path = Path(bound["signature_path"])
    journal_dir: Path = Path(bound["journal_dir"])
    desktop_dir: Path = Path(bound["desktop_dir"])
    tmux_dir: Path = Path(bound["tmux_dir"])
    dest: Destination = bound["dest"]
    bootstrap_file: Optional[Path] = Path(bound["bootstrap_file"]) if bound.get("bootstrap_file") else None
    receipt_path: Optional[Path] = Path(bound["expected_latest_receipt"]) if bound.get("expected_latest_receipt") else None
    expect_latest_absent = bool(bound.get("expect_latest_absent", False))

    is_fixture_dest = (
        type(dest) is FixtureDestination
        and getattr(dest, "_build_token", None) is FIXTURE_BUILD_TOKEN
    )
    if not is_fixture_dest and (receipt_path is None) == (not expect_latest_absent):
        raise Refusal(SCHEMA_INVALID, "production publish requires exactly one latest expectation")

    # 1. Capture release sources into private 0700 directory
    with capture_release_sources(
        manifest_path=manifest_path,
        signature_path=signature_path,
        journal_dir=journal_dir,
        desktop_dir=desktop_dir,
        tmux_dir=tmux_dir,
        bootstrap_file=bootstrap_file,
        receipt_path=receipt_path,
    ) as snapshot:

        # 2. Fire post-capture hook only for authorized FixtureDestination
        if is_fixture_dest and dest._post_capture_hook is not None:
            dest._post_capture_hook()

        # 3. Semantic operations from SNAPSHOT paths only
        manifest_bytes = snapshot.manifest_path.read_bytes()
        signature_bytes = snapshot.signature_path.read_bytes()

        manifest_obj = load_platform_manifest_bytes(
            manifest_bytes,
            canonical_refusal=RELEASE_COHERENCE,
        )
        canonical_manifest = manifest_bytes

        receipt_obj = None
        if snapshot.receipt_path:
            receipt_obj = parse_json_strict(snapshot.receipt_path.read_bytes())
            if not isinstance(receipt_obj, dict) or receipt_obj.get("schema_version") != 1:
                raise Refusal(SCHEMA_INVALID, "invalid recut receipt")

        version = manifest_obj["version"]
        lane = manifest_obj["lane"]
        platform_key_id = manifest_obj["platform_key_id"]

        # Resolve platform pin
        if is_fixture_dest and dest.platform_pin is not None:
            platform_pin = dest.platform_pin
        else:
            platform_pin = require_production_platform_pin()

        if platform_key_id != platform_pin.key_id:
            raise Refusal(
                SIGNATURE_PIN_MISMATCH,
                f"manifest platform_key_id '{platform_key_id}' does not match verification pin '{platform_pin.key_id}'",
            )

        verify_minisign_signature(platform_pin, snapshot.manifest_path, snapshot.signature_path)

        if receipt_obj is not None:
            base_obj = _verify_base_release(dest, receipt_obj, platform_pin, snapshot.snapshot_root)
            _validate_recut_receipt(receipt_obj, snapshot, manifest_obj, manifest_bytes, base_obj)

        # Resolve native pins
        if is_fixture_dest and dest.pinset is not None:
            pins = dest.pinset
        else:
            pins = embedded_pins()

        # Ingest components from snapshot trees
        # Expected updates origin for journal
        expected_origin = "https://updates.solstone.app"
        ingested_j = ingest_journal(
            native_dir=snapshot.journal_dir,
            lane=lane,
            origin=expected_origin,
            bootstrap_file=snapshot.bootstrap_file,
            pin=pins.journal,
        )
        ingested_d = ingest_desktop(
            native_dir=snapshot.desktop_dir,
            pin=pins.desktop,
        )
        ingested_t = ingest_tmux(
            native_dir=snapshot.tmux_dir,
            pin=pins.tmux,
        )

        # Require provenance bootstrap URL to match exact production format
        j_manifest_comp = manifest_obj["components"].get("journal", {})
        j_provenance = j_manifest_comp.get("provenance", {})
        j_boot_url = j_provenance.get("bootstrap", {}).get("url", "")
        expected_boot_filename = f"solstone-journal-{ingested_j.version}-install.sh"
        expected_boot_url = f"https://updates.solstone.app/solstone-journal/{lane}/{ingested_j.version}/{expected_boot_filename}"
        if j_boot_url != expected_boot_url:
            raise Refusal(
                RELEASE_COHERENCE,
                f"platform manifest journal bootstrap url mismatch: expected '{expected_boot_url}', got '{j_boot_url}'",
            )

        # Compare snapshot payload facts to platform.json components
        scanned_components = {
            "journal": ingested_j,
            "desktop": ingested_d,
            "tmux": ingested_t,
        }

        for comp_name, scanned_comp in scanned_components.items():
            if comp_name not in manifest_obj["components"]:
                raise Refusal(RELEASE_COHERENCE, f"platform manifest missing component '{comp_name}'")
            comp_obj = manifest_obj["components"][comp_name]
            if comp_obj.get("version") != scanned_comp.version:
                raise Refusal(RELEASE_COHERENCE, f"component '{comp_name}' version mismatch: manifest {comp_obj.get('version')} vs scanned {scanned_comp.version}")

            for arch_name, scanned_variants in scanned_comp.arches.items():
                if arch_name not in comp_obj.get("arches", {}):
                    raise Refusal(RELEASE_COHERENCE, f"component '{comp_name}' missing arch '{arch_name}' in manifest")
                manifest_variants = comp_obj["arches"][arch_name]
                for var_type, scanned_var in scanned_variants.items():
                    if var_type not in manifest_variants:
                        raise Refusal(RELEASE_COHERENCE, f"component '{comp_name}' arch '{arch_name}' missing variant '{var_type}'")
                    mvar = manifest_variants[var_type]
                    # Verify fields
                    for field in ["filename", "sha256", "bytes", "native_target", "package_identity", "executable", "payload_build_id", "archive_inventory", "authority"]:
                        if mvar.get(field) != scanned_var.get(field):
                            raise Refusal(RELEASE_COHERENCE, f"component '{comp_name}' {arch_name}/{var_type} mismatch in {field}")

        # Prepare list of all immutable objects to claim and verify
        # Dictionary of key -> (bytes, content_type, cache_control)
        claims_to_make: dict[str, tuple[bytes, str, str]] = {}

        # 1. Native objects (claim ONLY ingest-identified native authority, sidecars, and variant artifacts)
        # Journal objects
        for target_arch in ["x86_64", "aarch64"]:
            arch_dir = snapshot.journal_dir / f"linux-{target_arch}"
            if not arch_dir.is_dir():
                arch_dir = snapshot.journal_dir

            j_ver = ingested_j.version
            j_files = {
                f"solstone-journal-{j_ver}-linux-{target_arch}.manifest.json",
                f"solstone-journal-{j_ver}-linux-{target_arch}.manifest.json.minisig",
                f"solstone-journal-{j_ver}-linux-{target_arch}.release",
                f"solstone-journal-{j_ver}-linux-{target_arch}.sha256",
            }
            if target_arch in ingested_j.arches:
                for var_data in ingested_j.arches[target_arch].values():
                    j_files.add(var_data["filename"])

            for fname in sorted(j_files):
                fpath = arch_dir / fname
                if fpath.is_file():
                    k = f"solstone/{lane}/{version}/{fname}"
                    b = fpath.read_bytes()
                    if fname.endswith(".json"):
                        ct = "application/json"
                    elif fname.endswith(".minisig") or fname.endswith(".tar.gz") or fname.endswith(".deb") or fname.endswith(".rpm"):
                        ct = "application/octet-stream"
                    else:
                        ct = "text/plain; charset=utf-8"
                    _add_claim(claims_to_make, k, (b, ct, "public, max-age=31536000, immutable"))

        # Desktop objects
        d_ver = ingested_d.version
        d_files = {
            f"solstone-linux-{d_ver}-linux-x86_64.rust-release-manifest.json",
            f"solstone-linux-{d_ver}-linux-x86_64.rust-release-manifest.json.minisig",
        }
        if "x86_64" in ingested_d.arches:
            for var_data in ingested_d.arches["x86_64"].values():
                d_files.add(var_data["filename"])

        for fname in sorted(d_files):
            fpath = snapshot.desktop_dir / fname
            if fpath.is_file():
                k = f"solstone/{lane}/{version}/{fname}"
                b = fpath.read_bytes()
                if fname.endswith(".json"):
                    ct = "application/json"
                elif fname.endswith(".minisig") or fname.endswith(".tar.gz") or fname.endswith(".deb") or fname.endswith(".rpm"):
                    ct = "application/octet-stream"
                else:
                    ct = "text/plain; charset=utf-8"
                _add_claim(claims_to_make, k, (b, ct, "public, max-age=31536000, immutable"))

        # Tmux objects
        t_ver = ingested_t.version
        t_files = {
            "SHA256SUMS",
            "SHA256SUMS.minisig",
        }
        for target_arch in ["x86_64", "aarch64"]:
            mapping = get_target_mapping(target_arch)
            t_files.add(f"solstone-tmux-{t_ver}-{mapping.musl_target}.target.json")
            if target_arch in ingested_t.arches:
                for var_data in ingested_t.arches[target_arch].values():
                    t_files.add(var_data["filename"])

        for fname in sorted(t_files):
            fpath = snapshot.tmux_dir / fname
            if fpath.is_file():
                k = f"solstone/{lane}/{version}/{fname}"
                b = fpath.read_bytes()
                if fname.endswith(".json"):
                    ct = "application/json"
                elif fname.endswith(".minisig") or fname.endswith(".tar.gz") or fname.endswith(".deb") or fname.endswith(".rpm"):
                    ct = "application/octet-stream"
                else:
                    ct = "text/plain; charset=utf-8"
                _add_claim(claims_to_make, k, (b, ct, "public, max-age=31536000, immutable"))

        # 2. Journal bootstrap object
        if snapshot.bootstrap_file and snapshot.bootstrap_file.is_file():
            boot_bytes = snapshot.bootstrap_file.read_bytes()
        else:
            boot_bytes = (snapshot.journal_dir / expected_boot_filename).read_bytes() if (snapshot.journal_dir / expected_boot_filename).is_file() else (snapshot.journal_dir / "linux-x86_64" / expected_boot_filename).read_bytes()

        boot_key = f"solstone-journal/{lane}/{ingested_j.version}/{expected_boot_filename}"
        bootstrap_claim = (boot_bytes, "application/octet-stream", "public, max-age=31536000, immutable")

        # 3. Platform pair
        manifest_key = f"solstone/{lane}/{version}/platform.json"
        sig_key = f"solstone/{lane}/{version}/platform.json.minisig"
        platform_manifest_claim = (canonical_manifest, "application/json", "public, max-age=31536000, immutable")
        platform_sig_claim = (signature_bytes, "application/octet-stream", "public, max-age=31536000, immutable")

        latest_key = f"solstone/{lane}/latest"
        latest_body = f"{version}\n".encode("utf-8")
        latest_ct = "text/plain; charset=utf-8"
        latest_cc = "no-store, max-age=0"

        # Fence the release against the exact pointer observed during preparation.
        current_latest = dest.get(latest_key)
        expected_etag: str
        legacy_equal = False
        if receipt_obj is not None:
            expected_base_body = receipt_obj["base"]["body"].encode("utf-8")
            expected_base_etag = receipt_obj["base"]["etag"]
            if current_latest.is_ok() and current_latest.body == latest_body:
                for k, (b, ct, cc) in sorted(claims_to_make.items()):
                    _verify_immutable_object(dest, k, b, ct, cc)
                _verify_immutable_object(dest, boot_key, bootstrap_claim[0], bootstrap_claim[1], bootstrap_claim[2])
                _verify_immutable_object(dest, manifest_key, platform_manifest_claim[0], platform_manifest_claim[1], platform_manifest_claim[2])
                _verify_immutable_object(dest, sig_key, platform_sig_claim[0], platform_sig_claim[1], platform_sig_claim[2])
                _verify_immutable_object(dest, latest_key, latest_body, latest_ct, latest_cc)
                return PublishReport(version, lane, manifest_key, sig_key, latest_key, False)
            if (
                not current_latest.is_ok()
                or current_latest.body != expected_base_body
                or current_latest.etag != expected_base_etag
            ):
                raise Refusal(RELEASE_COHERENCE, "latest moved since recut preparation; prepare again")
            expected_etag = expected_base_etag
        elif expect_latest_absent:
            if not current_latest.is_absent():
                raise Refusal(RELEASE_COHERENCE, "initial publication expected latest to be absent")
            expected_etag = ""
        else:
            # Compatibility for the sealed in-memory fixture only. Production callers
            # are required to provide an explicit expectation above.
            if current_latest.is_ok():
                curr_ver = (current_latest.body or b"").decode("utf-8", errors="strict").strip()
                cmp = compare_semver(version, curr_ver)
                if cmp < 0:
                    raise Refusal(ROLLBACK_REFUSED, f"incoming version {version} is older than current latest {curr_ver}")
                if cmp == 0:
                    legacy_equal = True
                expected_etag = current_latest.etag or ""
            elif current_latest.is_absent():
                expected_etag = ""
            else:
                raise Refusal(PUBLISH_INDETERMINATE, f"failed to read {latest_key}: {current_latest.status}")

        # Execute Phase 1: Claim natives
        for k, (b, ct, cc) in sorted(claims_to_make.items()):
            _claim_immutable_object(dest, k, b, ct, cc)

        # Execute Phase 2: Claim bootstrap
        _claim_immutable_object(dest, boot_key, bootstrap_claim[0], bootstrap_claim[1], bootstrap_claim[2])

        # Execute Phase 3: Claim platform pair
        _claim_immutable_object(dest, manifest_key, platform_manifest_claim[0], platform_manifest_claim[1], platform_manifest_claim[2])
        _claim_immutable_object(dest, sig_key, platform_sig_claim[0], platform_sig_claim[1], platform_sig_claim[2])

        # Execute Phase 4: Authoritative re-read and verify complete dependency set
        for k, (b, ct, cc) in sorted(claims_to_make.items()):
            _verify_immutable_object(dest, k, b, ct, cc)
        _verify_immutable_object(dest, boot_key, bootstrap_claim[0], bootstrap_claim[1], bootstrap_claim[2])
        _verify_immutable_object(dest, manifest_key, platform_manifest_claim[0], platform_manifest_claim[1], platform_manifest_claim[2])
        _verify_immutable_object(dest, sig_key, platform_sig_claim[0], platform_sig_claim[1], platform_sig_claim[2])

        if legacy_equal:
            _verify_immutable_object(dest, latest_key, latest_body, latest_ct, latest_cc)
            return PublishReport(version, lane, manifest_key, sig_key, latest_key, False)

        # Execute Phase 5 exactly once. Never adopt a newer pointer generation.
        cas_res = dest.compare_and_swap(
            key=latest_key,
            body=latest_body,
            expected_etag=expected_etag,
            content_type=latest_ct,
            cache_control=latest_cc,
        )
        if cas_res.is_ok():
            _verify_immutable_object(dest, latest_key, latest_body, latest_ct, latest_cc)
            return PublishReport(version, lane, manifest_key, sig_key, latest_key, True)

        reread = dest.get(latest_key)
        if reread.is_ok() and reread.body == latest_body:
            _verify_immutable_object(dest, latest_key, latest_body, latest_ct, latest_cc)
            return PublishReport(version, lane, manifest_key, sig_key, latest_key, False)
        if reread.is_ok() and reread.body is not None:
            observed = reread.body.decode("utf-8", errors="replace").strip()
            if compare_semver(version, observed) < 0:
                raise Refusal(ROLLBACK_REFUSED, f"incoming version {version} is older than current latest {observed}")
            raise Refusal(RELEASE_COHERENCE, f"candidate is not current; latest moved to {observed}")
        raise Refusal(PUBLISH_INDETERMINATE, f"CAS outcome indeterminate for {latest_key}: {cas_res.status}")
