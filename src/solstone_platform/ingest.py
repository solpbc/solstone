# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Native artifact and signed metadata ingestion for journal, desktop, and tmux."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Optional
from urllib.parse import urlparse

from solstone_platform.archive import scan_variant_archive
from solstone_platform.canonical import parse_json_strict
from solstone_platform.pins import MinisignPin, PinSet, embedded_pins
from solstone_platform.refusals import (
    DESKTOP_AARCH64,
    INCOMPLETE_VARIANTS,
    RELEASE_COHERENCE,
    SCHEMA_INVALID,
    SIGNATURE_PIN_MISMATCH,
    UNSAFE_FILENAME,
    URL_FRAGMENT,
    URL_INSECURE,
    URL_USERINFO,
    Refusal,
)
from solstone_platform.targets import (
    MUSL_TO_ARCH,
    NATIVE_TO_ARCH,
    TARGET_MAPPINGS,
    get_target_mapping,
)


def verify_minisign_signature(pubkey_pin: MinisignPin, message_path: Path, signature_path: Path) -> None:
    """Verify a file's detached signature using host minisign binary."""
    if not message_path.is_file():
        raise Refusal(RELEASE_COHERENCE, f"message file not found: {message_path}")
    if not signature_path.is_file():
        raise Refusal(RELEASE_COHERENCE, f"signature file not found: {signature_path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pub = Path(tmp_dir) / "key.pub"
        tmp_pub.write_text(pubkey_pin.to_minisign_pub_file_content(), encoding="utf-8")

        proc = subprocess.run(
            ["minisign", "-V", "-p", str(tmp_pub), "-m", str(message_path), "-x", str(signature_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
            raise Refusal(SIGNATURE_PIN_MISMATCH, f"minisign verification failed for {message_path.name}: {err_msg}")


def validate_bootstrap_url(url_str: str) -> None:
    """Validate bootstrap URL for security constraints."""
    parsed = urlparse(url_str)
    if parsed.username or parsed.password:
        raise Refusal(URL_USERINFO, f"URL contains userinfo: {url_str}")
    if parsed.fragment:
        raise Refusal(URL_FRAGMENT, f"URL contains fragment: {url_str}")
    if parsed.scheme != "https":
        if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise Refusal(URL_INSECURE, f"remote URL must use HTTPS: {url_str}")


def extract_bootstrap_contract_version(script_bytes: bytes) -> int:
    """Extract BOOTSTRAP_REVISION integer from script bytes."""
    text = script_bytes.decode("utf-8", errors="replace")
    for line in text.splitlines():
        match = re.match(r"^BOOTSTRAP_REVISION=([0-9]+)$", line.strip())
        if match:
            return int(match.group(1))
    raise Refusal(RELEASE_COHERENCE, "BOOTSTRAP_REVISION not found in bootstrap script")


def parse_release_sidecar(content: str) -> dict[str, str]:
    """Parse key=value pairs from a .release sidecar file."""
    result = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise Refusal(RELEASE_COHERENCE, f"malformed line in .release: '{line}'")
        k, v = line.split("=", 1)
        result[k.strip()] = v.strip()
    return result


@dataclass
class IngestedComponent:
    name: str
    version: str
    provenance: Optional[dict[str, Any]]
    arches: dict[str, dict[str, Any]]


def ingest_journal(
    native_dir: Path,
    lane: str,
    origin: str,
    bootstrap_file: Optional[Path],
    pin: MinisignPin,
) -> IngestedComponent:
    """Ingest solstone-journal release from linux-x86_64 and linux-aarch64 trees."""
    validate_bootstrap_url(origin)
    arches_data: dict[str, dict[str, Any]] = {}
    provenance_data: Optional[dict[str, Any]] = None
    component_version: Optional[str] = None

    for target_arch in ["x86_64", "aarch64"]:
        mapping = get_target_mapping(target_arch)
        arch_dir = native_dir / mapping.native_target
        if not arch_dir.is_dir():
            # Try flat structure where native_dir contains files directly
            arch_dir = native_dir

        # Find manifest file
        manifests = list(arch_dir.glob(f"solstone-journal-*-{mapping.native_target}.manifest.json"))
        if not manifests:
            raise Refusal(RELEASE_COHERENCE, f"no journal manifest found for {mapping.native_target} in {arch_dir}")
        manifest_path = manifests[0]
        sig_path = manifest_path.with_name(manifest_path.name + ".minisig")

        verify_minisign_signature(pin, manifest_path, sig_path)
        manifest_bytes = manifest_path.read_bytes()
        sig_bytes = sig_path.read_bytes()
        manifest_obj = parse_json_strict(manifest_bytes)

        if manifest_obj.get("product") != "solstone-journal":
            raise Refusal(RELEASE_COHERENCE, f"manifest product is '{manifest_obj.get('product')}', expected 'solstone-journal'")

        ver = manifest_obj.get("version")
        if not ver:
            raise Refusal(RELEASE_COHERENCE, "manifest missing version")
        if component_version is None:
            component_version = ver
        elif component_version != ver:
            raise Refusal(RELEASE_COHERENCE, f"version mismatch between arches: {component_version} vs {ver}")

        expected_basename = f"solstone-journal-{ver}-{mapping.native_target}"
        if manifest_path.name != f"{expected_basename}.manifest.json":
            raise Refusal(RELEASE_COHERENCE, f"manifest filename mismatch: {manifest_path.name}")

        release_path = arch_dir / f"{expected_basename}.release"
        if not release_path.is_file():
            raise Refusal(RELEASE_COHERENCE, f"missing {release_path.name}")
        release_bytes = release_path.read_bytes()
        release_dict = parse_release_sidecar(release_bytes.decode("utf-8", errors="replace"))

        # Check required v2 release keys
        required_v2_keys = ["bootstrap_sha256", "state_reader_min", "state_reader_max", "upgrade_epoch", "retention_window", "min_bootstrap_revision"]
        for rk in required_v2_keys:
            if rk not in release_dict:
                raise Refusal(RELEASE_COHERENCE, f"journal .release missing v2 key '{rk}'")

        if release_dict.get("upgrade_epoch") != "journal-v2":
            raise Refusal(RELEASE_COHERENCE, f"unsupported upgrade_epoch: {release_dict.get('upgrade_epoch')}")

        # Bootstrap verification
        if bootstrap_file and bootstrap_file.is_file():
            boot_bytes = bootstrap_file.read_bytes()
        else:
            boot_path = arch_dir / "install.sh"
            if not boot_path.is_file():
                boot_path = native_dir / "install.sh"
            if not boot_path.is_file():
                raise Refusal(RELEASE_COHERENCE, "bootstrap install.sh not found for journal")
            boot_bytes = boot_path.read_bytes()

        computed_boot_sha = hashlib.sha256(boot_bytes).hexdigest()
        if computed_boot_sha != release_dict["bootstrap_sha256"]:
            raise Refusal(RELEASE_COHERENCE, f"bootstrap script sha256 mismatch: {computed_boot_sha} vs release {release_dict['bootstrap_sha256']}")

        contract_version = extract_bootstrap_contract_version(boot_bytes)
        bootstrap_url = f"{origin.rstrip('/')}/solstone-journal/{lane}/{ver}/install.sh"
        validate_bootstrap_url(bootstrap_url)

        if provenance_data is None:
            provenance_data = {
                "bootstrap": {
                    "url": bootstrap_url,
                    "sha256": computed_boot_sha,
                    "contract_version": contract_version,
                },
                "upgrade_epoch": release_dict["upgrade_epoch"],
                "state_reader_min": release_dict["state_reader_min"],
                "state_reader_max": release_dict["state_reader_max"],
                "retention_window": int(release_dict["retention_window"]),
            }

        # Process variants: tree (.tar.gz), deb (.deb), rpm (.rpm)
        variants_dict: dict[str, Any] = {}
        files_map = manifest_obj.get("files", {})

        variant_files = {
            "tree": f"{expected_basename}.tar.gz",
            "deb": f"{expected_basename}.deb",
            "rpm": f"{expected_basename}.rpm",
        }

        for var_type, filename in variant_files.items():
            file_path = arch_dir / filename
            if not file_path.is_file():
                raise Refusal(INCOMPLETE_VARIANTS, f"missing variant archive {filename} for journal {target_arch}")

            file_bytes = file_path.read_bytes()
            computed_file_sha = hashlib.sha256(file_bytes).hexdigest()

            if filename not in files_map:
                raise Refusal(RELEASE_COHERENCE, f"manifest does not list member {filename}")
            if files_map[filename] != computed_file_sha:
                raise Refusal(RELEASE_COHERENCE, f"checksum mismatch for {filename}: computed {computed_file_sha} vs manifest {files_map[filename]}")

            scan_res = scan_variant_archive(file_path)
            exe_sha = scan_res.executable_sha256.get("journal")
            if not exe_sha:
                raise Refusal(RELEASE_COHERENCE, f"executable 'journal' not found in archive {filename}")

            pkg_id = scan_res.package_identity.to_dict() if scan_res.package_identity else None

            variants_dict[var_type] = {
                "filename": filename,
                "sha256": computed_file_sha,
                "bytes": len(file_bytes),
                "native_target": mapping.native_target,
                "package_identity": pkg_id,
                "executable": {
                    "name": "journal",
                    "sha256": exe_sha,
                    "version_command": ["journal", "--version"],
                },
                "payload_build_id": scan_res.payload_build_id,
                "archive_inventory": scan_res.inventory,
                "authority": {
                    "type": "journal-manifest-release-bootstrap",
                    "verifier_id": pin.verifier_id(),
                    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "signature_sha256": hashlib.sha256(sig_bytes).hexdigest(),
                    "release_sha256": hashlib.sha256(release_bytes).hexdigest(),
                    "bootstrap_sha256": computed_boot_sha,
                },
            }

        arches_data[target_arch] = variants_dict

    return IngestedComponent(
        name="journal",
        version=component_version or "",
        provenance=provenance_data,
        arches=arches_data,
    )


def ingest_desktop(native_dir: Path, pin: MinisignPin) -> IngestedComponent:
    """Ingest solstone-linux desktop release (x86_64 only)."""
    # Verify no aarch64 exists in desktop native dir
    aarch64_files = list(native_dir.glob("*aarch64*"))
    if aarch64_files:
        raise Refusal(DESKTOP_AARCH64, f"desktop component contains aarch64 artifacts: {[f.name for f in aarch64_files]}")

    manifests = list(native_dir.glob("solstone-linux-*-linux-x86_64.rust-release-manifest.json"))
    if not manifests:
        raise Refusal(RELEASE_COHERENCE, f"desktop rust-release-manifest not found in {native_dir}")
    manifest_path = manifests[0]
    sig_path = manifest_path.with_name(manifest_path.name + ".minisig")

    verify_minisign_signature(pin, manifest_path, sig_path)
    manifest_bytes = manifest_path.read_bytes()
    sig_bytes = sig_path.read_bytes()
    manifest_obj = parse_json_strict(manifest_bytes)

    if manifest_obj.get("product") != "solstone-linux":
        raise Refusal(RELEASE_COHERENCE, f"desktop manifest product mismatch: {manifest_obj.get('product')}")

    ver = manifest_obj.get("version")
    if not ver:
        raise Refusal(RELEASE_COHERENCE, "desktop manifest missing version")

    artifacts = manifest_obj.get("artifacts", [])
    artifact_map = {item["path"]: item for item in artifacts if "path" in item}

    # Identify tree, deb, rpm
    tar_item = next((item for item in artifacts if item["path"].endswith(".tar.gz")), None)
    deb_item = next((item for item in artifacts if item["path"].endswith(".deb")), None)
    rpm_item = next((item for item in artifacts if item["path"].endswith(".rpm")), None)

    if not tar_item or not deb_item or not rpm_item:
        raise Refusal(INCOMPLETE_VARIANTS, "desktop missing tree, deb, or rpm artifact in manifest")

    variants_dict: dict[str, Any] = {}
    for var_type, item in [("tree", tar_item), ("deb", deb_item), ("rpm", rpm_item)]:
        filename = item["path"]
        file_path = native_dir / filename
        if not file_path.is_file():
            raise Refusal(INCOMPLETE_VARIANTS, f"desktop artifact file {filename} not found on disk")

        file_bytes = file_path.read_bytes()
        computed_sha = hashlib.sha256(file_bytes).hexdigest()
        if computed_sha != item["sha256"] or len(file_bytes) != item["bytes"]:
            raise Refusal(RELEASE_COHERENCE, f"desktop artifact {filename} digest or length mismatch")

        scan_res = scan_variant_archive(file_path)
        exe_sha = scan_res.executable_sha256.get("solstone-linux")
        if not exe_sha:
            raise Refusal(RELEASE_COHERENCE, f"executable 'solstone-linux' not found in {filename}")

        pkg_id = scan_res.package_identity.to_dict() if scan_res.package_identity else None

        variants_dict[var_type] = {
            "filename": filename,
            "sha256": computed_sha,
            "bytes": len(file_bytes),
            "native_target": "linux-x86_64",
            "package_identity": pkg_id,
            "executable": {
                "name": "solstone-linux",
                "sha256": exe_sha,
                "version_command": ["solstone-linux", "--version"],
            },
            "payload_build_id": scan_res.payload_build_id,
            "archive_inventory": scan_res.inventory,
            "authority": {
                "type": "desktop-rust-release-manifest",
                "verifier_id": pin.verifier_id(),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "signature_sha256": hashlib.sha256(sig_bytes).hexdigest(),
            },
        }

    return IngestedComponent(
        name="desktop",
        version=ver,
        provenance=None,
        arches={"x86_64": variants_dict},
    )


def ingest_tmux(native_dir: Path, pin: MinisignPin) -> IngestedComponent:
    """Ingest solstone-tmux release (x86_64 and aarch64)."""
    sums_path = native_dir / "SHA256SUMS"
    sig_path = native_dir / "SHA256SUMS.minisig"
    if not sums_path.is_file() or not sig_path.is_file():
        raise Refusal(RELEASE_COHERENCE, f"tmux SHA256SUMS or SHA256SUMS.minisig missing in {native_dir}")

    verify_minisign_signature(pin, sums_path, sig_path)
    sums_bytes = sums_path.read_bytes()
    sig_bytes = sig_path.read_bytes()

    # Parse SHA256SUMS
    checksums: dict[str, str] = {}
    for line in sums_bytes.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            checksums[parts[1].lstrip("*")] = parts[0].lower()

    component_version: Optional[str] = None
    arches_data: dict[str, dict[str, Any]] = {}

    for target_arch in ["x86_64", "aarch64"]:
        mapping = get_target_mapping(target_arch)
        target_json_name = f"solstone-tmux-2.0.3-{mapping.musl_target}.target.json"
        # Find any matching target.json
        matching_targets = list(native_dir.glob(f"solstone-tmux-*-{mapping.musl_target}.target.json"))
        if not matching_targets:
            raise Refusal(RELEASE_COHERENCE, f"tmux target json missing for {mapping.musl_target}")

        target_json_path = matching_targets[0]
        if target_json_path.name not in checksums:
            raise Refusal(RELEASE_COHERENCE, f"{target_json_path.name} not in SHA256SUMS")

        target_bytes = target_json_path.read_bytes()
        computed_target_sha = hashlib.sha256(target_bytes).hexdigest()
        if computed_target_sha != checksums[target_json_path.name]:
            raise Refusal(RELEASE_COHERENCE, f"checksum mismatch for {target_json_path.name}")

        target_obj = parse_json_strict(target_bytes)
        ver = target_obj.get("product_version")
        if not ver:
            raise Refusal(RELEASE_COHERENCE, f"missing product_version in {target_json_path.name}")
        if component_version is None:
            component_version = ver
        elif component_version != ver:
            raise Refusal(RELEASE_COHERENCE, f"tmux version mismatch: {component_version} vs {ver}")

        # Artifacts from target.json
        target_artifacts = {item["name"]: item["sha256"] for item in target_obj.get("artifacts", [])}

        # Locate tree, deb, rpm
        tar_name = f"solstone-tmux-{ver}-{target_arch}-linux.tar.gz"
        deb_name = f"solstone-tmux_{ver}_{mapping.deb_arch}.deb"
        rpm_name = f"solstone-tmux-{ver}-1.{mapping.rpm_arch}.rpm"

        variants_dict: dict[str, Any] = {}
        for var_type, filename in [("tree", tar_name), ("deb", deb_name), ("rpm", rpm_name)]:
            file_path = native_dir / filename
            if not file_path.is_file():
                raise Refusal(INCOMPLETE_VARIANTS, f"tmux artifact {filename} not found on disk")

            file_bytes = file_path.read_bytes()
            computed_sha = hashlib.sha256(file_bytes).hexdigest()

            if filename not in checksums or checksums[filename] != computed_sha:
                raise Refusal(RELEASE_COHERENCE, f"checksum mismatch in SHA256SUMS for tmux artifact {filename}")
            if filename not in target_artifacts or target_artifacts[filename] != computed_sha:
                raise Refusal(RELEASE_COHERENCE, f"checksum mismatch in target.json for tmux artifact {filename}")

            scan_res = scan_variant_archive(file_path)
            exe_sha = scan_res.executable_sha256.get("solstone-tmux")
            if not exe_sha:
                raise Refusal(RELEASE_COHERENCE, f"executable 'solstone-tmux' not found in {filename}")

            pkg_id = scan_res.package_identity.to_dict() if scan_res.package_identity else None

            variants_dict[var_type] = {
                "filename": filename,
                "sha256": computed_sha,
                "bytes": len(file_bytes),
                "native_target": mapping.native_target,
                "package_identity": pkg_id,
                "executable": {
                    "name": "solstone-tmux",
                    "sha256": exe_sha,
                    "version_command": ["solstone-tmux", "--version"],
                },
                "payload_build_id": scan_res.payload_build_id,
                "archive_inventory": scan_res.inventory,
                "authority": {
                    "type": "tmux-sha256sums-target",
                    "verifier_id": pin.verifier_id(),
                    "sums_sha256": hashlib.sha256(sums_bytes).hexdigest(),
                    "signature_sha256": hashlib.sha256(sig_bytes).hexdigest(),
                    "target_json_sha256": computed_target_sha,
                },
            }

        arches_data[target_arch] = variants_dict

    return IngestedComponent(
        name="tmux",
        version=component_version or "",
        provenance=None,
        arches=arches_data,
    )
