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
    DUPLICATE_KEY,
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
from solstone_platform.witness import get_witness


def verify_minisign_signature(pubkey_pin: MinisignPin, message_path: Path, signature_path: Path) -> None:
    """Verify a file's detached signature using host minisign binary."""
    get_witness().record("pin_verify", str(message_path))
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
    """Extract BOOTSTRAP_CONTRACT_VERSION from exact bootstrap bytes."""
    get_witness().record("bootstrap_contract_extract")
    text = script_bytes.decode("utf-8", errors="replace")
    for line in text.splitlines():
        match = re.match(r"^BOOTSTRAP_CONTRACT_VERSION=([0-9]+)$", line.strip())
        if match:
            return int(match.group(1))
    raise Refusal(RELEASE_COHERENCE, "BOOTSTRAP_CONTRACT_VERSION not found in bootstrap script")


def extract_bootstrap_revision(script_bytes: bytes) -> int:
    """Extract BOOTSTRAP_REVISION from exact bootstrap bytes."""
    get_witness().record("bootstrap_revision_extract")
    text = script_bytes.decode("utf-8", errors="replace")
    for line in text.splitlines():
        match = re.match(r"^BOOTSTRAP_REVISION=([0-9]+)$", line.strip())
        if match:
            return int(match.group(1))
    raise Refusal(RELEASE_COHERENCE, "BOOTSTRAP_REVISION not found in bootstrap script")


def parse_release_sidecar(content: str) -> dict[str, str]:
    """Parse key=value pairs from a .release sidecar file."""
    get_witness().record("release_parse")
    result = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise Refusal(RELEASE_COHERENCE, f"malformed line in .release: '{line}'")
        raw_key, raw_value = line.split("=", 1)
        k = raw_key.strip()
        v = raw_value.strip()
        if k != raw_key or v != raw_value:
            raise Refusal(RELEASE_COHERENCE, f"non-canonical whitespace in .release key '{k}'")
        if not k or k in result:
            raise Refusal(RELEASE_COHERENCE, f"empty or duplicate key in .release: '{k}'")
        result[k] = v
    return result


def parse_sha256_sidecar(content: str) -> dict[str, str]:
    """Parse hex  filename lines from a .sha256 sidecar file."""
    get_witness().record("sha256_parse")
    result: dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise Refusal(RELEASE_COHERENCE, f"malformed line in .sha256 sidecar: '{line}'")
        sha, fname = parts[0].lower(), parts[1].lstrip("*").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise Refusal(RELEASE_COHERENCE, f"invalid sha256 in sidecar: '{sha}'")
        if fname in result:
            raise Refusal(RELEASE_COHERENCE, f"duplicate filename in .sha256 sidecar: '{fname}'")
        result[fname] = sha
    return result


def _semver_tuple(value: str, field: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", value)
    if not match:
        raise Refusal(RELEASE_COHERENCE, f"invalid {field}: {value}")
    return tuple(int(part) for part in match.groups())


def _positive_int(value: str, field: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise Refusal(RELEASE_COHERENCE, f"invalid {field}: {value}")
    return int(value)


def _require_one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise Refusal(RELEASE_COHERENCE, f"expected exactly one {description}, found {len(paths)}")
    return paths[0]


def _parse_tmux_checksums(content: bytes) -> dict[str, str]:
    checksums: dict[str, str] = {}
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise Refusal(RELEASE_COHERENCE, "tmux SHA256SUMS is not valid utf-8")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise Refusal(RELEASE_COHERENCE, f"malformed tmux SHA256SUMS line: '{line}'")
        sha = parts[0].lower()
        fname = parts[1].lstrip("*").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise Refusal(RELEASE_COHERENCE, f"invalid sha256 in tmux SHA256SUMS: '{sha}'")
        if not fname or "/" in fname or "\\" in fname or fname in {".", ".."}:
            raise Refusal(RELEASE_COHERENCE, f"unsafe filename in tmux SHA256SUMS: '{fname}'")
        if fname in checksums:
            raise Refusal(RELEASE_COHERENCE, f"duplicate entry in tmux SHA256SUMS: {fname}")
        checksums[fname] = sha
    return checksums


def _validate_desktop_artifacts(artifacts: Any) -> None:
    if not isinstance(artifacts, list):
        raise Refusal(RELEASE_COHERENCE, "desktop manifest artifacts must be a list")
    seen_paths: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            raise Refusal(RELEASE_COHERENCE, "desktop manifest artifact must be an object")
        path = item.get("path")
        digest = item.get("sha256")
        byte_length = item.get("bytes")
        if (
            not isinstance(path, str)
            or not path
            or "/" in path
            or "\\" in path
            or path in {".", ".."}
            or path in seen_paths
        ):
            raise Refusal(RELEASE_COHERENCE, f"duplicate or invalid artifact path in desktop manifest: {path}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise Refusal(RELEASE_COHERENCE, f"invalid artifact sha256 in desktop manifest: {digest}")
        if not isinstance(byte_length, int) or isinstance(byte_length, bool) or byte_length < 0:
            raise Refusal(RELEASE_COHERENCE, f"invalid artifact byte length in desktop manifest: {byte_length}")
        seen_paths.add(path)


def _parse_tmux_target_artifacts(artifacts: Any) -> dict[str, str]:
    if not isinstance(artifacts, list):
        raise Refusal(RELEASE_COHERENCE, "tmux target artifacts must be a list")
    result: dict[str, str] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            raise Refusal(RELEASE_COHERENCE, "tmux target artifact must be an object")
        name = item.get("name")
        digest = item.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or name in {".", ".."}
            or name in result
        ):
            raise Refusal(RELEASE_COHERENCE, f"duplicate or invalid artifact name in tmux target.json: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise Refusal(RELEASE_COHERENCE, f"invalid artifact sha in tmux target.json: {digest}")
        result[name] = digest
    return result


def _require_exact_artifact_names(actual: set[str], expected: set[str], description: str) -> None:
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise Refusal(RELEASE_COHERENCE, f"{description} artifact set mismatch: missing={missing}, extra={extra}")


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

    cross_arch_commit: Optional[str] = None
    cross_arch_lock: Optional[str] = None

    for target_arch in ["x86_64", "aarch64"]:
        mapping = get_target_mapping(target_arch)
        arch_dir = native_dir / mapping.native_target
        if not arch_dir.is_dir():
            # Try flat structure where native_dir contains files directly
            arch_dir = native_dir

        # Find manifest file
        manifests = list(arch_dir.glob(f"solstone-journal-*-{mapping.native_target}.manifest.json"))
        manifest_path = _require_one(manifests, f"journal manifest for {mapping.native_target}")
        sig_path = manifest_path.with_name(manifest_path.name + ".minisig")

        verify_minisign_signature(pin, manifest_path, sig_path)
        manifest_bytes = manifest_path.read_bytes()
        sig_bytes = sig_path.read_bytes()
        manifest_obj = parse_json_strict(manifest_bytes)
        get_witness().record("json_parse", str(manifest_path))

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
        if manifest_obj.get("target") != mapping.native_target:
            raise Refusal(RELEASE_COHERENCE, f"manifest target mismatch: {manifest_obj.get('target')}")

        files_map = manifest_obj.get("files")
        if not isinstance(files_map, dict) or any(
            not isinstance(name, str) or not isinstance(digest, str)
            for name, digest in files_map.items()
        ):
            raise Refusal(RELEASE_COHERENCE, "journal manifest files must be a filename-to-digest object")

        # Check .release sidecar
        release_path = arch_dir / f"{expected_basename}.release"
        if not release_path.is_file():
            raise Refusal(RELEASE_COHERENCE, f"missing {release_path.name}")
        release_bytes = release_path.read_bytes()
        release_sha = hashlib.sha256(release_bytes).hexdigest()
        if files_map.get(release_path.name) != release_sha:
            raise Refusal(RELEASE_COHERENCE, f"manifest does not bind exact release bytes: {release_path.name}")
        release_dict = parse_release_sidecar(release_bytes.decode("utf-8", errors="replace"))

        # Check required v2 release keys
        required_v2_keys = [
            "product",
            "version",
            "target",
            "commit",
            "lock_sha256",
            "bootstrap_contract_version",
            "bootstrap_filename",
            "state_reader_min",
            "state_reader_max",
            "upgrade_epoch",
            "retention_window",
            "min_bootstrap_revision",
        ]
        for rk in required_v2_keys:
            if rk not in release_dict:
                raise Refusal(RELEASE_COHERENCE, f"journal .release missing v2 key '{rk}'")

        # Validate commit format (40 lowercase hex) and cross-arch equality
        commit_val = release_dict["commit"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit_val):
            raise Refusal(RELEASE_COHERENCE, f"journal .release commit invalid format: '{commit_val}'")
        if cross_arch_commit is None:
            cross_arch_commit = commit_val
        elif cross_arch_commit != commit_val:
            raise Refusal(RELEASE_COHERENCE, f"journal commit mismatch across arches: {cross_arch_commit} vs {commit_val}")

        # Validate lock_sha256 format (64 lowercase hex) and cross-arch equality
        lock_val = release_dict["lock_sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", lock_val):
            raise Refusal(RELEASE_COHERENCE, f"journal .release lock_sha256 invalid format: '{lock_val}'")
        if cross_arch_lock is None:
            cross_arch_lock = lock_val
        elif cross_arch_lock != lock_val:
            raise Refusal(RELEASE_COHERENCE, f"journal lock_sha256 mismatch across arches: {cross_arch_lock} vs {lock_val}")

        if release_dict["product"] != "solstone-journal":
            raise Refusal(RELEASE_COHERENCE, f"journal .release product mismatch: {release_dict['product']}")
        if release_dict["version"] != ver:
            raise Refusal(RELEASE_COHERENCE, f"journal .release version mismatch: {release_dict['version']}")
        if release_dict["target"] != mapping.native_target:
            raise Refusal(RELEASE_COHERENCE, f"journal .release target mismatch: {release_dict['target']}")
        if release_dict.get("upgrade_epoch") != "journal-v2":
            raise Refusal(RELEASE_COHERENCE, f"unsupported upgrade_epoch: {release_dict.get('upgrade_epoch')}")

        contract_version = _positive_int(release_dict["bootstrap_contract_version"], "bootstrap_contract_version")
        if contract_version != 2:
            raise Refusal(RELEASE_COHERENCE, f"unsupported bootstrap_contract_version: {contract_version}")
        min_boot_rev = _positive_int(release_dict["min_bootstrap_revision"], "min_bootstrap_revision")
        retention_window = _positive_int(release_dict["retention_window"], "retention_window")
        reader_min = _semver_tuple(release_dict["state_reader_min"], "state_reader_min")
        reader_max = _semver_tuple(release_dict["state_reader_max"], "state_reader_max")
        current_version = _semver_tuple(ver, "journal version")
        if reader_min > reader_max or not (reader_min <= current_version <= reader_max):
            raise Refusal(RELEASE_COHERENCE, "journal version is outside its signed state reader range")

        # Bootstrap verification
        bootstrap_name = release_dict["bootstrap_filename"]
        expected_bootstrap_name = f"solstone-journal-{ver}-install.sh"
        if bootstrap_name != expected_bootstrap_name:
            raise Refusal(RELEASE_COHERENCE, f"bootstrap filename mismatch: {bootstrap_name}")
        if bootstrap_file and bootstrap_file.is_file():
            boot_bytes = bootstrap_file.read_bytes()
        else:
            boot_path = arch_dir / bootstrap_name
            if not boot_path.is_file():
                boot_path = native_dir / bootstrap_name
            if not boot_path.is_file():
                raise Refusal(RELEASE_COHERENCE, f"bootstrap {bootstrap_name} not found for journal")
            boot_bytes = boot_path.read_bytes()

        computed_boot_sha = hashlib.sha256(boot_bytes).hexdigest()
        if files_map.get(bootstrap_name) != computed_boot_sha:
            raise Refusal(RELEASE_COHERENCE, f"manifest does not bind exact bootstrap bytes: {bootstrap_name}")

        script_contract_version = extract_bootstrap_contract_version(boot_bytes)
        if script_contract_version != contract_version:
            raise Refusal(RELEASE_COHERENCE, "bootstrap contract version disagrees with .release")

        script_bootstrap_rev = extract_bootstrap_revision(boot_bytes)
        if script_bootstrap_rev < min_boot_rev:
            raise Refusal(RELEASE_COHERENCE, f"bootstrap revision {script_bootstrap_rev} < min_bootstrap_revision {min_boot_rev}")

        bootstrap_url = f"{origin.rstrip('/')}/solstone-journal/{lane}/{ver}/{bootstrap_name}"
        validate_bootstrap_url(bootstrap_url)

        arch_provenance = {
            "bootstrap": {
                "url": bootstrap_url,
                "sha256": computed_boot_sha,
                "contract_version": contract_version,
            },
            "upgrade_epoch": release_dict["upgrade_epoch"],
            "state_reader_min": release_dict["state_reader_min"],
            "state_reader_max": release_dict["state_reader_max"],
            "retention_window": retention_window,
        }
        if provenance_data is None:
            provenance_data = arch_provenance
        elif provenance_data != arch_provenance:
            raise Refusal(RELEASE_COHERENCE, "journal provenance differs between architectures")

        # Check .sha256 sidecar
        sha256_sidecar_path = arch_dir / f"{expected_basename}.sha256"
        if not sha256_sidecar_path.is_file():
            raise Refusal(RELEASE_COHERENCE, f"missing journal .sha256 sidecar: {sha256_sidecar_path.name}")
        sha256_sidecar_bytes = sha256_sidecar_path.read_bytes()
        if files_map.get(sha256_sidecar_path.name) != hashlib.sha256(sha256_sidecar_bytes).hexdigest():
            raise Refusal(RELEASE_COHERENCE, f"manifest does not bind exact .sha256 bytes: {sha256_sidecar_path.name}")
        try:
            sha256_sidecar_text = sha256_sidecar_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise Refusal(RELEASE_COHERENCE, f"journal .sha256 sidecar is not valid utf-8: {sha256_sidecar_path.name}")
        sidecar_map = parse_sha256_sidecar(sha256_sidecar_text)

        required_sidecar_members = [
            release_path.name,
            bootstrap_name,
            f"{expected_basename}.tar.gz",
            f"{expected_basename}.deb",
            f"{expected_basename}.rpm",
        ]
        for member in required_sidecar_members:
            if member not in sidecar_map:
                raise Refusal(RELEASE_COHERENCE, f"journal .sha256 sidecar missing member: {member}")

        if sidecar_map[release_path.name] != release_sha:
            raise Refusal(RELEASE_COHERENCE, f"journal .sha256 sidecar mismatch for {release_path.name}")
        if sidecar_map[bootstrap_name] != computed_boot_sha:
            raise Refusal(RELEASE_COHERENCE, f"journal .sha256 sidecar mismatch for {bootstrap_name}")

        # Process variants: tree (.tar.gz), deb (.deb), rpm (.rpm)
        variants_dict: dict[str, Any] = {}

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
            if sidecar_map[filename] != computed_file_sha:
                raise Refusal(RELEASE_COHERENCE, f"checksum mismatch in .sha256 sidecar for {filename}")

            scan_res = scan_variant_archive(file_path)
            get_witness().record("archive_scan", str(file_path))
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
                    "release_sha256": release_sha,
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
    manifest_path = _require_one(manifests, "desktop rust-release-manifest")
    sig_path = manifest_path.with_name(manifest_path.name + ".minisig")

    verify_minisign_signature(pin, manifest_path, sig_path)
    manifest_bytes = manifest_path.read_bytes()
    sig_bytes = sig_path.read_bytes()
    manifest_obj = parse_json_strict(manifest_bytes)
    get_witness().record("json_parse", str(manifest_path))

    if manifest_obj.get("product") != "solstone-linux":
        raise Refusal(RELEASE_COHERENCE, f"desktop manifest product mismatch: {manifest_obj.get('product')}")

    ver = manifest_obj.get("version")
    if not ver:
        raise Refusal(RELEASE_COHERENCE, "desktop manifest missing version")

    if manifest_obj.get("source_dirty") is not False:
        raise Refusal(RELEASE_COHERENCE, "desktop manifest source_dirty must be false")

    commit = manifest_obj.get("source_commit")
    if not commit or not re.fullmatch(r"[0-9a-f]{40}", str(commit)):
        raise Refusal(RELEASE_COHERENCE, "desktop manifest missing or invalid source_commit")

    triple = manifest_obj.get("target", {}).get("triple")
    if triple != "x86_64-unknown-linux-gnu":
        raise Refusal(RELEASE_COHERENCE, f"desktop target.triple mismatch: expected 'x86_64-unknown-linux-gnu', got '{triple}'")

    native_target = TARGET_MAPPINGS["x86_64"].native_target

    artifacts = manifest_obj.get("artifacts", [])
    _validate_desktop_artifacts(artifacts)

    expected_artifact_names = {
        "tree": f"solstone-linux-{ver}-linux-x86_64.tar.gz",
        "deb": f"solstone-linux_{ver}-1_amd64.deb",
        "rpm": f"solstone-linux-{ver}-1.x86_64.rpm",
    }
    items_by_name = {item["path"]: item for item in artifacts}
    _require_exact_artifact_names(
        set(items_by_name),
        set(expected_artifact_names.values()),
        "desktop",
    )

    variants_dict: dict[str, Any] = {}
    for var_type, filename in expected_artifact_names.items():
        item = items_by_name[filename]
        filename = item["path"]
        file_path = native_dir / filename
        if not file_path.is_file():
            raise Refusal(INCOMPLETE_VARIANTS, f"desktop artifact file {filename} not found on disk")

        file_bytes = file_path.read_bytes()
        computed_sha = hashlib.sha256(file_bytes).hexdigest()
        if computed_sha != item["sha256"] or len(file_bytes) != item["bytes"]:
            raise Refusal(RELEASE_COHERENCE, f"desktop artifact {filename} digest or length mismatch")

        scan_res = scan_variant_archive(file_path)
        get_witness().record("archive_scan", str(file_path))
        exe_sha = scan_res.executable_sha256.get("solstone-linux")
        if not exe_sha:
            raise Refusal(RELEASE_COHERENCE, f"executable 'solstone-linux' not found in {filename}")

        pkg_id = scan_res.package_identity.to_dict() if scan_res.package_identity else None

        variants_dict[var_type] = {
            "filename": filename,
            "sha256": computed_sha,
            "bytes": len(file_bytes),
            "native_target": native_target,
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

    checksums = _parse_tmux_checksums(sums_bytes)

    component_version: Optional[str] = None
    cross_arch_commit: Optional[str] = None
    arches_data: dict[str, dict[str, Any]] = {}

    for target_arch in ["x86_64", "aarch64"]:
        mapping = get_target_mapping(target_arch)
        # Find matching target.json
        matching_targets = list(native_dir.glob(f"solstone-tmux-*-{mapping.musl_target}.target.json"))
        target_json_path = _require_one(matching_targets, f"tmux target json for {mapping.musl_target}")
        if target_json_path.name not in checksums:
            raise Refusal(RELEASE_COHERENCE, f"{target_json_path.name} not in SHA256SUMS")

        target_bytes = target_json_path.read_bytes()
        computed_target_sha = hashlib.sha256(target_bytes).hexdigest()
        if computed_target_sha != checksums[target_json_path.name]:
            raise Refusal(RELEASE_COHERENCE, f"checksum mismatch for {target_json_path.name}")

        target_obj = parse_json_strict(target_bytes)
        get_witness().record("json_parse", str(target_json_path))

        ver = target_obj.get("product_version")
        if not ver:
            raise Refusal(RELEASE_COHERENCE, f"missing product_version in {target_json_path.name}")
        if component_version is None:
            component_version = ver
        elif component_version != ver:
            raise Refusal(RELEASE_COHERENCE, f"tmux version mismatch: {component_version} vs {ver}")

        # Check rust_target
        if target_obj.get("rust_target") != mapping.musl_target:
            raise Refusal(RELEASE_COHERENCE, f"tmux rust_target mismatch: expected {mapping.musl_target}, got {target_obj.get('rust_target')}")

        # Check source_commit and cross-arch equality
        commit_val = target_obj.get("source_commit")
        if not commit_val or not re.fullmatch(r"[0-9a-f]{40}", str(commit_val)):
            raise Refusal(RELEASE_COHERENCE, f"tmux target.json missing or invalid source_commit: '{commit_val}'")
        if cross_arch_commit is None:
            cross_arch_commit = commit_val
        elif cross_arch_commit != commit_val:
            raise Refusal(RELEASE_COHERENCE, f"tmux source_commit mismatch across arches: {cross_arch_commit} vs {commit_val}")

        # Check target artifacts with uniqueness
        target_artifacts = _parse_tmux_target_artifacts(target_obj.get("artifacts", []))


        # Locate tree, deb, rpm
        tar_name = f"solstone-tmux-{ver}-{target_arch}-linux.tar.gz"
        deb_name = f"solstone-tmux_{ver}_{mapping.deb_arch}.deb"
        rpm_name = f"solstone-tmux-{ver}-1.{mapping.rpm_arch}.rpm"
        _require_exact_artifact_names(
            set(target_artifacts),
            {tar_name, deb_name, rpm_name},
            f"tmux {mapping.musl_target}",
        )

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
            get_witness().record("archive_scan", str(file_path))
            exe_sha = scan_res.executable_sha256.get("solstone-tmux")
            if not exe_sha:
                raise Refusal(RELEASE_COHERENCE, f"executable 'solstone-tmux' not found in {filename}")

            # Verify target.json executable section against canonical tree archive
            if var_type == "tree":
                exe_obj = target_obj.get("executable", {})
                if exe_obj.get("name") != "solstone-tmux" or exe_obj.get("sha256") != exe_sha:
                    raise Refusal(RELEASE_COHERENCE, f"tmux executable identity/digest mismatch in target.json vs archive")


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
