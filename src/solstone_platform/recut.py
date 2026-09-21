# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Prepare a platform recut from already-published component releases."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.capture import MAX_AGGREGATE_BYTES, MAX_ARTIFACT_BYTES, MAX_METADATA_BYTES
from solstone_platform.destination import Destination
from solstone_platform.generate import generate_platform_manifest
from solstone_platform.ingest import ingest_desktop, ingest_journal, ingest_tmux, verify_minisign_signature
from solstone_platform.pins import embedded_pins, require_production_platform_pin
from solstone_platform.publish import compare_semver
from solstone_platform.refusals import (
    CAPTURE_SIZE_LIMIT_EXCEEDED,
    HTTP_3XX,
    HTTP_TIMEOUT,
    HTTP_TRUNCATED_2XX,
    RELEASE_COHERENCE,
    SCHEMA_INVALID,
    UNSAFE_FILENAME,
    Refusal,
)
from solstone_platform.schema import load_platform_manifest_bytes
from solstone_platform.targets import get_target_mapping
from solstone_platform.testdest import FIXTURE_BUILD_TOKEN, FixtureDestination


CANONICAL_ORIGIN = "https://updates.solstone.app"
KNOWN_COMPONENTS = frozenset({"journal", "desktop", "tmux"})
DOWNLOAD_TIMEOUT_SECONDS = 120
DOWNLOAD_OVERALL_TIMEOUT_SECONDS = 900
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Refusal(HTTP_3XX, f"redirect refused for {req.full_url}")


def _safe_name(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise Refusal(UNSAFE_FILENAME, f"unsafe remote filename: {value}")
    return value


class _Downloader:
    def __init__(self) -> None:
        self.opener = urllib.request.build_opener(_NoRedirect)
        self.total = 0
        self.sources: list[dict[str, Any]] = []

    def fetch(self, url: str, destination: Path, *, artifact: bool = False) -> bytes:
        started = time.monotonic()
        limit = MAX_ARTIFACT_BYTES if artifact else MAX_METADATA_BYTES
        request = urllib.request.Request(url, headers={"User-Agent": "solstone-platform-recut/1"})
        try:
            response = self.opener.open(request, timeout=DOWNLOAD_TIMEOUT_SECONDS)
        except Refusal:
            raise
        except urllib.error.HTTPError as err:
            if 300 <= err.code < 400:
                raise Refusal(HTTP_3XX, f"redirect refused for {url}") from err
            raise Refusal(RELEASE_COHERENCE, f"download failed for {url}: HTTP {err.code}") from err
        except (urllib.error.URLError, TimeoutError) as err:
            raise Refusal(HTTP_TIMEOUT, f"download failed for {url}: {err}") from err

        hasher = hashlib.sha256()
        observed = 0
        declared: int | None = None
        try:
            raw_length = response.headers.get("Content-Length")
            if raw_length is not None:
                try:
                    declared = int(raw_length)
                except ValueError as err:
                    raise Refusal(RELEASE_COHERENCE, f"invalid Content-Length for {url}") from err
                if declared < 0 or declared > limit or self.total + declared > MAX_AGGREGATE_BYTES:
                    raise Refusal(CAPTURE_SIZE_LIMIT_EXCEEDED, f"download size exceeds limits for {url}")

            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output:
                while True:
                    if time.monotonic() - started > DOWNLOAD_OVERALL_TIMEOUT_SECONDS:
                        raise Refusal(HTTP_TIMEOUT, f"download exceeded overall deadline for {url}")
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    observed += len(chunk)
                    if observed > limit or self.total + observed > MAX_AGGREGATE_BYTES:
                        raise Refusal(CAPTURE_SIZE_LIMIT_EXCEEDED, f"streamed download exceeds limits for {url}")
                    output.write(chunk)
                    hasher.update(chunk)
                output.flush()
                os.fsync(output.fileno())
        except TimeoutError as err:
            raise Refusal(HTTP_TIMEOUT, f"download timed out for {url}") from err
        finally:
            response.close()

        if declared is not None and observed != declared:
            raise Refusal(HTTP_TRUNCATED_2XX, f"download length mismatch for {url}: {observed} vs {declared}")
        self.total += observed
        digest = hasher.hexdigest()
        self.sources.append({"url": url, "path": "", "sha256": digest, "bytes": observed})
        return destination.read_bytes()


def _git_identity(repo_root: Path) -> str:
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=repo_root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()
        if Path(top).resolve() != repo_root.resolve():
            raise Refusal(RELEASE_COHERENCE, "prepare-recut must run from the solstone repository")
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=repo_root,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout
        if status:
            raise Refusal(RELEASE_COHERENCE, "prepare-recut requires a clean worktree, including untracked files")
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as err:
        raise Refusal(RELEASE_COHERENCE, "could not resolve clean repository HEAD") from err
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Refusal(RELEASE_COHERENCE, "repository HEAD is not a full lowercase commit hash")
    return commit


def _write_remote(downloader: _Downloader, url: str, path: Path, *, artifact: bool = False) -> bytes:
    data = downloader.fetch(url, path, artifact=artifact)
    downloader.sources[-1]["path"] = str(path)
    return data


def _download_journal(d: _Downloader, root: Path, version: str, prefix: str) -> Path:
    out = root / "journal"
    bootstrap_name = f"solstone-journal-{version}-install.sh"
    for arch in ("x86_64", "aarch64"):
        target = f"linux-{arch}"
        arch_dir = out / target
        stem = f"solstone-journal-{version}-{target}"
        manifest_name = f"{stem}.manifest.json"
        manifest = _write_remote(d, f"{prefix}/{manifest_name}", arch_dir / manifest_name)
        _write_remote(d, f"{prefix}/{manifest_name}.minisig", arch_dir / f"{manifest_name}.minisig")
        obj = parse_json_strict(manifest)
        files = obj.get("files") if isinstance(obj, dict) else None
        if not isinstance(files, dict):
            raise Refusal(RELEASE_COHERENCE, f"journal manifest {manifest_name} has no files map")
        for name in sorted(files):
            _safe_name(name)
            if name == bootstrap_name:
                continue
            _write_remote(d, f"{prefix}/{name}", arch_dir / name, artifact=name.endswith((".tar.gz", ".deb", ".rpm")))
    return out


def _download_desktop(d: _Downloader, root: Path, version: str, prefix: str) -> Path:
    out = root / "desktop"
    name = f"solstone-linux-{version}-linux-x86_64.rust-release-manifest.json"
    manifest = _write_remote(d, f"{prefix}/{name}", out / name)
    _write_remote(d, f"{prefix}/{name}.minisig", out / f"{name}.minisig")
    obj = parse_json_strict(manifest)
    artifacts = obj.get("artifacts") if isinstance(obj, dict) else None
    if not isinstance(artifacts, list):
        raise Refusal(RELEASE_COHERENCE, "desktop manifest has no artifacts list")
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise Refusal(RELEASE_COHERENCE, "desktop manifest has invalid artifact entry")
        artifact = _safe_name(item["path"])
        _write_remote(d, f"{prefix}/{artifact}", out / artifact, artifact=True)
    return out


def _download_tmux(d: _Downloader, root: Path, version: str, prefix: str) -> Path:
    out = root / "tmux"
    _write_remote(d, f"{prefix}/SHA256SUMS", out / "SHA256SUMS")
    _write_remote(d, f"{prefix}/SHA256SUMS.minisig", out / "SHA256SUMS.minisig")
    names: set[str] = set()
    for arch in ("x86_64", "aarch64"):
        target = get_target_mapping(arch).musl_target
        target_name = f"solstone-tmux-{version}-{target}.target.json"
        raw = _write_remote(d, f"{prefix}/{target_name}", out / target_name)
        obj = parse_json_strict(raw)
        artifacts = obj.get("artifacts") if isinstance(obj, dict) else None
        if not isinstance(artifacts, list):
            raise Refusal(RELEASE_COHERENCE, f"tmux target {target_name} has no artifacts list")
        for item in artifacts:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise Refusal(RELEASE_COHERENCE, "tmux target has invalid artifact entry")
            names.add(_safe_name(item["name"]))
    for name in sorted(names):
        _write_remote(d, f"{prefix}/{name}", out / name, artifact=True)
    return out


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.name in {"recut-receipt.json", "platform.json.minisig"}:
            continue
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise Refusal(UNSAFE_FILENAME, f"non-regular prepared path: {path}")
        if path.is_file():
            data = path.read_bytes()
            result.append({
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            })
    return result


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(destination)
    if error in {errno.ENOSYS, errno.EINVAL, errno.EXDEV}:
        raise Refusal(RELEASE_COHERENCE, "atomic no-replace rename is unavailable for output filesystem")
    raise OSError(error, os.strerror(error), str(destination))


def _validate_existing(existing: Path, staged: Path) -> bool:
    existing_files = _file_inventory(existing)
    staged_files = _file_inventory(staged)
    if existing_files != staged_files:
        return False
    return (existing / "recut-receipt.json").read_bytes() == (staged / "recut-receipt.json").read_bytes()


def prepare_recut(
    *,
    version: str,
    replacements: dict[str, str],
    output_dir: Path,
    repo_root: Path,
    dest: Destination,
    created_unix: int | None = None,
    origin: str = CANONICAL_ORIGIN,
) -> dict[str, Any]:
    if not replacements or not set(replacements).issubset(KNOWN_COMPONENTS):
        raise Refusal(SCHEMA_INVALID, "at least one known component replacement is required")
    output_dir = output_dir.resolve()
    repo_resolved = repo_root.resolve()
    if output_dir == Path("/tmp") or Path("/tmp") in output_dir.parents or output_dir == repo_resolved or repo_resolved in output_dir.parents:
        raise Refusal(UNSAFE_FILENAME, "recut output must be outside the repository and /tmp")

    source_commit = _git_identity(repo_root)
    latest = dest.get("solstone/release/latest")
    if not latest.is_ok() or latest.body is None or not latest.etag:
        raise Refusal(RELEASE_COHERENCE, f"could not read authoritative release latest: {latest.status}")
    try:
        base_version = latest.body.decode("utf-8").strip()
    except UnicodeDecodeError as err:
        raise Refusal(RELEASE_COHERENCE, "latest is not UTF-8") from err
    if compare_semver(version, base_version) <= 0:
        raise Refusal(RELEASE_COHERENCE, f"candidate {version} must be newer than base {base_version}")

    existing_receipt: dict[str, Any] | None = None
    if output_dir.exists():
        receipt_file = output_dir / "recut-receipt.json"
        if not receipt_file.is_file():
            raise Refusal(RELEASE_COHERENCE, f"existing recut output has no receipt: {output_dir}")
        parsed = parse_json_strict(receipt_file.read_bytes())
        if not isinstance(parsed, dict):
            raise Refusal(RELEASE_COHERENCE, "existing recut receipt is invalid")
        expected_base = parsed.get("base")
        expected_candidate = parsed.get("candidate")
        if (
            not isinstance(expected_candidate, dict)
            or expected_candidate.get("version") != version
            or parsed.get("replacements") != dict(sorted(replacements.items()))
            or parsed.get("source_commit") != source_commit
            or not isinstance(expected_base, dict)
            or expected_base.get("version") != base_version
            or expected_base.get("body") != latest.body.decode("utf-8")
            or expected_base.get("etag") != latest.etag
        ):
            raise Refusal(RELEASE_COHERENCE, "existing recut output belongs to a different preparation")
        existing_receipt = parsed

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.prepare-", dir=parent))
    os.chmod(stage, 0o700)
    try:
        base_manifest_result = dest.get(f"solstone/release/{base_version}/platform.json")
        base_sig_result = dest.get(f"solstone/release/{base_version}/platform.json.minisig")
        if not base_manifest_result.is_ok() or base_manifest_result.body is None or not base_sig_result.is_ok() or base_sig_result.body is None:
            raise Refusal(RELEASE_COHERENCE, "could not read signed base platform release")
        base_manifest_path = stage / ".base-platform.json"
        base_sig_path = stage / ".base-platform.json.minisig"
        base_manifest_path.write_bytes(base_manifest_result.body)
        base_sig_path.write_bytes(base_sig_result.body)
        is_fixture_dest = type(dest) is FixtureDestination and getattr(dest, "_build_token", None) is FIXTURE_BUILD_TOKEN
        platform_pin = dest.platform_pin if is_fixture_dest and dest.platform_pin is not None else require_production_platform_pin(repo_root)
        verify_minisign_signature(platform_pin, base_manifest_path, base_sig_path)
        base_obj = load_platform_manifest_bytes(base_manifest_result.body, canonical_refusal=RELEASE_COHERENCE)
        if (
            base_obj["version"] != base_version
            or base_obj["lane"] != "release"
            or base_obj["platform_key_id"] != platform_pin.key_id
        ):
            raise Refusal(RELEASE_COHERENCE, "base platform coordinate does not match signed manifest")

        component_versions = {name: base_obj["components"][name]["version"] for name in KNOWN_COMPONENTS}
        for name, replacement in replacements.items():
            if compare_semver(replacement, component_versions[name]) <= 0:
                raise Refusal(RELEASE_COHERENCE, f"{name} replacement must be newer than {component_versions[name]}")
            component_versions[name] = replacement

        downloader = _Downloader()
        base_prefix = f"{origin}/solstone/release/{base_version}"
        journal_prefix = f"{origin}/solstone-journal/release/{component_versions['journal']}" if "journal" in replacements else base_prefix
        desktop_prefix = f"{origin}/solstone-linux/release/{component_versions['desktop']}" if "desktop" in replacements else base_prefix
        tmux_prefix = f"{origin}/solstone-tmux/release/{component_versions['tmux']}" if "tmux" in replacements else base_prefix

        journal_dir = _download_journal(downloader, stage, component_versions["journal"], journal_prefix)
        desktop_dir = _download_desktop(downloader, stage, component_versions["desktop"], desktop_prefix)
        tmux_dir = _download_tmux(downloader, stage, component_versions["tmux"], tmux_prefix)

        bootstrap_name = f"solstone-journal-{component_versions['journal']}-install.sh"
        bootstrap_url = base_obj["components"]["journal"].get("provenance", {}).get("bootstrap", {}).get("url")
        if "journal" in replacements:
            bootstrap_url = f"{origin}/solstone-journal/release/{component_versions['journal']}/{bootstrap_name}"
        if not isinstance(bootstrap_url, str) or bootstrap_url != f"{origin}/solstone-journal/release/{component_versions['journal']}/{bootstrap_name}":
            raise Refusal(RELEASE_COHERENCE, "journal bootstrap provenance URL is not canonical")
        bootstrap_path = stage / "bootstrap" / bootstrap_name
        _write_remote(downloader, bootstrap_url, bootstrap_path)

        pins = dest.pinset if is_fixture_dest and dest.pinset is not None else embedded_pins()
        ingest_journal(journal_dir, "release", origin, bootstrap_path, pins.journal)
        ingest_desktop(desktop_dir, pins.desktop)
        ingest_tmux(tmux_dir, pins.tmux)

        if created_unix is not None:
            epoch = int(created_unix)
        elif existing_receipt is not None and isinstance(existing_receipt.get("created_unix"), int):
            epoch = int(existing_receipt["created_unix"])
        else:
            epoch = int(time.time())
        manifest = generate_platform_manifest(
            version=version,
            lane="release",
            created_unix=epoch,
            source_commit=source_commit,
            platform_key_id=platform_pin.key_id,
            repo_root=repo_root,
            journal_dir=journal_dir,
            desktop_dir=desktop_dir,
            tmux_dir=tmux_dir,
            journal_origin=origin,
            bootstrap_file=bootstrap_path,
            pins=pins,
        )
        manifest_path = stage / "platform.json"
        manifest_path.write_bytes(manifest)
        candidate = load_platform_manifest_bytes(manifest, canonical_refusal=RELEASE_COHERENCE)

        for field in ("schema_version", "protocol_version", "lane", "platform_key_id", "minimum_installer_revision"):
            if candidate[field] != base_obj[field]:
                raise Refusal(RELEASE_COHERENCE, f"recut changed protected top-level field {field}")
        for name in KNOWN_COMPONENTS:
            base_component = base_obj["components"][name]
            candidate_component = candidate["components"][name]
            if name not in replacements:
                if candidate_component != base_component:
                    raise Refusal(RELEASE_COHERENCE, f"recut changed unselected component {name}")
            else:
                for field in ("handler_contract_version", "install_entrypoint", "uninstall_service_entrypoint"):
                    if candidate_component.get(field) != base_component.get(field):
                        raise Refusal(RELEASE_COHERENCE, f"recut changed {name} handler field {field}")

        base_manifest_path.unlink()
        base_sig_path.unlink()
        prepared_files = _file_inventory(stage)
        source_records = []
        for item in downloader.sources:
            relative = str(Path(item["path"]).relative_to(stage))
            source_records.append({**item, "path": relative})
        receipt = {
            "schema_version": 1,
            "lane": "release",
            "base": {"version": base_version, "body": latest.body.decode("utf-8"), "etag": latest.etag},
            "candidate": {"version": version, "sha256": hashlib.sha256(manifest).hexdigest()},
            "replacements": dict(sorted(replacements.items())),
            "source_commit": source_commit,
            "created_unix": epoch,
            "sources": sorted(source_records, key=lambda item: (item["url"], item["path"])),
            "prepared_files": prepared_files,
        }
        receipt_bytes = canonical_json_bytes(receipt)
        (stage / "recut-receipt.json").write_bytes(receipt_bytes)

        for directory in sorted((p for p in stage.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True) + [stage]:
            fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

        try:
            _rename_noreplace(stage, output_dir)
        except FileExistsError:
            if _validate_existing(output_dir, stage):
                shutil.rmtree(stage)
            else:
                raise Refusal(RELEASE_COHERENCE, f"existing recut output differs: {output_dir}")
        return receipt
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise
