# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""APT repository builder and verifier for solstone Linux packages."""

from __future__ import annotations

import email.utils
import gzip
import hashlib
import io
import lzma
from pathlib import Path
import shutil
import tarfile

from solstone_platform.archive import _parse_ar_archive, scan_deb_file
from solstone_platform.gpg_sign import sign_apt_release, verify_openpgp_signature
from solstone_platform.pins import OpenPGPPin
from solstone_platform.refusals import RELEASE_COHERENCE, SCHEMA_INVALID, Refusal
from solstone_platform.targets import get_target_mapping


def _extract_deb_control_text(deb_path: Path) -> str:
    raw_data = deb_path.read_bytes()
    members = _parse_ar_archive(raw_data)
    control_key = next((k for k in members if k.startswith("control.tar")), None)
    if not control_key:
        raise Refusal(SCHEMA_INVALID, f"deb package missing control.tar: {deb_path.name}")
    with tarfile.open(fileobj=io.BytesIO(members[control_key]), mode="r:*") as ctf:
        for cmember in ctf.getmembers():
            cname = cmember.name.replace("\\", "/").split("/")[-1]
            if cname == "control" and cmember.isreg():
                cf = ctf.extractfile(cmember)
                if cf:
                    return cf.read().decode("utf-8", errors="strict")
    raise Refusal(SCHEMA_INVALID, f"deb package missing control file: {deb_path.name}")


def _format_packages_stanza(deb_path: Path, rel_pool_path: str) -> str:
    scan = scan_deb_file(deb_path)
    if scan.package_identity is None:
        raise Refusal(SCHEMA_INVALID, f"deb missing package identity: {deb_path.name}")

    control_text = _extract_deb_control_text(deb_path).strip()
    lines = control_text.splitlines()
    present_keys = {
        line.split(":", 1)[0].strip().lower()
        for line in lines
        if ":" in line and not line.startswith((" ", "\t"))
    }

    # Ensure standard Debian archive fields are present even for older component .debs
    extra_headers: list[str] = []
    if "section" not in present_keys:
        extra_headers.append("Section: utils")
    if "priority" not in present_keys:
        extra_headers.append("Priority: optional")
    if "homepage" not in present_keys:
        extra_headers.append("Homepage: https://solstone.app")

    # Insert extra headers before Description: so multi-line Description stays intact
    out_lines: list[str] = []
    inserted = False
    for line in lines:
        if not inserted and line.startswith("Description:"):
            out_lines.extend(extra_headers)
            inserted = True
        out_lines.append(line)
    if not inserted:
        out_lines.extend(extra_headers)

    raw_bytes = deb_path.read_bytes()
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    sha512_hex = hashlib.sha512(raw_bytes).hexdigest()

    out_lines.extend([
        f"Filename: {rel_pool_path}",
        f"Size: {len(raw_bytes)}",
        f"SHA256: {sha256_hex}",
        f"SHA512: {sha512_hex}",
    ])
    return "\n".join(out_lines) + "\n"


def _deterministic_gzip(data: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=9, mtime=0) as gz:
        gz.write(data)
    return buf.getvalue()


def build_apt_repository(
    *,
    repo_root_dir: Path,
    lane: str,
    coordinate: str,
    created_unix: int,
    arch_deb_packages: dict[str, list[tuple[str, Path]]],
    gnupghome: Path,
    openpgp_pin: OpenPGPPin,
    passphrase: str,
) -> tuple[list[str], list[str]]:
    """
    Build the APT repository under repo_root_dir / 'linux' / 'apt'.
    arch_deb_packages maps platform arch ('x86_64' | 'aarch64') -> list of (pkg_name, deb_path).
    Returns (immutable_rel_keys, pointer_rel_keys) relative to repo_root_dir.
    """
    apt_root = repo_root_dir / "linux" / "apt"
    dists_lane = apt_root / "dists" / lane
    dists_lane.mkdir(parents=True, exist_ok=True)

    immutable_keys: list[str] = []
    pointer_keys: list[str] = []
    deb_arches: list[str] = []
    index_entries: list[tuple[str, bytes]] = []

    for arch in sorted(arch_deb_packages.keys()):
        mapping = get_target_mapping(arch)
        deb_arch = mapping.deb_arch
        deb_arches.append(deb_arch)

        stanzas: list[str] = []
        for pkg_name, src_deb_path in sorted(arch_deb_packages[arch], key=lambda item: item[0]):
            filename = src_deb_path.name
            rel_pool = f"pool/{lane}/main/s/{pkg_name}/{filename}"
            dest_deb = apt_root / rel_pool
            dest_deb.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_deb_path, dest_deb)

            rel_key = f"linux/apt/{rel_pool}"
            if rel_key not in immutable_keys:
                immutable_keys.append(rel_key)
            stanzas.append(_format_packages_stanza(dest_deb, rel_pool))

        packages_raw = ("\n".join(stanzas)).encode("utf-8")
        packages_gz = _deterministic_gzip(packages_raw)
        packages_xz = lzma.compress(packages_raw, format=lzma.FORMAT_XZ, preset=6)

        bin_dir = dists_lane / "main" / f"binary-{deb_arch}"
        by_hash_dir = bin_dir / "by-hash" / "SHA256"
        by_hash_dir.mkdir(parents=True, exist_ok=True)

        for idx_name, idx_bytes in (
            ("Packages", packages_raw),
            ("Packages.gz", packages_gz),
            ("Packages.xz", packages_xz),
        ):
            idx_path = bin_dir / idx_name
            idx_path.write_bytes(idx_bytes)
            pointer_keys.append(f"linux/apt/dists/{lane}/main/binary-{deb_arch}/{idx_name}")
            index_entries.append((f"main/binary-{deb_arch}/{idx_name}", idx_bytes))

            digest = hashlib.sha256(idx_bytes).hexdigest()
            hash_path = by_hash_dir / digest
            hash_path.write_bytes(idx_bytes)
            hash_key = f"linux/apt/dists/{lane}/main/binary-{deb_arch}/by-hash/SHA256/{digest}"
            if hash_key not in immutable_keys:
                immutable_keys.append(hash_key)

    if not deb_arches:
        raise Refusal(RELEASE_COHERENCE, "no architectures provided for APT repository build")

    date_str = email.utils.formatdate(created_unix, usegmt=True)
    release_lines = [
        "Origin: sol pbc",
        "Label: solstone",
        f"Suite: {lane}",
        f"Codename: {lane}",
        f"Version: {coordinate}",
        f"Date: {date_str}",
        f"Architectures: {' '.join(deb_arches)}",
        "Components: main",
        "Description: Official solstone APT repository",
        "Acquire-By-Hash: yes",
        "SHA256:",
    ]
    for rel_idx, data in index_entries:
        release_lines.append(f" {hashlib.sha256(data).hexdigest()} {len(data):16d} {rel_idx}")
    release_lines.append("SHA512:")
    for rel_idx, data in index_entries:
        release_lines.append(f" {hashlib.sha512(data).hexdigest()} {len(data):16d} {rel_idx}")

    release_bytes = ("\n".join(release_lines) + "\n").encode("utf-8")
    release_path = dists_lane / "Release"
    inrelease_path = dists_lane / "InRelease"
    release_gpg_path = dists_lane / "Release.gpg"
    release_path.write_bytes(release_bytes)

    sign_apt_release(
        release_path=release_path,
        inrelease_path=inrelease_path,
        release_gpg_path=release_gpg_path,
        gnupghome=gnupghome,
        key_fingerprint=openpgp_pin.fingerprint,
        passphrase=passphrase,
        created_unix=created_unix,
    )

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

    pointer_keys.extend([
        f"linux/apt/dists/{lane}/Release",
        f"linux/apt/dists/{lane}/Release.gpg",
        f"linux/apt/dists/{lane}/InRelease",
    ])
    return sorted(immutable_keys), pointer_keys
