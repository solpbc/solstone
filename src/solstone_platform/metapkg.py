# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Build and verify the solstone workstation meta-package (.deb and .rpm)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from solstone_platform.archive import ArchiveScanResult, scan_deb_file, scan_rpm_file
from solstone_platform.refusals import GENERATION_FAILED, RELEASE_COHERENCE, SCHEMA_INVALID, Refusal
from solstone_platform.targets import get_target_mapping


@dataclass(frozen=True)
class BuiltMetaPackage:
    arch: str
    deb_path: Path
    deb_sha256: str
    deb_bytes: int
    deb_scan: ArchiveScanResult
    rpm_path: Path
    rpm_sha256: str
    rpm_bytes: int
    rpm_scan: ArchiveScanResult


def _build_deb_metapackage(
    *,
    output_dir: Path,
    pkg_version: str,
    deb_arch: str,
    journal_deb_version: str,
    desktop_deb_version: str,
    license_bytes: bytes,
    created_unix: int,
) -> Path:
    deb_filename = f"solstone_{pkg_version}_{deb_arch}.deb"
    deb_out = output_dir / deb_filename

    copyright_bytes = (
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n"
        "Upstream-Name: solstone\n"
        "Source: https://solstone.app\n"
        "\n"
        "Files: *\n"
        "Copyright: 2026 sol pbc\n"
        "License: AGPL-3.0-only\n"
    ).encode("utf-8")

    installed_kib = math.ceil(len(copyright_bytes) / 1024) + math.ceil(len(license_bytes) / 1024)

    control_text = (
        "Package: solstone\n"
        f"Version: {pkg_version}\n"
        f"Architecture: {deb_arch}\n"
        "Section: metapackages\n"
        "Priority: optional\n"
        "Maintainer: sol pbc <support@solstone.app>\n"
        "Homepage: https://solstone.app\n"
        f"Installed-Size: {installed_kib}\n"
        f"Depends: solstone-journal (>= {journal_deb_version}), solstone-linux (>= {desktop_deb_version})\n"
        "Description: your journal and the solstone app for linux\n"
        " installs your journal (solstone-journal) and the solstone app for\n"
        " linux (solstone-linux).\n"
    )

    with tempfile.TemporaryDirectory(prefix="solstone-metadeb-") as tmp_str:
        root = Path(tmp_str) / "pkg"
        debian_dir = root / "DEBIAN"
        doc_dir = root / "usr" / "share" / "doc" / "solstone"
        lic_dir = root / "usr" / "share" / "licenses" / "solstone"
        debian_dir.mkdir(parents=True)
        doc_dir.mkdir(parents=True)
        lic_dir.mkdir(parents=True)

        (debian_dir / "control").write_text(control_text, encoding="utf-8")
        (doc_dir / "copyright").write_bytes(copyright_bytes)
        (lic_dir / "LICENSE").write_bytes(license_bytes)

        for p in [root, *(d for d in root.rglob("*") if d.is_dir())]:
            os.chmod(p, 0o755)
        for f in [debian_dir / "control", doc_dir / "copyright", lic_dir / "LICENSE"]:
            os.chmod(f, 0o644)

        env = os.environ.copy()
        env["SOURCE_DATE_EPOCH"] = str(created_unix)
        proc = subprocess.run(
            [
                "/usr/bin/dpkg-deb",
                "-Zxz",
                "--root-owner-group",
                "--build",
                str(root),
                str(deb_out),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not deb_out.is_file():
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise Refusal(GENERATION_FAILED, f"dpkg-deb failed building {deb_filename}: {err}")

    return deb_out


def _build_rpm_metapackage(
    *,
    output_dir: Path,
    upstream_version: str,
    release_str: str,
    rpm_arch: str,
    journal_rpm_version: str,
    desktop_rpm_version: str,
    license_bytes: bytes,
    created_unix: int,
) -> Path:
    rpm_filename = f"solstone-{upstream_version}-{release_str}.{rpm_arch}.rpm"
    rpm_out = output_dir / rpm_filename

    with tempfile.TemporaryDirectory(prefix="solstone-metarpm-") as tmp_str:
        topdir = Path(tmp_str) / "rpmbuild"
        for sub in ("BUILD", "RPMS", "SOURCES", "SPECS", "SRPMS"):
            (topdir / sub).mkdir(parents=True)

        (topdir / "SOURCES" / "LICENSE").write_bytes(license_bytes)
        spec_path = topdir / "SPECS" / "solstone.spec"
        spec_text = (
            "%global _build_id_links none\n"
            "%global debug_package %{nil}\n"
            "%global _source_date_epoch_from_changelog 0\n"
            "%global clamp_mtime_to_source_date_epoch 1\n"
            "%global use_source_date_epoch_as_buildtime 1\n"
            "Name:           solstone\n"
            f"Version:        {upstream_version}\n"
            f"Release:        {release_str}\n"
            "Summary:        your journal and the solstone app for linux\n"
            "License:        AGPL-3.0-only\n"
            "Vendor:         sol pbc\n"
            "URL:            https://solstone.app\n"
            f"Requires:       solstone-journal >= {journal_rpm_version}\n"
            f"Requires:       solstone-linux >= {desktop_rpm_version}\n"
            "\n"
            "%description\n"
            "installs your journal (solstone-journal) and the solstone app for\n"
            "linux (solstone-linux).\n"
            "\n"
            "%prep\n"
            "\n"
            "%build\n"
            "\n"
            "%install\n"
            "install -d -m 0755 %{buildroot}/usr/share/licenses/solstone\n"
            "install -m 0644 %{_sourcedir}/LICENSE %{buildroot}/usr/share/licenses/solstone/LICENSE\n"
            "\n"
            "%files\n"
            "%dir /usr/share/licenses/solstone\n"
            "/usr/share/licenses/solstone/LICENSE\n"
        )
        spec_path.write_text(spec_text, encoding="utf-8")

        env = os.environ.copy()
        env["SOURCE_DATE_EPOCH"] = str(created_unix)
        proc = subprocess.run(
            [
                "/usr/bin/rpmbuild",
                "-bb",
                "--target",
                f"{rpm_arch}-linux",
                "--define",
                f"_topdir {topdir}",
                "--define",
                "_binary_payload w9.gzdio",
                "--define",
                "_buildhost localhost",
                str(spec_path),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        built_candidates = list((topdir / "RPMS").rglob("*.rpm"))
        if proc.returncode != 0 or len(built_candidates) != 1:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise Refusal(GENERATION_FAILED, f"rpmbuild failed building {rpm_filename}: {err}")

        shutil.copy2(built_candidates[0], rpm_out)

    return rpm_out


def build_metapackage_for_arch(
    *,
    output_dir: Path,
    arch: str,
    journal_version: str,
    catalogue_revision: int,
    journal_deb_version: str,
    desktop_deb_version: str,
    journal_rpm_version: str,
    desktop_rpm_version: str,
    repo_root: Path,
    created_unix: int,
) -> BuiltMetaPackage:
    """Build and verify both .deb and .rpm solstone meta-packages for one architecture."""
    if catalogue_revision < 1:
        raise Refusal(SCHEMA_INVALID, f"invalid catalogue_revision: {catalogue_revision}")

    mapping = get_target_mapping(arch)
    license_path = repo_root / "LICENSE"
    if not license_path.is_file():
        raise Refusal(SCHEMA_INVALID, f"repository LICENSE file missing at {license_path}")
    license_bytes = license_path.read_bytes()

    output_dir.mkdir(parents=True, exist_ok=True)
    pkg_version = f"{journal_version}-{catalogue_revision}"
    release_str = str(catalogue_revision)

    deb_path = _build_deb_metapackage(
        output_dir=output_dir,
        pkg_version=pkg_version,
        deb_arch=mapping.deb_arch,
        journal_deb_version=journal_deb_version,
        desktop_deb_version=desktop_deb_version,
        license_bytes=license_bytes,
        created_unix=created_unix,
    )
    rpm_path = _build_rpm_metapackage(
        output_dir=output_dir,
        upstream_version=journal_version,
        release_str=release_str,
        rpm_arch=mapping.rpm_arch,
        journal_rpm_version=journal_rpm_version,
        desktop_rpm_version=desktop_rpm_version,
        license_bytes=license_bytes,
        created_unix=created_unix,
    )

    deb_scan = scan_deb_file(deb_path)
    if (
        deb_scan.package_identity is None
        or deb_scan.package_identity.name != "solstone"
        or deb_scan.package_identity.version != pkg_version
        or deb_scan.package_identity.arch != mapping.deb_arch
    ):
        raise Refusal(RELEASE_COHERENCE, f"built deb meta-package identity mismatch: {deb_scan.package_identity}")

    rpm_scan = scan_rpm_file(rpm_path)
    if (
        rpm_scan.package_identity is None
        or rpm_scan.package_identity.name != "solstone"
        or rpm_scan.package_identity.version != pkg_version
        or rpm_scan.package_identity.arch != mapping.rpm_arch
    ):
        raise Refusal(RELEASE_COHERENCE, f"built rpm meta-package identity mismatch: {rpm_scan.package_identity}")

    deb_raw = deb_path.read_bytes()
    rpm_raw = rpm_path.read_bytes()
    return BuiltMetaPackage(
        arch=arch,
        deb_path=deb_path,
        deb_sha256=hashlib.sha256(deb_raw).hexdigest(),
        deb_bytes=len(deb_raw),
        deb_scan=deb_scan,
        rpm_path=rpm_path,
        rpm_sha256=hashlib.sha256(rpm_raw).hexdigest(),
        rpm_bytes=len(rpm_raw),
        rpm_scan=rpm_scan,
    )
