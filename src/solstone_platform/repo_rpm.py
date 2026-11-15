# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""RPM (rpm-md) repository builder and verifier for solstone Linux packages."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

from solstone_platform.archive import scan_rpm_file
from solstone_platform.gpg_sign import (
    sign_detached_asc,
    sign_rpm_file,
    verify_openpgp_signature,
    verify_signed_rpm,
)
from solstone_platform.pins import OpenPGPPin
from solstone_platform.refusals import GENERATION_FAILED, RELEASE_COHERENCE, Refusal
from solstone_platform.targets import get_target_mapping


def build_rpm_repository(
    *,
    repo_root_dir: Path,
    lane: str,
    created_unix: int,
    arch_rpm_packages: dict[str, list[tuple[str, Path]]],
    gnupghome: Path,
    openpgp_pin: OpenPGPPin,
    passphrase: str,
) -> tuple[list[str], list[str]]:
    """
    Build the RPM repository under repo_root_dir / 'linux' / 'rpm' / lane / <rpm_arch>.
    Signs each RPM with rpmsign --addsign, verifies payload byte-identity and rpmkeys --checksig,
    places signed RPMs at content-addressed paths Packages/<sha256[:16]>/<filename>.rpm,
    runs createrepo_c --unique-md-filenames, and signs + verifies repomd.xml.asc.
    Returns (immutable_rel_keys, pointer_rel_keys) relative to repo_root_dir.
    """
    if not arch_rpm_packages:
        raise Refusal(RELEASE_COHERENCE, "no architectures provided for RPM repository build")

    immutable_keys: list[str] = []
    pointer_keys: list[str] = []

    for arch in sorted(arch_rpm_packages.keys()):
        mapping = get_target_mapping(arch)
        rpm_arch = mapping.rpm_arch
        arch_dir = repo_root_dir / "linux" / "rpm" / lane / rpm_arch
        arch_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="solstone-rpmsign-") as tmp_str:
            tmp_dir = Path(tmp_str)
            for pkg_name, unsigned_rpm_path in sorted(arch_rpm_packages[arch], key=lambda item: item[0]):
                scan_rpm_file(unsigned_rpm_path)
                signed_tmp = tmp_dir / unsigned_rpm_path.name
                sign_rpm_file(
                    unsigned_rpm=unsigned_rpm_path,
                    signed_rpm=signed_tmp,
                    gnupghome=gnupghome,
                    key_fingerprint=openpgp_pin.fingerprint,
                    passphrase=passphrase,
                    created_unix=created_unix,
                )
                verify_signed_rpm(signed_tmp, openpgp_pin.asc_bytes)

                signed_bytes = signed_tmp.read_bytes()
                prefix16 = hashlib.sha256(signed_bytes).hexdigest()[:16]
                rel_pkg = f"Packages/{prefix16}/{unsigned_rpm_path.name}"
                dest_rpm = arch_dir / rel_pkg
                dest_rpm.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(signed_tmp, dest_rpm)
                immutable_keys.append(f"linux/rpm/{lane}/{rpm_arch}/{rel_pkg}")

        proc = subprocess.run(
            [
                "/usr/bin/createrepo_c",
                "--no-database",
                "--unique-md-filenames",
                "--checksum",
                "sha256",
                "--general-compress-type",
                "gz",
                "--revision",
                str(created_unix),
                "--set-timestamp-to-revision",
                str(arch_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise Refusal(GENERATION_FAILED, f"createrepo_c failed for {rpm_arch}: {err}")

        repodata_dir = arch_dir / "repodata"
        repomd_xml = repodata_dir / "repomd.xml"
        repomd_asc = repodata_dir / "repomd.xml.asc"
        if not repomd_xml.is_file():
            raise Refusal(GENERATION_FAILED, f"createrepo_c did not produce repomd.xml for {rpm_arch}")

        sign_detached_asc(
            data_path=repomd_xml,
            sig_asc_path=repomd_asc,
            gnupghome=gnupghome,
            key_fingerprint=openpgp_pin.fingerprint,
            passphrase=passphrase,
            created_unix=created_unix,
        )
        verify_openpgp_signature(
            gpg_bytes=openpgp_pin.gpg_bytes,
            expected_fingerprint=openpgp_pin.fingerprint,
            sig_or_inline_path=repomd_asc,
            signed_data_path=repomd_xml,
        )

        for md_file in sorted(repodata_dir.iterdir()):
            if not md_file.is_file():
                continue
            if md_file.name in ("repomd.xml", "repomd.xml.asc"):
                continue
            immutable_keys.append(f"linux/rpm/{lane}/{rpm_arch}/repodata/{md_file.name}")

        # Keep repomd.xml immediately followed by repomd.xml.asc per arch
        pointer_keys.append(f"linux/rpm/{lane}/{rpm_arch}/repodata/repomd.xml")
        pointer_keys.append(f"linux/rpm/{lane}/{rpm_arch}/repodata/repomd.xml.asc")

    return sorted(immutable_keys), pointer_keys
