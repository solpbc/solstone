# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unit and integration tests for APT and RPM package repository building, signing, and publishing."""

import gzip
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.request
import xml.etree.ElementTree as ET

from solstone_platform.archive import scan_deb_file, scan_rpm_file
from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.cli import main as cli_main
from solstone_platform.generate import generate_platform_manifest
from solstone_platform.gpg_sign import (
    ephemeral_openpgp_keypair,
    isolated_gnupghome,
    rpm_main_header_offset,
    sign_apt_release,
    sign_rpm_file,
    verify_openpgp_signature,
    verify_signed_rpm,
)
from solstone_platform.metapkg import build_metapackage_for_arch
from solstone_platform.pins import OpenPGPPin, PinSet
from solstone_platform.publish import publish_release
from solstone_platform.r2 import R2Config, R2Destination, classify_destination_key
from solstone_platform.refusals import (
    LANE_INVALID,
    PIN_MISMATCH,
    PRODUCTION_UNAVAILABLE,
    RELEASE_COHERENCE,
    ROLLBACK_REFUSED,
    SAME_VERSION_DIFFERENT_BYTES,
    SIGNATURE_PIN_MISMATCH,
    UNSAFE_FILENAME,
    Refusal,
)
from solstone_platform.repo_publish import build_repos, publish_repos
from solstone_platform.sign import ephemeral_keypair, sign_manifest
from solstone_platform.testdest import FixtureDestination
from tools.fixture_builder import create_tiny_deb, create_tiny_tar


def build_real_tiny_rpm(
    dest: Path,
    pkg_name: str,
    version: str,
    release: str,
    rpm_arch: str,
    exe_name: str,
    exe_bytes: bytes,
    created_unix: int = 1700000000,
) -> bytes:
    """Build a minimal real RPM package via rpmbuild so rpmsign and createrepo_c operate on authentic headers."""
    with tempfile.TemporaryDirectory(prefix="solstone-test-rpmbuild-") as tmp_str:
        topdir = Path(tmp_str) / "rpmbuild"
        for sub in ("BUILD", "RPMS", "SOURCES", "SPECS", "SRPMS"):
            (topdir / sub).mkdir(parents=True)
        (topdir / "SOURCES" / exe_name).write_bytes(exe_bytes)
        spec_path = topdir / "SPECS" / f"{pkg_name}.spec"
        spec_text = (
            "%global _build_id_links none\n"
            "%global debug_package %{nil}\n"
            "%global _source_date_epoch_from_changelog 0\n"
            "%global clamp_mtime_to_source_date_epoch 1\n"
            "%global use_source_date_epoch_as_buildtime 1\n"
            f"Name:           {pkg_name}\n"
            f"Version:        {version}\n"
            f"Release:        {release}\n"
            f"Summary:        Test package for {pkg_name}\n"
            "License:        AGPL-3.0-only\n"
            "Vendor:         sol pbc\n"
            "URL:            https://solstone.app\n"
            "AutoReqProv:    no\n"
            "\n"
            "%description\n"
            f"Synthetic test package for {pkg_name}.\n"
            "\n"
            "%prep\n"
            "\n"
            "%build\n"
            "\n"
            "%install\n"
            "install -d -m 0755 %{buildroot}/usr/bin\n"
            f"install -m 0755 %{{_sourcedir}}/{exe_name} %{{buildroot}}/usr/bin/{exe_name}\n"
            "\n"
            "%files\n"
            f"/usr/bin/{exe_name}\n"
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
            check=True,
        )
        built = list((topdir / "RPMS").rglob("*.rpm"))
        assert len(built) == 1, proc.stderr.decode("utf-8", errors="replace")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built[0], dest)
    return dest.read_bytes()


def build_repo_test_natives(
    target_dir: Path,
    journal_version: str = "2.0.76",
    desktop_version: str = "2.0.2",
) -> tuple[PinSet, dict[str, Path]]:
    """Build synthetic journal and desktop release directories with authentic rpmbuild RPMs for repo tests."""
    dirs = {
        "journal": target_dir / "journal",
        "desktop": target_dir / "desktop",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    with ephemeral_keypair("repo test journal") as (j_sec, _, j_pin), \
         ephemeral_keypair("repo test desktop") as (d_sec, _, d_pin), \
         ephemeral_keypair("repo test tmux") as (_, _, t_pin):
        pinset = PinSet(journal=j_pin, desktop=d_pin, tmux=t_pin)

        desktop_exe = f"#!/bin/sh\necho {desktop_version}\n".encode("utf-8")
        for arch, deb_arch, rpm_arch in [("x86_64", "amd64", "x86_64"), ("aarch64", "arm64", "aarch64")]:
            d_tar = create_tiny_tar(
                dirs["desktop"] / f"solstone-linux-{desktop_version}-linux-{arch}.tar.gz",
                {f"solstone-linux-{desktop_version}-linux-{arch}/bin/solstone-linux": desktop_exe},
            )
            d_deb = create_tiny_deb(
                dirs["desktop"] / f"solstone-linux_{desktop_version}-1_{deb_arch}.deb",
                "solstone-linux",
                f"{desktop_version}-1",
                deb_arch,
                "solstone-linux",
                desktop_exe,
            )
            d_rpm = build_real_tiny_rpm(
                dirs["desktop"] / f"solstone-linux-{desktop_version}-1.{rpm_arch}.rpm",
                "solstone-linux",
                desktop_version,
                "1",
                rpm_arch,
                "solstone-linux",
                desktop_exe,
            )
            d_manifest = {
                "schema_version": 1,
                "product": "solstone-linux",
                "version": desktop_version,
                "source_commit": "3a6bded610425b52a8ac875ae364d23f6e70ce7f",
                "source_dirty": False,
                "cargo_lock_sha256": "0" * 64,
                "rust": {"rustc_verbose": "1", "cargo_version": "1"},
                "target": {"kind": "compiled", "triple": f"{arch}-unknown-linux-gnu", "profile": "release", "features": []},
                "native_tools": {},
                "dependency_policy": {},
                "active_exceptions": [],
                "artifacts": [
                    {"path": f"solstone-linux-{desktop_version}-linux-{arch}.tar.gz", "sha256": hashlib.sha256(d_tar).hexdigest(), "bytes": len(d_tar)},
                    {"path": f"solstone-linux_{desktop_version}-1_{deb_arch}.deb", "sha256": hashlib.sha256(d_deb).hexdigest(), "bytes": len(d_deb)},
                    {"path": f"solstone-linux-{desktop_version}-1.{rpm_arch}.rpm", "sha256": hashlib.sha256(d_rpm).hexdigest(), "bytes": len(d_rpm)},
                ],
            }
            d_m_path = dirs["desktop"] / f"solstone-linux-{desktop_version}-linux-{arch}.rust-release-manifest.json"
            d_m_path.write_bytes(canonical_json_bytes(d_manifest))
            subprocess.run(
                ["minisign", "-S", "-W", "-s", str(d_sec), "-m", str(d_m_path), "-x", str(d_m_path.with_suffix(".json.minisig")), "-t", "solstone-linux release manifest"],
                check=True,
            )

        boot_script = (
            b"#!/bin/sh\n"
            b"BOOTSTRAP_REVISION=2\n"
            b"BOOTSTRAP_CONTRACT_VERSION=2\n"
            b"echo install\n"
        )
        boot_sha = hashlib.sha256(boot_script).hexdigest()
        journal_exe = f"#!/bin/sh\necho {journal_version}\n".encode("utf-8")
        bootstrap_name = f"solstone-journal-{journal_version}-install.sh"

        for arch, target, deb_arch, rpm_arch in [
            ("x86_64", "linux-x86_64", "amd64", "x86_64"),
            ("aarch64", "linux-aarch64", "arm64", "aarch64"),
        ]:
            arch_dir = dirs["journal"] / target
            arch_dir.mkdir(parents=True, exist_ok=True)
            (arch_dir / bootstrap_name).write_bytes(boot_script)
            j_tar = create_tiny_tar(
                arch_dir / f"solstone-journal-{journal_version}-{target}.tar.gz",
                {"usr/bin/solstone": journal_exe},
            )
            j_deb = create_tiny_deb(
                arch_dir / f"solstone-journal-{journal_version}-{target}.deb",
                "solstone-journal",
                journal_version,
                deb_arch,
                "solstone",
                journal_exe,
            )
            j_rpm = build_real_tiny_rpm(
                arch_dir / f"solstone-journal-{journal_version}-{target}.rpm",
                "solstone-journal",
                journal_version,
                "1",
                rpm_arch,
                "solstone",
                journal_exe,
            )
            tar_sha = hashlib.sha256(j_tar).hexdigest()
            deb_sha = hashlib.sha256(j_deb).hexdigest()
            rpm_sha = hashlib.sha256(j_rpm).hexdigest()

            rel_text = (
                f"product=solstone-journal\nversion={journal_version}\ntarget={target}\n"
                f"commit=3075c36b12fad469d4c9c0ab4555908fe8ecca1b\n"
                f"lock_sha256={'0' * 64}\n"
                f"upgrade_epoch=journal-v2\nretention_window=3\nmin_bootstrap_revision=2\n"
                f"bootstrap_contract_version=2\nbootstrap_filename={bootstrap_name}\n"
                f"state_reader_min=2.0.0\nstate_reader_max={journal_version}\n"
            )
            (arch_dir / f"solstone-journal-{journal_version}-{target}.release").write_text(rel_text, encoding="utf-8")
            sha256_lines = [
                f"{tar_sha}  solstone-journal-{journal_version}-{target}.tar.gz",
                f"{deb_sha}  solstone-journal-{journal_version}-{target}.deb",
                f"{rpm_sha}  solstone-journal-{journal_version}-{target}.rpm",
                f"{hashlib.sha256(rel_text.encode('utf-8')).hexdigest()}  solstone-journal-{journal_version}-{target}.release",
                f"{boot_sha}  {bootstrap_name}",
            ]
            sha256_bytes = ("\n".join(sha256_lines) + "\n").encode("utf-8")
            (arch_dir / f"solstone-journal-{journal_version}-{target}.sha256").write_bytes(sha256_bytes)

            j_manifest = {
                "product": "solstone-journal",
                "version": journal_version,
                "target": target,
                "files": {
                    f"solstone-journal-{journal_version}-{target}.release": hashlib.sha256(rel_text.encode("utf-8")).hexdigest(),
                    bootstrap_name: boot_sha,
                    f"solstone-journal-{journal_version}-{target}.tar.gz": tar_sha,
                    f"solstone-journal-{journal_version}-{target}.deb": deb_sha,
                    f"solstone-journal-{journal_version}-{target}.rpm": rpm_sha,
                    f"solstone-journal-{journal_version}-{target}.sha256": hashlib.sha256(sha256_bytes).hexdigest(),
                },
            }
            m_path = arch_dir / f"solstone-journal-{journal_version}-{target}.manifest.json"
            m_path.write_bytes(canonical_json_bytes(j_manifest))
            subprocess.run(
                ["minisign", "-S", "-W", "-s", str(j_sec), "-m", str(m_path), "-x", str(m_path.with_suffix(".json.minisig")), "-t", "solstone-journal release manifest"],
                check=True,
            )

        return pinset, dirs


class TestRepos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).parent.parent
        cls.p_kp_ctx = ephemeral_keypair("platform repo test key")
        cls.platform_sec, cls.platform_pub, cls.platform_pin = cls.p_kp_ctx.__enter__()

        cls.pgp_ctx = ephemeral_openpgp_keypair("test-openpgp-pass")
        cls.gpg_sec, cls.gpg_pass_file, raw_pgp_pin = cls.pgp_ctx.__enter__()
        cls.gpg_pass = "test-openpgp-pass"

        # Sign the ephemeral OpenPGP ASCII-armored public key with the ephemeral platform Minisign key
        cls.shared_tmp = tempfile.TemporaryDirectory(prefix="solstone-test-repos-cls-")
        shared_dir = Path(cls.shared_tmp.name)
        asc_file = shared_dir / "repo.asc"
        minisig_file = shared_dir / "repo.asc.minisig"
        asc_file.write_bytes(raw_pgp_pin.asc_bytes)
        subprocess.run(
            [
                "minisign",
                "-S",
                "-W",
                "-s",
                str(cls.platform_sec),
                "-m",
                str(asc_file),
                "-x",
                str(minisig_file),
                "-t",
                f"solstone linux package signing key {raw_pgp_pin.fingerprint}",
            ],
            check=True,
        )
        cls.openpgp_pin = OpenPGPPin(
            fingerprint=raw_pgp_pin.fingerprint,
            key_id=raw_pgp_pin.key_id,
            asc_bytes=raw_pgp_pin.asc_bytes,
            asc_minisig_bytes=minisig_file.read_bytes(),
            gpg_bytes=raw_pgp_pin.gpg_bytes,
        )

    @classmethod
    def tearDownClass(cls):
        cls.shared_tmp.cleanup()
        cls.pgp_ctx.__exit__(None, None, None)
        cls.p_kp_ctx.__exit__(None, None, None)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="solstone-test-repos-")
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_metapackage_builds_valid_deb_and_rpm_for_both_arches(self):
        for arch, expected_deb_arch, expected_rpm_arch in [
            ("x86_64", "amd64", "x86_64"),
            ("aarch64", "arm64", "aarch64"),
        ]:
            built = build_metapackage_for_arch(
                output_dir=self.root / f"meta-{arch}",
                arch=arch,
                journal_version="2.0.76",
                catalogue_revision=2,
                journal_deb_version="2.0.76",
                desktop_deb_version="2.0.2-1",
                journal_rpm_version="2.0.76-1",
                desktop_rpm_version="2.0.2-1",
                repo_root=self.repo_root,
                created_unix=1700000000,
            )
            self.assertEqual(built.deb_path.name, f"solstone_2.0.76-2_{expected_deb_arch}.deb")
            self.assertEqual(built.rpm_path.name, f"solstone-2.0.76-2.{expected_rpm_arch}.rpm")
            self.assertEqual(built.deb_scan.package_identity.name, "solstone")
            self.assertEqual(built.deb_scan.package_identity.version, "2.0.76-2")
            self.assertEqual(built.deb_scan.package_identity.arch, expected_deb_arch)
            self.assertEqual(built.rpm_scan.package_identity.name, "solstone")
            self.assertEqual(built.rpm_scan.package_identity.version, "2.0.76-2")
            self.assertEqual(built.rpm_scan.package_identity.arch, expected_rpm_arch)

    def test_rpm_signing_preserves_general_header_and_payload_bytes(self):
        unsigned_rpm = self.root / "unsigned.rpm"
        signed_rpm = self.root / "signed.rpm"
        build_real_tiny_rpm(
            unsigned_rpm,
            "solstone-journal",
            "2.0.76",
            "1",
            "x86_64",
            "solstone",
            b"#!/bin/sh\necho hi\n",
        )
        unsigned_bytes = unsigned_rpm.read_bytes()
        unsigned_offset = rpm_main_header_offset(unsigned_bytes)

        with isolated_gnupghome(
            secret_key_path=self.gpg_sec,
            passphrase=self.gpg_pass,
            selected_pin=self.openpgp_pin,
        ) as gnupghome:
            sign_rpm_file(
                unsigned_rpm=unsigned_rpm,
                signed_rpm=signed_rpm,
                gnupghome=gnupghome,
                key_fingerprint=self.openpgp_pin.fingerprint,
                passphrase=self.gpg_pass,
                created_unix=1700000000,
            )

        signed_bytes = signed_rpm.read_bytes()
        signed_offset = rpm_main_header_offset(signed_bytes)
        self.assertNotEqual(signed_bytes[:signed_offset], unsigned_bytes[:unsigned_offset])
        self.assertEqual(signed_bytes[signed_offset:], unsigned_bytes[unsigned_offset:])

        verify_signed_rpm(signed_rpm, self.openpgp_pin.asc_bytes)
        with self.assertRaises(Refusal) as ctx:
            verify_signed_rpm(unsigned_rpm, self.openpgp_pin.asc_bytes)
        self.assertEqual(ctx.exception.name, SIGNATURE_PIN_MISMATCH)

    def test_build_and_publish_repos_end_to_end_and_idempotency(self):
        natives_dir = self.root / "natives"
        pinset, dirs = build_repo_test_natives(natives_dir, journal_version="2.0.76", desktop_version="2.0.40")

        manifest_bytes = generate_platform_manifest(
            version=None,
            lane="release",
            created_unix=1700000000,
            source_commit="3075c36b12fad469d4c9c0ab4555908fe8ecca1b",
            platform_key_id=self.platform_pin.key_id,
            repo_root=self.repo_root,
            journal_dir=dirs["journal"],
            desktop_dir=dirs["desktop"],
            journal_origin="https://updates.solstone.app",
            pins=pinset,
            catalogue_revision=1,
        )
        manifest_path = self.root / "platform.json"
        sig_path = self.root / "platform.json.minisig"
        manifest_path.write_bytes(manifest_bytes)
        sig_bytes = sign_manifest(
            manifest_bytes=manifest_bytes,
            secret_key_path=self.platform_sec,
            selected_pin=self.platform_pin,
            passphrase_callback=lambda: "",
            is_production=False,
            acknowledge_production=False,
            repo_root=self.repo_root,
        )
        sig_path.write_bytes(sig_bytes)

        dest = FixtureDestination(pinset=pinset, platform_pin=self.platform_pin)

        # Publishing repos before platform release is published must refuse
        repo_out = self.root / "staged-repos"
        repo_manifest = build_repos(
            manifest_path=manifest_path,
            signature_path=sig_path,
            journal_dir=dirs["journal"],
            desktop_dir=dirs["desktop"],
            output_dir=repo_out,
            gpg_secret_key_path=self.gpg_sec,
            passphrase_callback=lambda: self.gpg_pass,
            repo_root=self.repo_root,
            selected_platform_pin=self.platform_pin,
            selected_openpgp_pin=self.openpgp_pin,
        )
        self.assertEqual(repo_manifest["version"], "2.0.76-r1")
        self.assertEqual(repo_manifest["lane"], "release")

        with self.assertRaises(Refusal) as pre_ctx:
            publish_repos(
                repo_dir=repo_out,
                dest=dest,
                repo_root=self.repo_root,
                selected_platform_pin=self.platform_pin,
                selected_openpgp_pin=self.openpgp_pin,
            )
        self.assertEqual(pre_ctx.exception.name, RELEASE_COHERENCE)

        # Now publish the platform release first
        pub_report = publish_release(
            manifest_path=manifest_path,
            signature_path=sig_path,
            journal_dir=dirs["journal"],
            desktop_dir=dirs["desktop"],
            dest=dest,
            expect_latest_absent=True,
        )
        self.assertTrue(pub_report.latest_promoted)

        # Now publish the staged repositories
        repo_report = publish_repos(
            repo_dir=repo_out,
            dest=dest,
            repo_root=self.repo_root,
            selected_platform_pin=self.platform_pin,
            selected_openpgp_pin=self.openpgp_pin,
        )
        self.assertEqual(repo_report.version, "2.0.76-r1")
        self.assertEqual(repo_report.lane, "release")
        self.assertEqual(repo_report.pointers_promoted, repo_report.pointer_count)
        self.assertGreater(repo_report.immutable_count, 0)

        # Re-running publish_repos on identical release is idempotent (0 pointers promoted)
        second_report = publish_repos(
            repo_dir=repo_out,
            dest=dest,
            repo_root=self.repo_root,
            selected_platform_pin=self.platform_pin,
            selected_openpgp_pin=self.openpgp_pin,
        )
        self.assertEqual(second_report.pointers_promoted, 0)

        # Tampering with a staged file must be refused
        tampered_dir = self.root / "tampered-repos"
        shutil.copytree(repo_out, tampered_dir)
        inrelease_file = tampered_dir / "linux/apt/dists/release/InRelease"
        inrelease_file.write_bytes(inrelease_file.read_bytes() + b"\n")
        with self.assertRaises(Refusal) as tamp_ctx:
            publish_repos(
                repo_dir=tampered_dir,
                dest=dest,
                repo_root=self.repo_root,
                selected_platform_pin=self.platform_pin,
                selected_openpgp_pin=self.openpgp_pin,
            )
        self.assertEqual(tamp_ctx.exception.name, SAME_VERSION_DIFFERENT_BYTES)

        # Adding an uninventoried file under linux/ must be refused
        extra_dir = self.root / "extra-repos"
        shutil.copytree(repo_out, extra_dir)
        (extra_dir / "linux/apt/dists/release/extra.txt").write_text("rogue", encoding="utf-8")
        with self.assertRaises(Refusal) as extra_ctx:
            publish_repos(
                repo_dir=extra_dir,
                dest=dest,
                repo_root=self.repo_root,
                selected_platform_pin=self.platform_pin,
                selected_openpgp_pin=self.openpgp_pin,
            )
        self.assertEqual(extra_ctx.exception.name, RELEASE_COHERENCE)

        # Rollback refusal when remote InRelease has a newer Version
        dest.objects["linux/apt/dists/release/InRelease"].body = (
            dest.objects["linux/apt/dists/release/InRelease"]
            .body.replace(b"Version: 2.0.76-r1\n", b"Version: 2.0.99-r1\n")
        )
        with self.assertRaises(Refusal) as rb_ctx:
            publish_repos(
                repo_dir=repo_out,
                dest=dest,
                repo_root=self.repo_root,
                selected_platform_pin=self.platform_pin,
                selected_openpgp_pin=self.openpgp_pin,
            )
        self.assertEqual(rb_ctx.exception.name, ROLLBACK_REFUSED)

        # Live HTTP verification with real apt-get and rpm clients + tamper rejection
        self._verify_live_http_apt_and_rpm_clients(repo_out)

    def _verify_live_http_apt_and_rpm_clients(self, repo_out: Path) -> None:
        class QuietHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(repo_out), **kwargs)

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # 1. Real apt-get update + apt-get --download-only install solstone
            apt_work = self.root / "apt-client"
            for rel in (
                "etc/apt/sources.list.d",
                "etc/apt/preferences.d",
                "etc/apt/apt.conf.d",
                "etc/apt/keyrings",
                "var/lib/apt/lists/partial",
                "var/cache/apt/archives/partial",
                "var/lib/dpkg/updates",
                "var/lib/dpkg/info",
                "var/log",
            ):
                (apt_work / rel).mkdir(parents=True, exist_ok=True)
            (apt_work / "var/lib/dpkg/status").write_text("", encoding="utf-8")
            keyring_path = apt_work / "etc/apt/keyrings/solstone-archive-keyring.gpg"
            shutil.copy2(repo_out / "linux/keys/solstone-archive-keyring.gpg", keyring_path)
            (apt_work / "etc/apt/sources.list.d/solstone.sources").write_text(
                "Types: deb\n"
                f"URIs: http://127.0.0.1:{port}/linux/apt\n"
                "Suites: release\n"
                "Components: main\n"
                "Architectures: amd64\n"
                f"Signed-By: {keyring_path}\n",
                encoding="utf-8",
            )
            import pwd
            current_user = pwd.getpwuid(os.getuid()).pw_name
            apt_opts = [
                "-o", f"Dir={apt_work}",
                "-o", f"Dir::State::status={apt_work / 'var/lib/dpkg/status'}",
                "-o", "APT::Architecture=amd64",
                "-o", "APT::Get::Assume-Yes=true",
                "-o", f"APT::Sandbox::User={current_user}",
                "-o", "Debug::NoLocking=1",
            ]
            upd = subprocess.run(["/usr/bin/apt-get", *apt_opts, "update"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(upd.returncode, 0, upd.stderr.decode("utf-8", errors="replace"))

            inst = subprocess.run(
                ["/usr/bin/apt-get", *apt_opts, "--download-only", "install", "solstone"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(inst.returncode, 0, inst.stderr.decode("utf-8", errors="replace"))
            downloaded_debs = {p.name for p in (apt_work / "var/cache/apt/archives").glob("*.deb")}
            self.assertEqual(
                downloaded_debs,
                {
                    "solstone_2.0.76-1_amd64.deb",
                    "solstone-journal_2.0.76_amd64.deb",
                    "solstone-linux_2.0.40-1_amd64.deb",
                },
            )

            # Negative APT check: wrong Signed-By key must cause apt-get update to fail
            for cached in (apt_work / "var/lib/apt/lists").glob("*"):
                if cached.is_file():
                    cached.unlink()
            with ephemeral_openpgp_keypair("wrong-apt-key") as (_, _, wrong_pin):
                wrong_keyring = apt_work / "etc/apt/keyrings/wrong.gpg"
                wrong_keyring.write_bytes(wrong_pin.gpg_bytes)
                (apt_work / "etc/apt/sources.list.d/solstone.sources").write_text(
                    "Types: deb\n"
                    f"URIs: http://127.0.0.1:{port}/linux/apt\n"
                    "Suites: release\n"
                    "Components: main\n"
                    "Architectures: amd64\n"
                    f"Signed-By: {wrong_keyring}\n",
                    encoding="utf-8",
                )
                bad_upd = subprocess.run(
                    ["/usr/bin/apt-get", *apt_opts, "update"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.assertNotEqual(bad_upd.returncode, 0)

            # 2. Real RPM repository client fetch, repomd.xml.asc gpgv check, rpmkeys --checksig, and librpm dependency solver
            rpm_work = self.root / "rpm-client"
            rpmdb = rpm_work / "rpmdb"
            dl_dir = rpm_work / "dl"
            rpmdb.mkdir(parents=True)
            dl_dir.mkdir(parents=True)

            base_rpm_url = f"http://127.0.0.1:{port}/linux/rpm/release/x86_64/"
            repomd_xml = dl_dir / "repomd.xml"
            repomd_asc = dl_dir / "repomd.xml.asc"
            repomd_xml.write_bytes(urllib.request.urlopen(base_rpm_url + "repodata/repomd.xml").read())
            repomd_asc.write_bytes(urllib.request.urlopen(base_rpm_url + "repodata/repomd.xml.asc").read())
            verify_openpgp_signature(
                gpg_bytes=self.openpgp_pin.gpg_bytes,
                expected_fingerprint=self.openpgp_pin.fingerprint,
                sig_or_inline_path=repomd_asc,
                signed_data_path=repomd_xml,
            )

            ns = {"repo": "http://linux.duke.edu/metadata/repo", "common": "http://linux.duke.edu/metadata/common"}
            tree = ET.parse(repomd_xml)
            primary_href = None
            primary_sha = None
            for data_el in tree.getroot().findall("repo:data", ns):
                if data_el.attrib.get("type") == "primary":
                    primary_href = data_el.find("repo:location", ns).attrib["href"]
                    primary_sha = data_el.find("repo:checksum", ns).text.strip()
            self.assertIsNotNone(primary_href)
            raw_primary = urllib.request.urlopen(base_rpm_url + primary_href).read()
            self.assertEqual(hashlib.sha256(raw_primary).hexdigest(), primary_sha)

            ptree = ET.fromstring(gzip.decompress(raw_primary))
            rpm_files: dict[str, Path] = {}
            for pkg_el in ptree.findall("common:package", ns):
                name = pkg_el.find("common:name", ns).text
                href = pkg_el.find("common:location", ns).attrib["href"]
                sha = pkg_el.find("common:checksum", ns).text.strip()
                rpm_bytes = urllib.request.urlopen(base_rpm_url + href).read()
                self.assertEqual(hashlib.sha256(rpm_bytes).hexdigest(), sha)
                out_rpm = dl_dir / Path(href).name
                out_rpm.write_bytes(rpm_bytes)
                rpm_files[name] = out_rpm

            self.assertEqual(set(rpm_files.keys()), {"solstone", "solstone-journal", "solstone-linux"})

            key_asc_file = dl_dir / "key.asc"
            key_asc_file.write_bytes(
                urllib.request.urlopen(f"http://127.0.0.1:{port}/linux/keys/solstone-archive-keyring.asc").read()
            )
            subprocess.run(["/usr/bin/rpmkeys", "--dbpath", str(rpmdb), "--import", str(key_asc_file)], check=True)
            for rpm_path in rpm_files.values():
                chk = subprocess.run(
                    ["/usr/bin/rpmkeys", "--dbpath", str(rpmdb), "--checksig", str(rpm_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                self.assertIn("digests signatures OK", chk.stdout.decode("utf-8"))

            # Installing solstone meta-package alone without its dependencies must fail librpm dependency resolution
            alone = subprocess.run(
                ["/usr/bin/rpm", "--dbpath", str(rpmdb), "--test", "-Uvh", str(rpm_files["solstone"])],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(alone.returncode, 0)

            # Installing all 3 packages together satisfies librpm dependency resolution
            together = subprocess.run(
                ["/usr/bin/rpm", "--dbpath", str(rpmdb), "--test", "-Uvh", *(str(p) for p in rpm_files.values())],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(together.returncode, 0, together.stderr.decode("utf-8", errors="replace"))
        finally:
            server.shutdown()
            server.server_close()

    def test_destination_key_confinement_for_apt_rpm_and_keys(self):
        # Valid immutable keys
        valid_immutable = [
            "linux/apt/pool/release/main/s/solstone/solstone_2.0.76-1_amd64.deb",
            "linux/apt/pool/release/main/s/solstone-journal/solstone-journal-2.0.76-linux-x86_64.deb",
            "linux/apt/pool/release/main/s/solstone-linux/solstone-linux_2.0.2-1_arm64.deb",
            f"linux/apt/dists/release/main/binary-amd64/by-hash/SHA256/{'a' * 64}",
            f"linux/rpm/release/x86_64/Packages/{'b' * 16}/solstone-2.0.76-1.x86_64.rpm",
            f"linux/rpm/release/aarch64/repodata/{'c' * 64}-primary.xml.gz",
            f"linux/rpm/release/aarch64/repodata/{'d' * 64}-filelists.xml.gz",
            f"linux/rpm/release/aarch64/repodata/{'e' * 64}-other.xml.gz",
        ]
        for k in valid_immutable:
            self.assertEqual(classify_destination_key(k), "immutable", k)

        # Valid pointer keys
        valid_pointers = [
            "linux/keys/solstone-archive-keyring.gpg",
            "linux/keys/solstone-archive-keyring.asc",
            "linux/keys/solstone-archive-keyring.asc.minisig",
            "linux/apt/dists/release/Release",
            "linux/apt/dists/release/InRelease",
            "linux/apt/dists/release/Release.gpg",
            "linux/apt/dists/release/main/binary-amd64/Packages",
            "linux/apt/dists/release/main/binary-amd64/Packages.gz",
            "linux/apt/dists/release/main/binary-arm64/Packages.xz",
            "linux/rpm/release/x86_64/repodata/repomd.xml",
            "linux/rpm/release/aarch64/repodata/repomd.xml.asc",
        ]
        for k in valid_pointers:
            self.assertEqual(classify_destination_key(k), "pointer", k)

        # Invalid keys
        invalid_keys = [
            ("linux/keys/evil.gpg", UNSAFE_FILENAME),
            ("linux/apt/pool/badlane/main/s/solstone/solstone_1.0.0_amd64.deb", LANE_INVALID),
            ("linux/apt/pool/release/main/s/other/other_1.0.0_amd64.deb", UNSAFE_FILENAME),
            ("linux/apt/dists/release/main/binary-i386/Packages", UNSAFE_FILENAME),
            ("linux/apt/dists/release/main/binary-amd64/by-hash/SHA256/nothex", UNSAFE_FILENAME),
            ("linux/rpm/release/x86_64/Packages/short/solstone-1.0.0-1.x86_64.rpm", UNSAFE_FILENAME),
            ("linux/rpm/release/x86_64/repodata/other.xml", UNSAFE_FILENAME),
        ]
        for k, expected_refusal in invalid_keys:
            with self.assertRaises(Refusal, msg=k) as ctx:
                classify_destination_key(k)
            self.assertEqual(ctx.exception.name, expected_refusal, k)

        # R2Destination admission checks for pointer vs immutable keys
        r2 = R2Destination(R2Config(endpoint="https://example.invalid", bucket="updates"))
        with self.assertRaises(Refusal) as ctx:
            r2._admit_request(
                "PUT",
                "linux/apt/pool/release/main/s/solstone/solstone_2.0.76-1_amd64.deb",
                {"If-Match": '"etag"'},
            )
        self.assertEqual(ctx.exception.name, UNSAFE_FILENAME)

        with self.assertRaises(Refusal) as ctx:
            r2._admit_request(
                "PUT",
                "linux/apt/dists/release/InRelease",
                {"If-Match": '"etag"', "If-None-Match": "*"},
            )
        self.assertEqual(ctx.exception.name, UNSAFE_FILENAME)

    def test_cli_build_and_publish_repos_require_production_acknowledgement(self):
        rc = cli_main([
            "publish-repos",
            "--repo-dir",
            str(self.root),
        ])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
