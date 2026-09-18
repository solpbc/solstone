# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test Families 2 & 3: Verification, signature tampering, skip-signature, duplicate-key, schema strictness."""

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from solstone_platform.generate import generate_platform_manifest
from solstone_platform.pins import PinSet, embedded_pins, load_pin_file
from solstone_platform.refusals import Refusal
from solstone_platform.sign import ephemeral_keypair, sign_manifest
from tests.install_test_helpers import LoopbackServer, make_v2_bootstrap_script, setup_test_release_server
from tools.build_installer import build_installer
from tools.fixture_builder import build_tiny_natives

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInstallVerify(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.prefix = self.work_dir / "prefix"
        self.prefix.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_signature_succeeds(self):
        with ephemeral_keypair("test verify") as (sec, pub, pin):
            server, _ = setup_test_release_server(self.work_dir, sec, pin)
            try:
                installer = self.work_dir / "install.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                )

                proc = subprocess.run(
                    [str(installer), "--list", "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"List failed: {proc.stderr}\n{proc.stdout}")
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["root_code"], "list")
                self.assertEqual(res["verification_layers"], "minisign+digest")
            finally:
                server.stop()

    def test_corrupt_signature_refuses(self):
        with ephemeral_keypair("test verify corrupt") as (sec, pub, pin):
            server, _ = setup_test_release_server(self.work_dir, sec, pin, corrupt_signature=True)
            try:
                installer = self.work_dir / "install.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                )

                proc = subprocess.run(
                    [str(installer), "--list", "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "refusal")
                self.assertEqual(res["root_code"], "signature-invalid")
            finally:
                server.stop()

    def test_skip_signature_bypasses_minisig(self):
        with ephemeral_keypair("test verify skip") as (sec, pub, pin):
            server, _ = setup_test_release_server(self.work_dir, sec, pin, corrupt_signature=True)
            try:
                installer = self.work_dir / "install.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                )

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "journal", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Install failed: {proc.stderr}\n{proc.stdout}")
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["verification_layers"], "digest-matched; signatures skipped")
            finally:
                server.stop()

    def test_skip_signature_still_refuses_digest_mismatch(self):
        with ephemeral_keypair("test verify digest mismatch") as (sec, pub, pin):
            server, server_root = setup_test_release_server(self.work_dir, sec, pin)
            try:
                # Corrupt the bootstrap file on the server
                j_bootstrap = (
                    server_root
                    / "solstone-journal"
                    / "release"
                    / "2.0.6"
                    / "solstone-journal-2.0.6-install.sh"
                )
                j_bootstrap.write_bytes(b"tampered bootstrap script\n")

                installer = self.work_dir / "install.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                )

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "journal", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "refusal")
                self.assertEqual(res["root_code"], "digest-mismatch")
            finally:
                server.stop()

    def test_duplicate_key_refuses(self):
        with ephemeral_keypair("test verify dup") as (sec, pub, pin):
            server, _ = setup_test_release_server(self.work_dir, sec, pin, duplicate_key=True)
            try:
                installer = self.work_dir / "install.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                )

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "journal", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "refusal")
                self.assertEqual(res["root_code"], "duplicate-key")
            finally:
                server.stop()

    def test_corrupt_manifest_refuses(self):
        with ephemeral_keypair("test verify corrupt json") as (sec, pub, pin):
            server, _ = setup_test_release_server(self.work_dir, sec, pin, corrupt_manifest=True)
            try:
                installer = self.work_dir / "install.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                )

                proc = subprocess.run(
                    [str(installer), "--skip-signature", "--components", "journal", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "refusal")
                self.assertEqual(res["root_code"], "schema-invalid")
            finally:
                server.stop()

    def test_production_build_requires_exact_checked_in_pin(self):
        out_file = self.work_dir / "prod_install.sh"
        proc = subprocess.run(
            ["python3", "tools/build_installer.py", "--production", "-o", str(out_file)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        pin = load_pin_file(REPO_ROOT / "pins" / "platform.pub")
        script = out_file.read_text(encoding="utf-8")
        self.assertIn(f'PLATFORM_KEY_ID="{pin.key_id}"', script)
        self.assertIn(f'PLATFORM_PUBKEY="{pin.pubkey}"', script)
        self.assertIn("TEST_SEAM=0", script)

        fixture_repo = self.work_dir / "missing-pin-repo"
        fixture_repo.mkdir()
        shutil.copy2(REPO_ROOT / "install.sh.in", fixture_repo / "install.sh.in")
        shutil.copytree(REPO_ROOT / "compat", fixture_repo / "compat")
        shutil.copytree(REPO_ROOT / "pins", fixture_repo / "pins")
        (fixture_repo / "pins" / "platform.pub").unlink()
        with self.assertRaisesRegex(Refusal, "production platform pin is absent"):
            build_installer(
                repo_root=fixture_repo,
                output_path=self.work_dir / "must-not-exist.sh",
                is_production=True,
            )

        shutil.copy2(REPO_ROOT / "pins" / "platform.pub", fixture_repo / "pins" / "platform.pub")
        (fixture_repo / "pins" / "platform.keyid").write_text("0000000000000000\n", encoding="utf-8")
        with self.assertRaisesRegex(Refusal, "platform key ID mismatch"):
            build_installer(
                repo_root=fixture_repo,
                output_path=self.work_dir / "must-not-exist.sh",
                is_production=True,
            )

    def test_native_desktop_pin_matches_checked_in_fixture(self):
        with ephemeral_keypair("test desktop pin") as (sec, pub, pin):
            server_root = self.work_dir / "www_desktop"
            server_root.mkdir(parents=True, exist_ok=True)
            server = LoopbackServer(server_root)
            server.start()
            try:
                # Build tiny journal (v2 bootstrap)
                v2_boot = make_v2_bootstrap_script(revision=2)
                tiny_pinset, native_dirs = build_tiny_natives(
                    target_dir=self.work_dir / "tiny_natives",
                    bootstrap_script=v2_boot,
                )

                embedded = embedded_pins()
                mixed_pinset = PinSet(
                    journal=tiny_pinset.journal,
                    desktop=embedded.desktop,
                    tmux=embedded.tmux,
                )

                version = "2.0.3"
                lane = "release"
                manifest_bytes = generate_platform_manifest(
                    version=version,
                    lane=lane,
                    created_unix=1773820000,
                    source_commit="3075c36b12fad469d4c9c0ab4555908fe8ecca1b",
                    platform_key_id=pin.key_id,
                    repo_root=REPO_ROOT,
                    journal_dir=native_dirs["journal"],
                    desktop_dir=REPO_ROOT / "testdata" / "native" / "desktop" / "2.0.3",
                    tmux_dir=REPO_ROOT / "testdata" / "native" / "tmux" / "2.0.3",
                    journal_origin=server.origin,
                    pins=mixed_pinset,
                )

                lane_dir = server_root / "solstone" / lane
                ver_dir = lane_dir / version
                ver_dir.mkdir(parents=True, exist_ok=True)

                (lane_dir / "latest").write_text(f"{version}\n", encoding="utf-8")
                (ver_dir / "platform.json").write_bytes(manifest_bytes)

                sig_bytes = sign_manifest(
                    manifest_bytes=manifest_bytes,
                    secret_key_path=sec,
                    selected_pin=pin,
                    repo_root=REPO_ROOT,
                    is_production=False,
                )
                (ver_dir / "platform.json.minisig").write_bytes(sig_bytes)

                # Copy real desktop tree archive
                real_desktop_dir = REPO_ROOT / "testdata" / "native" / "desktop" / "2.0.3"
                for p in real_desktop_dir.glob("*.tar.gz"):
                    shutil.copy2(p, ver_dir / p.name)

                installer = self.work_dir / "install_desktop_pin.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                )

                # Run installer with --components desktop without --skip-signature
                proc = subprocess.run(
                    [str(installer), "--components", "desktop", "--dry-run", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, f"Install failed: {proc.stderr}\n{proc.stdout}")
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "success")
                self.assertEqual(res["root_code"], "dry-run-completed")
                self.assertIn("desktop", res["components"])
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
