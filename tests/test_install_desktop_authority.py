# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Installer-side verification of Desktop's native release authority."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from solstone_platform.canonical import canonical_json_bytes
from solstone_platform.generate import generate_platform_manifest
from solstone_platform.pins import DESKTOP_KEY_ID, DESKTOP_PUBKEY, PinSet, embedded_pins
from solstone_platform.sign import ephemeral_keypair, sign_manifest
from tests.install_test_helpers import LoopbackServer, setup_fake_sudo
from tools.build_installer import build_installer
from tools.fixture_builder import build_tiny_natives


REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_VERSION = "2.0.0"
DESKTOP_VERSION = "2.0.3"
HELPER_SCRIPT = REPO_ROOT / "helpers" / "solstone-pkg-helper.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DesktopFixture:
    def __init__(self, root: Path, platform_sec: Path, platform_pub: Path, platform_pin, desktop_sec: Path | None, desktop_pin, real: bool = False):
        self.root = root
        self.platform_sec = platform_sec
        self.platform_pin = platform_pin
        self.desktop_sec = desktop_sec
        self.desktop_pin = desktop_pin
        self.web_root = root / "www"
        self.server = LoopbackServer(self.web_root)
        self.server.start()

        tiny_pins, native_dirs = build_tiny_natives(root / "natives")
        if real:
            self.desktop_dir = REPO_ROOT / "testdata" / "native" / "desktop" / DESKTOP_VERSION
        else:
            self.desktop_dir = native_dirs["desktop"]
            manifest = next(self.desktop_dir.glob("*.rust-release-manifest.json"))
            subprocess.run(
                [
                    "minisign", "-S", "-W", "-s", str(desktop_sec),
                    "-m", str(manifest), "-x", str(manifest.with_name(manifest.name + ".minisig")),
                    "-t", "solstone-linux test release manifest",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        pins = PinSet(journal=tiny_pins.journal, desktop=desktop_pin, tmux=tiny_pins.tmux)
        manifest_bytes = generate_platform_manifest(
            version=PLATFORM_VERSION,
            lane="release",
            created_unix=1773820000,
            source_commit="3075c36b12fad469d4c9c0ab4555908fe8ecca1b",
            platform_key_id=platform_pin.key_id,
            repo_root=REPO_ROOT,
            journal_dir=native_dirs["journal"],
            desktop_dir=self.desktop_dir,
            tmux_dir=native_dirs["tmux"],
            journal_origin=self.server.origin,
            pins=pins,
        )
        self.platform = json.loads(manifest_bytes.decode("utf-8"))
        self.version_dir = self.web_root / "solstone" / "release" / PLATFORM_VERSION
        self.version_dir.mkdir(parents=True)
        (self.version_dir.parent / "latest").write_text(f"{PLATFORM_VERSION}\n", encoding="utf-8")
        for source in self.desktop_dir.iterdir():
            if source.is_file():
                shutil.copy2(source, self.version_dir / source.name)

        self.write_platform()
        self.installer = root / "install.sh"
        build_installer(
            repo_root=REPO_ROOT,
            output_path=self.installer,
            platform_pub_path=platform_pub,
            platform_key_id=platform_pin.key_id,
            origin=self.server.origin,
        )
        if not real:
            rendered = self.installer.read_text(encoding="utf-8")
            rendered = rendered.replace(f'DESKTOP_KEY_ID="{DESKTOP_KEY_ID}"', f'DESKTOP_KEY_ID="{desktop_pin.key_id}"')
            rendered = rendered.replace(f'DESKTOP_PUBKEY="{DESKTOP_PUBKEY}"', f'DESKTOP_PUBKEY="{desktop_pin.pubkey}"')
            self.installer.write_text(rendered, encoding="utf-8")
        self.installer.chmod(0o755)

    def close(self):
        self.server.stop()

    @property
    def native_manifest(self) -> Path:
        return self.version_dir / f"solstone-linux-{DESKTOP_VERSION}-linux-x86_64.rust-release-manifest.json"

    @property
    def native_signature(self) -> Path:
        return self.native_manifest.with_name(self.native_manifest.name + ".minisig")

    def sign_native(self):
        if self.desktop_sec is None:
            raise AssertionError("real producer fixture cannot be re-signed")
        subprocess.run(
            [
                "minisign", "-S", "-W", "-s", str(self.desktop_sec),
                "-m", str(self.native_manifest), "-x", str(self.native_signature),
                "-t", "solstone-linux test release manifest",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def refresh_outer(self):
        for route in self.platform["components"]["desktop"]["arches"]["x86_64"].values():
            route["authority"]["manifest_sha256"] = sha256(self.native_manifest)
            route["authority"]["signature_sha256"] = sha256(self.native_signature)

    def resign_native(self):
        self.sign_native()
        self.refresh_outer()
        self.write_platform()

    def write_native(self, obj):
        self.native_manifest.write_bytes(canonical_json_bytes(obj))
        self.resign_native()

    def write_platform(self):
        manifest_bytes = canonical_json_bytes(self.platform)
        (self.version_dir / "platform.json").write_bytes(manifest_bytes)
        signature = sign_manifest(
            manifest_bytes=manifest_bytes,
            secret_key_path=self.platform_sec,
            selected_pin=self.platform_pin,
            repo_root=REPO_ROOT,
            is_production=False,
        )
        (self.version_dir / "platform.json.minisig").write_bytes(signature)

    def run(self, *args: str, path: str | None = None, extra_env: dict[str, str] | None = None):
        env = os.environ.copy()
        if path is not None:
            env["PATH"] = path
        if extra_env:
            env.update(extra_env)
        prefix = self.root / f"prefix-{len(self.server.request_paths)}"
        self.server.request_paths.clear()
        proc = subprocess.run(
            [str(self.installer), *args, "--prefix", str(prefix), "--json"],
            capture_output=True,
            text=True,
            env=env,
        )
        return proc, prefix, list(self.server.request_paths)


class TestInstallDesktopAuthority(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, label: str, real: bool = False) -> DesktopFixture:
        platform_keys = ephemeral_keypair(f"desktop platform {label}")
        p_sec, p_pub, p_pin = platform_keys.__enter__()
        self.addCleanup(platform_keys.__exit__, None, None, None)
        if real:
            d_sec = None
            d_pin = embedded_pins().desktop
        else:
            desktop_keys = ephemeral_keypair(f"desktop native {label}")
            d_sec, _, d_pin = desktop_keys.__enter__()
            self.addCleanup(desktop_keys.__exit__, None, None, None)
        fixture = DesktopFixture(self.work_dir / label, p_sec, p_pub, p_pin, d_sec, d_pin, real=real)
        self.addCleanup(fixture.close)
        return fixture

    @staticmethod
    def result(proc: subprocess.CompletedProcess) -> dict:
        return json.loads(proc.stdout.strip())

    @staticmethod
    def selected_artifact(obj: dict, route: str = "tree") -> dict:
        suffix = {"tree": ".tar.gz", "deb": ".deb", "rpm": ".rpm"}[route]
        return next(item for item in obj["artifacts"] if item["path"].endswith(suffix))

    def test_real_producer_bytes_and_exact_publish_paths(self):
        fixture = self.fixture("real", real=True)
        proc, prefix, requests = fixture.run("--components", "desktop", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        base = f"/solstone/release/{PLATFORM_VERSION}/solstone-linux-{DESKTOP_VERSION}-linux-x86_64.rust-release-manifest.json"
        self.assertEqual(
            requests,
            [
                "/solstone/release/latest",
                f"/solstone/release/{PLATFORM_VERSION}/platform.json",
                f"/solstone/release/{PLATFORM_VERSION}/platform.json.minisig",
                base,
                base + ".minisig",
            ],
        )
        self.assertFalse(prefix.exists())

    def test_digest_checks_precede_native_consumers(self):
        fixture = self.fixture("digest-order")
        shim_dir = self.work_dir / "digest-shims"
        shim_dir.mkdir()
        mini_log = self.work_dir / "minisign.log"
        awk_log = self.work_dir / "awk.log"
        real_minisign = shutil.which("minisign")
        real_awk = shutil.which("awk")
        self.assertIsNotNone(real_minisign)
        self.assertIsNotNone(real_awk)
        (shim_dir / "minisign").write_text(
            f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{mini_log}"\nexec "{real_minisign}" "$@"\n',
            encoding="utf-8",
        )
        (shim_dir / "awk").write_text(
            f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{awk_log}"\nexec "{real_awk}" "$@"\n',
            encoding="utf-8",
        )
        (shim_dir / "minisign").chmod(0o755)
        (shim_dir / "awk").chmod(0o755)
        path = f"{shim_dir}:{os.environ['PATH']}"

        for route in fixture.platform["components"]["desktop"]["arches"]["x86_64"].values():
            route["authority"]["manifest_sha256"] = "0" * 64
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "desktop", "--dry-run", path=path)
        self.assertEqual(self.result(proc)["root_code"], "digest-mismatch")
        self.assertEqual(len(mini_log.read_text().splitlines()), 1)
        self.assertFalse(any("schema_mode=desktop" in line for line in awk_log.read_text().splitlines()))

        mini_log.unlink()
        awk_log.unlink()
        fixture.refresh_outer()
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "desktop", "--dry-run", path=path)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(len(mini_log.read_text().splitlines()), 2)
        self.assertTrue(any("schema_mode=desktop" in line for line in awk_log.read_text().splitlines()))

        mini_log.unlink()
        awk_log.unlink()
        for route in fixture.platform["components"]["desktop"]["arches"]["x86_64"].values():
            route["authority"]["signature_sha256"] = "0" * 64
        fixture.write_platform()
        proc, _, _ = fixture.run("--skip-signature", "--components", "desktop", "--dry-run", path=path)
        self.assertEqual(self.result(proc)["root_code"], "digest-mismatch")
        self.assertFalse(mini_log.exists())
        self.assertFalse(any("schema_mode=desktop" in line for line in awk_log.read_text().splitlines()))

    def test_native_signature_and_verifier_stage(self):
        fixture = self.fixture("signature")
        fixture.native_signature.write_bytes(b"not a minisign signature\n")
        fixture.refresh_outer()
        fixture.write_platform()
        proc, prefix, requests = fixture.run("--components", "desktop", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "signature-invalid")
        self.assertFalse(prefix.exists())
        self.assertFalse(any(path.endswith((".tar.gz", ".deb", ".rpm")) for path in requests))

        fixture = self.fixture("verifier-stage")
        fake_bin = self.work_dir / "one-use-path"
        fake_bin.mkdir()
        required = ("awk", "cat", "chmod", "curl", "dd", "grep", "head", "mkdir", "mktemp", "od", "rm", "sed", "sha256sum", "tr", "uname", "wc")
        for command in required:
            target = shutil.which(command)
            self.assertIsNotNone(target, command)
            (fake_bin / command).symlink_to(target)
        real_minisign = shutil.which("minisign")
        (fake_bin / "minisign").write_text(
            f'#!/bin/sh\nrm -f "$0"\nexec "{real_minisign}" "$@"\n',
            encoding="utf-8",
        )
        (fake_bin / "minisign").chmod(0o755)
        proc, prefix, requests = fixture.run("--components", "desktop", "--dry-run", path=str(fake_bin))
        self.assertEqual(self.result(proc)["root_code"], "verifier-missing")
        self.assertTrue(any(path.endswith(".rust-release-manifest.json.minisig") for path in requests))
        self.assertFalse(any(path.endswith((".tar.gz", ".deb", ".rpm")) for path in requests))
        self.assertFalse(prefix.exists())

    def test_typed_security_fields_and_semantic_bindings(self):
        mutations = {
            "schema-value": (lambda o: o.__setitem__("schema_version", 2), "schema-invalid"),
            "schema-type": (lambda o: o.__setitem__("schema_version", "1"), "schema-invalid"),
            "product": (lambda o: o.__setitem__("product", "other"), "release-coherence"),
            "version": (lambda o: o.__setitem__("version", "9.9.9"), "release-coherence"),
            "dirty": (lambda o: o.__setitem__("source_dirty", True), "release-coherence"),
            "dirty-type": (lambda o: o.__setitem__("source_dirty", "false"), "schema-invalid"),
            "commit-value": (lambda o: o.__setitem__("source_commit", "a" * 39), "schema-invalid"),
            "commit-type": (lambda o: o.__setitem__("source_commit", 7), "schema-invalid"),
            "lock-value": (lambda o: o.__setitem__("cargo_lock_sha256", "b" * 63), "schema-invalid"),
            "lock-type": (lambda o: o.__setitem__("cargo_lock_sha256", 7), "schema-invalid"),
            "target-kind": (lambda o: o["target"].__setitem__("kind", "other"), "release-coherence"),
            "target-profile": (lambda o: o["target"].__setitem__("profile", "debug"), "release-coherence"),
            "target-triple": (lambda o: o["target"].__setitem__("triple", "aarch64-unknown-linux-gnu"), "release-coherence"),
            "artifacts-type": (lambda o: o.__setitem__("artifacts", {}), "schema-invalid"),
            "member-type": (lambda o: o["artifacts"].__setitem__(0, "not-an-object"), "schema-invalid"),
            "bytes-type": (lambda o: self.selected_artifact(o).__setitem__("bytes", str(self.selected_artifact(o)["bytes"])), "schema-invalid"),
        }
        for label, (mutate, code) in mutations.items():
            with self.subTest(label=label):
                fixture = self.fixture(f"typed-{label}")
                obj = json.loads(fixture.native_manifest.read_text(encoding="utf-8"))
                mutate(obj)
                fixture.write_native(obj)
                proc, prefix, requests = fixture.run("--components", "desktop", "--dry-run")
                self.assertEqual(self.result(proc)["root_code"], code, proc.stdout + proc.stderr)
                self.assertFalse(prefix.exists())
                self.assertFalse(any(path.endswith((".tar.gz", ".deb", ".rpm")) for path in requests))

    def test_duplicate_keys_members_and_unique_twin(self):
        fixture = self.fixture("duplicate-key")
        raw = fixture.native_manifest.read_bytes().replace(b'"schema_version":1,', b'"schema_version":1,"schema_version":1,', 1)
        fixture.native_manifest.write_bytes(raw)
        fixture.resign_native()
        proc, _, _ = fixture.run("--components", "desktop", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "schema-invalid")

        for label, alter in (("exact", False), ("mixed", True)):
            with self.subTest(duplicate_member=label):
                fixture = self.fixture(f"duplicate-member-{label}")
                obj = json.loads(fixture.native_manifest.read_text(encoding="utf-8"))
                duplicate = dict(self.selected_artifact(obj))
                if alter:
                    duplicate["sha256"] = "f" * 64
                obj["artifacts"].append(duplicate)
                fixture.write_native(obj)
                proc, _, _ = fixture.run("--components", "desktop", "--dry-run")
                self.assertEqual(self.result(proc)["root_code"], "schema-invalid")

        fixture = self.fixture("unique-twin")
        proc, _, _ = fixture.run("--components", "desktop", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_route_member_and_value_coherence(self):
        for route in ("tree", "deb", "rpm"):
            with self.subTest(route=route):
                fixture = self.fixture(f"route-{route}")
                obj = json.loads(fixture.native_manifest.read_text(encoding="utf-8"))
                selected = self.selected_artifact(obj, route)
                other = next(item for item in obj["artifacts"] if item is not selected)
                selected["path"] = other["path"]
                fixture.write_native(obj)
                proc, _, _ = fixture.run("--components", "desktop", "--dry-run", "--route", route)
                self.assertEqual(self.result(proc)["root_code"], "release-coherence")

        for field, value in (("sha256", "f" * 64), ("bytes", 1)):
            with self.subTest(field=field):
                fixture = self.fixture(f"artifact-{field}")
                obj = json.loads(fixture.native_manifest.read_text(encoding="utf-8"))
                self.selected_artifact(obj)[field] = value
                fixture.write_native(obj)
                proc, _, _ = fixture.run("--components", "desktop", "--dry-run")
                self.assertEqual(self.result(proc)["root_code"], "release-coherence")

    def test_skip_signature_still_enforces_native_semantics(self):
        fixture = self.fixture("skip-semantic")
        obj = json.loads(fixture.native_manifest.read_text(encoding="utf-8"))
        obj["product"] = "other"
        fixture.write_native(obj)
        proc, _, _ = fixture.run("--skip-signature", "--components", "desktop", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "release-coherence")

        fixture = self.fixture("skip-pin")
        for route in fixture.platform["components"]["desktop"]["arches"]["x86_64"].values():
            route["authority"]["verifier_id"] = "minisign:0000000000000000"
        fixture.write_platform()
        proc, _, _ = fixture.run("--skip-signature", "--components", "desktop", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "pin-mismatch")

    def test_failure_is_pre_mutation_and_valid_package_reaches_helper(self):
        fixture = self.fixture("pre-mutation")
        obj = json.loads(fixture.native_manifest.read_text(encoding="utf-8"))
        obj["product"] = "other"
        fixture.write_native(obj)
        proc, prefix, requests = fixture.run("--components", "desktop")
        self.assertEqual(self.result(proc)["root_code"], "release-coherence")
        self.assertFalse(prefix.exists())
        self.assertFalse(any(path.endswith((".tar.gz", ".deb", ".rpm")) for path in requests))

        fixture = self.fixture("package-twin")
        bin_dir = self.work_dir / "package-bin"
        setup_fake_sudo(bin_dir)
        lock_dir = self.work_dir / "package-lock"
        etc_root = self.work_dir / "package-etc"
        fake_db = self.work_dir / "package-db"
        for directory in (lock_dir, etc_root, fake_db):
            directory.mkdir()
        env = {
            "SOLSTONE_LOCK_DIR": str(lock_dir),
            "SOLSTONE_ETC_ROOT": str(etc_root),
            "SOLSTONE_FAKE_PKG_DB": str(fake_db),
            "SOLSTONE_HELPER": str(HELPER_SCRIPT),
        }
        proc, _, requests = fixture.run(
            "--skip-signature", "--route", "deb", "--components", "desktop",
            path=f"{bin_dir}:{os.environ['PATH']}", extra_env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(any(path.endswith(".deb") for path in requests))
        self.assertTrue((fake_db / "install.log").is_file())

    def test_native_authority_caps(self):
        manifest_limit = 65536
        signature_limit = 16384

        fixture = self.fixture("manifest-cap")
        fixture.native_manifest.write_bytes(fixture.native_manifest.read_bytes() + b" " * (manifest_limit - fixture.native_manifest.stat().st_size))
        fixture.resign_native()
        proc, _, _ = fixture.run("--components", "desktop", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        fixture.native_manifest.write_bytes(fixture.native_manifest.read_bytes() + b" ")
        fixture.refresh_outer()
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "desktop", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "response-too-large")

        fixture = self.fixture("signature-cap")
        fixture.native_signature.write_bytes(fixture.native_signature.read_bytes() + b" " * (signature_limit - fixture.native_signature.stat().st_size))
        fixture.refresh_outer()
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "desktop", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        fixture.native_signature.write_bytes(fixture.native_signature.read_bytes() + b" ")
        fixture.refresh_outer()
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "desktop", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "response-too-large")


if __name__ == "__main__":
    unittest.main()
