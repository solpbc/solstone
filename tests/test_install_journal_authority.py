# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Installer-side verification of Journal's native release authority."""

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
from solstone_platform.pins import DESKTOP_KEY_ID, DESKTOP_PUBKEY, JOURNAL_KEY_ID, JOURNAL_PUBKEY, TMUX_KEY_ID, TMUX_PUBKEY, PinSet
from solstone_platform.sign import ephemeral_keypair, sign_manifest
from tests.install_test_helpers import LoopbackServer
from tools.build_installer import build_installer
from tools.fixture_builder import build_tiny_natives


REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_VERSION = "2.0.0"
JOURNAL_VERSION = "2.0.8"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class JournalFixture:
    def __init__(self, root: Path, platform_sec: Path, platform_pub: Path, platform_pin, journal_sec: Path, journal_pin):
        self.root = root
        self.platform_sec = platform_sec
        self.platform_pub = platform_pub
        self.platform_pin = platform_pin
        self.journal_sec = journal_sec
        self.journal_pin = journal_pin
        self.web_root = root / "www"
        self.server = LoopbackServer(self.web_root)
        self.server.start()

        source = REPO_ROOT / "testdata" / "native" / "journal-v2"
        self.journal_dir = root / "journal-v2"
        shutil.copytree(source, self.journal_dir)
        for manifest in self.journal_dir.rglob("*.manifest.json"):
            self._sign_native(manifest)

        tiny_pins, native_dirs = build_tiny_natives(root / "other-natives")
        pins = PinSet(journal=journal_pin, desktop=tiny_pins.desktop, tmux=tiny_pins.tmux)
        manifest_bytes = generate_platform_manifest(
            version=PLATFORM_VERSION,
            lane="release",
            created_unix=1773820000,
            source_commit="3075c36b12fad469d4c9c0ab4555908fe8ecca1b",
            platform_key_id=platform_pin.key_id,
            repo_root=REPO_ROOT,
            journal_dir=self.journal_dir,
            desktop_dir=native_dirs["desktop"],
            tmux_dir=native_dirs["tmux"],
            journal_origin=self.server.origin,
            pins=pins,
        )
        self.platform = json.loads(manifest_bytes.decode("utf-8"))
        self.version_dir = self.web_root / "solstone" / "release" / PLATFORM_VERSION
        self.version_dir.mkdir(parents=True, exist_ok=True)
        latest_dir = self.version_dir.parent
        (latest_dir / "latest").write_text(f"{PLATFORM_VERSION}\n", encoding="utf-8")

        for source_file in self.journal_dir.rglob("*"):
            if source_file.is_file():
                shutil.copy2(source_file, self.version_dir / source_file.name)
        for component in (native_dirs["desktop"], native_dirs["tmux"]):
            for source_file in component.rglob("*"):
                if source_file.is_file() and (
                    source_file.suffix in (".gz", ".deb", ".rpm")
                    or ".rust-release-manifest.json" in source_file.name
                    or source_file.name in ("SHA256SUMS", "SHA256SUMS.minisig")
                    or source_file.name.endswith(".target.json")
                ):
                    shutil.copy2(source_file, self.version_dir / source_file.name)

        bootstrap = next((self.journal_dir / "linux-x86_64").glob("*-install.sh"))
        bootstrap_dir = self.web_root / "solstone-journal" / "release" / JOURNAL_VERSION
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bootstrap, bootstrap_dir / bootstrap.name)

        self.write_platform()
        self.installer = root / "install.sh"
        build_installer(
            repo_root=REPO_ROOT,
            output_path=self.installer,
            platform_pub_path=platform_pub,
            platform_key_id=platform_pin.key_id,
            origin=self.server.origin,
        )
        rendered = self.installer.read_text(encoding="utf-8")
        rendered = rendered.replace(f'JOURNAL_KEY_ID="{JOURNAL_KEY_ID}"', f'JOURNAL_KEY_ID="{journal_pin.key_id}"')
        rendered = rendered.replace(f'JOURNAL_PUBKEY="{JOURNAL_PUBKEY}"', f'JOURNAL_PUBKEY="{journal_pin.pubkey}"')
        rendered = rendered.replace(f'DESKTOP_KEY_ID="{DESKTOP_KEY_ID}"', f'DESKTOP_KEY_ID="{tiny_pins.desktop.key_id}"')
        rendered = rendered.replace(f'DESKTOP_PUBKEY="{DESKTOP_PUBKEY}"', f'DESKTOP_PUBKEY="{tiny_pins.desktop.pubkey}"')
        rendered = rendered.replace(f'TMUX_KEY_ID="{TMUX_KEY_ID}"', f'TMUX_KEY_ID="{tiny_pins.tmux.key_id}"')
        rendered = rendered.replace(f'TMUX_PUBKEY="{TMUX_PUBKEY}"', f'TMUX_PUBKEY="{tiny_pins.tmux.pubkey}"')
        self.installer.write_text(rendered, encoding="utf-8")
        self.installer.chmod(0o755)

    def close(self):
        self.server.stop()

    def _sign_native(self, manifest: Path):
        subprocess.run(
            [
                "minisign",
                "-S",
                "-W",
                "-s",
                str(self.journal_sec),
                "-m",
                str(manifest),
                "-x",
                str(manifest.with_suffix(".json.minisig")),
                "-t",
                "solstone-journal test release manifest",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def native_path(self, arch: str, suffix: str) -> Path:
        target = f"linux-{arch}"
        return self.version_dir / f"solstone-journal-{JOURNAL_VERSION}-{target}{suffix}"

    def refresh_outer(self, arch: str):
        manifest = self.native_path(arch, ".manifest.json")
        signature = self.native_path(arch, ".manifest.json.minisig")
        release = self.native_path(arch, ".release")
        bootstrap = self.version_dir / f"solstone-journal-{JOURNAL_VERSION}-install.sh"
        if not bootstrap.exists():
            bootstrap = self.web_root / "solstone-journal" / "release" / JOURNAL_VERSION / bootstrap.name
        for route in self.platform["components"]["journal"]["arches"][arch].values():
            authority = route["authority"]
            authority["manifest_sha256"] = sha256(manifest)
            authority["signature_sha256"] = sha256(signature)
            authority["release_sha256"] = sha256(release)
            authority["bootstrap_sha256"] = sha256(bootstrap)

    def resign_native(self, arch: str):
        self._sign_native(self.native_path(arch, ".manifest.json"))
        self.refresh_outer(arch)

    def rebind_sums(self, arch: str):
        manifest_path = self.native_path(arch, ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sums = self.native_path(arch, ".sha256")
        manifest["files"][sums.name] = sha256(sums)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        self.resign_native(arch)

    def rebind_release(self, arch: str):
        release = self.native_path(arch, ".release")
        sums = self.native_path(arch, ".sha256")
        release_digest = sha256(release)
        lines = sums.read_text(encoding="utf-8").splitlines()
        rebound = []
        for line in lines:
            name = line[66:]
            rebound.append(f"{release_digest}  {name}" if name == release.name else line)
        sums.write_text("\n".join(rebound) + "\n", encoding="utf-8")
        manifest_path = self.native_path(arch, ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][release.name] = release_digest
        manifest["files"][sums.name] = sha256(sums)
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        self.resign_native(arch)

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

    def run(self, *args: str, arch: str = "x86_64", path: str | None = None):
        env = os.environ.copy()
        if arch == "aarch64":
            fake_bin = self.root / "fake-aarch64-bin"
            fake_bin.mkdir(exist_ok=True)
            uname = fake_bin / "uname"
            uname.write_text("#!/bin/sh\necho aarch64\n", encoding="utf-8")
            uname.chmod(0o755)
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
        if path is not None:
            env["PATH"] = path
        prefix = self.root / f"prefix-{arch}-{len(self.server.request_paths)}"
        self.server.request_paths.clear()
        proc = subprocess.run(
            [str(self.installer), *args, "--prefix", str(prefix), "--json"],
            capture_output=True,
            text=True,
            env=env,
        )
        return proc, prefix, list(self.server.request_paths)


class TestInstallJournalAuthority(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, label: str):
        platform_keys = ephemeral_keypair(f"platform {label}")
        journal_keys = ephemeral_keypair(f"journal {label}")
        p_sec, p_pub, p_pin = platform_keys.__enter__()
        j_sec, _, j_pin = journal_keys.__enter__()
        fixture = JournalFixture(self.work_dir / label, p_sec, p_pub, p_pin, j_sec, j_pin)
        self.addCleanup(platform_keys.__exit__, None, None, None)
        self.addCleanup(journal_keys.__exit__, None, None, None)
        self.addCleanup(fixture.close)
        return fixture

    def test_real_producer_bytes_and_exact_publish_paths(self):
        fixture = self.fixture("real-producer")
        for arch in ("x86_64", "aarch64"):
            with self.subTest(arch=arch):
                proc, prefix, requests = fixture.run("--components", "journal", "--dry-run", arch=arch)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                target = f"linux-{arch}"
                base = f"/solstone/release/{PLATFORM_VERSION}/solstone-journal-{JOURNAL_VERSION}-{target}"
                self.assertEqual(
                    requests,
                    [
                        "/solstone/release/latest",
                        f"/solstone/release/{PLATFORM_VERSION}/platform.json",
                        f"/solstone/release/{PLATFORM_VERSION}/platform.json.minisig",
                        f"{base}.manifest.json",
                        f"{base}.manifest.json.minisig",
                        f"{base}.release",
                        f"{base}.sha256",
                    ],
                )
                self.assertFalse(prefix.exists())

    def test_journal_aliases_share_signature_preflight(self):
        fixture = self.fixture("aliases")
        for selection in ("journal", "cli", "capture"):
            with self.subTest(selection=selection, state="valid"):
                proc, _, _ = fixture.run("--components", selection, "--dry-run")
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

        signature = fixture.native_path("x86_64", ".manifest.json.minisig")
        signature_lines = signature.read_bytes().splitlines(keepends=True)
        signed_line = bytearray(signature_lines[1])
        signed_line[0] = ord("A") if signed_line[0] != ord("A") else ord("B")
        signature_lines[1] = bytes(signed_line)
        signature.write_bytes(b"".join(signature_lines))
        fixture.refresh_outer("x86_64")
        fixture.write_platform()
        for selection in ("journal", "cli", "capture"):
            with self.subTest(selection=selection, state="tampered"):
                proc, prefix, requests = fixture.run("--components", selection, "--dry-run")
                self.assertNotEqual(proc.returncode, 0)
                self.assertEqual(json.loads(proc.stdout)["root_code"], "signature-invalid")
                self.assertFalse(prefix.exists())
                self.assertFalse(any("install.sh" in path for path in requests))

    def test_manifest_semantic_mismatch_is_not_hidden_by_crypto(self):
        fixture = self.fixture("semantic")
        manifest_path = fixture.native_path("x86_64", ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["product"] = "wrong-product"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        fixture.resign_native("x86_64")
        fixture.write_platform()

        proc, prefix, requests = fixture.run("--components", "journal", "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["root_code"], "release-coherence")
        self.assertFalse(prefix.exists())
        self.assertFalse(any("install.sh" in path for path in requests))

    def test_sha256_sidecar_bytes_are_manifest_bound(self):
        fixture = self.fixture("sidecar-digest")
        sums = fixture.native_path("x86_64", ".sha256")
        lines = sums.read_text(encoding="utf-8").splitlines()
        sums.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

        proc, prefix, _ = fixture.run("--components", "journal", "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["root_code"], "digest-mismatch")
        self.assertFalse(prefix.exists())

    def test_skip_signature_keeps_pin_and_semantic_binding(self):
        fixture = self.fixture("skip-signature")
        for route in fixture.platform["components"]["journal"]["arches"]["x86_64"].values():
            route["authority"]["verifier_id"] = "minisign:0000000000000000"
        fixture.write_platform()
        proc, prefix, _ = fixture.run("--skip-signature", "--components", "journal", "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["root_code"], "pin-mismatch")
        self.assertFalse(prefix.exists())

    def test_each_route_binds_its_selected_artifact(self):
        for route in ("tree", "deb", "rpm"):
            with self.subTest(route=route):
                fixture = self.fixture(f"route-{route}")
                manifest_path = fixture.native_path("x86_64", ".manifest.json")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                selected = fixture.platform["components"]["journal"]["arches"]["x86_64"][route]["filename"]
                manifest["files"][selected] = "f" * 64
                manifest_path.write_bytes(canonical_json_bytes(manifest))
                fixture.resign_native("x86_64")
                fixture.write_platform()

                proc, prefix, requests = fixture.run("--route", route, "--components", "journal", "--dry-run")
                self.assertNotEqual(proc.returncode, 0)
                self.assertEqual(json.loads(proc.stdout)["root_code"], "release-coherence")
                self.assertFalse(prefix.exists())
                self.assertFalse(any("install.sh" in path for path in requests))

    def test_route_refuses_another_valid_signed_member(self):
        fixture = self.fixture("route-member-swap")
        routes = fixture.platform["components"]["journal"]["arches"]["x86_64"]
        deb_name, deb_sha = routes["deb"]["filename"], routes["deb"]["sha256"]
        routes["deb"]["filename"], routes["deb"]["sha256"] = routes["rpm"]["filename"], routes["rpm"]["sha256"]
        routes["rpm"]["filename"], routes["rpm"]["sha256"] = deb_name, deb_sha
        fixture.write_platform()

        proc, prefix, _ = fixture.run("--route", "deb", "--components", "journal", "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["root_code"], "release-coherence")
        self.assertFalse(prefix.exists())

    def test_release_and_native_formats_reject_ambiguity(self):
        cases = (
            "manifest-duplicate",
            "manifest-files-type",
            "release-duplicate",
            "release-noncanonical",
            "release-commit-shape",
            "release-lock-shape",
            "sums-duplicate",
        )
        for case in cases:
            with self.subTest(case=case):
                fixture = self.fixture(case)
                manifest_path = fixture.native_path("x86_64", ".manifest.json")
                release = fixture.native_path("x86_64", ".release")
                sums = fixture.native_path("x86_64", ".sha256")
                if case == "manifest-duplicate":
                    raw = manifest_path.read_bytes()
                    raw = raw.replace(b'  "product": "solstone-journal",', b'  "product": "solstone-journal",\n  "product": "solstone-journal",', 1)
                    manifest_path.write_bytes(raw)
                    fixture.resign_native("x86_64")
                elif case == "manifest-files-type":
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["files"] = []
                    manifest_path.write_bytes(canonical_json_bytes(manifest))
                    fixture.resign_native("x86_64")
                elif case == "release-duplicate":
                    release.write_bytes(release.read_bytes() + b"product=solstone-journal\n")
                    fixture.rebind_release("x86_64")
                elif case == "release-noncanonical":
                    release.write_bytes(release.read_bytes() + b"invalid line\n")
                    fixture.rebind_release("x86_64")
                elif case == "release-commit-shape":
                    release.write_text(
                        release.read_text(encoding="utf-8").replace(
                            "commit=3aabaf0ddad4e535e917973de6bdf7a36c324247",
                            "commit=NOT-A-COMMIT",
                        ),
                        encoding="utf-8",
                    )
                    fixture.rebind_release("x86_64")
                elif case == "release-lock-shape":
                    release.write_text(
                        release.read_text(encoding="utf-8").replace(
                            "lock_sha256=1d7bcb50f6b969f35bdad3fc37c6df1d355d142b24f582267b3394c45c7496b0",
                            "lock_sha256=NOT-A-DIGEST",
                        ),
                        encoding="utf-8",
                    )
                    fixture.rebind_release("x86_64")
                else:
                    first = sums.read_text(encoding="utf-8").splitlines()[0]
                    sums.write_text(sums.read_text(encoding="utf-8") + first + "\n", encoding="utf-8")
                    fixture.rebind_sums("x86_64")
                fixture.write_platform()

                proc, prefix, _ = fixture.run("--components", "journal", "--dry-run")
                self.assertNotEqual(proc.returncode, 0)
                self.assertEqual(json.loads(proc.stdout)["root_code"], "schema-invalid")
                self.assertFalse(prefix.exists())

    def test_release_provenance_mismatch_reaches_semantic_check(self):
        fixture = self.fixture("release-coherence")
        release = fixture.native_path("x86_64", ".release")
        text = release.read_text(encoding="utf-8").replace("upgrade_epoch=journal-v2", "upgrade_epoch=wrong-epoch")
        release.write_text(text, encoding="utf-8")
        fixture.rebind_release("x86_64")
        fixture.write_platform()

        proc, prefix, _ = fixture.run("--components", "journal", "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["root_code"], "release-coherence")
        self.assertFalse(prefix.exists())

    def test_skip_signature_still_checks_signature_digest_and_semantics(self):
        fixture = self.fixture("skip-digest")
        signature = fixture.native_path("x86_64", ".manifest.json.minisig")
        signature.write_bytes(signature.read_bytes() + b"changed")
        proc, prefix, _ = fixture.run("--skip-signature", "--components", "journal", "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["root_code"], "digest-mismatch")
        self.assertFalse(prefix.exists())

        fixture = self.fixture("skip-semantic")
        manifest_path = fixture.native_path("x86_64", ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["target"] = "linux-aarch64"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        fixture.resign_native("x86_64")
        fixture.write_platform()
        proc, prefix, _ = fixture.run("--skip-signature", "--components", "journal", "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["root_code"], "release-coherence")
        self.assertFalse(prefix.exists())

    def test_native_verifier_missing_after_platform_verification(self):
        fixture = self.fixture("verifier-missing")
        fake_bin = self.work_dir / "one-use-path"
        fake_bin.mkdir()
        required = (
            "awk", "cat", "chmod", "curl", "dd", "grep", "head", "mkdir", "mktemp", "od", "rm", "sed",
            "sha256sum", "tr", "uname", "wc",
        )
        for command in required:
            target = shutil.which(command)
            self.assertIsNotNone(target, command)
            (fake_bin / command).symlink_to(target)
        real_minisign = shutil.which("minisign")
        self.assertIsNotNone(real_minisign)
        shim = fake_bin / "minisign"
        shim.write_text(f'#!/bin/sh\nrm -f "$0"\nexec "{real_minisign}" "$@"\n', encoding="utf-8")
        shim.chmod(0o755)

        proc, prefix, requests = fixture.run(
            "--components", "journal", "--dry-run", path=str(fake_bin)
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["root_code"], "verifier-missing")
        self.assertIn(f"/solstone/release/{PLATFORM_VERSION}/platform.json.minisig", requests)
        self.assertTrue(any(path.endswith(".manifest.json.minisig") for path in requests))
        self.assertFalse(any("install.sh" in path for path in requests))
        self.assertFalse(prefix.exists())

    def test_native_authority_file_caps_accept_n_and_refuse_n_plus_one(self):
        manifest_limit = 65536
        small_limit = 16384

        fixture = self.fixture("cap-manifest")
        manifest_path = fixture.native_path("x86_64", ".manifest.json")
        manifest_path.write_bytes(manifest_path.read_bytes() + b" " * (manifest_limit - manifest_path.stat().st_size))
        self.assertEqual(manifest_path.stat().st_size, manifest_limit)
        fixture.resign_native("x86_64")
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "journal", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
        fixture.resign_native("x86_64")
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "journal", "--dry-run")
        self.assertEqual(json.loads(proc.stdout)["root_code"], "response-too-large")

        fixture = self.fixture("cap-signature")
        signature = fixture.native_path("x86_64", ".manifest.json.minisig")
        lines = signature.read_bytes().splitlines(keepends=True)
        lines[0] = lines[0].rstrip(b"\n") + b"x" * (small_limit - signature.stat().st_size) + b"\n"
        signature.write_bytes(b"".join(lines))
        self.assertEqual(signature.stat().st_size, small_limit)
        fixture.refresh_outer("x86_64")
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "journal", "--dry-run")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["root_code"], "signature-invalid")
        lines = signature.read_bytes().splitlines(keepends=True)
        lines[0] = lines[0].rstrip(b"\n") + b"x\n"
        signature.write_bytes(b"".join(lines))
        fixture.refresh_outer("x86_64")
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "journal", "--dry-run")
        self.assertEqual(json.loads(proc.stdout)["root_code"], "response-too-large")

        fixture = self.fixture("cap-release")
        release = fixture.native_path("x86_64", ".release")
        release_bytes = release.read_bytes()
        padded_epoch = b"journal-v2" + b"e" * (small_limit - len(release_bytes))
        release.write_bytes(release_bytes.replace(b"upgrade_epoch=journal-v2", b"upgrade_epoch=" + padded_epoch, 1))
        fixture.platform["components"]["journal"]["provenance"]["upgrade_epoch"] = padded_epoch.decode("ascii")
        self.assertEqual(release.stat().st_size, small_limit)
        fixture.rebind_release("x86_64")
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "journal", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        release.write_bytes(release.read_bytes().replace(b"upgrade_epoch=", b"upgrade_epoch=x", 1))
        fixture.platform["components"]["journal"]["provenance"]["upgrade_epoch"] = "x" + padded_epoch.decode("ascii")
        fixture.rebind_release("x86_64")
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "journal", "--dry-run")
        self.assertEqual(json.loads(proc.stdout)["root_code"], "response-too-large")

        fixture = self.fixture("cap-sums")
        sums = fixture.native_path("x86_64", ".sha256")
        expanded = bytearray(sums.read_bytes())
        index = 0
        while manifest_limit - len(expanded) > 167:
            name = f"extra-{index:06d}".ljust(33, "x")
            expanded.extend(("0" * 64 + "  " + name + "\n").encode("ascii"))
            index += 1
        remaining = manifest_limit - len(expanded)
        final_name_len = remaining - 67
        self.assertGreaterEqual(final_name_len, 1)
        final_name = ("final" + "z" * final_name_len)[:final_name_len]
        expanded.extend(("0" * 64 + "  " + final_name + "\n").encode("ascii"))
        sums.write_bytes(bytes(expanded))
        self.assertEqual(sums.stat().st_size, manifest_limit)
        fixture.rebind_sums("x86_64")
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "journal", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        sums.write_bytes(sums.read_bytes() + b" ")
        fixture.rebind_sums("x86_64")
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "journal", "--dry-run")
        self.assertEqual(json.loads(proc.stdout)["root_code"], "response-too-large")


if __name__ == "__main__":
    unittest.main()
