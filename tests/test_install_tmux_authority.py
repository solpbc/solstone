# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Installer-side verification of Tmux's native release authority."""

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
from solstone_platform.pins import TMUX_KEY_ID, TMUX_PUBKEY, PinSet, embedded_pins
from solstone_platform.sign import ephemeral_keypair, sign_manifest
from tests.install_test_helpers import LoopbackServer, setup_fake_sudo
from tools.build_installer import build_installer
from tools.fixture_builder import build_tiny_natives


REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_VERSION = "2.0.0"
TMUX_VERSION = "2.0.3"
HELPER_SCRIPT = REPO_ROOT / "helpers" / "solstone-pkg-helper.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TmuxFixture:
    def __init__(self, root: Path, platform_sec: Path, platform_pub: Path, platform_pin, tmux_sec: Path | None, tmux_pin, real: bool = False):
        self.root = root
        self.platform_sec = platform_sec
        self.platform_pin = platform_pin
        self.tmux_sec = tmux_sec
        self.tmux_pin = tmux_pin
        self.web_root = root / "www"
        self.server = LoopbackServer(self.web_root)
        self.server.start()

        tiny_pins, native_dirs = build_tiny_natives(root / "natives")
        if real:
            self.tmux_dir = REPO_ROOT / "testdata" / "native" / "tmux" / TMUX_VERSION
        else:
            self.tmux_dir = native_dirs["tmux"]
            self._sign_file(self.tmux_dir / "SHA256SUMS", self.tmux_dir / "SHA256SUMS.minisig")
        pins = PinSet(journal=tiny_pins.journal, desktop=tiny_pins.desktop, tmux=tmux_pin)
        manifest_bytes = generate_platform_manifest(
            version=PLATFORM_VERSION,
            lane="release",
            created_unix=1773820000,
            source_commit="3075c36b12fad469d4c9c0ab4555908fe8ecca1b",
            platform_key_id=platform_pin.key_id,
            repo_root=REPO_ROOT,
            journal_dir=native_dirs["journal"],
            desktop_dir=native_dirs["desktop"],
            tmux_dir=self.tmux_dir,
            journal_origin=self.server.origin,
            pins=pins,
        )
        self.platform = json.loads(manifest_bytes.decode("utf-8"))
        self.version_dir = self.web_root / "solstone" / "release" / PLATFORM_VERSION
        self.version_dir.mkdir(parents=True)
        (self.version_dir.parent / "latest").write_text(f"{PLATFORM_VERSION}\n", encoding="utf-8")
        for source in self.tmux_dir.iterdir():
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
            rendered = rendered.replace(f'TMUX_KEY_ID="{TMUX_KEY_ID}"', f'TMUX_KEY_ID="{tmux_pin.key_id}"')
            rendered = rendered.replace(f'TMUX_PUBKEY="{TMUX_PUBKEY}"', f'TMUX_PUBKEY="{tmux_pin.pubkey}"')
            self.installer.write_text(rendered, encoding="utf-8")
        self.installer.chmod(0o755)

    def close(self):
        self.server.stop()

    @property
    def sums(self) -> Path:
        return self.version_dir / "SHA256SUMS"

    @property
    def signature(self) -> Path:
        return self.version_dir / "SHA256SUMS.minisig"

    def target(self, arch: str) -> Path:
        rust_target = f"{arch}-unknown-linux-musl"
        return self.version_dir / f"solstone-tmux-{TMUX_VERSION}-{rust_target}.target.json"

    def _sign_file(self, message: Path, signature: Path):
        if self.tmux_sec is None:
            raise AssertionError("real producer fixture cannot be re-signed")
        subprocess.run(
            [
                "minisign", "-S", "-W", "-s", str(self.tmux_sec),
                "-m", str(message), "-x", str(signature),
                "-t", "solstone-tmux test SHA256SUMS",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def sign_sums(self):
        self._sign_file(self.sums, self.signature)

    def refresh_outer(self):
        for arch, routes in self.platform["components"]["tmux"]["arches"].items():
            target_sha = sha256(self.target(arch))
            for route in routes.values():
                route["authority"]["sums_sha256"] = sha256(self.sums)
                route["authority"]["signature_sha256"] = sha256(self.signature)
                route["authority"]["target_json_sha256"] = target_sha

    def write_target(self, arch: str, obj: dict):
        self.write_target_bytes(arch, canonical_json_bytes(obj))

    def write_target_bytes(self, arch: str, content: bytes):
        target = self.target(arch)
        target.write_bytes(content)
        target_sha = sha256(target)
        lines = self.sums.read_text(encoding="utf-8").splitlines()
        rebound = [f"{target_sha}  {target.name}" if line[66:] == target.name else line for line in lines]
        self.sums.write_text("\n".join(rebound) + "\n", encoding="utf-8")
        self.sign_sums()
        self.refresh_outer()
        self.write_platform()

    def write_sums(self, content: bytes):
        self.sums.write_bytes(content)
        self.sign_sums()
        self.refresh_outer()
        self.write_platform()

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

    def run(self, *args: str, arch: str = "x86_64", path: str | None = None, extra_env: dict[str, str] | None = None):
        env = os.environ.copy()
        if arch == "aarch64":
            env["SOLSTONE_TEST_HOST_ARCH"] = "aarch64"
            if path is not None:
                env["PATH"] = path
        elif path is not None:
            env["PATH"] = path
        if extra_env:
            env.update(extra_env)
        prefix = self.root / f"prefix-{arch}-{len(self.server.request_paths)}"
        self.server.request_paths.clear()
        proc = subprocess.run(
            [str(self.installer), *args, "--prefix", str(prefix), "--json"],
            capture_output=True,
            text=True,
            env=env,
        )
        return proc, prefix, list(self.server.request_paths)


class TestInstallTmuxAuthority(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, label: str, real: bool = False) -> TmuxFixture:
        platform_keys = ephemeral_keypair(f"tmux platform {label}")
        p_sec, p_pub, p_pin = platform_keys.__enter__()
        self.addCleanup(platform_keys.__exit__, None, None, None)
        if real:
            t_sec = None
            t_pin = embedded_pins().tmux
        else:
            tmux_keys = ephemeral_keypair(f"tmux native {label}")
            t_sec, _, t_pin = tmux_keys.__enter__()
            self.addCleanup(tmux_keys.__exit__, None, None, None)
        fixture = TmuxFixture(self.work_dir / label, p_sec, p_pub, p_pin, t_sec, t_pin, real=real)
        self.addCleanup(fixture.close)
        return fixture

    @staticmethod
    def result(proc: subprocess.CompletedProcess) -> dict:
        return json.loads(proc.stdout.strip())

    @staticmethod
    def selected_artifact(obj: dict, route: str = "tree") -> dict:
        suffix = {"tree": ".tar.gz", "deb": ".deb", "rpm": ".rpm"}[route]
        return next(item for item in obj["artifacts"] if item["name"].endswith(suffix))

    def test_real_producer_bytes_and_exact_publish_paths(self):
        fixture = self.fixture("real", real=True)
        for arch in ("x86_64", "aarch64"):
            with self.subTest(arch=arch):
                proc, prefix, requests = fixture.run("--components", "tmux", "--dry-run", arch=arch)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                base = f"/solstone/release/{PLATFORM_VERSION}"
                target = f"solstone-tmux-{TMUX_VERSION}-{arch}-unknown-linux-musl.target.json"
                self.assertEqual(
                    requests,
                    [
                        "/solstone/release/latest",
                        f"{base}/platform.json",
                        f"{base}/platform.json.minisig",
                        f"{base}/SHA256SUMS",
                        f"{base}/SHA256SUMS.minisig",
                        f"{base}/{target}",
                    ],
                )
                self.assertFalse(prefix.exists())

    def test_package_postprocessing_does_not_require_raw_binary_digest(self):
        fixture = self.fixture("package-postprocessing")
        target = json.loads(fixture.target("x86_64").read_text())
        target["executable"]["sha256"] = "a" * 64
        fixture.write_target("x86_64", target)
        package, _, _ = fixture.run("--components", "tmux", "--route", "rpm", "--dry-run")
        self.assertEqual(package.returncode, 0, package.stderr + package.stdout)
        tree, _, _ = fixture.run("--components", "tmux", "--route", "tree", "--dry-run")
        self.assertEqual(self.result(tree)["root_code"], "release-coherence")
        target["executable"]["sha256"] = "not-a-digest"
        fixture.write_target("x86_64", target)
        invalid, _, _ = fixture.run("--components", "tmux", "--route", "rpm", "--dry-run")
        self.assertEqual(self.result(invalid)["root_code"], "schema-invalid")

    def test_digest_checks_precede_native_consumers(self):
        fixture = self.fixture("digest-order")
        shim_dir = self.work_dir / "digest-shims"
        shim_dir.mkdir()
        mini_log = self.work_dir / "minisign.log"
        awk_log = self.work_dir / "awk.log"
        real_minisign = shutil.which("minisign")
        real_awk = shutil.which("awk")
        (shim_dir / "minisign").write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{mini_log}"\nexec "{real_minisign}" "$@"\n', encoding="utf-8")
        (shim_dir / "awk").write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{awk_log}"\nexec "{real_awk}" "$@"\n', encoding="utf-8")
        (shim_dir / "minisign").chmod(0o755)
        (shim_dir / "awk").chmod(0o755)
        path = f"{shim_dir}:{os.environ['PATH']}"

        authority = fixture.platform["components"]["tmux"]["arches"]["x86_64"]["tree"]["authority"]
        for field in ("sums_sha256", "signature_sha256", "target_json_sha256"):
            with self.subTest(field=field):
                original = authority[field]
                authority[field] = "0" * 64
                fixture.write_platform()
                proc, _, _ = fixture.run("--components", "tmux", "--dry-run", path=path)
                self.assertEqual(self.result(proc)["root_code"], "digest-mismatch")
                self.assertEqual(len(mini_log.read_text().splitlines()), 1)
                native_lines = [line for line in awk_log.read_text().splitlines() if "parse_tmux_sums" in line or "schema_mode=tmux" in line]
                self.assertEqual(native_lines, [])
                authority[field] = original
                mini_log.unlink()
                awk_log.unlink()

        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "tmux", "--dry-run", path=path)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(len(mini_log.read_text().splitlines()), 2)
        native_lines = [line for line in awk_log.read_text().splitlines() if "parse_tmux_sums" in line or "schema_mode=tmux" in line]
        self.assertEqual(len(native_lines), 2)

        mini_log.unlink()
        awk_log.unlink()
        authority["sums_sha256"] = "0" * 64
        fixture.write_platform()
        proc, _, _ = fixture.run("--skip-signature", "--components", "tmux", "--dry-run", path=path)
        self.assertEqual(self.result(proc)["root_code"], "digest-mismatch")
        self.assertFalse(mini_log.exists())
        native_lines = [line for line in awk_log.read_text().splitlines() if "parse_tmux_sums" in line or "schema_mode=tmux" in line]
        self.assertEqual(native_lines, [])

    def test_signature_and_verifier_stage(self):
        fixture = self.fixture("signature")
        fixture.signature.write_bytes(b"not a minisign signature\n")
        fixture.refresh_outer()
        fixture.write_platform()
        proc, prefix, requests = fixture.run("--components", "tmux", "--dry-run")
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
        (fake_bin / "minisign").write_text(f'#!/bin/sh\nrm -f "$0"\nexec "{real_minisign}" "$@"\n', encoding="utf-8")
        (fake_bin / "minisign").chmod(0o755)
        proc, prefix, requests = fixture.run("--components", "tmux", "--dry-run", path=str(fake_bin))
        self.assertEqual(self.result(proc)["root_code"], "verifier-missing")
        self.assertTrue(any(path.endswith("SHA256SUMS.minisig") for path in requests))
        self.assertFalse(any(path.endswith((".tar.gz", ".deb", ".rpm")) for path in requests))
        self.assertFalse(prefix.exists())

    def test_target_types_and_semantics(self):
        mutations = {
            "schema-value": (lambda o: o.__setitem__("schema_version", 2), "schema-invalid"),
            "schema-type": (lambda o: o.__setitem__("schema_version", "1"), "schema-invalid"),
            "version": (lambda o: o.__setitem__("product_version", "9.9.9"), "release-coherence"),
            "version-type": (lambda o: o.__setitem__("product_version", 7), "schema-invalid"),
            "commit-value": (lambda o: o.__setitem__("source_commit", "a" * 39), "schema-invalid"),
            "commit-type": (lambda o: o.__setitem__("source_commit", 7), "schema-invalid"),
            "target": (lambda o: o.__setitem__("rust_target", "aarch64-unknown-linux-musl"), "release-coherence"),
            "target-type": (lambda o: o.__setitem__("rust_target", 7), "schema-invalid"),
            "executable-type": (lambda o: o.__setitem__("executable", []), "schema-invalid"),
            "exec-name": (lambda o: o["executable"].__setitem__("name", "other"), "release-coherence"),
            "exec-name-type": (lambda o: o["executable"].__setitem__("name", 7), "schema-invalid"),
            "exec-sha": (lambda o: o["executable"].__setitem__("sha256", "f" * 64), "release-coherence"),
            "exec-sha-type": (lambda o: o["executable"].__setitem__("sha256", 7), "schema-invalid"),
            "artifacts-type": (lambda o: o.__setitem__("artifacts", {}), "schema-invalid"),
            "member-type": (lambda o: o["artifacts"].__setitem__(0, "not-an-object"), "schema-invalid"),
            "name-type": (lambda o: self.selected_artifact(o).__setitem__("name", 7), "schema-invalid"),
            "sha-type": (lambda o: self.selected_artifact(o).__setitem__("sha256", 7), "schema-invalid"),
        }
        for label, (mutate, code) in mutations.items():
            with self.subTest(label=label):
                fixture = self.fixture(f"typed-{label}")
                obj = json.loads(fixture.target("x86_64").read_text(encoding="utf-8"))
                mutate(obj)
                fixture.write_target("x86_64", obj)
                proc, prefix, requests = fixture.run("--components", "tmux", "--dry-run")
                self.assertEqual(self.result(proc)["root_code"], code, proc.stdout + proc.stderr)
                self.assertFalse(prefix.exists())
                self.assertFalse(any(path.endswith((".tar.gz", ".deb", ".rpm")) for path in requests))

    def test_signed_sums_strict_grammar(self):
        mutations = {
            "blank": lambda lines: lines.insert(1, ""),
            "one-space": lambda lines: lines.__setitem__(0, lines[0].replace("  ", " ", 1)),
            "tab": lambda lines: lines.__setitem__(0, lines[0].replace("  ", "\t", 1)),
            "uppercase": lambda lines: lines.__setitem__(0, lines[0][:64].upper() + lines[0][64:]),
            "unsafe": lambda lines: lines.__setitem__(0, lines[0][:66] + "../unsafe"),
            "malformed": lambda lines: lines.__setitem__(0, "g" + lines[0][1:]),
            "duplicate": lambda lines: lines.append(lines[0]),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                fixture = self.fixture(f"sums-{label}")
                lines = fixture.sums.read_text(encoding="utf-8").splitlines()
                mutate(lines)
                fixture.write_sums(("\n".join(lines) + "\n").encode("utf-8"))
                proc, _, _ = fixture.run("--components", "tmux", "--dry-run")
                self.assertEqual(self.result(proc)["root_code"], "schema-invalid")

    def test_duplicate_keys_members_routes_and_unique_twin(self):
        fixture = self.fixture("duplicate-key")
        target = fixture.target("x86_64")
        raw = target.read_bytes().replace(b'"schema_version":1,', b'"schema_version":1,"schema_version":1,', 1)
        target.write_bytes(raw)
        target_sha = sha256(target)
        lines = fixture.sums.read_text(encoding="utf-8").splitlines()
        fixture.sums.write_text("\n".join(f"{target_sha}  {target.name}" if line[66:] == target.name else line for line in lines) + "\n", encoding="utf-8")
        fixture.sign_sums()
        fixture.refresh_outer()
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "tmux", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "schema-invalid")

        for label, alter in (("exact", False), ("mixed", True)):
            with self.subTest(duplicate_member=label):
                fixture = self.fixture(f"duplicate-member-{label}")
                obj = json.loads(fixture.target("x86_64").read_text(encoding="utf-8"))
                duplicate = dict(self.selected_artifact(obj))
                if alter:
                    duplicate["sha256"] = "f" * 64
                obj["artifacts"].append(duplicate)
                fixture.write_target("x86_64", obj)
                proc, _, _ = fixture.run("--components", "tmux", "--dry-run")
                self.assertEqual(self.result(proc)["root_code"], "schema-invalid")

        for route in ("tree", "deb", "rpm"):
            with self.subTest(route=route):
                fixture = self.fixture(f"route-{route}")
                obj = json.loads(fixture.target("x86_64").read_text(encoding="utf-8"))
                selected = self.selected_artifact(obj, route)
                selected["name"] = next(item["name"] for item in obj["artifacts"] if item is not selected)
                fixture.write_target("x86_64", obj)
                proc, _, _ = fixture.run("--components", "tmux", "--dry-run", "--route", route)
                self.assertEqual(self.result(proc)["root_code"], "release-coherence")

        fixture = self.fixture("unique-twin")
        proc, _, _ = fixture.run("--components", "tmux", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

        fixture = self.fixture("escaped-key")
        raw = fixture.target("x86_64").read_bytes().replace(
            b'"product_version":"2.0.3",',
            b'"product_version":"2.0.3","product\\u005fversion":"2.0.3",',
            1,
        )
        fixture.write_target_bytes("x86_64", raw)
        proc, _, _ = fixture.run("--components", "tmux", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "schema-invalid")

        fixture = self.fixture("escaped-selected-name")
        obj = json.loads(fixture.target("x86_64").read_text(encoding="utf-8"))
        selected = dict(self.selected_artifact(obj))
        obj["artifacts"].append(selected)
        raw = canonical_json_bytes(obj)
        encoded_name = json.dumps(selected["name"], separators=(",", ":")).encode("utf-8")
        escaped_name = encoded_name.replace(b"-", b"\\u002d", 1)
        before, marker, after = raw.rpartition(encoded_name)
        self.assertTrue(marker)
        fixture.write_target_bytes("x86_64", before + escaped_name + after)
        proc, _, _ = fixture.run("--components", "tmux", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "schema-invalid")

    def test_skip_signature_keeps_pin_and_semantics(self):
        fixture = self.fixture("skip-semantic")
        obj = json.loads(fixture.target("x86_64").read_text(encoding="utf-8"))
        obj["product_version"] = "9.9.9"
        fixture.write_target("x86_64", obj)
        proc, _, _ = fixture.run("--skip-signature", "--components", "tmux", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "release-coherence")

        fixture = self.fixture("skip-pin")
        for route in fixture.platform["components"]["tmux"]["arches"]["x86_64"].values():
            route["authority"]["verifier_id"] = "minisign:0000000000000000"
        fixture.write_platform()
        proc, _, _ = fixture.run("--skip-signature", "--components", "tmux", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "pin-mismatch")

    def test_failure_is_pre_mutation_and_valid_package_reaches_helper(self):
        fixture = self.fixture("pre-mutation")
        obj = json.loads(fixture.target("x86_64").read_text(encoding="utf-8"))
        obj["product_version"] = "9.9.9"
        fixture.write_target("x86_64", obj)
        proc, prefix, requests = fixture.run("--components", "tmux")
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
            "--skip-signature", "--route", "deb", "--components", "tmux",
            path=f"{bin_dir}:{os.environ['PATH']}", extra_env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(any(path.endswith(".deb") for path in requests))
        self.assertTrue((fake_db / "install.log").is_file())

    def test_native_authority_caps(self):
        fixture = self.fixture("sums-cap")
        fixture.sums.write_bytes(fixture.sums.read_bytes() + b"\n" * (65536 - fixture.sums.stat().st_size))
        fixture.sign_sums()
        fixture.refresh_outer()
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "tmux", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "schema-invalid")
        fixture.sums.write_bytes(fixture.sums.read_bytes() + b"\n")
        fixture.sign_sums()
        fixture.refresh_outer()
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "tmux", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "response-too-large")

        fixture = self.fixture("target-cap")
        target = fixture.target("x86_64")
        target.write_bytes(target.read_bytes() + b" " * (65536 - target.stat().st_size))
        obj = json.loads(target.read_text(encoding="utf-8"))
        fixture.write_target("x86_64", obj)
        target.write_bytes(target.read_bytes() + b" " * (65536 - target.stat().st_size))
        target_sha = sha256(target)
        lines = fixture.sums.read_text(encoding="utf-8").splitlines()
        fixture.sums.write_text("\n".join(f"{target_sha}  {target.name}" if line[66:] == target.name else line for line in lines) + "\n", encoding="utf-8")
        fixture.sign_sums()
        fixture.refresh_outer()
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "tmux", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        target.write_bytes(target.read_bytes() + b" ")
        target_sha = sha256(target)
        lines = fixture.sums.read_text(encoding="utf-8").splitlines()
        fixture.sums.write_text("\n".join(f"{target_sha}  {target.name}" if line[66:] == target.name else line for line in lines) + "\n", encoding="utf-8")
        fixture.sign_sums()
        fixture.refresh_outer()
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "tmux", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "response-too-large")

        fixture = self.fixture("signature-cap")
        fixture.signature.write_bytes(fixture.signature.read_bytes() + b" " * (16384 - fixture.signature.stat().st_size))
        fixture.refresh_outer()
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "tmux", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        fixture.signature.write_bytes(fixture.signature.read_bytes() + b" ")
        fixture.refresh_outer()
        fixture.write_platform()
        proc, _, _ = fixture.run("--components", "tmux", "--dry-run")
        self.assertEqual(self.result(proc)["root_code"], "response-too-large")


if __name__ == "__main__":
    unittest.main()
