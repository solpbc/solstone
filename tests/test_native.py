# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from solstone_platform.canonical import canonical_json_bytes
from solstone_platform.generate import generate_platform_manifest

from solstone_platform.ingest import (
    _parse_tmux_checksums,
    _parse_tmux_target_artifacts,
    _require_exact_artifact_names,
    _validate_desktop_artifacts,
    ingest_desktop,
    ingest_journal,
    ingest_tmux,
)
from solstone_platform.pins import embedded_pins, load_pin_file
from solstone_platform.schema import load_platform_manifest_bytes
from solstone_platform.sign import ephemeral_keypair
from solstone_platform.refusals import (
    RELEASE_COHERENCE,
    SCHEMA_INVALID,
    SIGNATURE_PIN_MISMATCH,
    URL_FRAGMENT,
    URL_INSECURE,
    URL_USERINFO,
    Refusal,
)
from tools.fixture_builder import build_tiny_natives


class TestNativeIngest(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent
        self.pins = embedded_pins()

    def test_ingest_checked_in_desktop(self):
        desktop_dir = self.repo_root / "testdata" / "native" / "desktop" / "2.0.3"
        comp = ingest_desktop(desktop_dir, self.pins.desktop)
        self.assertEqual(comp.name, "desktop")
        self.assertEqual(comp.version, "2.0.3")
        self.assertIn("x86_64", comp.arches)
        self.assertNotIn("aarch64", comp.arches)
        self.assertEqual(len(comp.arches["x86_64"]), 3)  # tree, deb, rpm

    def test_ingest_checked_in_tmux(self):
        tmux_dir = self.repo_root / "testdata" / "native" / "tmux" / "2.0.3"
        comp = ingest_tmux(tmux_dir, self.pins.tmux)
        self.assertEqual(comp.name, "tmux")
        self.assertEqual(comp.version, "2.0.3")
        self.assertIn("x86_64", comp.arches)
        self.assertIn("aarch64", comp.arches)
        self.assertEqual(len(comp.arches["x86_64"]), 3)
        self.assertEqual(len(comp.arches["aarch64"]), 3)

    def test_ingest_desktop_with_wrong_pin_fails(self):
        desktop_dir = self.repo_root / "testdata" / "native" / "desktop" / "2.0.3"
        with self.assertRaises(Refusal) as ctx:
            ingest_desktop(desktop_dir, self.pins.tmux)
        self.assertEqual(ctx.exception.name, SIGNATURE_PIN_MISMATCH)

    def test_ingest_journal_2_0_6_metadata_refuses_coherence(self):
        journal_dir = self.repo_root / "testdata" / "native" / "journal" / "2.0.6"
        if journal_dir.is_dir():
            # Ingesting real 2.0.6 metadata should verify signature and fail coherence due to missing state_reader fields
            with self.assertRaises(Refusal) as ctx:
                ingest_journal(
                    native_dir=journal_dir,
                    lane="release",
                    origin="https://updates.solstone.app",
                    bootstrap_file=None,
                    pin=self.pins.journal,
                )
            self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_ingest_journal_escaped_coordinate_userinfo(self):
        journal_dir = self.repo_root / "testdata" / "native" / "journal" / "2.0.6"
        if journal_dir.is_dir():
            with self.assertRaises(Refusal) as ctx:
                ingest_journal(
                    native_dir=journal_dir,
                    lane="release",
                    origin="https://user:pass@127.0.0.1",
                    bootstrap_file=None,
                    pin=self.pins.journal,
                )
            self.assertEqual(ctx.exception.name, URL_USERINFO)

    def test_ingest_journal_escaped_coordinate_fragment(self):
        journal_dir = self.repo_root / "testdata" / "native" / "journal" / "2.0.6"
        if journal_dir.is_dir():
            with self.assertRaises(Refusal) as ctx:
                ingest_journal(
                    native_dir=journal_dir,
                    lane="release",
                    origin="https://127.0.0.1/base#frag",
                    bootstrap_file=None,
                    pin=self.pins.journal,
                )
            self.assertEqual(ctx.exception.name, URL_FRAGMENT)

    def test_ingest_journal_v2_producer_fixture(self):
        j2_dir = self.repo_root / "testdata" / "native" / "journal-v2"
        fixture_pin = load_pin_file(j2_dir / "fixture.pub")
        comp = ingest_journal(
            native_dir=j2_dir,
            lane="release",
            origin="https://updates.solstone.app",
            bootstrap_file=None,
            pin=fixture_pin,
        )
        self.assertEqual(comp.name, "journal")
        self.assertEqual(comp.version, "2.0.8")
        self.assertEqual(set(comp.arches), {"x86_64", "aarch64"})
        self.assertEqual(comp.provenance["bootstrap"]["contract_version"], 2)
        self.assertEqual(
            comp.provenance["bootstrap"]["url"],
            "https://updates.solstone.app/solstone-journal/release/2.0.8/solstone-journal-2.0.8-install.sh",
        )
        self.assertEqual(comp.provenance["upgrade_epoch"], "journal-v2")
        self.assertEqual(comp.provenance["state_reader_min"], "1.0.0")
        self.assertEqual(comp.provenance["state_reader_max"], "2.0.8")
        self.assertEqual(comp.provenance["retention_window"], 3)

    def test_ingest_journal_v2_fixture_refuses_production_pin(self):
        j2_dir = self.repo_root / "testdata" / "native" / "journal-v2"
        with self.assertRaises(Refusal) as ctx:
            ingest_journal(
                native_dir=j2_dir,
                lane="release",
                origin="https://updates.solstone.app",
                bootstrap_file=None,
                pin=self.pins.journal,
            )
        self.assertEqual(ctx.exception.name, SIGNATURE_PIN_MISMATCH)

    def test_ingest_journal_v2_refuses_signed_manifest_tamper(self):
        source = self.repo_root / "testdata" / "native" / "journal-v2"
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = Path(tmp_dir) / "journal-v2"
            shutil.copytree(source, fixture)
            manifest = fixture / "linux-x86_64" / "solstone-journal-2.0.8-linux-x86_64.manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b" ")
            with self.assertRaises(Refusal) as ctx:
                ingest_journal(
                    native_dir=fixture,
                    lane="release",
                    origin="https://updates.solstone.app",
                    bootstrap_file=None,
                    pin=load_pin_file(fixture / "fixture.pub"),
                )
            self.assertEqual(ctx.exception.name, SIGNATURE_PIN_MISMATCH)

    def test_ingest_journal_v2_refuses_bootstrap_tamper(self):
        source = self.repo_root / "testdata" / "native" / "journal-v2"
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = Path(tmp_dir) / "journal-v2"
            shutil.copytree(source, fixture)
            bootstrap = fixture / "linux-aarch64" / "solstone-journal-2.0.8-install.sh"
            bootstrap.write_bytes(bootstrap.read_bytes() + b"\n# tampered\n")
            with self.assertRaises(Refusal) as ctx:
                ingest_journal(
                    native_dir=fixture,
                    lane="release",
                    origin="https://updates.solstone.app",
                    bootstrap_file=None,
                    pin=load_pin_file(fixture / "fixture.pub"),
                )
            self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_ingest_journal_v2_refuses_unbound_sidecar_bytes(self):
        source = self.repo_root / "testdata" / "native" / "journal-v2"
        with tempfile.TemporaryDirectory() as tmp_dir:
            fixture = Path(tmp_dir) / "journal-v2"
            shutil.copytree(source, fixture)
            sidecar = fixture / "linux-x86_64" / "solstone-journal-2.0.8-linux-x86_64.sha256"
            sidecar.write_bytes(sidecar.read_bytes() + b"# semantically empty tamper\n")
            with self.assertRaises(Refusal) as ctx:
                ingest_journal(
                    native_dir=fixture,
                    lane="release",
                    origin="https://updates.solstone.app",
                    bootstrap_file=None,
                    pin=load_pin_file(fixture / "fixture.pub"),
                )
            self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_native_metadata_selection_refuses_ambiguity(self):
        cases = [
            (
                "journal-v2",
                self.repo_root / "testdata" / "native" / "journal-v2",
                "linux-x86_64/solstone-journal-2.0.8-linux-x86_64.manifest.json",
                "linux-x86_64/solstone-journal-9.9.9-linux-x86_64.manifest.json",
            ),
            (
                "desktop",
                self.repo_root / "testdata" / "native" / "desktop" / "2.0.3",
                "solstone-linux-2.0.3-linux-x86_64.rust-release-manifest.json",
                "solstone-linux-9.9.9-linux-x86_64.rust-release-manifest.json",
            ),
            (
                "tmux",
                self.repo_root / "testdata" / "native" / "tmux" / "2.0.3",
                "solstone-tmux-2.0.3-x86_64-unknown-linux-musl.target.json",
                "solstone-tmux-9.9.9-x86_64-unknown-linux-musl.target.json",
            ),
        ]
        for component, source, original, duplicate in cases:
            with self.subTest(component=component), tempfile.TemporaryDirectory() as tmp_dir:
                fixture = Path(tmp_dir) / component
                shutil.copytree(source, fixture)
                duplicate_path = fixture / duplicate
                duplicate_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fixture / original, duplicate_path)
                with self.assertRaises(Refusal) as ctx:
                    if component == "journal-v2":
                        ingest_journal(
                            native_dir=fixture,
                            lane="release",
                            origin="https://updates.solstone.app",
                            bootstrap_file=None,
                            pin=load_pin_file(fixture / "fixture.pub"),
                        )
                    elif component == "desktop":
                        ingest_desktop(fixture, self.pins.desktop)
                    else:
                        ingest_tmux(fixture, self.pins.tmux)
                self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_duplicate_digests_and_lengths_are_allowed_for_distinct_names(self):
        digest = "a" * 64
        _validate_desktop_artifacts(
            [
                {"path": "one.tar.gz", "sha256": digest, "bytes": 12},
                {"path": "two.deb", "sha256": digest, "bytes": 12},
            ]
        )
        self.assertEqual(
            _parse_tmux_target_artifacts(
                [
                    {"name": "one.tar.gz", "sha256": digest},
                    {"name": "two.deb", "sha256": digest},
                ]
            ),
            {"one.tar.gz": digest, "two.deb": digest},
        )
        self.assertEqual(
            _parse_tmux_checksums(f"{digest}  one.tar.gz\n{digest}  two.deb\n".encode()),
            {"one.tar.gz": digest, "two.deb": digest},
        )

    def test_native_artifact_sets_refuse_missing_or_extra_members(self):
        expected = {"one.tar.gz", "two.deb", "three.rpm"}
        _require_exact_artifact_names(expected, expected, "fixture")
        for actual in [expected - {"two.deb"}, expected | {"extra.tar.gz"}]:
            with self.subTest(actual=actual), self.assertRaises(Refusal) as ctx:
                _require_exact_artifact_names(actual, expected, "fixture")
            self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_tmux_checksums_refuse_malformed_and_duplicate_names(self):
        digest = "b" * 64
        invalid_inputs = [
            b"not-a-checksum-line\n",
            b"z" * 64 + b"  artifact.tar.gz\n",
            f"{digest}  ../artifact.tar.gz\n".encode(),
            f"{digest}  artifact.tar.gz\n{digest}  artifact.tar.gz\n".encode(),
        ]
        for content in invalid_inputs:
            with self.subTest(content=content), self.assertRaises(Refusal) as ctx:
                _parse_tmux_checksums(content)
            self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_ingest_unsigned_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            manifest = tmp_p / ".rust-release-manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaises(Refusal) as ctx:
                ingest_desktop(tmp_p, self.pins.desktop)
            self.assertIn(ctx.exception.name, (SIGNATURE_PIN_MISMATCH, SCHEMA_INVALID, RELEASE_COHERENCE))


class TestDesktopAarch64Ingest(unittest.TestCase):
    """Desktop carries x86_64 always and aarch64 when the release has a signed aarch64 set."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        keys = ephemeral_keypair("desktop aarch64 ingest")
        self.secret, _, self.pin = keys.__enter__()
        self.addCleanup(keys.__exit__, None, None, None)
        self.addCleanup(self.temp.cleanup)

    def build(self, arches=("x86_64", "aarch64"), label="natives"):
        @contextmanager
        def retained(*args, **kwargs):
            yield self.secret, None, self.pin

        with patch("tools.fixture_builder.ephemeral_keypair", retained):
            pins, dirs = build_tiny_natives(self.root / label, desktop_arches=arches)
        return pins, dirs

    def manifest(self, desktop: Path, arch: str) -> Path:
        return desktop / f"solstone-linux-2.0.3-linux-{arch}.rust-release-manifest.json"

    def rewrite(self, desktop: Path, arch: str, change) -> None:
        path = self.manifest(desktop, arch)
        obj = json.loads(path.read_text(encoding="utf-8"))
        change(obj)
        path.write_bytes(canonical_json_bytes(obj))
        subprocess.run(["minisign", "-S", "-W", "-s", str(self.secret), "-m", str(path),
                        "-x", str(path) + ".minisig"], check=True, capture_output=True)

    def refuses(self, desktop: Path, code: str = RELEASE_COHERENCE) -> None:
        with self.assertRaises(Refusal) as ctx:
            ingest_desktop(desktop, self.pin)
        self.assertEqual(ctx.exception.name, code, str(ctx.exception))

    def test_both_arches_ingest_with_their_own_names_and_authority(self):
        pins, dirs = self.build()
        comp = ingest_desktop(dirs["desktop"], self.pin)
        self.assertEqual(set(comp.arches), {"x86_64", "aarch64"})
        arm = comp.arches["aarch64"]
        self.assertEqual(
            {route: entry["filename"] for route, entry in arm.items()},
            {
                "tree": "solstone-linux-2.0.3-linux-aarch64.tar.gz",
                "deb": "solstone-linux_2.0.3-1_arm64.deb",
                "rpm": "solstone-linux-2.0.3-1.aarch64.rpm",
            },
        )
        self.assertEqual({entry["native_target"] for entry in arm.values()}, {"linux-aarch64"})
        self.assertEqual(arm["deb"]["package_identity"]["arch"], "arm64")
        self.assertEqual(arm["rpm"]["package_identity"]["arch"], "aarch64")
        self.assertNotEqual(arm["tree"]["authority"]["manifest_sha256"], comp.arches["x86_64"]["tree"]["authority"]["manifest_sha256"])

        manifest = generate_platform_manifest(
            version="2.0.3", lane="release", created_unix=1700000000,
            source_commit="3075c36b12fad469d4c9c0ab4555908fe8ecca1b", platform_key_id=self.pin.key_id,
            repo_root=Path(__file__).parent.parent, journal_dir=dirs["journal"], desktop_dir=dirs["desktop"],
            tmux_dir=dirs["tmux"], journal_origin="https://updates.solstone.app", pins=pins,
        )
        obj = load_platform_manifest_bytes(manifest)
        self.assertEqual(set(obj["components"]["desktop"]["arches"]), {"x86_64", "aarch64"})

    def test_x86_64_only_release_stays_x86_64_only(self):
        _, dirs = self.build(arches=("x86_64",))
        self.assertEqual(set(ingest_desktop(dirs["desktop"], self.pin).arches), {"x86_64"})

    def test_x86_64_is_required(self):
        _, dirs = self.build(arches=("aarch64",))
        self.refuses(dirs["desktop"])

    def test_aarch64_payloads_without_signed_manifest_refuse(self):
        _, dirs = self.build()
        for name in ("manifest", "signature"):
            with self.subTest(missing=name):
                desktop = self.root / f"stray-{name}"
                shutil.copytree(dirs["desktop"], desktop)
                manifest = self.manifest(desktop, "aarch64")
                manifest.with_name(manifest.name + ".minisig").unlink()
                if name == "manifest":
                    manifest.unlink()
                self.refuses(desktop)

    def test_aarch64_set_is_verified_like_x86_64(self):
        _, dirs = self.build()
        cases = {
            "x86-triple": lambda o: o["target"].__setitem__("triple", "x86_64-unknown-linux-gnu"),
            "version": lambda o: o.__setitem__("version", "2.0.4"),
            "commit": lambda o: o.__setitem__("source_commit", "b" * 40),
            "dirty": lambda o: o.__setitem__("source_dirty", True),
            "amd64-names": lambda o: [item.__setitem__("path", item["path"].replace("arm64", "amd64").replace("aarch64", "x86_64")) for item in o["artifacts"]],
            "digest": lambda o: o["artifacts"][0].__setitem__("sha256", "f" * 64),
        }
        for label, change in cases.items():
            with self.subTest(case=label):
                desktop = self.root / f"mutated-{label}"
                shutil.copytree(dirs["desktop"], desktop)
                self.rewrite(desktop, "aarch64", change)
                self.refuses(desktop)

        desktop = self.root / "foreign-signature"
        shutil.copytree(dirs["desktop"], desktop)
        with ephemeral_keypair("foreign desktop key") as (foreign, _, _):
            manifest = self.manifest(desktop, "aarch64")
            subprocess.run(["minisign", "-S", "-W", "-s", str(foreign), "-m", str(manifest),
                            "-x", str(manifest) + ".minisig"], check=True, capture_output=True)
        self.refuses(desktop, SIGNATURE_PIN_MISMATCH)

        desktop = self.root / "payload-bytes"
        shutil.copytree(dirs["desktop"], desktop)
        deb = desktop / "solstone-linux_2.0.3-1_arm64.deb"
        deb.write_bytes(deb.read_bytes() + b"x")
        self.refuses(desktop)


if __name__ == "__main__":
    unittest.main()
