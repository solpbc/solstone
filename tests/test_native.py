# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path
import shutil
import tempfile
import unittest

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
from solstone_platform.refusals import (
    RELEASE_COHERENCE,
    SCHEMA_INVALID,
    SIGNATURE_PIN_MISMATCH,
    URL_FRAGMENT,
    URL_INSECURE,
    URL_USERINFO,
    Refusal,
)


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


if __name__ == "__main__":
    unittest.main()
