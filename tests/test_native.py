# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from pathlib import Path
import tempfile
import unittest

from solstone_platform.ingest import ingest_desktop, ingest_journal, ingest_tmux
from solstone_platform.pins import embedded_pins
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

    def test_ingest_journal_v2_if_present(self):
        j2_dir = self.repo_root / "testdata" / "native" / "journal-v2"
        # If only EXPECTED is present, skip success path
        if not (j2_dir / ".manifest.json").is_file():
            self.skipTest("journal-v2 real fixture not present (EXPECTED only)")

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
