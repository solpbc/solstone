# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Unit tests for race-safe, content-blind source capture module."""

import os
from pathlib import Path
import shutil
import tempfile
import unittest

import solstone_platform.capture as capture
from solstone_platform.capture import (
    MAX_TOTAL_ENTRIES,
    capture_release_sources,
    last_private_parent,
)
from solstone_platform.refusals import (
    CAPTURE_DEPTH_EXCEEDED,
    CAPTURE_ENTRY_LIMIT_EXCEEDED,
    CAPTURE_HARDLINK_ALIAS,
    CAPTURE_READ_INTERRUPTED,
    CAPTURE_SIZE_LIMIT_EXCEEDED,
    CAPTURE_SPECIAL_FILE,
    CAPTURE_SYMLINK_TRAVERSAL,
    Refusal,
    SOURCE_CHANGED_DURING_CAPTURE,
)


class TestCapture(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Setup basic mock sources
        self.manifest_file = self.root / "platform.json"
        self.manifest_file.write_text('{"version": "1.0.0"}', encoding="utf-8")

        self.sig_file = self.root / "platform.json.minisig"
        self.sig_file.write_text("untrusted comment: sig\nABCD\n", encoding="utf-8")

        self.journal_dir = self.root / "journal"
        self.journal_dir.mkdir()
        (self.journal_dir / "solstone-journal-1.0.0-install.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (self.journal_dir / "unrelated.txt").write_text("some unrelated data", encoding="utf-8")

        j_x86 = self.journal_dir / "linux-x86_64"
        j_x86.mkdir()
        (j_x86 / "solstone-journal-1.0.0-linux-x86_64.release").write_text("commit=1234\n", encoding="utf-8")

        self.desktop_dir = self.root / "desktop"
        self.desktop_dir.mkdir()
        (self.desktop_dir / "solstone-linux-1.0.0-linux-x86_64.rust-release-manifest.json").write_text("{}", encoding="utf-8")
        (self.desktop_dir / "extra.dat").write_text("extra payload bytes", encoding="utf-8")

        self.tmux_dir = self.root / "tmux"
        self.tmux_dir.mkdir()
        (self.tmux_dir / "SHA256SUMS").write_text("hash  file\n", encoding="utf-8")

        self.bootstrap_file = self.root / "solstone-journal-1.0.0-install.sh"
        self.bootstrap_file.write_text("#!/bin/sh\necho test\n", encoding="utf-8")

        capture._between_inspect_and_open = None
        capture._interrupted_read_hook = None

    def tearDown(self):
        capture._between_inspect_and_open = None
        capture._interrupted_read_hook = None
        self.temp_dir.cleanup()

    def test_clean_capture_and_cleanup(self):
        snapshot = capture_release_sources(
            manifest_path=self.manifest_file,
            signature_path=self.sig_file,
            journal_dir=self.journal_dir,
            desktop_dir=self.desktop_dir,
            tmux_dir=self.tmux_dir,
            bootstrap_file=self.bootstrap_file,
        )
        snap_root = snapshot.snapshot_root
        self.assertTrue(snap_root.is_dir())
        self.assertEqual(last_private_parent(), snap_root)

        # Verify files captured in snapshot
        self.assertTrue(snapshot.manifest_path.is_file())
        self.assertTrue(snapshot.signature_path.is_file())
        self.assertTrue((snapshot.journal_dir / "unrelated.txt").is_file())
        self.assertTrue((snapshot.desktop_dir / "extra.dat").is_file())
        self.assertTrue((snapshot.journal_dir / "linux-x86_64" / "solstone-journal-1.0.0-linux-x86_64.release").is_file())
        self.assertTrue(snapshot.bootstrap_file.is_file())

        snapshot.cleanup()
        self.assertFalse(snap_root.exists())

    def test_refuse_symlink_in_tree(self):
        symlink_file = self.desktop_dir / "symlink.txt"
        symlink_file.symlink_to(self.manifest_file)

        with self.assertRaises(Refusal) as ctx:
            capture_release_sources(
                manifest_path=self.manifest_file,
                signature_path=self.sig_file,
                journal_dir=self.journal_dir,
                desktop_dir=self.desktop_dir,
                tmux_dir=self.tmux_dir,
            )
        self.assertEqual(ctx.exception.name, CAPTURE_SYMLINK_TRAVERSAL)
        self.assertFalse(last_private_parent().exists())

    def test_refuse_hardlink_alias(self):
        hardlink_file = self.desktop_dir / "hardlink.txt"
        os.link(self.manifest_file, hardlink_file)

        with self.assertRaises(Refusal) as ctx:
            capture_release_sources(
                manifest_path=self.manifest_file,
                signature_path=self.sig_file,
                journal_dir=self.journal_dir,
                desktop_dir=self.desktop_dir,
                tmux_dir=self.tmux_dir,
            )
        self.assertEqual(ctx.exception.name, CAPTURE_HARDLINK_ALIAS)
        self.assertFalse(last_private_parent().exists())

    def test_refuse_depth_exceeded_in_desktop(self):
        subdir = self.desktop_dir / "nested"
        subdir.mkdir()
        (subdir / "file.txt").write_text("nested", encoding="utf-8")

        with self.assertRaises(Refusal) as ctx:
            capture_release_sources(
                manifest_path=self.manifest_file,
                signature_path=self.sig_file,
                journal_dir=self.journal_dir,
                desktop_dir=self.desktop_dir,
                tmux_dir=self.tmux_dir,
            )
        self.assertEqual(ctx.exception.name, CAPTURE_DEPTH_EXCEEDED)
        self.assertFalse(last_private_parent().exists())

    def test_refuse_depth_exceeded_in_journal(self):
        subdir = self.journal_dir / "linux-x86_64" / "nested"
        subdir.mkdir()
        (subdir / "file.txt").write_text("nested", encoding="utf-8")

        with self.assertRaises(Refusal) as ctx:
            capture_release_sources(
                manifest_path=self.manifest_file,
                signature_path=self.sig_file,
                journal_dir=self.journal_dir,
                desktop_dir=self.desktop_dir,
                tmux_dir=self.tmux_dir,
            )
        self.assertEqual(ctx.exception.name, CAPTURE_DEPTH_EXCEEDED)
        self.assertFalse(last_private_parent().exists())

    def test_refuse_entry_limit_exceeded(self):
        for i in range(MAX_TOTAL_ENTRIES + 5):
            (self.tmux_dir / f"entry_{i:04d}.txt").write_text("x", encoding="utf-8")

        with self.assertRaises(Refusal) as ctx:
            capture_release_sources(
                manifest_path=self.manifest_file,
                signature_path=self.sig_file,
                journal_dir=self.journal_dir,
                desktop_dir=self.desktop_dir,
                tmux_dir=self.tmux_dir,
            )
        self.assertEqual(ctx.exception.name, CAPTURE_ENTRY_LIMIT_EXCEEDED)
        self.assertFalse(last_private_parent().exists())

    def test_refuse_metadata_size_limit_exceeded(self):
        large_file = self.tmux_dir / "huge_metadata.json"
        large_file.write_bytes(b"A" * (4 * 1024 * 1024 + 1024))

        with self.assertRaises(Refusal) as ctx:
            capture_release_sources(
                manifest_path=self.manifest_file,
                signature_path=self.sig_file,
                journal_dir=self.journal_dir,
                desktop_dir=self.desktop_dir,
                tmux_dir=self.tmux_dir,
            )
        self.assertEqual(ctx.exception.name, CAPTURE_SIZE_LIMIT_EXCEEDED)
        self.assertFalse(last_private_parent().exists())

    def test_race_regular_to_symlink_between_inspect_and_open(self):
        def replace_with_symlink(source_class: str, rel_path: str):
            if source_class == "desktop" and rel_path == "extra.dat":
                p = self.desktop_dir / "extra.dat"
                p.unlink()
                p.symlink_to(self.manifest_file)

        capture._between_inspect_and_open = replace_with_symlink

        with self.assertRaises(Refusal) as ctx:
            capture_release_sources(
                manifest_path=self.manifest_file,
                signature_path=self.sig_file,
                journal_dir=self.journal_dir,
                desktop_dir=self.desktop_dir,
                tmux_dir=self.tmux_dir,
            )
        self.assertEqual(ctx.exception.name, CAPTURE_SYMLINK_TRAVERSAL)
        self.assertFalse(last_private_parent().exists())

    def test_race_regular_to_symlink_success_twin(self):
        # Without mutation, identical sources capture successfully
        snap = capture_release_sources(
            manifest_path=self.manifest_file,
            signature_path=self.sig_file,
            journal_dir=self.journal_dir,
            desktop_dir=self.desktop_dir,
            tmux_dir=self.tmux_dir,
        )
        self.assertTrue(snap.snapshot_root.exists())
        snap.cleanup()

    def test_race_inode_replaced_between_inspect_and_open(self):
        def replace_inode(source_class: str, rel_path: str):
            if source_class == "desktop" and rel_path == "extra.dat":
                p = self.desktop_dir / "extra.dat"
                hold_fd = os.open(str(p), os.O_RDONLY)
                p.unlink()
                p.write_text("recreated with same name but new inode", encoding="utf-8")
                os.close(hold_fd)

        capture._between_inspect_and_open = replace_inode

        with self.assertRaises(Refusal) as ctx:
            capture_release_sources(
                manifest_path=self.manifest_file,
                signature_path=self.sig_file,
                journal_dir=self.journal_dir,
                desktop_dir=self.desktop_dir,
                tmux_dir=self.tmux_dir,
            )
        self.assertEqual(ctx.exception.name, SOURCE_CHANGED_DURING_CAPTURE)
        self.assertFalse(last_private_parent().exists())

    def test_race_parent_directory_replaced_between_inspect_and_open(self):
        def replace_parent(source_class: str, rel_path: str):
            if source_class == "journal" and "solstone-journal" in rel_path:
                old_sub = self.journal_dir / "linux-x86_64"
                shutil.rmtree(old_sub)
                old_sub.mkdir()
                (old_sub / "solstone-journal-1.0.0-linux-x86_64.release").write_text("commit=1234\n", encoding="utf-8")

        capture._between_inspect_and_open = replace_parent

        with self.assertRaises(Refusal) as ctx:
            capture_release_sources(
                manifest_path=self.manifest_file,
                signature_path=self.sig_file,
                journal_dir=self.journal_dir,
                desktop_dir=self.desktop_dir,
                tmux_dir=self.tmux_dir,
            )
        self.assertEqual(ctx.exception.name, SOURCE_CHANGED_DURING_CAPTURE)
        self.assertFalse(last_private_parent().exists())

    def test_race_inplace_rewrite_same_inode_length_detected(self):
        def tamper_hook(source_class: str, rel_path: str):
            if "SHA256SUMS" in rel_path:
                with open(self.tmux_dir / "SHA256SUMS", "r+b") as f:
                    f.seek(0)
                    f.write(b"tampered")

        capture._interrupted_read_hook = tamper_hook

        with self.assertRaises(Refusal) as ctx:
            capture_release_sources(
                manifest_path=self.manifest_file,
                signature_path=self.sig_file,
                journal_dir=self.journal_dir,
                desktop_dir=self.desktop_dir,
                tmux_dir=self.tmux_dir,
            )
        self.assertEqual(ctx.exception.name, SOURCE_CHANGED_DURING_CAPTURE)
        self.assertFalse(last_private_parent().exists())

    def test_interrupted_read_per_source_class(self):
        source_classes = [
            "platform_manifest",
            "platform_signature",
            "journal",
            "desktop",
            "tmux",
            "bootstrap",
        ]
        for target_class in source_classes:
            with self.subTest(target_class=target_class):
                def interrupt(s_class: str, rel_path: str):
                    if s_class == target_class:
                        raise Refusal(CAPTURE_READ_INTERRUPTED, f"simulated interrupted read on {target_class}")

                capture._interrupted_read_hook = interrupt

                with self.assertRaises(Refusal) as ctx:
                    capture_release_sources(
                        manifest_path=self.manifest_file,
                        signature_path=self.sig_file,
                        journal_dir=self.journal_dir,
                        desktop_dir=self.desktop_dir,
                        tmux_dir=self.tmux_dir,
                        bootstrap_file=self.bootstrap_file,
                    )
                self.assertEqual(ctx.exception.name, CAPTURE_READ_INTERRUPTED)
                self.assertFalse(last_private_parent().exists())


if __name__ == "__main__":
    unittest.main()
