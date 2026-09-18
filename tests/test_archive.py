# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from solstone_platform.archive import (
    scan_deb_file,
    scan_rpm_file,
    scan_tar_file,
    scan_variant_archive,
)
from solstone_platform.refusals import (
    ARCHIVE_ABSOLUTE_PATH,
    ARCHIVE_DEVICE,
    ARCHIVE_DUPLICATE_MEMBER,
    ARCHIVE_FIFO,
    ARCHIVE_HARDLINK_ESCAPE,
    ARCHIVE_PACKAGE_SCRIPT,
    ARCHIVE_PACKAGE_TRIGGER,
    ARCHIVE_PARENT_TRAVERSAL,
    ARCHIVE_SYMLINK_ESCAPE,
    ARCHIVE_SYMLINK_THEN_CHILD,
    Refusal,
)
from tools.fixture_builder import create_tiny_deb, create_tiny_tar


class TestArchive(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).parent.parent
        self.desktop_rpm = self.repo_root / "testdata" / "native" / "desktop" / "2.0.3" / "solstone-linux-2.0.3-1.x86_64.rpm"

    def test_scan_real_rpm(self):
        if not self.desktop_rpm.is_file():
            self.skipTest("desktop rpm fixture not present")
        res = scan_rpm_file(self.desktop_rpm)
        self.assertIsNotNone(res.package_identity)
        self.assertEqual(res.package_identity.name, "solstone-linux")
        self.assertEqual(res.package_identity.arch, "x86_64")
        self.assertIn("solstone-linux", res.executable_sha256)
        self.assertTrue(len(res.inventory) > 0)

    def test_scan_tiny_tar_and_deb(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tar_file = tmp_path / "test.tar.gz"
            create_tiny_tar(tar_file, {"usr/bin/mybin": b"binary-content", "usr/share/doc.txt": b"doc-content"})
            tar_res = scan_tar_file(tar_file)
            self.assertEqual(len(tar_res.inventory), 2)
            self.assertIn("mybin", tar_res.executable_sha256)

            deb_file = tmp_path / "test.deb"
            create_tiny_deb(deb_file, "mypkg", "1.0.0", "amd64", "mybin", b"binary-content")
            deb_res = scan_deb_file(deb_file)
            self.assertIsNotNone(deb_res.package_identity)
            self.assertEqual(deb_res.package_identity.name, "mypkg")
            self.assertEqual(deb_res.package_identity.version, "1.0.0")
            self.assertEqual(deb_res.package_identity.arch, "amd64")

    def test_archive_absolute_path_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_file = Path(tmp_dir) / "bad.tar.gz"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                ti = tarfile.TarInfo(name="/etc/passwd")
                ti.size = 5
                tf.addfile(ti, io.BytesIO(b"hello"))
            tar_file.write_bytes(buf.getvalue())
            with self.assertRaises(Refusal) as ctx:
                scan_tar_file(tar_file)
            self.assertEqual(ctx.exception.name, ARCHIVE_ABSOLUTE_PATH)

    def test_archive_parent_traversal_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_file = Path(tmp_dir) / "bad.tar.gz"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                ti = tarfile.TarInfo(name="usr/bin/../../etc/passwd")
                ti.size = 5
                tf.addfile(ti, io.BytesIO(b"hello"))
            tar_file.write_bytes(buf.getvalue())
            with self.assertRaises(Refusal) as ctx:
                scan_tar_file(tar_file)
            self.assertEqual(ctx.exception.name, ARCHIVE_PARENT_TRAVERSAL)

    def test_archive_symlink_escape_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_file = Path(tmp_dir) / "bad.tar.gz"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                ti = tarfile.TarInfo(name="usr/bin/sym")
                ti.type = tarfile.SYMTYPE
                ti.linkname = "../../../etc/shadow"
                tf.addfile(ti)
            tar_file.write_bytes(buf.getvalue())
            with self.assertRaises(Refusal) as ctx:
                scan_tar_file(tar_file)
            self.assertEqual(ctx.exception.name, ARCHIVE_SYMLINK_ESCAPE)

    def test_archive_hardlink_escape_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_file = Path(tmp_dir) / "bad.tar.gz"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                ti = tarfile.TarInfo(name="usr/bin/link")
                ti.type = tarfile.LNKTYPE
                ti.linkname = "../../../etc/shadow"
                tf.addfile(ti)
            tar_file.write_bytes(buf.getvalue())
            with self.assertRaises(Refusal) as ctx:
                scan_tar_file(tar_file)
            self.assertEqual(ctx.exception.name, ARCHIVE_HARDLINK_ESCAPE)

    def test_archive_device_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_file = Path(tmp_dir) / "bad.tar.gz"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                ti = tarfile.TarInfo(name="usr/bin/dev")
                ti.type = tarfile.CHRTYPE
                tf.addfile(ti)
            tar_file.write_bytes(buf.getvalue())
            with self.assertRaises(Refusal) as ctx:
                scan_tar_file(tar_file)
            self.assertEqual(ctx.exception.name, ARCHIVE_DEVICE)

    def test_archive_fifo_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_file = Path(tmp_dir) / "bad.tar.gz"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                ti = tarfile.TarInfo(name="usr/bin/fifo")
                ti.type = tarfile.FIFOTYPE
                tf.addfile(ti)
            tar_file.write_bytes(buf.getvalue())
            with self.assertRaises(Refusal) as ctx:
                scan_tar_file(tar_file)
            self.assertEqual(ctx.exception.name, ARCHIVE_FIFO)

    def test_archive_symlink_then_child_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_file = Path(tmp_dir) / "bad.tar.gz"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                ti_sym = tarfile.TarInfo(name="usr/local")
                ti_sym.type = tarfile.SYMTYPE
                ti_sym.linkname = "bin"
                tf.addfile(ti_sym)

                ti_child = tarfile.TarInfo(name="usr/local/secret")
                ti_child.size = 4
                tf.addfile(ti_child, io.BytesIO(b"data"))
            tar_file.write_bytes(buf.getvalue())
            with self.assertRaises(Refusal) as ctx:
                scan_tar_file(tar_file)
            self.assertEqual(ctx.exception.name, ARCHIVE_SYMLINK_THEN_CHILD)

    def test_archive_duplicate_member_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_file = Path(tmp_dir) / "bad.tar.gz"
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                ti1 = tarfile.TarInfo(name="usr/bin/app")
                ti1.size = 4
                tf.addfile(ti1, io.BytesIO(b"ver1"))
                ti2 = tarfile.TarInfo(name="usr/bin/app")
                ti2.size = 4
                tf.addfile(ti2, io.BytesIO(b"ver2"))
            tar_file.write_bytes(buf.getvalue())
            with self.assertRaises(Refusal) as ctx:
                scan_tar_file(tar_file)
            self.assertEqual(ctx.exception.name, ARCHIVE_DUPLICATE_MEMBER)

    def test_deb_maintainer_script_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            deb_file = Path(tmp_dir) / "bad.deb"
            cbuf = io.BytesIO()
            with tarfile.open(fileobj=cbuf, mode="w:gz") as ctf:
                ti_ctrl = tarfile.TarInfo(name="control")
                ti_ctrl.size = 20
                ctf.addfile(ti_ctrl, io.BytesIO(b"Package: test\nVersion: 1\n"))
                ti_post = tarfile.TarInfo(name="postinst")
                ti_post.size = 10
                ctf.addfile(ti_post, io.BytesIO(b"#!/bin/sh\n"))
            cbytes = cbuf.getvalue()

            dbuf = io.BytesIO()
            with tarfile.open(fileobj=dbuf, mode="w:gz") as dtf:
                ti = tarfile.TarInfo(name="usr/bin/app")
                ti.size = 4
                dtf.addfile(ti, io.BytesIO(b"test"))
            dbytes = dbuf.getvalue()

            def ar_entry(name: str, data: bytes) -> bytes:
                hdr = f"{name:<16}{'0':<12}{'0':<6}{'0':<6}{'100644':<8}{len(data):<10}`\n".encode("ascii")
                pad = b"\n" if len(data) % 2 != 0 else b""
                return hdr + data + pad

            ar_data = b"!<arch>\n" + ar_entry("debian-binary", b"2.0\n") + ar_entry("control.tar.gz", cbytes) + ar_entry("data.tar.gz", dbytes)
            deb_file.write_bytes(ar_data)

            with self.assertRaises(Refusal) as ctx:
                scan_deb_file(deb_file)
            self.assertEqual(ctx.exception.name, ARCHIVE_PACKAGE_SCRIPT)

    def test_deb_triggers_refusal(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            deb_file = Path(tmp_dir) / "bad_trig.deb"
            cbuf = io.BytesIO()
            with tarfile.open(fileobj=cbuf, mode="w:gz") as ctf:
                ti_ctrl = tarfile.TarInfo(name="control")
                ti_ctrl.size = 20
                ctf.addfile(ti_ctrl, io.BytesIO(b"Package: test\nVersion: 1\n"))
                ti_trig = tarfile.TarInfo(name="triggers")
                ti_trig.size = 10
                ctf.addfile(ti_trig, io.BytesIO(b"interest x\n"))
            cbytes = cbuf.getvalue()

            dbuf = io.BytesIO()
            with tarfile.open(fileobj=dbuf, mode="w:gz") as dtf:
                ti = tarfile.TarInfo(name="usr/bin/app")
                ti.size = 4
                dtf.addfile(ti, io.BytesIO(b"test"))
            dbytes = dbuf.getvalue()

            def ar_entry(name: str, data: bytes) -> bytes:
                hdr = f"{name:<16}{'0':<12}{'0':<6}{'0':<6}{'100644':<8}{len(data):<10}`\n".encode("ascii")
                pad = b"\n" if len(data) % 2 != 0 else b""
                return hdr + data + pad

            ar_data = b"!<arch>\n" + ar_entry("debian-binary", b"2.0\n") + ar_entry("control.tar.gz", cbytes) + ar_entry("data.tar.gz", dbytes)
            deb_file.write_bytes(ar_data)

            with self.assertRaises(Refusal) as ctx:
                scan_deb_file(deb_file)
            self.assertEqual(ctx.exception.name, ARCHIVE_PACKAGE_TRIGGER)


if __name__ == "__main__":
    unittest.main()
