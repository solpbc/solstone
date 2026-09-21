# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for published-component platform recut preparation."""

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from solstone_platform.recut import _Downloader, _rename_noreplace, prepare_recut
from solstone_platform.refusals import HTTP_3XX, SCHEMA_INVALID, Refusal


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/payload")
            self.end_headers()
            return
        body = b"verified bytes"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class TestRecut(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.root = Path(self.temp.name)
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_streamed_download_records_exact_bytes(self):
        downloader = _Downloader()
        destination = self.root / "payload"
        result = downloader.fetch(f"{self.origin}/payload", destination)
        self.assertEqual(result, b"verified bytes")
        self.assertEqual(destination.read_bytes(), result)
        self.assertEqual(downloader.sources[0]["bytes"], len(result))

    def test_redirect_is_refused(self):
        with self.assertRaises(Refusal) as ctx:
            _Downloader().fetch(f"{self.origin}/redirect", self.root / "payload")
        self.assertEqual(ctx.exception.name, HTTP_3XX)

    def test_atomic_no_replace_never_overwrites(self):
        first = self.root / "first"
        second = self.root / "second"
        final = self.root / "final"
        first.mkdir()
        second.mkdir()
        (first / "value").write_text("first", encoding="utf-8")
        (second / "value").write_text("second", encoding="utf-8")
        _rename_noreplace(first, final)
        with self.assertRaises(FileExistsError):
            _rename_noreplace(second, final)
        self.assertEqual((final / "value").read_text(encoding="utf-8"), "first")

    def test_zero_replacements_refuses_before_destination_use(self):
        class NeverDestination:
            def get(self, key):
                raise AssertionError("destination must not be used")

        with self.assertRaises(Refusal) as ctx:
            prepare_recut(
                version="2.0.1",
                replacements={},
                output_dir=self.root / "candidate",
                repo_root=Path(__file__).parent.parent,
                dest=NeverDestination(),
            )
        self.assertEqual(ctx.exception.name, SCHEMA_INVALID)


if __name__ == "__main__":
    unittest.main()
