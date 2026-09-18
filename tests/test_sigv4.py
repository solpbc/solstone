# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading
import unittest

from solstone_platform.canonical import parse_json_strict
from solstone_platform.destination import ResultStatus
from solstone_platform.r2 import R2Config, R2Destination, compute_sigv4_headers
from solstone_platform.refusals import (
    HTTP_3XX,
    LANE_INVALID,
    Refusal,
    UNSAFE_FILENAME,
)


class MockS3Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/bucket/solstone/staging/latest":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1/redirected")
            self.end_headers()
        elif self.path == "/bucket/solstone/dev/latest":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")
        elif self.path == "/bucket/solstone/release/2.0.3/platform.json":
            self.send_response(200)
            self.send_header("ETag", '"testetag"')
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            self.end_headers()
            self.wfile.write(b'{"lane":"release"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_PUT(self):
        if self.headers.get("If-Match") == '"wrong-etag"':
            self.send_response(412)
            self.end_headers()
        elif self.headers.get("If-None-Match") == "*" and "exists" in self.path:
            self.send_response(412)
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("ETag", '"put-etag"')
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress server stderr in tests


class TestSigV4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), MockS3Handler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.sigv4_dir = Path(__file__).parent.parent / "testdata" / "sigv4"
        self.config = R2Config(
            endpoint=f"http://127.0.0.1:{self.port}",
            bucket="bucket",
            region="us-east-1",
            access_key_id="TESTKEYID",
            secret_access_key="TESTSECRETKEY",
        )
        self.dest = R2Destination(self.config)

    def test_get_vanilla_iam_golden_vector(self):
        vec_path = self.sigv4_dir / "get-vanilla.json"
        vec = parse_json_strict(vec_path.read_bytes())

        computed_headers = compute_sigv4_headers(
            method=vec["method"],
            url=vec["url"],
            headers=vec["headers"],
            body=vec["body"].encode("utf-8"),
            region=vec["region"],
            service=vec["service"],
            access_key=vec["access_key"],
            secret_key=vec["secret_key"],
            amz_datetime=vec["datetime"],
        )
        self.assertEqual(computed_headers["authorization"], vec["authorization_header"])
        self.assertEqual(vec["signature"], "5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7")

    def test_post_vanilla_vector(self):
        vec_path = self.sigv4_dir / "post-vanilla.json"
        vec = parse_json_strict(vec_path.read_bytes())

        computed_headers = compute_sigv4_headers(
            method=vec["method"],
            url=vec["url"],
            headers=vec["headers"],
            body=vec["body"].encode("utf-8"),
            region=vec["region"],
            service=vec["service"],
            access_key=vec["access_key"],
            secret_key=vec["secret_key"],
            amz_datetime=vec["datetime"],
        )
        self.assertEqual(computed_headers["authorization"], vec["authorization_header"])

    def test_key_confinement_escapes(self):
        with self.assertRaises(Refusal) as ctx:
            self.dest.get("escaped/solstone/release/latest")
        self.assertEqual(ctx.exception.name, UNSAFE_FILENAME)

        with self.assertRaises(Refusal) as ctx:
            self.dest.get("solstone/release/../dev/latest")
        self.assertEqual(ctx.exception.name, UNSAFE_FILENAME)

        with self.assertRaises(Refusal) as ctx:
            self.dest.get("solstone/invalidlane/latest")
        self.assertEqual(ctx.exception.name, LANE_INVALID)

    def test_3xx_redirect_refused(self):
        with self.assertRaises(Refusal) as ctx:
            self.dest._send_request("GET", "solstone/staging/latest")
        self.assertEqual(ctx.exception.name, HTTP_3XX)

    def test_missing_etag_returns_malformed_etag(self):
        res = self.dest.get("solstone/dev/latest")
        self.assertEqual(res.status, ResultStatus.MALFORMED_ETAG)

    def test_put_precondition_failed_412(self):
        res = self.dest.put_if_absent(
            "solstone/release/2.0.3/exists.tar.gz",
            b"{}",
            content_type="application/octet-stream",
            cache_control="public",
        )
        self.assertEqual(res.status, ResultStatus.PRECONDITION_FAILED)

    def test_compare_and_swap_on_immutable_key_refused_at_admission(self):
        with self.assertRaises(Refusal) as ctx:
            self.dest.compare_and_swap(
                "solstone/release/2.0.3/platform.json",
                b"{}",
                expected_etag='"wrong-etag"',
                content_type="application/json",
                cache_control="no-store",
            )
        self.assertEqual(ctx.exception.name, UNSAFE_FILENAME)



if __name__ == "__main__":
    unittest.main()
