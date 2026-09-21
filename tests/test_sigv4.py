# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading
import unittest

from solstone_platform.canonical import parse_json_strict
from solstone_platform.destination import ResultStatus
from solstone_platform.r2 import (
    R2_READ_CHUNK_BYTES,
    R2_REQUEST_TIMEOUT_SECONDS,
    R2Config,
    R2Destination,
    compute_sigv4_headers,
)
from solstone_platform.refusals import (
    HTTP_3XX,
    HTTP_TIMEOUT,
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

    def test_compare_and_swap_preserves_transport_ambiguity(self):
        def timeout(*args, **kwargs):
            raise Refusal(HTTP_TIMEOUT, "simulated ambiguous timeout")

        self.dest._send_request = timeout
        result = self.dest.compare_and_swap(
            "solstone/release/latest",
            b"2.0.1\n",
            expected_etag='"base"',
            content_type="text/plain; charset=utf-8",
            cache_control="no-store, max-age=0",
        )
        self.assertEqual(result.status, ResultStatus.INDETERMINATE)

    def test_requests_allow_large_artifact_upload_window(self):
        class CapturingOpener:
            timeout = None

            def open(self, request, timeout):
                self.timeout = timeout
                return self

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            status = 200
            headers = {"ETag": '"testetag"'}

            @staticmethod
            def read():
                return b""

        opener = CapturingOpener()
        self.dest.opener = opener
        self.dest._send_request("GET", "solstone/release/latest")
        self.assertEqual(opener.timeout, R2_REQUEST_TIMEOUT_SECONDS)
        self.assertEqual(opener.timeout, 900)

    def test_immutable_get_assembles_bounded_ranges(self):
        class Response:
            def __init__(self, body, start, end, total):
                self.body = body
                self.status = 206
                self.headers = {
                    "Content-Length": str(len(body)),
                    "Content-Range": f"bytes {start}-{end}/{total}",
                    "ETag": '"stable"',
                    "Content-Type": "application/octet-stream",
                    "Cache-Control": "public, max-age=31536000, immutable",
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return self.body

        class RangedOpener:
            def __init__(self, payload):
                self.payload = payload
                self.requests = []

            def open(self, request, timeout):
                self.requests.append(request)
                range_header = request.get_header("Range")
                start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
                start = int(start_text)
                end = min(int(end_text), len(self.payload) - 1)
                return Response(self.payload[start : end + 1], start, end, len(self.payload))

        payload = b"a" * (R2_READ_CHUNK_BYTES + 7)
        opener = RangedOpener(payload)
        self.dest.opener = opener
        result = self.dest.get("solstone/release/2.0.0/large.tar.gz")
        self.assertTrue(result.is_ok())
        self.assertEqual(result.body, payload)
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual(opener.requests[1].get_header("If-match"), '"stable"')



if __name__ == "__main__":
    unittest.main()
