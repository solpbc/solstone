# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Test Families 4 & 5: Transport security, URL validation, and installer revision floors."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import unittest

from solstone_platform.pins import load_pin_file
from solstone_platform.refusals import Refusal
from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import LoopbackServer, setup_test_release_server
from tools.build_installer import DEFAULT_ORIGIN, build_installer

REPO_ROOT = Path(__file__).resolve().parent.parent


class RouteHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.request_paths.append(self.path)
        route = self.server.routes.get(self.path)
        if callable(route):
            route(self)
            return
        if route is None:
            route = (404, {}, b"")
        status, headers, body = route
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, format, *args):
        pass


class RouteServer:
    def __init__(self):
        self.httpd = HTTPServer(("127.0.0.1", 0), RouteHandler)
        self.httpd.request_paths = []
        self.httpd.routes = {}
        self.origin = f"http://127.0.0.1:{self.httpd.server_port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def routes(self):
        return self.httpd.routes

    @property
    def request_paths(self):
        return self.httpd.request_paths

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestInstallTransport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)
        self.prefix = self.work_dir / "prefix"
        self.prefix.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def _build(self, pub: Path, key_id: str, *, origin: str | None = None) -> Path:
        installer = self.work_dir / "install.sh"
        build_installer(
            repo_root=REPO_ROOT,
            output_path=installer,
            platform_pub_path=pub,
            platform_key_id=key_id,
            origin=origin,
        )
        return installer

    def _run(
        self,
        installer: Path,
        *args: str,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ):
        return subprocess.run(
            [str(installer), *args, "--json"],
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
        )

    def test_build_inputs_refuse_before_output_mutation(self):
        out = self.work_dir / "existing.sh"
        out.write_text("preserve-me\n", encoding="utf-8")
        production_pin = load_pin_file(REPO_ROOT / "pins" / "platform.pub")
        production_overrides = (
            {"platform_pub_path": REPO_ROOT / "pins" / "platform.pub"},
            {"platform_key_id": production_pin.key_id},
            {"origin": DEFAULT_ORIGIN},
            {"override_installer_revision": 1},
        )
        for override in production_overrides:
            with self.subTest(production_override=override):
                with self.assertRaises(Refusal):
                    build_installer(REPO_ROOT, out, is_production=True, **override)
                self.assertEqual(out.read_text(encoding="utf-8"), "preserve-me\n")

        with ephemeral_keypair("build input validation") as (_sec, pub, pin):
            invalid_inputs = (
                {"origin": "http://127.0.0.1:080"},
                {"origin": "http://127.0.0.1:65536"},
                {"origin": "http://localhost:8080"},
                {"origin": "http://127.0.0.1:8080;touch /tmp/no"},
                {"platform_key_id": "0" * 15},
                {"platform_key_id": "Ｇ" * 16},
                {"platform_key_id": "0" * 16},
                {"override_installer_revision": 0},
                {"override_installer_revision": -1},
                {"override_installer_revision": True},
                {"override_installer_revision": "1"},
            )
            for invalid in invalid_inputs:
                with self.subTest(fixture_input=invalid):
                    kwargs = {
                        "repo_root": REPO_ROOT,
                        "output_path": out,
                        "platform_pub_path": pub,
                        "platform_key_id": pin.key_id,
                    }
                    kwargs.update(invalid)
                    with self.assertRaises(Refusal):
                        build_installer(**kwargs)
                    self.assertEqual(out.read_text(encoding="utf-8"), "preserve-me\n")

            build_installer(
                REPO_ROOT,
                out,
                platform_pub_path=pub,
                platform_key_id=pin.key_id.lower(),
                origin="http://127.0.0.1:65535",
                override_installer_revision=1,
            )
            self.assertIn('DEFAULT_ORIGIN="http://127.0.0.1:65535"', out.read_text(encoding="utf-8"))

    def test_runtime_coordinates_refuse_before_network(self):
        with ephemeral_keypair("runtime coordinate validation") as (_sec, pub, pin):
            server_root = self.work_dir / "empty-server"
            server_root.mkdir()
            server = LoopbackServer(server_root)
            server.start()
            try:
                installer = self._build(pub, pin.key_id, origin=server.origin)
                cases = (
                    ("origin path", ("--origin", f"{server.origin}/extra"), "origin-invalid"),
                    ("lane", ("--lane", "release/../dev"), "lane-invalid"),
                    ("version leading zero", ("--version", "02.0.0"), "version-invalid"),
                    ("version extra", ("--version", "2.0.0/x"), "version-invalid"),
                    ("version long", ("--version", f"{'1' * 21}.0.0"), "version-invalid"),
                )
                for name, args, code in cases:
                    with self.subTest(name=name):
                        server.request_paths.clear()
                        proc = self._run(installer, *args, "--components", "journal")
                        self.assertNotEqual(proc.returncode, 0)
                        self.assertEqual(json.loads(proc.stdout)["root_code"], code)
                        self.assertEqual(server.request_paths, [])

                (self.work_dir / "1").touch()
                proc = self._run(
                    installer,
                    "--version",
                    "?.?.?",
                    "--components",
                    "journal",
                    cwd=self.work_dir,
                )
                self.assertEqual(json.loads(proc.stdout)["root_code"], "version-invalid")
                self.assertEqual(server.request_paths, [])
            finally:
                server.stop()

    def test_latest_shape_and_size(self):
        with ephemeral_keypair("latest transport") as (sec, pub, pin):
            server, root = setup_test_release_server(self.work_dir, sec, pin)
            try:
                installer = self._build(pub, pin.key_id, origin=server.origin)
                latest = root / "solstone" / "release" / "latest"
                for body in (b"2.0.0", b"2.0.0\n", b"2.0.0\r\n"):
                    with self.subTest(valid=body):
                        latest.write_bytes(body)
                        server.request_paths.clear()
                        proc = self._run(installer, "--skip-signature", "--list")
                        self.assertEqual(proc.returncode, 0, proc.stderr)
                        self.assertEqual(len(server.request_paths), 3)

                invalid_bodies = (
                    b"2.0.0\nsecond-line",
                    b"02.0.0\n",
                    b"2.0.\x00\n",
                    b"2.0.0\r",
                    b"1" * 65,
                )
                for body in invalid_bodies:
                    with self.subTest(invalid=body):
                        latest.write_bytes(body)
                        server.request_paths.clear()
                        proc = self._run(installer, "--skip-signature", "--list")
                        self.assertNotEqual(proc.returncode, 0)
                        self.assertIn(json.loads(proc.stdout)["root_code"], {"latest-invalid", "response-too-large"})
                        self.assertEqual(len(server.request_paths), 1)
            finally:
                server.stop()

    def _recording_curl_env(self) -> tuple[dict[str, str], Path]:
        bin_dir = self.work_dir / "curl-bin"
        bin_dir.mkdir(exist_ok=True)
        log_path = self.work_dir / "curl.log"
        wrapper = bin_dir / "curl"
        wrapper.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$*\" >> \"$TRANSPORT_LOG\"\n"
            "exec /usr/bin/curl \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env['PATH']}"
        env["TRANSPORT_LOG"] = str(log_path)
        return env, log_path

    def _recording_wget_env(self) -> tuple[dict[str, str], Path]:
        bin_dir = self.work_dir / "wget-bin"
        bin_dir.mkdir(exist_ok=True)
        log_path = self.work_dir / "wget.log"
        wget_path = shutil.which("wget")
        self.assertIsNotNone(wget_path)
        wrapper = bin_dir / "wget"
        wrapper.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$*\" >> \"$TRANSPORT_LOG\"\n"
            f"exec {wget_path} \"$@\"\n",
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
        for command in ("uname", "mktemp", "chmod", "cat", "rm", "wc", "sed"):
            command_path = shutil.which(command)
            self.assertIsNotNone(command_path)
            (bin_dir / command).symlink_to(command_path)
        env = os.environ.copy()
        env["PATH"] = str(bin_dir)
        env["TRANSPORT_LOG"] = str(log_path)
        return env, log_path

    def test_catalogue_caps_and_client_flags(self):
        with ephemeral_keypair("catalogue caps") as (sec, pub, pin):
            server, root = setup_test_release_server(self.work_dir, sec, pin)
            try:
                installer = self._build(pub, pin.key_id, origin=server.origin)
                manifest = root / "solstone" / "release" / "2.0.0" / "platform.json"
                signature = root / "solstone" / "release" / "2.0.0" / "platform.json.minisig"
                valid_manifest = manifest.read_bytes()

                manifest.write_bytes(b"{" * 4194304)
                server.request_paths.clear()
                exact = self._run(installer, "--version", "2.0.0", "--skip-signature", "--list")
                self.assertEqual(json.loads(exact.stdout)["root_code"], "schema-invalid")
                self.assertEqual(len(server.request_paths), 2)

                curl_env, curl_log = self._recording_curl_env()
                manifest.write_bytes(b"{" * 4194305)
                server.request_paths.clear()
                oversized = self._run(
                    installer,
                    "--version",
                    "2.0.0",
                    "--skip-signature",
                    "--list",
                    env=curl_env,
                )
                self.assertEqual(json.loads(oversized.stdout)["root_code"], "response-too-large")
                self.assertEqual(len(server.request_paths), 1)
                curl_args = curl_log.read_text(encoding="utf-8")
                for flag in ("-q", "--connect-timeout 10", "--max-time 120", "--retry 0", "--max-redirs 0"):
                    self.assertIn(flag, curl_args)

                manifest.write_bytes(valid_manifest)
                signature.write_bytes(b"s" * 16384)
                server.request_paths.clear()
                exact_sig = self._run(installer, "--version", "2.0.0", "--skip-signature", "--list")
                self.assertEqual(exact_sig.returncode, 0, exact_sig.stderr)
                self.assertEqual(len(server.request_paths), 2)

                signature.write_bytes(b"s" * 16385)
                wget_env, wget_log = self._recording_wget_env()
                server.request_paths.clear()
                oversized_sig = self._run(
                    installer,
                    "--version",
                    "2.0.0",
                    "--skip-signature",
                    "--list",
                    env=wget_env,
                )
                self.assertEqual(json.loads(oversized_sig.stdout)["root_code"], "response-too-large")
                self.assertEqual(len(server.request_paths), 2)
                wget_args = wget_log.read_text(encoding="utf-8")
                for flag in ("--no-config", "--tries=1", "--connect-timeout=10", "--timeout=120", "--max-redirect=0"):
                    self.assertIn(flag, wget_args)
            finally:
                server.stop()

    def test_redirect_confinement_hop_limit_and_wget_refusal(self):
        with ephemeral_keypair("redirect transport") as (_sec, pub, pin):
            first = RouteServer()
            second = RouteServer()
            first.start()
            second.start()
            try:
                installer = self._build(pub, pin.key_id, origin=first.origin)
                platform_path = "/solstone/release/2.0.0/platform.json"
                signature_path = f"{platform_path}.minisig"
                minimal_catalogue = json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_version": 1,
                        "version": "2.0.0",
                        "lane": "release",
                        "platform_key_id": pin.key_id,
                        "minimum_installer_revision": 1,
                    },
                    separators=(",", ":"),
                ).encode("utf-8")

                first.routes.update(
                    {
                        platform_path: (302, {"Location": f"{first.origin}/platform-target"}, b""),
                        "/platform-target": (200, {}, minimal_catalogue),
                        signature_path: (200, {}, b"signature"),
                    }
                )
                proc = self._run(installer, "--version", "2.0.0", "--skip-signature", "--list")
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(json.loads(proc.stdout)["root_code"], "list")
                self.assertEqual(first.request_paths, [platform_path, "/platform-target", signature_path])

                first.request_paths.clear()
                first.routes[platform_path] = (302, {"Location": f"{first.origin}/hop-two"}, b"")
                first.routes["/hop-two"] = (302, {"Location": f"{first.origin}/hop-three"}, b"")
                proc = self._run(installer, "--version", "2.0.0", "--skip-signature", "--list")
                self.assertEqual(json.loads(proc.stdout)["root_code"], "redirect-refused")
                self.assertEqual(first.request_paths, [platform_path, "/hop-two"])

                first.request_paths.clear()
                second.request_paths.clear()
                second.routes["/foreign"] = (200, {}, b"{}")
                first.routes[platform_path] = (302, {"Location": f"{second.origin}/foreign"}, b"")
                proc = self._run(installer, "--version", "2.0.0", "--skip-signature", "--list")
                self.assertEqual(json.loads(proc.stdout)["root_code"], "redirect-refused")
                self.assertEqual(first.request_paths, [platform_path])
                self.assertEqual(second.request_paths, [])

                first.request_paths.clear()
                first.routes[platform_path] = (302, {"Location": f"{first.origin}/platform-target"}, b"")
                wget_env, _wget_log = self._recording_wget_env()
                proc = self._run(
                    installer,
                    "--version",
                    "2.0.0",
                    "--skip-signature",
                    "--list",
                    env=wget_env,
                )
                self.assertEqual(json.loads(proc.stdout)["root_code"], "fetch-failed")
                self.assertEqual(first.request_paths, [platform_path])
            finally:
                first.stop()
                second.stop()

    def test_curl_refuses_scheme_change_before_second_hop(self):
        with ephemeral_keypair("scheme redirect") as (_sec, pub, pin):
            installer = self._build(pub, pin.key_id, origin="http://127.0.0.1:4444")
            bin_dir = self.work_dir / "scheme-bin"
            bin_dir.mkdir()
            log_path = self.work_dir / "scheme-curl.log"
            wrapper = bin_dir / "curl"
            wrapper.write_text(
                "#!/bin/sh\n"
                "printf 'call\\n' >> \"$TRANSPORT_LOG\"\n"
                "header=''\n"
                "dest=''\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  case \"$1\" in\n"
                "    -D) header=\"$2\"; shift 2 ;;\n"
                "    -o) dest=\"$2\"; shift 2 ;;\n"
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                "printf 'HTTP/1.1 302 Found\\r\\nLocation: https://127.0.0.1:4444/target\\r\\n\\r\\n' > \"$header\"\n"
                ": > \"$dest\"\n"
                "printf '302'\n",
                encoding="utf-8",
            )
            wrapper.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}:{env['PATH']}"
            env["TRANSPORT_LOG"] = str(log_path)
            proc = self._run(
                installer,
                "--version",
                "2.0.0",
                "--skip-signature",
                "--list",
                env=env,
            )
            self.assertEqual(json.loads(proc.stdout)["root_code"], "redirect-refused")
            self.assertEqual(log_path.read_text(encoding="utf-8").splitlines(), ["call"])

    def test_signal_cleanup_during_catalogue_fetch(self):
        with ephemeral_keypair("signal cleanup") as (_sec, pub, pin):
            server = RouteServer()
            request_started = threading.Event()
            release_request = threading.Event()

            def hold_request(handler):
                request_started.set()
                release_request.wait(timeout=5)
                handler.send_response(200)
                handler.end_headers()
                try:
                    handler.wfile.write(b"2.0.0\n")
                except BrokenPipeError:
                    pass

            server.routes["/solstone/release/latest"] = hold_request
            server.start()
            try:
                installer = self._build(pub, pin.key_id, origin=server.origin)
                for caught_signal in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
                    with self.subTest(signal=caught_signal):
                        request_started.clear()
                        release_request.clear()
                        before = set(Path("/var/tmp").glob("solstone-install.*"))
                        proc = subprocess.Popen(
                            [str(installer), "--skip-signature", "--list", "--json"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            start_new_session=True,
                        )
                        self.assertTrue(request_started.wait(timeout=3))
                        os.killpg(proc.pid, caught_signal)
                        proc.communicate(timeout=3)
                        release_request.set()
                        self.assertNotEqual(proc.returncode, 0)
                        deadline = time.monotonic() + 1
                        while time.monotonic() < deadline:
                            if set(Path("/var/tmp").glob("solstone-install.*")) <= before:
                                break
                            time.sleep(0.02)
                        self.assertLessEqual(set(Path("/var/tmp").glob("solstone-install.*")), before)
            finally:
                release_request.set()
                server.stop()

    def test_insecure_remote_http_refuses(self):
        with ephemeral_keypair("test transport") as (sec, pub, pin):
            installer = self.work_dir / "install.sh"
            build_installer(
                repo_root=REPO_ROOT,
                output_path=installer,
                platform_pub_path=pub,
                platform_key_id=pin.key_id,
            )

            proc = subprocess.run(
                [str(installer), "--origin", "http://example.com", "--components", "journal", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            res = json.loads(proc.stdout.strip())
            self.assertEqual(res["status"], "refusal")
            self.assertEqual(res["root_code"], "url-insecure")

    def test_unapproved_https_host_refuses(self):
        with ephemeral_keypair("test transport https") as (sec, pub, pin):
            installer = self.work_dir / "install.sh"
            build_installer(
                repo_root=REPO_ROOT,
                output_path=installer,
                platform_pub_path=pub,
                platform_key_id=pin.key_id,
            )

            proc = subprocess.run(
                [str(installer), "--origin", "https://unauthorized.solstone.app", "--components", "journal", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            res = json.loads(proc.stdout.strip())
            self.assertEqual(res["status"], "refusal")
            self.assertEqual(res["root_code"], "url-insecure")

    def test_url_userinfo_refuses(self):
        with ephemeral_keypair("test transport userinfo") as (sec, pub, pin):
            installer = self.work_dir / "install.sh"
            build_installer(
                repo_root=REPO_ROOT,
                output_path=installer,
                platform_pub_path=pub,
                platform_key_id=pin.key_id,
            )

            proc = subprocess.run(
                [str(installer), "--origin", "http://user:pass@127.0.0.1:8080", "--components", "journal", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            res = json.loads(proc.stdout.strip())
            self.assertEqual(res["status"], "refusal")
            self.assertEqual(res["root_code"], "url-userinfo")

    def test_url_fragment_refuses(self):
        with ephemeral_keypair("test transport frag") as (sec, pub, pin):
            installer = self.work_dir / "install.sh"
            build_installer(
                repo_root=REPO_ROOT,
                output_path=installer,
                platform_pub_path=pub,
                platform_key_id=pin.key_id,
            )

            proc = subprocess.run(
                [str(installer), "--origin", "http://127.0.0.1:8080/path#frag", "--components", "journal", "--json"],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            res = json.loads(proc.stdout.strip())
            self.assertEqual(res["status"], "refusal")
            self.assertEqual(res["root_code"], "url-fragment")

    def test_installer_revision_floors(self):
        with ephemeral_keypair("test rev floor") as (sec, pub, pin):
            # Manifest requires minimum_installer_revision = 2
            server, _ = setup_test_release_server(self.work_dir, sec, pin, min_installer_revision=2)
            try:
                # 1. Installer with revision 1 against min 2 -> Refuses revision-too-old
                installer_v1 = self.work_dir / "install_v1.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer_v1,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                    override_installer_revision=1,
                )

                proc = subprocess.run(
                    [str(installer_v1), "--skip-signature", "--components", "journal", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(proc.returncode, 0)
                res = json.loads(proc.stdout.strip())
                self.assertEqual(res["status"], "refusal")
                self.assertEqual(res["root_code"], "revision-too-old")

                # 2. Installer with revision 2 against min 2 -> Allowed
                installer_v2 = self.work_dir / "install_v2.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer_v2,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                    override_installer_revision=2,
                )

                proc2 = subprocess.run(
                    [str(installer_v2), "--skip-signature", "--components", "journal", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc2.returncode, 0, f"Failed: {proc2.stderr}")
                res2 = json.loads(proc2.stdout.strip())
                self.assertEqual(res2["status"], "success")

                # 3. Installer with revision 3 against min 2 -> Allowed
                installer_v3 = self.work_dir / "install_v3.sh"
                build_installer(
                    repo_root=REPO_ROOT,
                    output_path=installer_v3,
                    platform_pub_path=pub,
                    platform_key_id=pin.key_id,
                    origin=server.origin,
                    override_installer_revision=3,
                )

                proc3 = subprocess.run(
                    [str(installer_v3), "--skip-signature", "--components", "journal", "--prefix", str(self.prefix), "--json"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc3.returncode, 0, f"Failed: {proc3.stderr}")
                res3 = json.loads(proc3.stdout.strip())
                self.assertEqual(res3["status"], "success")
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
