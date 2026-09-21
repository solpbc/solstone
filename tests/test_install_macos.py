# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""macOS contract tests for native app-only installation."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest

from solstone_platform.sign import ephemeral_keypair
from tests.install_test_helpers import write_path_stub
from tools.build_installer import build_installer


REPO_ROOT = Path(__file__).resolve().parent.parent


class MacDownloadHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.request_paths.append(self.path)
        redirects = {
            "/download/journal/latest": "/journal-macos/journal.dmg",
            "/download/macos/latest": "/solstone-macos/solstone.dmg",
        }
        if self.path in redirects:
            self.send_response(302)
            location = self.server.redirect_override or self.server.origin + redirects[self.path]
            self.send_header("Location", location)
            self.end_headers()
            return
        if self.path in redirects.values():
            body = (self.path + "\n").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, format, *args):
        pass


class MacDownloadServer:
    def __init__(self):
        self.httpd = HTTPServer(("127.0.0.1", 0), MacDownloadHandler)
        self.origin = f"http://127.0.0.1:{self.httpd.server_port}"
        self.httpd.origin = self.origin
        self.httpd.request_paths = []
        self.httpd.redirect_override = None
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def request_paths(self):
        return self.httpd.request_paths

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestInstallMacOS(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.applications = self.root / "Applications"
        self.fake_bin = self.root / "fake-bin"
        self.home.mkdir()
        self.applications.mkdir()
        self.server = MacDownloadServer()
        self.server.start()
        self.addCleanup(self.server.stop)
        self._write_tool_doubles()

        keys = ephemeral_keypair("mac installer")
        sec, pub, pin = keys.__enter__()
        self.addCleanup(keys.__exit__, None, None, None)
        del sec
        self.installer = self.root / "install.sh"
        build_installer(
            repo_root=REPO_ROOT,
            output_path=self.installer,
            platform_pub_path=pub,
            platform_key_id=pin.key_id,
            origin=self.server.origin,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _write_tool_doubles(self):
        write_path_stub(self.fake_bin, "sw_vers", "printf '%s\\n' \"${SOLSTONE_FAKE_MAC_VERSION:-15.0}\"\n")
        write_path_stub(
            self.fake_bin,
            "codesign",
            "if [ \"$1\" = -dvvv ]; then\n"
            "  for arg do target=$arg; done\n"
            "  case \"$target\" in\n"
            "    *journal.app) identifier=app.solstone.journal ;;\n"
            "    *solstone.app) identifier=app.solstone.observer ;;\n"
            "    *) exit 2 ;;\n"
            "  esac\n"
            "  team=7QCG8V4M6H\n"
            "  [ \"${SOLSTONE_FAKE_BAD_TEAM:-0}\" = 0 ] || team=WRONGTEAM\n"
            "  printf 'Identifier=%s\\nTeamIdentifier=%s\\n' \"$identifier\" \"$team\" >&2\n"
            "fi\n"
            "exit 0\n",
        )
        write_path_stub(self.fake_bin, "spctl", "exit \"${SOLSTONE_FAKE_SPCTL_EXIT:-0}\"\n")
        write_path_stub(
            self.fake_bin,
            "hdiutil",
            "mode=$1\n"
            "shift\n"
            "case \"$mode\" in\n"
            "  attach)\n"
            "    dmg=$1\n"
            "    shift\n"
            "    mount=\n"
            "    while [ $# -gt 0 ]; do\n"
            "      if [ \"$1\" = -mountpoint ]; then mount=$2; shift 2; else shift; fi\n"
            "    done\n"
            "    case \"$dmg\" in *journal.dmg) app=journal.app ;; *) app=solstone.app ;; esac\n"
            "    mkdir -p \"$mount/$app\"\n"
            "    printf verified > \"$mount/$app/payload\"\n"
            "    ;;\n"
            "  detach) ;;\n"
            "  *) exit 2 ;;\n"
            "esac\n",
        )
        write_path_stub(self.fake_bin, "ditto", "cp -R \"$1\" \"$2\"\n")

    def run_installer(self, *args, extra_env=None):
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(self.home),
                "SOLSTONE_TEST_HOST_OS": "Darwin",
                "SOLSTONE_TEST_HOST_ARCH": "arm64",
                "SOLSTONE_TEST_CODESIGN": str(self.fake_bin / "codesign"),
                "SOLSTONE_TEST_SPCTL": str(self.fake_bin / "spctl"),
                "SOLSTONE_TEST_HDIUTIL": str(self.fake_bin / "hdiutil"),
                "SOLSTONE_TEST_DITTO": str(self.fake_bin / "ditto"),
                "SOLSTONE_TEST_SW_VERS": str(self.fake_bin / "sw_vers"),
                "SOLSTONE_TEST_APPLICATIONS": str(self.applications),
                "SOLSTONE_TEST_MACOS_ORIGIN": self.server.origin,
            }
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [str(self.installer), *args, "--json"],
            capture_output=True,
            text=True,
            env=env,
        )

    @staticmethod
    def result(proc):
        return json.loads(proc.stdout.strip())

    def test_installs_both_verified_apps_without_cli_runtime(self):
        proc = self.run_installer("--components", "journal,app")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        result = self.result(proc)
        self.assertEqual(result["status"], "success")
        self.assertEqual(set(result["components"]), {"journal", "app"})
        self.assertEqual(result["components"]["journal"]["route"], "app")
        self.assertTrue((self.applications / "journal.app" / "payload").is_file())
        self.assertTrue((self.applications / "solstone.app" / "payload").is_file())
        self.assertFalse((self.home / ".local").exists())
        self.assertFalse((self.home / "Library" / "LaunchAgents").exists())

    def test_existing_app_is_verified_and_left_unchanged(self):
        journal = self.applications / "journal.app"
        journal.mkdir()
        marker = journal / "owner-data"
        marker.write_text("keep", encoding="utf-8")
        proc = self.run_installer("--components", "journal")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(self.result(proc)["components"]["journal"]["status"], "unchanged")
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertEqual(self.server.request_paths, [])

    def test_installs_journal_app_beside_legacy_cli_without_changing_it(self):
        legacy = self.home / ".local" / "solstone-journal" / "install-receipt"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("owner-state", encoding="utf-8")
        proc = self.run_installer("--components", "journal")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        result = self.result(proc)
        self.assertEqual(result["status"], "success")
        self.assertTrue((self.applications / "journal.app").is_dir())
        self.assertEqual(legacy.read_text(encoding="utf-8"), "owner-state")
        self.assertFalse((self.home / "Library" / "LaunchAgents").exists())

    def test_solstone_app_can_install_beside_legacy_journal(self):
        legacy = self.home / ".local" / "bin" / "journal"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("legacy", encoding="utf-8")
        proc = self.run_installer("--components", "app")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue((self.applications / "solstone.app").is_dir())
        self.assertEqual(legacy.read_text(encoding="utf-8"), "legacy")

    def test_upgrade_flag_refuses_because_apps_own_updates(self):
        proc = self.run_installer("--components", "journal", "--upgrade")
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(self.result(proc)["root_code"], "app-updates-in-app")
        self.assertEqual(self.server.request_paths, [])

    def test_intel_mac_refuses(self):
        proc = self.run_installer(
            "--components",
            "journal",
            extra_env={"SOLSTONE_TEST_HOST_ARCH": "x86_64"},
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(self.result(proc)["root_code"], "unsupported-arch")
        self.assertEqual(self.server.request_paths, [])

    def test_dry_run_verifies_without_installing(self):
        proc = self.run_installer("--components", "all", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        result = self.result(proc)
        self.assertEqual(result["root_code"], "dry-run-completed")
        self.assertEqual(result["components"]["journal"]["status"], "planned")
        self.assertFalse((self.applications / "journal.app").exists())
        self.assertFalse((self.applications / "solstone.app").exists())

    def test_dry_run_verifies_current_downloads_when_apps_exist(self):
        (self.applications / "journal.app").mkdir()
        (self.applications / "solstone.app").mkdir()
        journal_marker = self.applications / "journal.app" / "owner-state"
        journal_marker.write_text("keep", encoding="utf-8")
        proc = self.run_installer("--components", "all", "--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(journal_marker.read_text(encoding="utf-8"), "keep")
        self.assertEqual(
            self.server.request_paths,
            [
                "/download/journal/latest",
                "/journal-macos/journal.dmg",
                "/download/macos/latest",
                "/solstone-macos/solstone.dmg",
            ],
        )

    def test_wrong_signing_team_refuses(self):
        proc = self.run_installer(
            "--components",
            "journal",
            extra_env={"SOLSTONE_FAKE_BAD_TEAM": "1"},
        )
        self.assertNotEqual(proc.returncode, 0)
        result = self.result(proc)
        self.assertEqual(result["root_code"], "identity-mismatch")
        self.assertFalse((self.applications / "journal.app").exists())

    def test_redirect_outside_release_channel_refuses(self):
        self.server.httpd.redirect_override = "https://example.com/journal.dmg"
        proc = self.run_installer("--components", "journal")
        self.assertNotEqual(proc.returncode, 0)
        result = self.result(proc)
        self.assertEqual(result["root_code"], "redirect-refused")
        self.assertFalse((self.applications / "journal.app").exists())


if __name__ == "__main__":
    unittest.main()
