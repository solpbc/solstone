# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Tests for published-component platform recut preparation."""

from contextlib import ExitStack, contextmanager
import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.parse
import urllib.request

from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.generate import generate_platform_manifest
from solstone_platform.pins import PinSet
from solstone_platform.publish import publish_release
from solstone_platform.recut import _Downloader, _rename_noreplace, prepare_recut
from solstone_platform.refusals import HTTP_3XX, RELEASE_COHERENCE, SCHEMA_INVALID, Refusal
from solstone_platform.schema import load_platform_manifest_bytes, validate_arch_retention
from solstone_platform.sign import ephemeral_keypair, sign_manifest
from solstone_platform.testdest import FixtureDestination
from tests.install_test_helpers import LoopbackServer
from tools.fixture_builder import build_tiny_natives


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/payload")
            self.end_headers()
            return
        if self.path in {"/missing", "/forbidden"}:
            self.send_response(404 if self.path == "/missing" else 403)
            self.send_header("Content-Length", "0")
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

    def test_optional_fetch_absorbs_only_404(self):
        downloader = _Downloader()
        self.assertIsNone(downloader.fetch(f"{self.origin}/missing", self.root / "missing", optional=True))
        self.assertFalse((self.root / "missing").exists())
        self.assertEqual(downloader.sources, [])
        for path, optional in (("/missing", False), ("/forbidden", True)):
            with self.subTest(path=path, optional=optional), self.assertRaises(Refusal) as ctx:
                downloader.fetch(f"{self.origin}{path}", self.root / path.strip("/"), optional=optional)
            self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

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


ORIGIN = "https://updates.solstone.app"
ARM_DESKTOP = {"solstone-linux-{v}-linux-aarch64.rust-release-manifest.json",
               "solstone-linux-{v}-linux-aarch64.rust-release-manifest.json.minisig",
               "solstone-linux-{v}-linux-aarch64.tar.gz", "solstone-linux_{v}-1_arm64.deb",
               "solstone-linux-{v}-1.aarch64.rpm"}


class TestDesktopAarch64Recut(unittest.TestCase):
    """prepare-recut and publish carry desktop aarch64 when present and never drop it once carried."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir="/var/tmp")))
        self.secret, _, self.pin = self.stack.enter_context(ephemeral_keypair("recut platform"))
        self.native_keys = self.stack.enter_context(ephemeral_keypair("recut natives"))
        self.pinset = PinSet(journal=self.native_keys[2], desktop=self.native_keys[2], tmux=self.native_keys[2])
        self.repo = self.root / "repo"
        self.repo.mkdir()
        for name in ("compat", "contracts", "pins"):
            shutil.copytree(Path(__file__).parent.parent / name, self.repo / name)
        for argv in (["git", "init", "-q"], ["git", "add", "."],
                     ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"]):
            subprocess.run(argv, cwd=self.repo, check=True, capture_output=True)
        self.www = self.root / "www"
        self.www.mkdir()
        self.server = LoopbackServer(self.www)
        self.server.start()
        self.addCleanup(self.server.stop)
        self.dest = FixtureDestination(pinset=self.pinset, platform_pin=self.pin)
        self.sequence = 0

    def natives(self, desktop: str, tmux: str, desktop_arches) -> dict[str, Path]:
        self.sequence += 1

        @contextmanager
        def retained(*args, **kwargs):
            yield self.native_keys

        with patch("tools.fixture_builder.ephemeral_keypair", retained):
            _, dirs = build_tiny_natives(self.root / f"natives-{self.sequence}", desktop_version=desktop,
                                         tmux_version=tmux, desktop_arches=desktop_arches)
        return dirs

    def publish_base(self, desktop_arches, desktop_version: str = "2.0.37", *, schema_1: bool = False) -> dict:
        dirs = self.natives(desktop_version, "2.0.14", desktop_arches)
        data = generate_platform_manifest(version="2.0.6" if schema_1 else None,
                catalogue_revision=None if schema_1 else 1, lane="release", created_unix=1700000000,
                minimum_installer_revision=1,
                source_commit="3075c36b12fad469d4c9c0ab4555908fe8ecca1b", platform_key_id=self.pin.key_id,
                repo_root=self.repo, journal_dir=dirs["journal"], desktop_dir=dirs["desktop"], tmux_dir=dirs["tmux"],
                journal_origin=ORIGIN, pins=self.pinset)
        path = self.root / f"base-{self.sequence}"
        path.mkdir()
        (path / "platform.json").write_bytes(data)
        self.sign(path)
        publish_release(manifest_path=path / "platform.json", signature_path=path / "platform.json.minisig",
                        journal_dir=dirs["journal"], desktop_dir=dirs["desktop"], dest=self.dest)
        self.sync()
        return load_platform_manifest_bytes(data)

    def add_release(self, component: str, version: str, desktop_arches=("x86_64",)) -> None:
        dirs = self.natives(version, version, desktop_arches)
        target = self.www / {"desktop": "solstone-linux", "tmux": "solstone-tmux", "journal": "solstone-journal"}[component] / "release" / version
        target.mkdir(parents=True, exist_ok=True)
        for source in dirs[component].iterdir():
            if source.is_dir():
                shutil.copytree(source, target / source.name, dirs_exist_ok=True)
            else:
                shutil.copyfile(source, target / source.name)

    def add_journal_release(self, version: str) -> None:
        self.sequence += 1

        @contextmanager
        def retained(*args, **kwargs):
            yield self.native_keys

        with patch("tools.fixture_builder.ephemeral_keypair", retained):
            _, dirs = build_tiny_natives(self.root / f"natives-{self.sequence}", journal_version=version)
        target = self.www / "solstone-journal" / "release" / version
        target.mkdir(parents=True, exist_ok=True)
        for source in dirs["journal"].rglob("*"):
            if source.is_file():
                shutil.copyfile(source, target / source.name)

    def assert_refused_before_preparing(self, coordinate: str, replacements: dict[str, str], name: str = RELEASE_COHERENCE):
        before = sorted(path.name for path in self.root.iterdir())
        with self.assertRaises(Refusal) as ctx:
            self.prepare(coordinate, replacements)
        self.assertEqual(ctx.exception.name, name)
        # No candidate output and no staging directory was left behind.
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), before)
        return ctx.exception

    def manifest(self, path: Path) -> dict:
        return load_platform_manifest_bytes((path / "platform.json").read_bytes())

    def sign(self, path: Path) -> None:
        (path / "platform.json.minisig").write_bytes(
            sign_manifest((path / "platform.json").read_bytes(), self.secret, self.pin, passphrase_callback=lambda: ""))

    def sync(self) -> None:
        for key, obj in self.dest.objects.items():
            target = self.www / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(obj.body)

    def prepare(self, coordinate: str, replacements: dict[str, str]) -> Path:
        inner = urllib.request.build_opener()
        server_origin = self.server.origin

        class OfflineOpener:
            def open(self, request, timeout):
                parsed = urllib.parse.urlsplit(request.full_url)
                if parsed.netloc != "updates.solstone.app":
                    raise AssertionError("recut attempted an unexpected origin")
                return inner.open(urllib.request.Request(server_origin + parsed.path), timeout=timeout)

        path = self.root / f"prepared-{coordinate}"
        with patch("solstone_platform.recut.urllib.request.build_opener", return_value=OfflineOpener()):
            prepare_recut(version=coordinate, replacements=replacements, output_dir=path, repo_root=self.repo,
                          dest=self.dest, created_unix=1700000001, origin=ORIGIN)
        self.sign(path)
        return path

    def publish(self, path: Path):
        bootstrap_files = list((path / "bootstrap").glob("*.sh"))
        bootstrap_file = bootstrap_files[0] if bootstrap_files else None
        return publish_release(manifest_path=path / "platform.json", signature_path=path / "platform.json.minisig",
                               journal_dir=path / "journal", desktop_dir=path / "desktop",
                               dest=self.dest, bootstrap_file=bootstrap_file,
                               expected_latest_receipt=path / "recut-receipt.json")

    def desktop_arches(self, path: Path) -> set[str]:
        return set(load_platform_manifest_bytes((path / "platform.json").read_bytes())["components"]["desktop"]["arches"])

    def test_desktop_replacement_adds_aarch64_and_publish_claims_it(self):
        self.publish_base(("x86_64",))
        self.add_release("desktop", "2.0.38", ("x86_64", "aarch64"))
        prepared = self.prepare("2.0.6-r2", {"desktop": "2.0.38"})
        self.assertEqual(self.desktop_arches(prepared), {"x86_64", "aarch64"})
        self.assertEqual(self.publish(prepared).version, "2.0.6-r2")
        claimed = {key.rsplit("/", 1)[1] for key in self.dest.objects if key.startswith("solstone/release/2.0.6-r2/")}
        self.assertTrue({name.format(v="2.0.38") for name in ARM_DESKTOP} <= claimed)
        self.assertIn("solstone-linux-2.0.38-linux-x86_64.rust-release-manifest.json", claimed)

        # Once carried, an unrelated recut re-reads aarch64 from the base catalogue and keeps it exact.
        self.sync()
        self.add_release("desktop", "2.0.39", ("x86_64", "aarch64"))
        kept = self.prepare("2.0.6-r3", {"desktop": "2.0.39"})
        self.assertEqual(self.desktop_arches(kept), {"x86_64", "aarch64"})
        self.assertEqual(self.publish(kept).version, "2.0.6-r3")

        # A newer desktop release without aarch64 would drop it: refused before anything is prepared.
        self.sync()
        self.add_release("desktop", "2.0.40")
        with self.assertRaises(Refusal) as ctx:
            self.prepare("2.0.6-r4", {"desktop": "2.0.40"})
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)
        self.assertFalse((self.root / "prepared-2.0.6-r4").exists())

    def test_desktop_release_without_aarch64_stays_x86_64_only(self):
        self.publish_base(("x86_64",))
        self.add_release("desktop", "2.0.38")
        prepared = self.prepare("2.0.6-r2", {"desktop": "2.0.38"})
        self.assertEqual(self.desktop_arches(prepared), {"x86_64"})
        receipt = parse_json_strict((prepared / "recut-receipt.json").read_bytes())
        self.assertFalse(any("aarch64" in source["url"] for source in receipt["sources"] if "/solstone-linux/" in source["url"]))
        self.assertEqual(self.publish(prepared).version, "2.0.6-r2")

    def test_retention_is_checked_independently_of_preparation(self):
        base = self.publish_base(("x86_64", "aarch64"))
        candidate = copy.deepcopy(base)
        validate_arch_retention(base, candidate)
        del candidate["components"]["desktop"]["arches"]["aarch64"]
        with self.assertRaises(Refusal) as ctx:
            validate_arch_retention(base, candidate)
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)
        validate_arch_retention(candidate, base)

    def test_journal_recut_on_a_tmux_base_with_an_old_carried_desktop_refuses_before_preparing(self):
        self.publish_base(("x86_64",), desktop_version="2.0.14")
        self.add_journal_release("2.0.7")
        minimum = (self.repo / "compat" / "terminal_capture_desktop_minimum").read_text(encoding="utf-8").strip()
        refusal = self.assert_refused_before_preparing("2.0.7-r1", {"journal": "2.0.7"})
        self.assertIn(minimum, str(refusal))

    def test_journal_recut_on_a_tmux_base_with_a_carried_desktop_at_the_minimum_retires_tmux(self):
        base = self.publish_base(("x86_64",), desktop_version="2.0.37")
        self.add_journal_release("2.0.7")
        prepared = self.prepare("2.0.7-r1", {"journal": "2.0.7"})
        candidate = self.manifest(prepared)
        self.assertEqual(set(candidate["components"]), {"journal", "desktop"})
        self.assertEqual(candidate["minimum_installer_revision"], 20)
        self.assertEqual(candidate["components"]["desktop"], base["components"]["desktop"])
        self.assertEqual(candidate["components"]["journal"]["version"], "2.0.7")
        receipt = parse_json_strict((prepared / "recut-receipt.json").read_bytes())
        self.assertEqual(receipt.get("retired_components"), ["tmux"])

    def test_retirement_at_the_minimum_publishes_no_tmux_object_and_the_steady_state_follows(self):
        self.publish_base(("x86_64",), desktop_version="2.0.14")
        minimum = (self.repo / "compat" / "terminal_capture_desktop_minimum").read_text(encoding="utf-8").strip()
        self.add_release("desktop", minimum)
        prepared = self.prepare("2.0.6-r2", {"desktop": minimum})
        candidate = self.manifest(prepared)
        self.assertEqual(set(candidate["components"]), {"journal", "desktop"})
        self.assertEqual(candidate["minimum_installer_revision"], 20)
        self.assertEqual(candidate["components"]["desktop"]["version"], minimum)
        self.assertEqual(parse_json_strict((prepared / "recut-receipt.json").read_bytes()).get("retired_components"), ["tmux"])

        existing = {key: obj.body for key, obj in self.dest.objects.items()}
        self.assertEqual(self.publish(prepared).version, "2.0.6-r2")
        new_keys = [key for key in self.dest.objects if key.startswith("solstone/release/2.0.6-r2/")]
        self.assertTrue(new_keys)
        self.assertFalse([key for key in new_keys if "tmux" in key])
        for key, body in existing.items():
            if key == "solstone/release/latest":
                continue
            self.assertEqual(self.dest.objects[key].body, body, key)
        self.assertEqual(self.dest.get("solstone/release/latest").body, b"2.0.6-r2\n")

        # From the tmux-less floor-20 base: a journal recut succeeds and stays without tmux.
        self.sync()
        self.add_journal_release("2.0.7")
        steady = self.prepare("2.0.7-r1", {"journal": "2.0.7"})
        steady_manifest = self.manifest(steady)
        self.assertEqual(set(steady_manifest["components"]), {"journal", "desktop"})
        self.assertEqual(steady_manifest["minimum_installer_revision"], 20)
        self.assertNotIn("retired_components", parse_json_strict((steady / "recut-receipt.json").read_bytes()))
        self.assertEqual(self.publish(steady).version, "2.0.7-r1")
        self.assertFalse([key for key in self.dest.objects if key.startswith("solstone/release/2.0.7-r1/") and "tmux" in key])

        # tmux is never a replacement on a tmux-less base either.
        self.sync()
        self.assert_refused_before_preparing("2.0.7-r2", {"tmux": "2.0.15"}, SCHEMA_INVALID)

    def test_schema_1_base_recut_follows_the_retirement_rule(self):
        self.publish_base(("x86_64",), desktop_version="2.0.14", schema_1=True)
        self.add_journal_release("2.0.7")
        self.assert_refused_before_preparing("2.0.7-r1", {"journal": "2.0.7"})

        minimum = (self.repo / "compat" / "terminal_capture_desktop_minimum").read_text(encoding="utf-8").strip()
        self.add_release("desktop", minimum)
        prepared = self.prepare("2.0.6-r1", {"desktop": minimum})
        candidate = self.manifest(prepared)
        self.assertEqual(candidate["schema_version"], 2)
        self.assertEqual(candidate["minimum_installer_revision"], 20)
        self.assertEqual(set(candidate["components"]), {"journal", "desktop"})
        self.assertEqual(self.publish(prepared).version, "2.0.6-r1")

    def test_recut_tmux_replace_refused(self):
        # tmux is never a replacement, refused before the destination is read
        with self.assertRaises(Refusal) as ctx:
            self.prepare("2.0.6-r2", {"tmux": "2.0.15"})
        self.assertEqual(ctx.exception.name, SCHEMA_INVALID)

    def test_recut_carried_desktop_below_minimum_refuses(self):
        self.publish_base(("x86_64",), desktop_version="2.0.14")
        self.add_release("desktop", "2.0.36")
        with self.assertRaises(Refusal) as ctx:
            self.prepare("2.0.6-r2", {"desktop": "2.0.36"})
        self.assertEqual(ctx.exception.name, RELEASE_COHERENCE)

    def test_recut_tmux_retirement_at_minimum_succeeds(self):
        self.publish_base(("x86_64",), desktop_version="2.0.37")
        self.add_release("desktop", "2.0.38")
        prepared = self.prepare("2.0.6-r2", {"desktop": "2.0.38"})
        manifest = parse_json_strict((prepared / "platform.json").read_bytes())
        self.assertEqual(manifest["minimum_installer_revision"], 20)
        self.assertNotIn("tmux", manifest["components"])
        receipt = parse_json_strict((prepared / "recut-receipt.json").read_bytes())
        self.assertEqual(receipt.get("retired_components"), ["tmux"])


if __name__ == "__main__":
    unittest.main()
