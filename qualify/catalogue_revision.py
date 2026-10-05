# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Explicit offline qualification: PYTHONPATH=src:. python3 qualify/catalogue_revision.py -v.

The frozen revision-7 template is byte-for-byte source from commit
3560842f416642ed0d00552f1f7335aa9820e03b, SHA256
6159d066b8253b46c4547b5574d0c8f8cb331ed42ba57d2cf9ed33ef594728c8.
"""

from contextlib import ExitStack, contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import urllib.parse
import urllib.request

from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.destination import PutResult, GetResult, ResultStatus
from solstone_platform.generate import generate_platform_manifest
from solstone_platform.pins import PinSet
from solstone_platform.publish import publish_release
from solstone_platform.recut import prepare_recut
from solstone_platform.refusals import Refusal
from solstone_platform.schema import catalogue_coordinate, load_platform_manifest_bytes, parse_catalogue_coordinate
from solstone_platform.sign import ephemeral_keypair, sign_manifest
from solstone_platform.testdest import FixtureDestination, StoredObject
from tests.install_test_helpers import HermeticInstallerTestCase, LoopbackServer, receipt_section, setup_test_release_server
from tests.test_install_package_lifecycle import TestInstallPackageLifecycle as PackageDriver
from tools.build_installer import build_installer
from tools.fixture_builder import build_tiny_natives

ROOT = Path(__file__).resolve().parent.parent
ORIGIN = "https://updates.solstone.app"
LATEST = "solstone/release/latest"


class PublicationQualification(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.temp = self.stack.enter_context(tempfile.TemporaryDirectory(dir="/var/tmp"))
        self.root = Path(self.temp)
        self.keys = [self.stack.enter_context(ephemeral_keypair("catalogue qualification")) for _ in range(4)]
        self.secret, self.public, self.pin = self.keys[0]
        self.pinset = PinSet(journal=self.keys[1][2], desktop=self.keys[2][2], tmux=self.keys[3][2])
        self.repo = self.root / "repo"
        self.repo.mkdir()
        for name in ("compat", "contracts", "pins"):
            shutil.copytree(ROOT / name, self.repo / name)
        for argv in (["git", "init", "-q"], ["git", "add", "."],
                     ["git", "-c", "user.name=Qualification", "-c", "user.email=qualification@example.invalid",
                      "commit", "-qm", "Fixture contracts"]):
            subprocess.run(argv, cwd=self.repo, check=True, capture_output=True)
        self.www = self.root / "www"
        self.www.mkdir()
        self.server = LoopbackServer(self.www)
        self.server.start()
        self.stack.callback(self.server.stop)
        self.dest = FixtureDestination(pinset=self.pinset, platform_pin=self.pin)
        self.sequence = 0
        self.base = self.release("9.0.0", journal="2.0.33", desktop="2.0.14", tmux="2.0.14", floor=1)
        self.publish(self.base)
        self.sync_destination()
        self.add_native_release("2.0.33", "2.0.15", "2.0.15")
        self.add_native_release("2.0.34", "2.0.15", "2.0.15")

    def tearDown(self):
        self.stack.close()

    def natives(self, journal, desktop, tmux):
        self.sequence += 1
        selected = iter(self.keys[1:])

        @contextmanager
        def retained_keys(*args, **kwargs):
            yield next(selected)

        with patch("tools.fixture_builder.ephemeral_keypair", retained_keys):
            pins, dirs = build_tiny_natives(self.root / f"natives-{self.sequence}",
                                           journal_version=journal, desktop_version=desktop, tmux_version=tmux)
        self.assertEqual(pins, self.pinset)
        return dirs

    def release(self, coordinate, *, journal, desktop, tmux, floor=8):
        _, _, revision = parse_catalogue_coordinate(coordinate)
        dirs = self.natives(journal, desktop, tmux)
        data = generate_platform_manifest(version=coordinate if revision is None else None,
                catalogue_revision=revision, lane="release", created_unix=1700000000,
                source_commit="3075c36b12fad469d4c9c0ab4555908fe8ecca1b", platform_key_id=self.pin.key_id,
                repo_root=self.repo, journal_dir=dirs["journal"], desktop_dir=dirs["desktop"], tmux_dir=dirs["tmux"],
                journal_origin=ORIGIN, pins=self.pinset)
        obj = parse_json_strict(data)
        obj["minimum_installer_revision"] = floor
        path = self.root / f"release-{self.sequence}"
        path.mkdir()
        for name, directory in dirs.items():
            shutil.copytree(directory, path / name)
        (path / "platform.json").write_bytes(canonical_json_bytes(obj))
        self.sign(path)
        return path

    def sign(self, path):
        (path / "platform.json.minisig").write_bytes(sign_manifest((path / "platform.json").read_bytes(),
                self.secret, self.pin, passphrase_callback=lambda: ""))

    def publish(self, path):
        receipt = path / "recut-receipt.json"
        bootstrap = list((path / "bootstrap").glob("*.sh"))
        return publish_release(manifest_path=path / "platform.json", signature_path=path / "platform.json.minisig",
                journal_dir=path / "journal", desktop_dir=path / "desktop", tmux_dir=path / "tmux", dest=self.dest,
                bootstrap_file=bootstrap[0] if bootstrap else None,
                **({"expected_latest_receipt": receipt} if receipt.is_file() else {}))

    def sync_destination(self):
        for key, obj in self.dest.objects.items():
            target = self.www / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(obj.body)

    def add_native_release(self, journal, desktop, tmux):
        dirs = self.natives(journal, desktop, tmux)
        for name, version in (("journal", journal), ("desktop", desktop), ("tmux", tmux)):
            prefix = {"journal": "solstone-journal", "desktop": "solstone-linux", "tmux": "solstone-tmux"}[name]
            target = self.www / prefix / "release" / version
            target.mkdir(parents=True, exist_ok=True)
            for source in dirs[name].rglob("*"):
                if source.is_file():
                    shutil.copyfile(source, target / source.name)

    def prepare(self, coordinate, replacements, *, name=None):
        # Keep signed production URL facts while transporting through a real loopback HTTP server.
        original_build = urllib.request.build_opener
        inner = original_build()
        server_origin = self.server.origin

        class OfflineOpener:
            def open(self, request, timeout):
                parsed = urllib.parse.urlsplit(request.full_url)
                if parsed.scheme != "https" or parsed.netloc != "updates.solstone.app":
                    raise AssertionError("qualification attempted an unexpected origin")
                local = urllib.request.Request(server_origin + parsed.path, headers=dict(request.headers))
                return inner.open(local, timeout=timeout)

        path = self.root / (name or ("prepared-" + coordinate + "-" + str(len(list(self.root.glob('prepared-*'))))))
        with patch("solstone_platform.recut.urllib.request.build_opener", return_value=OfflineOpener()):
            prepare_recut(version=coordinate, replacements=replacements, output_dir=path, repo_root=self.repo,
                          dest=self.dest, created_unix=1700000001, origin=ORIGIN)
        self.sign(path)
        return path

    def refuse(self, function, code, *args, **kwargs):
        with self.assertRaises(Refusal) as result:
            function(*args, **kwargs)
        self.assertEqual(result.exception.name, code, str(result.exception))

    def test_signed_desktop_then_tmux_then_journal_sequence_and_exact_retry(self):
        first = self.prepare("2.0.33-r1", {"desktop": "2.0.15"})
        first_obj = load_platform_manifest_bytes((first / "platform.json").read_bytes())
        base_obj = load_platform_manifest_bytes((self.base / "platform.json").read_bytes())
        self.assertEqual(first_obj["components"]["journal"], base_obj["components"]["journal"])
        self.assertEqual(first_obj["components"]["tmux"], base_obj["components"]["tmux"])
        self.assertEqual(first_obj["minimum_installer_revision"], 8)
        self.assertEqual(self.publish(first).version, "2.0.33-r1")
        self.assertFalse(self.publish(first).latest_promoted)
        self.sync_destination()
        second = self.prepare("2.0.33-r2", {"tmux": "2.0.15"})
        self.assertTrue(self.publish(second).latest_promoted)
        self.sync_destination()
        third = self.prepare("2.0.34-r1", {"journal": "2.0.34"})
        self.assertTrue(self.publish(third).latest_promoted)
        self.assertEqual(self.dest.get(LATEST).body, b"2.0.34-r1\n")
        cas = [item for item in self.dest.ledger if item.method == "compare_and_swap"]
        self.assertEqual(len(cas), 4)  # Initial fixture base and three prepared publications.

    def test_first_tmux_only_migration_and_positive_gap(self):
        candidate = self.prepare("2.0.33-r7", {"tmux": "2.0.15"})
        self.assertTrue(self.publish(candidate).latest_promoted)
        self.assertEqual(self.dest.get(LATEST).body, b"2.0.33-r7\n")

    def test_preparation_rejects_wrong_core_nonincrease_and_floor_twin(self):
        self.refuse(self.prepare, "release-coherence", "2.0.32-r1", {"tmux": "2.0.15"})
        self.refuse(self.prepare, "release-coherence", "9.0.1", {"tmux": "2.0.15"})
        first = self.prepare("2.0.33-r1", {"tmux": "2.0.15"})
        self.publish(first)
        self.sync_destination()
        for coordinate in ("2.0.33-r1", "2.0.32-r9", "9.0.1"):
            self.refuse(self.prepare, "release-coherence", coordinate, {"desktop": "2.0.15"})

    def mutate_candidate(self, path, change):
        obj = parse_json_strict((path / "platform.json").read_bytes())
        change(obj)
        data = canonical_json_bytes(obj)
        (path / "platform.json").write_bytes(data)
        receipt = parse_json_strict((path / "recut-receipt.json").read_bytes())
        receipt["candidate"]["version"] = catalogue_coordinate(obj)
        receipt["candidate"]["sha256"] = hashlib.sha256(data).hexdigest()
        for item in receipt["prepared_files"]:
            if item["path"] == "platform.json":
                item.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        (path / "recut-receipt.json").write_bytes(canonical_json_bytes(receipt))
        self.sign(path)

    def test_independent_publisher_refuses_transition_twins_before_writes(self):
        cases = [lambda obj: obj.update(minimum_installer_revision=7),
                 lambda obj: obj.update(protocol_version=2),
                 lambda obj: obj.update(lane="staging"),
                 lambda obj: obj["components"]["journal"]["provenance"].update(retention_window=99)]
        for number, change in enumerate(cases):
            candidate = self.prepare("2.0.33-r1", {"desktop": "2.0.15"}, name=f"twin-{number}")
            # Invalid protocol cannot pass the signer, which is itself an independent boundary.
            if number == 1:
                self.refuse(self.mutate_candidate, "schema-invalid", candidate, change)
                continue
            self.mutate_candidate(candidate, change)
            self.dest.ledger.clear()
            self.refuse(self.publish, "release-coherence", candidate)
            self.assertFalse(any(item.method in {"put_if_absent", "compare_and_swap"} for item in self.dest.ledger))
        first = self.prepare("2.0.33-r1", {"tmux": "2.0.15"})
        self.publish(first)
        self.sync_destination()
        candidate = self.prepare("2.0.33-r2", {"desktop": "2.0.15"})
        self.mutate_candidate(candidate, lambda obj: obj.update(minimum_installer_revision=9))
        self.dest.ledger.clear()
        self.refuse(self.publish, "release-coherence", candidate)
        self.assertFalse(any(item.method == "put_if_absent" for item in self.dest.ledger))

    def test_competing_candidates_refuse_stale_base_without_claims(self):
        first = self.prepare("2.0.33-r1", {"desktop": "2.0.15"})
        competing = self.prepare("2.0.33-r2", {"tmux": "2.0.15"})
        self.publish(first)
        self.dest.ledger.clear()
        self.refuse(self.publish, "release-coherence", competing)
        self.assertFalse(any(item.method in {"put_if_absent", "compare_and_swap"} for item in self.dest.ledger))
        self.assertEqual(self.dest.get(LATEST).body, b"2.0.33-r1\n")

    def test_competing_publication_between_read_and_cas_never_demotes(self):
        first = self.prepare("2.0.33-r1", {"desktop": "2.0.15"})
        competing = self.prepare("2.0.33-r2", {"tmux": "2.0.15"})
        saved_cas = self.dest.compare_and_swap
        requests = []
        def observed_cas(**kwargs):
            requests.append(kwargs.copy())
            return saved_cas(**kwargs)
        self.dest.cas_pre_hook = lambda: self.publish(competing)
        with patch.object(self.dest, "compare_and_swap", side_effect=observed_cas):
            self.refuse(self.publish, "rollback-refused", first)
        first_writes = [request for request in requests if request["body"] == b"2.0.33-r1\n"]
        self.assertEqual(len(first_writes), 1)
        receipt = parse_json_strict((first / "recut-receipt.json").read_bytes())
        self.assertEqual(first_writes[0]["expected_etag"], receipt["base"]["etag"])
        self.assertEqual(self.dest.get(LATEST).body, b"2.0.33-r2\n")

    def test_partial_manifest_claim_original_retry_and_conflicting_bytes(self):
        candidate = self.prepare("2.0.33-r1", {"desktop": "2.0.15"})
        original = (candidate / "platform.json").read_bytes()
        saved_put = self.dest.put_if_absent

        def fail_signature(**kwargs):
            if kwargs["key"].endswith("/platform.json.minisig"):
                return PutResult(status=ResultStatus.SERVER_ERROR)
            return saved_put(**kwargs)

        with patch.object(self.dest, "put_if_absent", side_effect=fail_signature):
            with self.assertRaises(Refusal):
                self.publish(candidate)
        key = "solstone/release/2.0.33-r1/platform.json"
        self.assertEqual(self.dest.objects[key].body, original)
        self.assertEqual(self.dest.get(LATEST).body, b"9.0.0\n")
        self.mutate_candidate(candidate, lambda obj: obj.update(created_unix=1700000002))
        self.refuse(self.publish, "same-version-different-bytes", candidate)
        self.assertEqual(self.dest.objects[key].body, original)
        (candidate / "platform.json").write_bytes(original)
        receipt = parse_json_strict((candidate / "recut-receipt.json").read_bytes())
        receipt["candidate"]["sha256"] = hashlib.sha256(original).hexdigest()
        for item in receipt["prepared_files"]:
            if item["path"] == "platform.json":
                item.update(sha256=hashlib.sha256(original).hexdigest(), bytes=len(original))
        (candidate / "recut-receipt.json").write_bytes(canonical_json_bytes(receipt))
        self.sign(candidate)
        self.assertTrue(self.publish(candidate).latest_promoted)
        self.assertFalse(self.publish(candidate).latest_promoted)

    def test_abandon_first_and_new_core_attempts_at_higher_revision(self):
        for first_coordinate, next_coordinate, replacements in (
                ("2.0.33-r1", "2.0.33-r3", {"desktop": "2.0.15"}),
                ("2.0.34-r1", "2.0.34-r4", {"journal": "2.0.34"})):
            self.sync_destination()
            failed = self.prepare(first_coordinate, replacements)
            saved_put = self.dest.put_if_absent
            def fail_signature(**kwargs):
                if kwargs["key"] == f"solstone/release/{first_coordinate}/platform.json.minisig":
                    return PutResult(status=ResultStatus.SERVER_ERROR)
                return saved_put(**kwargs)
            with patch.object(self.dest, "put_if_absent", side_effect=fail_signature):
                with self.assertRaises(Refusal):
                    self.publish(failed)
            abandoned_key = f"solstone/release/{first_coordinate}/platform.json"
            abandoned = self.dest.objects[abandoned_key]
            following = self.prepare(next_coordinate, replacements)
            self.assertTrue(self.publish(following).latest_promoted)
            self.assertEqual(self.dest.objects[abandoned_key], abandoned)
            self.assertEqual(self.dest.get(LATEST).body, (next_coordinate + "\n").encode())

    def test_indeterminate_cas_never_adopts_new_generation(self):
        candidate = self.prepare("2.0.33-r1", {"desktop": "2.0.15"})
        self.dest.ledger.clear()
        self.dest.fail_next_cas = ResultStatus.TIMEOUT
        def unreadable_latest():
            self.dest.get_overrides[LATEST] = GetResult(status=ResultStatus.TIMEOUT)
        self.dest.cas_pre_hook = unreadable_latest
        self.refuse(self.publish, "publish-indeterminate", candidate)
        self.assertEqual(sum(item.method == "compare_and_swap" for item in self.dest.ledger), 1)
        self.assertEqual(self.dest.get(LATEST).body, b"9.0.0\n")
        self.assertTrue(self.publish(candidate).latest_promoted)

    def test_complete_dependencies_precede_single_cas(self):
        candidate = self.prepare("2.0.33-r1", {"desktop": "2.0.15"})
        saved_put = self.dest.put_if_absent
        def corrupt_dependency(**kwargs):
            result = saved_put(**kwargs)
            if kwargs["key"].endswith(".deb"):
                obj = self.dest.objects[kwargs["key"]]
                self.dest.objects[kwargs["key"]] = StoredObject(b"corrupt", obj.etag, obj.content_type, obj.cache_control)
            return result
        self.dest.ledger.clear()
        with patch.object(self.dest, "put_if_absent", side_effect=corrupt_dependency):
            with self.assertRaises(Refusal):
                self.publish(candidate)
        self.assertFalse(any(item.method == "compare_and_swap" for item in self.dest.ledger))
        self.assertEqual(self.dest.get(LATEST).body, b"9.0.0\n")


class ReaderQualification(HermeticInstallerTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir="/var/tmp")))
        self.secret, self.public, self.pin = self.stack.enter_context(ephemeral_keypair("reader qualification"))
        self.server, self.www = setup_test_release_server(self.root, self.secret, self.pin,
                                                        version="9.0.0", min_installer_revision=1)
        self.stack.callback(self.server.stop)
        self.installer = self.root / "reader.sh"
        build_installer(ROOT, self.installer, platform_pub_path=self.public,
                        platform_key_id=self.pin.key_id, origin=self.server.origin)
        self.base = parse_json_strict((self.www / "solstone/release/9.0.0/platform.json").read_bytes())
        self.env = dict(os.environ, HOME=str(self.root / "home"), XDG_DATA_HOME=str(self.root / "data"))
        Path(self.env["HOME"]).mkdir()
        self.prefix = self.root / "prefix"
        self.prefix.mkdir()
        self.sentinel = self.prefix / "owner-record"
        self.sentinel.write_bytes(b"synthetic owner record\n")

    def tearDown(self):
        self.stack.close()

    def signed(self, obj, coordinate, latest=None):
        target = self.www / "solstone/release" / coordinate
        target.mkdir(parents=True, exist_ok=True)
        manifest = target / "platform.json"
        manifest.write_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
        # Sign deliberately malformed fixtures directly, bypassing the production semantic signer.
        subprocess.run(["minisign", "-S", "-s", str(self.secret), "-m", str(manifest),
                        "-x", str(target / "platform.json.minisig"), "-t", "reader qualification"],
                       input=b"\n", check=True, capture_output=True)
        (self.www / "solstone/release/latest").write_bytes(latest if latest is not None else (coordinate + "\n").encode())
        return manifest.read_bytes()

    def run_reader(self, *extra):
        result = subprocess.run(["sh", str(self.installer), "--list", "--json", *extra],
                                env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(self.sentinel.read_bytes(), b"synthetic owner record\n")
        self.assertEqual(list(self.prefix.iterdir()), [self.sentinel])
        return result, json.loads(result.stdout)

    def revised(self, revision=1):
        obj = copy.deepcopy(self.base)
        obj.update(schema_version=2, version=obj["components"]["journal"]["version"],
                   catalogue_revision=revision, minimum_installer_revision=8)
        return obj

    def test_signed_legacy_and_revised_listing_and_distinct_floor(self):
        for obj, coordinate in ((self.base, "9.0.0"), (self.revised(9), "2.0.6-r9")):
            self.signed(obj, coordinate)
            result, report = self.run_reader()
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(report["platform_version"], coordinate)
            self.assertEqual(report["verification_layers"], "minisign+digest")
        obj = self.revised(9)
        obj["minimum_installer_revision"] = 9
        self.signed(obj, "2.0.6-r9")
        result, report = self.run_reader()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report["root_code"], "revision-too-old")

    def test_signed_scalar_type_and_identity_parity(self):
        changes = [("schema_version", "2"), ("schema_version", True), ("protocol_version", "1"),
                   ("protocol_version", True), ("catalogue_revision", "1"), ("catalogue_revision", True),
                   ("catalogue_revision", 1.0), ("catalogue_revision", 0), ("catalogue_revision", -1),
                   ("catalogue_revision", 10**20), ("version", "2.0.7"),
                   ("minimum_installer_revision", "8"), ("minimum_installer_revision", True)]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                obj = self.revised()
                obj[key] = value
                raw = self.signed(obj, "2.0.6-r1")
                with self.assertRaises(Refusal):
                    load_platform_manifest_bytes(raw)
                result, report = self.run_reader()
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(report["root_code"], {"schema-invalid", "version-invalid", "release-coherence"})
        obj = self.revised()
        self.signed(obj, "2.0.6-r2")
        result, report = self.run_reader()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report["root_code"], "release-coherence")
        obj = copy.deepcopy(self.base)
        obj["catalogue_revision"] = 1
        self.signed(obj, "9.0.0")
        self.assertNotEqual(self.run_reader()[0].returncode, 0)

    def test_maximum_coordinate_crlf_and_invalid_framing(self):
        n = "9" * 20
        coordinate = f"{n}.{n}.{n}-r{n}"
        self.assertEqual(len(coordinate), 84)
        obj = self.revised(int(n))
        obj["version"] = obj["components"]["journal"]["version"] = f"{n}.{n}.{n}"
        for framing in (coordinate.encode(), (coordinate + "\n").encode(), (coordinate + "\r\n").encode()):
            self.signed(obj, coordinate, framing)
            result, report = self.run_reader()
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(report["platform_version"], coordinate)
        invalid = [b"2.0.6-r0\n", b"2.0.6-r01\n", b"2.0.6-r1\n\n", b"2.0.6-r1\x00\n",
                   b"2.0.6-r1\r", b"2.0.6-r1 \n", b"2.0.6-rc1\n", "2.0.6-r١\n".encode(),
                   b"2.0.6-r" + b"9" * 21 + b"\n", (coordinate + "\r\n\n").encode()]
        for framing in invalid:
            with self.subTest(framing=framing):
                (self.www / "solstone/release/latest").write_bytes(framing)
                self.server.request_paths.clear()
                result, report = self.run_reader()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(report["root_code"], {"latest-invalid", "response-too-large"})
                self.assertEqual(self.server.request_paths, ["/solstone/release/latest"])

    def test_frozen_revision7_refuses_after_only_pointer(self):
        template = ROOT / "qualify/install.sh.in.rev7"
        self.assertEqual(hashlib.sha256(template.read_bytes()).hexdigest(),
                         "6159d066b8253b46c4547b5574d0c8f8cb331ed42ba57d2cf9ed33ef594728c8")
        old = self.root / "reader7.sh"
        build_installer(ROOT, old, platform_pub_path=self.public, platform_key_id=self.pin.key_id,
                        origin=self.server.origin, override_installer_revision=7, template_path=template)
        self.signed(self.revised(), "2.0.6-r1")
        self.server.request_paths.clear()
        result = subprocess.run(["sh", str(old), "--list", "--json"], env=self.env,
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["root_code"], "latest-invalid")
        self.assertEqual(self.server.request_paths, ["/solstone/release/latest"])
        self.assertEqual(list(self.prefix.iterdir()), [self.sentinel])


class PackageQualification(HermeticInstallerTestCase):
    def setUp(self):
        self.driver = PackageDriver(methodName="runTest")
        self.driver.setUp()
        self.stack = ExitStack()
        self.secret, self.public, self.pin = self.stack.enter_context(ephemeral_keypair("package qualification"))
        self.server, self.www = setup_test_release_server(self.driver.work_dir, self.secret, self.pin,
                                                        version="9.0.0", min_installer_revision=1)
        self.stack.callback(self.server.stop)
        self.old = self.driver.work_dir / "old.sh"
        self.new = self.driver.work_dir / "new.sh"
        build_installer(ROOT, self.old, platform_pub_path=self.public, platform_key_id=self.pin.key_id,
                        origin=self.server.origin, override_installer_revision=7,
                        template_path=ROOT / "qualify/install.sh.in.rev7")
        build_installer(ROOT, self.new, platform_pub_path=self.public, platform_key_id=self.pin.key_id,
                        origin=self.server.origin)
        self.sentinel = self.driver.work_dir / "tree-data/synthetic-owner-record"
        self.sentinel.parent.mkdir(parents=True, exist_ok=True)
        self.sentinel.write_bytes(b"owner data\n")

    def tearDown(self):
        self.stack.close()
        self.driver.tearDown()

    def run_install(self, reader, coordinate, components="cli", *extra):
        result = subprocess.run(["sh", str(reader), "--skip-signature", "--route", "deb", "--components", components,
                "--version", coordinate, "--prefix", str(self.driver.prefix), "--no-start", "--json", *extra],
                env=self.driver.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(self.sentinel.read_bytes(), b"owner data\n")
        return result

    def revision(self, number):
        base_dir = self.www / "solstone/release/9.0.0"
        target = self.www / "solstone/release" / f"2.0.6-r{number}"
        shutil.copytree(base_dir, target)
        obj = parse_json_strict((target / "platform.json").read_bytes())
        obj.update(schema_version=2, version="2.0.6", catalogue_revision=number, minimum_installer_revision=8)
        # An unselected desktop shipment changes catalogue facts while CLI native facts remain exact.
        obj["components"]["desktop"]["version"] = "2.0.4"
        (target / "platform.json").write_bytes(canonical_json_bytes(obj))
        return f"2.0.6-r{number}"

    def test_old_receipt_global_only_refresh_failure_retry_and_exact_rerun(self):
        first = self.run_install(self.old, "9.0.0")
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        old = self.driver.receipt.read_bytes()
        component = receipt_section(old, "component:cli")
        installs = self.driver.installs()
        setup = self.driver.setup_records()
        coordinate = self.revision(1)
        self.driver.env["SOLSTONE_TEST_FAIL_PACKAGE_GLOBAL_REFRESH"] = "1"
        failed = self.run_install(self.new, coordinate, "cli", "--upgrade")
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(json.loads(failed.stdout)["root_code"], "receipt-write-failed")
        self.assertEqual(self.driver.receipt.read_bytes(), old)
        self.assertEqual(self.driver.installs(), installs)
        self.assertEqual(self.driver.setup_records(), setup)
        self.driver.env.pop("SOLSTONE_TEST_FAIL_PACKAGE_GLOBAL_REFRESH")
        success = self.run_install(self.new, coordinate, "cli", "--upgrade")
        self.assertEqual(success.returncode, 0, success.stderr + success.stdout)
        current = self.driver.receipt.read_bytes()
        self.assertIn(b"platform_version=2.0.6-r1\n", current)
        self.assertIn(b"installer_revision=8\n", current)
        self.assertEqual(receipt_section(current, "component:cli"), component)
        self.assertEqual(self.driver.installs(), installs)
        self.assertEqual(self.driver.setup_records(), setup)
        retry = self.run_install(self.new, coordinate, "cli", "--upgrade")
        self.assertEqual(retry.returncode, 0, retry.stderr + retry.stdout)
        self.assertEqual(self.driver.receipt.read_bytes(), current)
        removed = self.run_install(self.new, coordinate, "cli", "--uninstall")
        self.assertEqual(removed.returncode, 0, removed.stderr + removed.stdout)

    def test_mixed_tree_success_does_not_suppress_matching_package_refresh(self):
        first = self.run_install(self.old, "9.0.0")
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        tree = self.run_install(self.old, "9.0.0", "tmux", "--route", "tree")
        self.assertEqual(tree.returncode, 0, tree.stderr + tree.stdout)
        package_section = receipt_section(self.driver.receipt.read_bytes(), "component:cli")
        installs = self.driver.installs()
        setup = self.driver.setup_records()
        coordinate = self.revision(1)
        migrated = subprocess.run(["sh", str(self.new), "--skip-signature", "--components", "cli,tmux",
                "--upgrade", "--version", coordinate, "--prefix", str(self.driver.prefix), "--no-start", "--json"],
                env=self.driver.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(self.sentinel.read_bytes(), b"owner data\n")
        self.assertEqual(migrated.returncode, 0, migrated.stderr + migrated.stdout)
        tree_receipt = self.driver.work_dir / "tree-data/solstone/install.conf"
        for receipt in (self.driver.receipt, tree_receipt):
            self.assertIn(b"platform_version=2.0.6-r1\n", receipt.read_bytes())
        self.assertEqual(receipt_section(self.driver.receipt.read_bytes(), "component:cli"), package_section)
        self.assertEqual(self.driver.installs(), installs)
        self.assertEqual(self.driver.setup_records(), setup)

    def test_literal_rpm_release_suffix_and_explicit_history(self):
        first = self.run_install(self.old, "9.0.0", "cli", "--route", "rpm")
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        section = receipt_section(self.driver.receipt.read_bytes(), "component:cli")
        self.assertIn(b"package_version=2.0.6-1\n", section)
        coordinate = self.revision(1)
        migrated = self.run_install(self.new, coordinate, "cli", "--route", "rpm", "--upgrade")
        self.assertEqual(migrated.returncode, 0, migrated.stderr + migrated.stdout)
        self.assertEqual(receipt_section(self.driver.receipt.read_bytes(), "component:cli"), section)
        history = self.run_install(self.new, "9.0.0", "cli", "--route", "rpm", "--upgrade")
        self.assertEqual(history.returncode, 0, history.stderr + history.stdout)
        self.assertEqual(receipt_section(self.driver.receipt.read_bytes(), "component:cli"), section)
        self.assertEqual(len(self.driver.installs()), 1)

    def test_unselected_pending_prior_custody_survives_global_refresh(self):
        first = self.run_install(self.old, "9.0.0", "cli,tmux")
        self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
        original = self.driver.receipt.read_bytes()
        tmux = receipt_section(original, "component:tmux")
        pending = tmux.replace(b"phase=complete", b"phase=payload").replace(b"status=installed", b"status=pending")
        pending += b"prior_platform_version=8.0.0\nprior_status=installed\n"
        self.driver.receipt.write_bytes(original.replace(tmux, pending))
        coordinate = self.revision(1)
        result = self.run_install(self.new, coordinate, "cli", "--upgrade")
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertEqual(receipt_section(self.driver.receipt.read_bytes(), "component:tmux"), pending)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for case in (PublicationQualification, ReaderQualification, PackageQualification):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
