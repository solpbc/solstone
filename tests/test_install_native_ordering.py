# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""All selected native authorities must preflight before any mutation."""

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest

from solstone_platform.sign import ephemeral_keypair
from tests.test_install_journal_authority import JournalFixture, JOURNAL_VERSION, PLATFORM_VERSION


REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER_SCRIPT = REPO_ROOT / "helpers" / "solstone-pkg-helper.sh"


def snapshot_tree(root: Path) -> dict[str, tuple[int, bytes]]:
    result: dict[str, tuple[int, bytes]] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        mode = stat.S_IMODE(path.lstat().st_mode)
        result[relative] = (mode, path.read_bytes() if path.is_file() else b"")
    return result


class TestInstallNativeOrdering(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/var/tmp")
        self.work_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_native_authorities_precede_payload_lock_and_helper(self):
        platform_keys = ephemeral_keypair("ordering platform")
        journal_keys = ephemeral_keypair("ordering journal")
        p_sec, p_pub, p_pin = platform_keys.__enter__()
        j_sec, _, j_pin = journal_keys.__enter__()
        self.addCleanup(platform_keys.__exit__, None, None, None)
        self.addCleanup(journal_keys.__exit__, None, None, None)
        fixture = JournalFixture(self.work_dir / "fixture", p_sec, p_pub, p_pin, j_sec, j_pin)
        self.addCleanup(fixture.close)

        prefix = self.work_dir / "prefix"
        etc_root = self.work_dir / "etc"
        helper_state = self.work_dir / "helper-state"
        lock_dir = self.work_dir / "package-lock"
        seeded = self.work_dir / "seeded"
        for directory in (prefix / "share" / "solstone", prefix / "bin", etc_root / "solstone", helper_state, lock_dir, seeded):
            directory.mkdir(parents=True, exist_ok=True)
        (prefix / ".solstone-platform.lock").write_bytes(b"tree-lock-sentinel\n")
        (prefix / "share" / "solstone" / "install.conf").write_bytes(b"tree-receipt-sentinel\n")
        (prefix / "bin" / "existing-owner").write_bytes(b"prefix-sentinel\n")
        (etc_root / "solstone" / "install.conf").write_bytes(b"package-receipt-sentinel\n")
        (helper_state / "install.log").write_bytes(b"helper-ledger-sentinel\n")
        (lock_dir / "lock").write_bytes(b"package-lock-sentinel\n")
        (seeded / "marker").write_bytes(b"ordering-snapshot-sentinel\n")
        observed_roots = {
            "prefix": prefix,
            "etc": etc_root,
            "helper": helper_state,
            "package-lock": lock_dir,
            "seeded": seeded,
        }
        before = {name: snapshot_tree(root) for name, root in observed_roots.items()}

        shim_dir = self.work_dir / "shims"
        shim_dir.mkdir()
        minisign_log = self.work_dir / "minisign-invocations.log"
        flock_log = self.work_dir / "flock-invocations.log"
        helper_log = self.work_dir / "helper-invocations.log"
        real_minisign = shutil.which("minisign")
        real_flock = shutil.which("flock")
        self.assertIsNotNone(real_minisign)
        self.assertIsNotNone(real_flock)
        minisign_shim = shim_dir / "minisign"
        minisign_shim.write_text(
            f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{minisign_log}"\nexec "{real_minisign}" "$@"\n',
            encoding="utf-8",
        )
        minisign_shim.chmod(0o755)
        flock_shim = shim_dir / "flock"
        flock_shim.write_text(
            f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{flock_log}"\nexec "{real_flock}" "$@"\n',
            encoding="utf-8",
        )
        flock_shim.chmod(0o755)
        helper_shim = self.work_dir / "helper-shim"
        helper_shim.write_text(
            "#!/bin/sh\n"
            f'tmp="{self.work_dir}/helper-stdin.$$"\n'
            'trap \'rm -f "$tmp"\' 0\n'
            'cat > "$tmp"\n'
            f'printf "START %s\\n" "$*" >> "{helper_log}"\n'
            f'cat "$tmp" >> "{helper_log}"\n'
            f'exec "{HELPER_SCRIPT}" "$@" < "$tmp"\n',
            encoding="utf-8",
        )
        helper_shim.chmod(0o755)

        env = os.environ.copy()
        env["PATH"] = f"{shim_dir}:{env['PATH']}"
        env["SOLSTONE_HELPER"] = str(helper_shim)
        env["SOLSTONE_LOCK_DIR"] = str(lock_dir)
        env["SOLSTONE_ETC_ROOT"] = str(etc_root)
        env["SOLSTONE_FAKE_PKG_DB"] = str(helper_state)
        env["XDG_DATA_HOME"] = str(prefix / "share")

        target_name = f"solstone-tmux-2.0.3-x86_64-unknown-linux-musl.target.json"
        target = fixture.version_dir / target_name
        valid_target = target.read_bytes()
        target.write_bytes(valid_target + b" ")

        expected_requests = [
            "/solstone/release/latest",
            f"/solstone/release/{PLATFORM_VERSION}/platform.json",
            f"/solstone/release/{PLATFORM_VERSION}/platform.json.minisig",
            f"/solstone/release/{PLATFORM_VERSION}/solstone-journal-{JOURNAL_VERSION}-linux-x86_64.manifest.json",
            f"/solstone/release/{PLATFORM_VERSION}/solstone-journal-{JOURNAL_VERSION}-linux-x86_64.manifest.json.minisig",
            f"/solstone/release/{PLATFORM_VERSION}/solstone-journal-{JOURNAL_VERSION}-linux-x86_64.release",
            f"/solstone/release/{PLATFORM_VERSION}/solstone-journal-{JOURNAL_VERSION}-linux-x86_64.sha256",
            f"/solstone/release/{PLATFORM_VERSION}/solstone-linux-2.0.3-linux-x86_64.rust-release-manifest.json",
            f"/solstone/release/{PLATFORM_VERSION}/solstone-linux-2.0.3-linux-x86_64.rust-release-manifest.json.minisig",
            f"/solstone/release/{PLATFORM_VERSION}/SHA256SUMS",
            f"/solstone/release/{PLATFORM_VERSION}/SHA256SUMS.minisig",
            f"/solstone/release/{PLATFORM_VERSION}/{target_name}",
        ]

        def run(*extra: str):
            fixture.server.request_paths.clear()
            return subprocess.run(
                [
                    str(fixture.installer), "--components", "journal,desktop,tmux",
                    "--prefix", str(prefix), *extra, "--json",
                ],
                capture_output=True,
                text=True,
                env=env,
            )

        proc = run()
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["root_code"], "digest-mismatch")
        self.assertEqual(fixture.server.request_paths, expected_requests)
        verifier_lines = minisign_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(verifier_lines), 3)
        for verified_name in (
            "platform.json",
            f"solstone-journal-{JOURNAL_VERSION}-linux-x86_64.manifest.json",
            "solstone-linux-2.0.3-linux-x86_64.rust-release-manifest.json",
        ):
            self.assertEqual(sum(f"/{verified_name} " in line for line in verifier_lines), 1)
        self.assertFalse(any("/SHA256SUMS " in line for line in verifier_lines))
        self.assertFalse(flock_log.exists())
        self.assertFalse(helper_log.exists())
        self.assertEqual({name: snapshot_tree(root) for name, root in observed_roots.items()}, before)

        target.write_bytes(valid_target)
        minisign_log.unlink()
        proc = run("--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["root_code"], "dry-run-completed")
        self.assertEqual(fixture.server.request_paths, expected_requests)
        verifier_lines = minisign_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(verifier_lines), 4)
        for verified_name in (
            "platform.json",
            f"solstone-journal-{JOURNAL_VERSION}-linux-x86_64.manifest.json",
            "solstone-linux-2.0.3-linux-x86_64.rust-release-manifest.json",
            "SHA256SUMS",
        ):
            self.assertEqual(sum(f"/{verified_name} " in line for line in verifier_lines), 1)
        self.assertFalse(flock_log.exists())
        self.assertFalse(helper_log.exists())
        self.assertEqual({name: snapshot_tree(root) for name, root in observed_roots.items()}, before)


if __name__ == "__main__":
    unittest.main()
