# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Detached Minisign signer and ephemeral fixture identity manager."""

from contextlib import contextmanager
import getpass
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Callable, Generator, Optional

from solstone_platform.canonical import canonical_json_bytes, parse_json_strict
from solstone_platform.pins import (
    MinisignPin,
    load_pin_file,
    parse_minisign_pub,
    require_production_platform_pin,
)
from solstone_platform.refusals import (
    FIXTURE_KEY_REFUSED,
    PIN_MISMATCH,
    PRODUCTION_UNAVAILABLE,
    SCHEMA_INVALID,
    SIGNATURE_PIN_MISMATCH,
    Refusal,
)
from solstone_platform.schema import validate_platform_manifest


def derive_public_key_from_secret(
    secret_key_path: Path,
    passphrase: str = "",
) -> MinisignPin:
    """Derive public key directly from the secret key using minisign -R."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pub = Path(tmp_dir) / "derived.pub"
        cmd = ["minisign", "-R", "-s", str(secret_key_path), "-p", str(tmp_pub)]
        if not passphrase:
            cmd.append("-W")

        proc = subprocess.run(
            cmd,
            input=f"{passphrase}\n".encode("utf-8") if passphrase else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not tmp_pub.is_file():
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise Refusal(PIN_MISMATCH, f"failed to derive public key from secret: {err}")

        return load_pin_file(tmp_pub)


def sign_manifest(
    manifest_bytes: bytes,
    secret_key_path: Path,
    selected_pin: MinisignPin,
    passphrase_callback: Optional[Callable[[], str]] = None,
    trusted_comment: str = "solstone platform release manifest",
    is_production: bool = False,
    acknowledge_production: bool = False,
    repo_root: Optional[Path] = None,
) -> bytes:
    """Sign manifest bytes with minisign and return detached signature bytes."""
    parsed_manifest = parse_json_strict(manifest_bytes)
    validate_platform_manifest(parsed_manifest)
    if canonical_json_bytes(parsed_manifest) != manifest_bytes:
        raise Refusal(SCHEMA_INVALID, "platform manifest must be canonical before signing")
    if parsed_manifest["platform_key_id"] != selected_pin.key_id:
        raise Refusal(PIN_MISMATCH, "manifest platform_key_id does not match selected signing pin")

    if passphrase_callback is not None:
        passphrase = passphrase_callback()
    elif not sys.stdin.isatty():
        passphrase = ""
    else:
        passphrase = getpass.getpass("Enter minisign passphrase: ")

    # Always derive public key directly from secret key and verify against selected pin before signing
    derived_pin = derive_public_key_from_secret(secret_key_path, passphrase)
    if derived_pin.pubkey != selected_pin.pubkey or derived_pin.key_id != selected_pin.key_id:
        raise Refusal(PIN_MISMATCH, f"secret key derives pin {derived_pin.key_id} ({derived_pin.pubkey}), does not match selected pin {selected_pin.key_id} ({selected_pin.pubkey})")

    if is_production:
        if repo_root is None:
            raise Refusal(PRODUCTION_UNAVAILABLE, "repo_root required for production signing")
        if not acknowledge_production:
            raise Refusal(PRODUCTION_UNAVAILABLE, "--acknowledge-production required for production signing")
        if os.environ.get("SOLSTONE_PLATFORM_PRODUCTION") != "ack":
            raise Refusal(PRODUCTION_UNAVAILABLE, "SOLSTONE_PLATFORM_PRODUCTION=ack environment variable required")

        # Check for fixture markers in secret key path or file content
        sec_content = secret_key_path.read_text(encoding="utf-8", errors="replace") if secret_key_path.is_file() else ""
        if "fixture" in str(secret_key_path).lower() or "test key" in str(secret_key_path).lower() or "fixture" in sec_content.lower() or "test key" in sec_content.lower():
            raise Refusal(FIXTURE_KEY_REFUSED, "fixture secret key rejected for production signing")

        prod_pin = require_production_platform_pin(repo_root)
        if derived_pin.key_id != prod_pin.key_id or derived_pin.pubkey != prod_pin.pubkey:
            raise Refusal(PIN_MISMATCH, "derived secret key is not the production platform pin")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_manifest = Path(tmp_dir) / "platform.json"
        tmp_manifest.write_bytes(manifest_bytes)
        tmp_sig = Path(tmp_dir) / "platform.json.minisig"

        cmd = [
            "minisign",
            "-S",
            "-s", str(secret_key_path),
            "-m", str(tmp_manifest),
            "-x", str(tmp_sig),
            "-t", trusted_comment,
        ]
        if not passphrase:
            cmd.append("-W")
        proc = subprocess.run(
            cmd,
            input=f"{passphrase}\n".encode("utf-8") if passphrase else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise Refusal(SIGNATURE_PIN_MISMATCH, f"signing failed: {err}")

        # Verify generated signature immediately against selected_pin
        tmp_pub = Path(tmp_dir) / "selected.pub"
        tmp_pub.write_text(selected_pin.to_minisign_pub_file_content(), encoding="utf-8")

        vproc = subprocess.run(
            ["minisign", "-V", "-p", str(tmp_pub), "-m", str(tmp_manifest), "-x", str(tmp_sig)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if vproc.returncode != 0:
            raise Refusal(PIN_MISMATCH, "generated signature did not verify against selected pin")

        return tmp_sig.read_bytes()


@contextmanager
def ephemeral_keypair(comment: str = "fixture test key") -> Generator[tuple[Path, Path, MinisignPin], None, None]:
    """Generate ephemeral Minisign keypair in secure tempdir (0700) with automatic cleanup."""
    tmp_dir = tempfile.mkdtemp(prefix="solstone-key-")
    os.chmod(tmp_dir, 0o700)

    def _sig_handler(signum, frame):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        signal.default_int_handler(signum, frame)

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
        sec_path = Path(tmp_dir) / "test.key"
        pub_path = Path(tmp_dir) / "test.pub"

        proc = subprocess.run(
            ["minisign", "-G", "-W", "-p", str(pub_path), "-s", str(sec_path), "-c", comment],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        pin = load_pin_file(pub_path)
        yield sec_path, pub_path, pin
    finally:
        try:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
        except Exception:
            pass
        shutil.rmtree(tmp_dir, ignore_errors=True)
