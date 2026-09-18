# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Command-line interface for platform release manifest generation, signing, and publication."""

import argparse
import os
from pathlib import Path
import stat
import sys

from solstone_platform.generate import generate_platform_manifest
from solstone_platform.pins import (
    embedded_pins,
    load_pin_file,
    require_production_platform_pin,
)
from solstone_platform.publish import publish_release
from solstone_platform.r2 import R2Config, R2Destination
from solstone_platform.redact import redact_sensitive_text
from solstone_platform.refusals import (
    PASSPHRASE_SOURCE_INVALID,
    PRODUCTION_UNAVAILABLE,
    Refusal,
)
from solstone_platform.sign import sign_manifest


def get_repo_root() -> Path:
    # Walk up to locate compat/minimum_installer_revision or Makefile
    curr = Path.cwd()
    while curr != curr.parent:
        if (curr / "compat" / "minimum_installer_revision").is_file():
            return curr
        curr = curr.parent
    return Path.cwd()


def read_passphrase_file(path: Path) -> str:
    """Read one private, caller-owned passphrase line without exposing it in argv."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as err:
        raise Refusal(PASSPHRASE_SOURCE_INVALID, "passphrase file must be a readable non-symlink file") from err
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise Refusal(PASSPHRASE_SOURCE_INVALID, "passphrase file must be a regular file")
        if file_stat.st_uid != os.geteuid():
            raise Refusal(PASSPHRASE_SOURCE_INVALID, "passphrase file must be owned by the caller")
        if file_stat.st_mode & 0o077:
            raise Refusal(PASSPHRASE_SOURCE_INVALID, "passphrase file must not be accessible by group or other")
        raw = os.read(fd, 4097)
        if len(raw) > 4096:
            raise Refusal(PASSPHRASE_SOURCE_INVALID, "passphrase file exceeds 4096 bytes")
        text = raw.decode("utf-8")
    except UnicodeDecodeError as err:
        raise Refusal(PASSPHRASE_SOURCE_INVALID, "passphrase file could not be read as UTF-8") from err
    finally:
        os.close(fd)
    if text.endswith("\n"):
        text = text[:-1]
    if not text or "\n" in text or "\r" in text or "\x00" in text:
        raise Refusal(PASSPHRASE_SOURCE_INVALID, "passphrase file must contain exactly one non-empty line")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="solstone_platform",
        description="Solstone Platform Release Generator & Publisher",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # generate subcommand
    p_gen = subparsers.add_parser("generate", help="Generate canonical platform.json manifest")
    p_gen.add_argument("--version", required=True, help="Platform version (strict X.Y.Z)")
    p_gen.add_argument("--lane", required=True, choices=["release", "staging", "dev"], help="Release lane")
    p_gen.add_argument("--created-unix", type=int, required=True, help="Unix timestamp in seconds")
    p_gen.add_argument("--source-commit", required=True, help="40-character lowercase commit hash")
    p_gen.add_argument("--platform-key-id", help="16-character uppercase hex platform key ID")
    p_gen.add_argument("--platform-pub", type=Path, help="Path to platform.pub key file")
    p_gen.add_argument("--journal-dir", type=Path, required=True, help="Path to solstone-journal release dir")
    p_gen.add_argument("--desktop-dir", type=Path, required=True, help="Path to solstone-linux release dir")
    p_gen.add_argument("--tmux-dir", type=Path, required=True, help="Path to solstone-tmux release dir")
    p_gen.add_argument("--journal-origin", required=True, help="Journal origin URL (https:// or loopback)")
    p_gen.add_argument("--bootstrap-file", type=Path, help="Explicit path to bootstrap install.sh")
    p_gen.add_argument("--out", type=Path, help="Output file path (default stdout)")

    # sign subcommand
    p_sign = subparsers.add_parser("sign", help="Sign platform.json manifest with Minisign")
    p_sign.add_argument("--manifest", type=Path, required=True, help="Path to platform.json")
    p_sign.add_argument("--secret-key", type=Path, required=True, help="Path to minisign secret key")
    p_sign.add_argument("--passphrase-file", type=Path, help="Private file containing the encrypted-key passphrase")
    p_sign.add_argument("--platform-pub", type=Path, help="Path to platform public key file")
    p_sign.add_argument("--out", type=Path, help="Output signature file (default platform.json.minisig)")
    p_sign.add_argument("--acknowledge-production", action="store_true", help="Acknowledge production signing")

    # publish subcommand
    p_pub = subparsers.add_parser("publish", help="Publish release manifest and promote latest pointer")
    p_pub.add_argument("--manifest", type=Path, required=True, help="Path to platform.json")
    p_pub.add_argument("--signature", type=Path, required=True, help="Path to platform.json.minisig")
    p_pub.add_argument("--journal-dir", type=Path, required=True, help="Path to solstone-journal release dir")
    p_pub.add_argument("--desktop-dir", type=Path, required=True, help="Path to solstone-linux release dir")
    p_pub.add_argument("--tmux-dir", type=Path, required=True, help="Path to solstone-tmux release dir")
    p_pub.add_argument("--bootstrap-file", type=Path, help="Explicit path to bootstrap install.sh")
    p_pub.add_argument("--acknowledge-production", action="store_true", help="Acknowledge production publishing")

    args = parser.parse_args(argv)
    repo_root = get_repo_root()

    try:
        if args.subcommand == "generate":
            key_id = args.platform_key_id
            if not key_id and args.platform_pub:
                pin = load_pin_file(args.platform_pub)
                key_id = pin.key_id
            if not key_id:
                # In production mode, require platform pin
                if os.environ.get("SOLSTONE_PLATFORM_PRODUCTION") == "ack":
                    prod_pin = require_production_platform_pin(repo_root)
                    key_id = prod_pin.key_id
                else:
                    raise Refusal(PRODUCTION_UNAVAILABLE, "--platform-key-id or --platform-pub required")

            manifest_bytes = generate_platform_manifest(
                version=args.version,
                lane=args.lane,
                created_unix=args.created_unix,
                source_commit=args.source_commit,
                platform_key_id=key_id,
                repo_root=repo_root,
                journal_dir=args.journal_dir,
                desktop_dir=args.desktop_dir,
                tmux_dir=args.tmux_dir,
                journal_origin=args.journal_origin,
                bootstrap_file=args.bootstrap_file,
                pins=embedded_pins(),  # CLI always uses embedded native pins
            )
            if args.out:
                args.out.write_bytes(manifest_bytes)
            else:
                sys.stdout.buffer.write(manifest_bytes)

        elif args.subcommand == "sign":
            manifest_bytes = args.manifest.read_bytes()
            production_pin = require_production_platform_pin(repo_root)
            if args.platform_pub:
                selected_pin = load_pin_file(args.platform_pub)
            else:
                selected_pin = production_pin
            is_production = selected_pin == production_pin
            if is_production and not args.acknowledge_production:
                raise Refusal(PRODUCTION_UNAVAILABLE, "--acknowledge-production required for production signing")
            if is_production and os.environ.get("SOLSTONE_PLATFORM_PRODUCTION") != "ack":
                raise Refusal(PRODUCTION_UNAVAILABLE, "SOLSTONE_PLATFORM_PRODUCTION=ack environment variable required")

            sig_bytes = sign_manifest(
                manifest_bytes=manifest_bytes,
                secret_key_path=args.secret_key,
                selected_pin=selected_pin,
                passphrase_callback=(
                    (lambda: read_passphrase_file(args.passphrase_file))
                    if args.passphrase_file
                    else None
                ),
                is_production=is_production,
                acknowledge_production=args.acknowledge_production,
                repo_root=repo_root,
            )
            out_path = args.out or args.manifest.with_name(args.manifest.name + ".minisig")
            out_path.write_bytes(sig_bytes)

        elif args.subcommand == "publish":
            if not args.acknowledge_production:
                raise Refusal(PRODUCTION_UNAVAILABLE, "--acknowledge-production required for production publishing")
            if os.environ.get("SOLSTONE_PLATFORM_PRODUCTION") != "ack":
                raise Refusal(PRODUCTION_UNAVAILABLE, "SOLSTONE_PLATFORM_PRODUCTION=ack environment variable required")

            config = R2Config.from_env()
            if not config.endpoint or not config.bucket:
                raise Refusal(PRODUCTION_UNAVAILABLE, "R2 destination configuration missing in environment")

            dest = R2Destination(config)
            report = publish_release(
                manifest_path=args.manifest,
                signature_path=args.signature,
                journal_dir=args.journal_dir,
                desktop_dir=args.desktop_dir,
                tmux_dir=args.tmux_dir,
                dest=dest,
                bootstrap_file=args.bootstrap_file,
            )
            print(f"Published {report.version} on lane '{report.lane}' (latest promoted: {report.latest_promoted})")

        return 0

    except Refusal as err:
        sys.stderr.write(f"Refusal: {redact_sensitive_text(str(err))}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Error: {redact_sensitive_text(str(err))}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
