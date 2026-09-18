# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Command-line interface for platform release manifest generation, signing, and publication."""

import argparse
import os
from pathlib import Path
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
    p_sign.add_argument("--platform-pub", type=Path, help="Path to platform public key file")
    p_sign.add_argument("--out", type=Path, help="Output signature file (default platform.json.minisig)")
    p_sign.add_argument("--acknowledge-production", action="store_true", help="Acknowledge production signing")

    # publish subcommand
    p_pub = subparsers.add_parser("publish", help="Publish release manifest and promote latest pointer")
    p_pub.add_argument("--manifest", type=Path, required=True, help="Path to platform.json")
    p_pub.add_argument("--signature", type=Path, required=True, help="Path to platform.json.minisig")
    p_pub.add_argument("--platform-pub", type=Path, help="Path to platform public key file")
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
            if args.platform_pub:
                selected_pin = load_pin_file(args.platform_pub)
            else:
                selected_pin = require_production_platform_pin(repo_root)

            sig_bytes = sign_manifest(
                manifest_bytes=manifest_bytes,
                secret_key_path=args.secret_key,
                selected_pin=selected_pin,
                is_production=args.acknowledge_production,
                acknowledge_production=args.acknowledge_production,
                repo_root=repo_root,
            )
            out_path = args.out or args.manifest.with_name(args.manifest.name + ".minisig")
            out_path.write_bytes(sig_bytes)

        elif args.subcommand == "publish":
            manifest_bytes = args.manifest.read_bytes()
            sig_bytes = args.signature.read_bytes()

            # Production gate for publish
            if args.acknowledge_production or not args.platform_pub:
                if not args.acknowledge_production:
                    raise Refusal(PRODUCTION_UNAVAILABLE, "--acknowledge-production required for production publishing")
                if os.environ.get("SOLSTONE_PLATFORM_PRODUCTION") != "ack":
                    raise Refusal(PRODUCTION_UNAVAILABLE, "SOLSTONE_PLATFORM_PRODUCTION=ack environment variable required")
                selected_pin = require_production_platform_pin(repo_root)
            else:
                selected_pin = load_pin_file(args.platform_pub)

            config = R2Config.from_env()
            if not config.endpoint or not config.bucket:
                raise Refusal(PRODUCTION_UNAVAILABLE, "R2 destination configuration missing in environment")

            dest = R2Destination(config)
            report = publish_release(
                manifest_bytes=manifest_bytes,
                signature_bytes=sig_bytes,
                selected_pin=selected_pin,
                dest=dest,
                key_prefix=config.key_prefix,
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
