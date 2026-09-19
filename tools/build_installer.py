#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Build engine and pin substitution rail for the Solstone POSIX platform installer."""

import argparse
from pathlib import Path
import re
import sys

from solstone_platform.pins import (
    DESKTOP_KEY_ID,
    DESKTOP_PUBKEY,
    JOURNAL_KEY_ID,
    JOURNAL_PUBKEY,
    TMUX_KEY_ID,
    TMUX_PUBKEY,
    load_pin_file,
    parse_minisign_pub,
)
from solstone_platform.refusals import PIN_MISMATCH, PRODUCTION_UNAVAILABLE, Refusal

DEFAULT_ORIGIN = "https://updates.solstone.app"
LOOPBACK_ORIGIN_RE = re.compile(r"http://127\.0\.0\.1:([1-9][0-9]{0,4})", re.ASCII)
KEY_ID_RE = re.compile(r"[0-9A-Fa-f]{16}", re.ASCII)


def validate_origin(origin: str) -> str:
    if origin == DEFAULT_ORIGIN:
        return origin
    match = LOOPBACK_ORIGIN_RE.fullmatch(origin)
    if match is None or int(match.group(1)) > 65535:
        raise Refusal("origin-invalid", f"unsupported installer origin: {origin}")
    return origin


def load_revisions(repo_root: Path) -> tuple[int, int]:
    inst_rev_path = repo_root / "compat" / "installer_revision"
    min_rev_path = repo_root / "compat" / "minimum_installer_revision"

    if not inst_rev_path.is_file():
        raise Refusal("compat-missing", f"missing {inst_rev_path}")
    if not min_rev_path.is_file():
        raise Refusal("compat-missing", f"missing {min_rev_path}")

    try:
        inst_rev = int(inst_rev_path.read_text(encoding="utf-8").strip())
        min_rev = int(min_rev_path.read_text(encoding="utf-8").strip())
        return inst_rev, min_rev
    except ValueError as err:
        raise Refusal("compat-invalid", f"invalid revision number: {err}") from err


def verify_native_pins(repo_root: Path) -> dict[str, tuple[str, str]]:
    pins_dir = repo_root / "pins"
    j_pin = load_pin_file(pins_dir / "journal.pub")
    d_pin = load_pin_file(pins_dir / "desktop.pub")
    t_pin = load_pin_file(pins_dir / "tmux.pub")

    if j_pin.key_id != JOURNAL_KEY_ID or j_pin.pubkey != JOURNAL_PUBKEY:
        raise Refusal(PIN_MISMATCH, "journal.pub does not match pins.py constants bit-identically")
    if d_pin.key_id != DESKTOP_KEY_ID or d_pin.pubkey != DESKTOP_PUBKEY:
        raise Refusal(PIN_MISMATCH, "desktop.pub does not match pins.py constants bit-identically")
    if t_pin.key_id != TMUX_KEY_ID or t_pin.pubkey != TMUX_PUBKEY:
        raise Refusal(PIN_MISMATCH, "tmux.pub does not match pins.py constants bit-identically")

    return {
        "journal": (j_pin.key_id, j_pin.pubkey),
        "desktop": (d_pin.key_id, d_pin.pubkey),
        "tmux": (t_pin.key_id, t_pin.pubkey),
    }


def embedded_runtime(repo_root: Path) -> str:
    """Stage the exact reviewed runtime; never execute helpers from caller cwd."""
    paths = [
        "helpers/solstone-pkg-helper.sh",
        "handlers/desktop/v1/install-desktop",
        "handlers/desktop/v1/uninstall-desktop-service",
        "handlers/tmux/v1/install-tmux",
        "handlers/tmux/v1/uninstall-tmux-service",
    ]
    lines = ["init_bundled_runtime() {", '    BUNDLED_RUNTIME="${SCRATCH_DIR}/runtime"']
    for index, relative in enumerate(paths):
        content = (repo_root / relative).read_text(encoding="utf-8")
        delimiter = f"SOLSTONE_RUNTIME_FILE_{index}_END"
        if delimiter in content.splitlines() or not content.endswith("\n"):
            raise Refusal("runtime-invalid", f"invalid embedding boundary: {relative}")
        target = f'"$BUNDLED_RUNTIME/{relative}"'
        parent = str(Path(relative).parent)
        lines.extend([
            f'    mkdir -p "$BUNDLED_RUNTIME/{parent}" || report_exit refusal runtime-unavailable "could not stage installer runtime"',
            f"    if ! cat > {target} <<'{delimiter}'",
            content[:-1],
            delimiter,
            '    then report_exit refusal runtime-unavailable "could not write installer runtime"; fi',
            f'    chmod 0700 {target} || report_exit refusal runtime-unavailable "could not prepare installer runtime"',
        ])
    lines.append("}")
    return "\n".join(lines)


def build_installer(
    repo_root: Path,
    output_path: Path,
    platform_pub_path: Path | None = None,
    platform_key_id: str | None = None,
    origin: str | None = None,
    override_installer_revision: int | None = None,
    is_production: bool = False,
) -> Path:
    template_path = repo_root / "install.sh.in"
    if not template_path.is_file():
        raise Refusal("template-missing", f"template file not found: {template_path}")

    if is_production and any(
        value is not None
        for value in (platform_pub_path, platform_key_id, origin, override_installer_revision)
    ):
        raise Refusal(PRODUCTION_UNAVAILABLE, "production build forbids test seam overrides")

    resolved_origin = validate_origin(DEFAULT_ORIGIN if origin is None else origin)
    if override_installer_revision is not None and (
        type(override_installer_revision) is not int or override_installer_revision <= 0
    ):
        raise Refusal("installer-revision-invalid", "installer revision must be a positive integer")
    if platform_key_id is not None and KEY_ID_RE.fullmatch(platform_key_id) is None:
        raise Refusal("platform-key-id-invalid", "platform key ID must be exactly 16 ASCII hex digits")

    inst_rev, min_rev = load_revisions(repo_root)
    if override_installer_revision is not None:
        inst_rev = override_installer_revision

    native_pins = verify_native_pins(repo_root)

    # Determine platform pin and test seam
    if is_production:
        test_seam = "0"
        prod_pub = repo_root / "pins" / "platform.pub"
        prod_keyid = repo_root / "pins" / "platform.keyid"
        if not prod_pub.is_file() or not prod_keyid.is_file():
            raise Refusal(
                PRODUCTION_UNAVAILABLE,
                "production platform pin is absent (pins/platform.pub and pins/platform.keyid required for production publication)",
            )
        plat_pin = load_pin_file(prod_pub)
        expected_id = prod_keyid.read_text(encoding="utf-8").strip().upper()
        if plat_pin.key_id != expected_id:
            raise Refusal(PIN_MISMATCH, f"platform key ID mismatch: {plat_pin.key_id} vs {expected_id}")
        plat_key_id = plat_pin.key_id
        plat_pubkey = plat_pin.pubkey
    else:
        test_seam = "1"
        if platform_pub_path is not None:
            plat_pin = load_pin_file(platform_pub_path)
            plat_key_id = platform_key_id.upper() if platform_key_id is not None else plat_pin.key_id
            plat_pubkey = plat_pin.pubkey
        else:
            # Check if production pins exist, otherwise refuse unless test seam key is provided
            prod_pub = repo_root / "pins" / "platform.pub"
            prod_keyid = repo_root / "pins" / "platform.keyid"
            if prod_pub.is_file() and prod_keyid.is_file():
                plat_pin = load_pin_file(prod_pub)
                plat_key_id = plat_pin.key_id
                plat_pubkey = plat_pin.pubkey
            else:
                raise Refusal(
                    PRODUCTION_UNAVAILABLE,
                    "platform public key not provided (use --platform-pub or supply pins/platform.pub)",
                )

        if platform_key_id is not None and plat_key_id != plat_pin.key_id:
            raise Refusal(PIN_MISMATCH, "supplied platform key ID does not match selected public pin")

    template_content = template_path.read_text(encoding="utf-8")

    substitutions = {
        "@INSTALLER_REVISION@": str(inst_rev),
        "@MINIMUM_INSTALLER_REVISION@": str(min_rev),
        "@PLATFORM_KEY_ID@": plat_key_id,
        "@PLATFORM_PUBKEY@": plat_pubkey,
        "@JOURNAL_KEY_ID@": native_pins["journal"][0],
        "@JOURNAL_PUBKEY@": native_pins["journal"][1],
        "@DESKTOP_KEY_ID@": native_pins["desktop"][0],
        "@DESKTOP_PUBKEY@": native_pins["desktop"][1],
        "@TMUX_KEY_ID@": native_pins["tmux"][0],
        "@TMUX_PUBKEY@": native_pins["tmux"][1],
        "@DEFAULT_ORIGIN@": resolved_origin,
        "@TEST_SEAM@": test_seam,
        "@BUNDLED_RUNTIME@": embedded_runtime(repo_root),
    }

    result = template_content
    for placeholder, val in substitutions.items():
        if placeholder not in result:
            raise Refusal("placeholder-missing", f"template missing expected placeholder '{placeholder}'")
        result = result.replace(placeholder, val)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result, encoding="utf-8")
    output_path.chmod(0o755)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Solstone POSIX platform installer")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output script path")
    parser.add_argument("--platform-pub", type=Path, default=None, help="Test seam: path to platform public key")
    parser.add_argument("--platform-keyid", type=str, default=None, help="Test seam: platform key ID")
    parser.add_argument("--origin", type=str, default=None, help="Test seam: default download origin")
    parser.add_argument("--installer-revision", type=int, default=None, help="Test seam: override installer revision")
    parser.add_argument("--production", action="store_true", help="Build production installer")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    out_path = args.output or (repo_root / "dist" / "install.sh")

    try:
        build_installer(
            repo_root=repo_root,
            output_path=out_path,
            platform_pub_path=args.platform_pub,
            platform_key_id=args.platform_keyid,
            origin=args.origin,
            override_installer_revision=args.installer_revision,
            is_production=args.production,
        )
        print(f"Built installer: {out_path}")
        return 0
    except Refusal as err:
        print(f"ERROR: {err.name}: {err.detail or ''}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
