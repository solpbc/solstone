# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Embedded native public key pins and pin loaders."""

from dataclasses import dataclass
import importlib.resources
from pathlib import Path
import re
from typing import Optional

from solstone_platform.refusals import PIN_MISMATCH, PRODUCTION_UNAVAILABLE, Refusal

# Production embedded native public keys
JOURNAL_KEY_ID = "B44073BF49E0D944"
JOURNAL_PUBKEY = "RWRE2eBJv3NAtN0mF5+kqygYyP/ocYNw1Ng9yJhAKgyTflNV9NabMMjq"

DESKTOP_KEY_ID = "19EC9FAF7B331CF2"
DESKTOP_PUBKEY = "RWTyHDN7r5/sGTjcpaSzR+tGcH324jnxrsd7dRnfK7Qn/FAbAzU1JyGe"

TMUX_KEY_ID = "365708FAD9F80092"
TMUX_PUBKEY = "RWSSAPjZ+ghXNvb4ExBLSd59dQMtjqW+xIZcl9MWfpWvjsTws6sBPEZz"

PLATFORM_KEY_ID = "2938B1EBDC1E3876"
PLATFORM_PUBKEY = "RWR2OB7c67E4KfRo4OnyOoXnvfOl+sum7TG6LscqXmN8mv/Q55nlBzCD"


@dataclass(frozen=True)
class MinisignPin:
    key_id: str
    pubkey: str

    def verifier_id(self) -> str:
        return f"minisign:{self.key_id}"

    def to_minisign_pub_file_content(self) -> str:
        return f"untrusted comment: minisign public key {self.key_id}\n{self.pubkey}\n"


@dataclass(frozen=True)
class PinSet:
    journal: MinisignPin
    desktop: MinisignPin
    tmux: MinisignPin
    platform: Optional[MinisignPin] = None


def parse_minisign_pub(content: str) -> MinisignPin:
    """Parse a Minisign .pub file content and extract key ID and public key string."""
    lines = [line.strip() for line in content.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise Refusal(PIN_MISMATCH, "invalid minisign public key file format")

    comment_line = lines[0]
    pubkey_line = lines[1]

    match = re.search(r"key:?\s+([0-9A-Fa-f]{16})", comment_line)
    if not match:
        raise Refusal(PIN_MISMATCH, f"could not parse 16-char hex key id from comment: '{comment_line}'")

    key_id = match.group(1).upper()
    return MinisignPin(key_id=key_id, pubkey=pubkey_line)


def load_pin_file(path: Path) -> MinisignPin:
    if not path.is_file():
        raise Refusal(PIN_MISMATCH, f"pin file not found: {path}")
    return parse_minisign_pub(path.read_text(encoding="utf-8"))


def load_packaged_pin(name: str) -> MinisignPin:
    """Load a pin directly from package resources."""
    try:
        res = importlib.resources.files("solstone_platform.pin_resources")
        pub_file = res / f"{name}.pub"
        content = pub_file.read_text(encoding="utf-8")
        return parse_minisign_pub(content)
    except Exception as err:
        raise Refusal(PRODUCTION_UNAVAILABLE, f"packaged pin '{name}' unavailable: {err}") from err


def _verify_constant_synchronization() -> None:
    """Ensure in-code constants exactly match packaged pin resources."""
    res_pins = {
        "journal": (JOURNAL_KEY_ID, JOURNAL_PUBKEY),
        "desktop": (DESKTOP_KEY_ID, DESKTOP_PUBKEY),
        "tmux": (TMUX_KEY_ID, TMUX_PUBKEY),
        "platform": (PLATFORM_KEY_ID, PLATFORM_PUBKEY),
    }
    for name, (expected_id, expected_pub) in res_pins.items():
        pkg_pin = load_packaged_pin(name)
        if pkg_pin.key_id != expected_id or pkg_pin.pubkey != expected_pub:
            raise Refusal(PIN_MISMATCH, f"packaged pin for {name} does not match constant: {pkg_pin.key_id} vs {expected_id}")

    # Also verify platform.keyid
    try:
        res = importlib.resources.files("solstone_platform.pin_resources")
        keyid_file = res / "platform.keyid"
        keyid_val = keyid_file.read_text(encoding="utf-8").strip().upper()
        if keyid_val != PLATFORM_KEY_ID:
            raise Refusal(PIN_MISMATCH, f"packaged platform.keyid mismatch: {keyid_val} vs {PLATFORM_KEY_ID}")
    except Exception as err:
        raise Refusal(PRODUCTION_UNAVAILABLE, f"packaged platform.keyid unavailable: {err}") from err


# Verify at import time
_verify_constant_synchronization()


def embedded_pins() -> PinSet:
    """Return the official embedded production pins for native components."""
    return PinSet(
        journal=load_packaged_pin("journal"),
        desktop=load_packaged_pin("desktop"),
        tmux=load_packaged_pin("tmux"),
        platform=load_packaged_pin("platform"),
    )


def require_production_platform_pin(repo_root: Optional[Path] = None) -> MinisignPin:
    """Load production platform pin from packaged resources, ignoring caller repo_root."""
    return load_packaged_pin("platform")
