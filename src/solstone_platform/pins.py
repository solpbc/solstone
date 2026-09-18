# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Embedded native public key pins and pin loaders."""

from dataclasses import dataclass
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

    match = re.search(r"key\s+([0-9A-Fa-f]{16})", comment_line)
    if not match:
        raise Refusal(PIN_MISMATCH, f"could not parse 16-char hex key id from comment: '{comment_line}'")

    key_id = match.group(1).upper()
    return MinisignPin(key_id=key_id, pubkey=pubkey_line)


def load_pin_file(path: Path) -> MinisignPin:
    if not path.is_file():
        raise Refusal(PIN_MISMATCH, f"pin file not found: {path}")
    return parse_minisign_pub(path.read_text(encoding="utf-8"))


def embedded_pins() -> PinSet:
    """Return the official embedded production pins for native components."""
    return PinSet(
        journal=MinisignPin(key_id=JOURNAL_KEY_ID, pubkey=JOURNAL_PUBKEY),
        desktop=MinisignPin(key_id=DESKTOP_KEY_ID, pubkey=DESKTOP_PUBKEY),
        tmux=MinisignPin(key_id=TMUX_KEY_ID, pubkey=TMUX_PUBKEY),
        platform=None,  # Production platform pin is deliberately absent
    )


def require_production_platform_pin(repo_root: Path) -> MinisignPin:
    """Load production platform pin, refusing if absent."""
    pub_path = repo_root / "pins" / "platform.pub"
    keyid_path = repo_root / "pins" / "platform.keyid"
    if not pub_path.is_file() or not keyid_path.is_file():
        raise Refusal(
            PRODUCTION_UNAVAILABLE,
            "production platform pin is absent (pins/platform.pub and pins/platform.keyid required for production publication)",
        )
    pin = load_pin_file(pub_path)
    expected_id = keyid_path.read_text(encoding="utf-8").strip()
    if pin.key_id != expected_id.upper():
        raise Refusal(PIN_MISMATCH, f"platform key ID mismatch: file has {pin.key_id}, keyid has {expected_id}")
    return pin
