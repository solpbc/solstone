#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Synchronize pinned public keys from repo pins/ to packaged resources and pins.py constants."""

import importlib.resources
from pathlib import Path
import re
import sys

from solstone_platform.pins import parse_minisign_pub


def sync_pins(repo_root: Path) -> None:
    pins_dir = repo_root / "pins"
    pkg_res_dir = repo_root / "src" / "solstone_platform" / "pin_resources"
    pkg_res_dir.mkdir(parents=True, exist_ok=True)

    pin_names = ["journal", "desktop", "tmux", "platform"]
    parsed_pins = {}

    for name in pin_names:
        pub_file = pins_dir / f"{name}.pub"
        if not pub_file.is_file():
            sys.stderr.write(f"Error: missing {pub_file}\n")
            sys.exit(1)
        content = pub_file.read_text(encoding="utf-8")
        pin = parse_minisign_pub(content)
        parsed_pins[name] = pin
        (pkg_res_dir / f"{name}.pub").write_text(content, encoding="utf-8")

    keyid_file = pins_dir / "platform.keyid"
    if not keyid_file.is_file():
        sys.stderr.write(f"Error: missing {keyid_file}\n")
        sys.exit(1)
    keyid_content = keyid_file.read_text(encoding="utf-8").strip()
    (pkg_res_dir / "platform.keyid").write_text(f"{keyid_content}\n", encoding="utf-8")

    # Update pins.py constant block
    pins_py = repo_root / "src" / "solstone_platform" / "pins.py"
    content = pins_py.read_text(encoding="utf-8")

    block = (
        f'# Production embedded native public keys\n'
        f'JOURNAL_KEY_ID = "{parsed_pins["journal"].key_id}"\n'
        f'JOURNAL_PUBKEY = "{parsed_pins["journal"].pubkey}"\n\n'
        f'DESKTOP_KEY_ID = "{parsed_pins["desktop"].key_id}"\n'
        f'DESKTOP_PUBKEY = "{parsed_pins["desktop"].pubkey}"\n\n'
        f'TMUX_KEY_ID = "{parsed_pins["tmux"].key_id}"\n'
        f'TMUX_PUBKEY = "{parsed_pins["tmux"].pubkey}"\n\n'
        f'PLATFORM_KEY_ID = "{parsed_pins["platform"].key_id}"\n'
        f'PLATFORM_PUBKEY = "{parsed_pins["platform"].pubkey}"\n'
    )

    pattern = r'# Production embedded native public keys\n(?:[A-Z_]+ = "[^\n]*"\n|\n)+'
    if re.search(pattern, content):
        new_content = re.sub(pattern, block, content, count=1)
    else:
        new_content = content

    pins_py.write_text(new_content, encoding="utf-8")
    print("Pins synchronized successfully.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    sync_pins(root)
