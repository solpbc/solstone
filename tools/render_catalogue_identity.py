#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

"""Render platform schema and splice generated catalogue identity into installer template."""

from pathlib import Path
import sys

# Ensure src is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from solstone_platform.schema import (
    render_platform_schema,
    render_posix_catalogue_keys,
    render_posix_identity_fragment,
)


def render_all(repo_root: Path = REPO_ROOT) -> None:
    schema_path = repo_root / "schema" / "platform.v1.json"
    template_path = repo_root / "install.sh.in"

    # 1. Render schema
    schema_content = render_platform_schema()
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(schema_content, encoding="utf-8")

    # 2. Splice template
    template_text = template_path.read_text(encoding="utf-8")

    identity_fragment = render_posix_identity_fragment()
    catalogue_keys = render_posix_catalogue_keys()

    # Splice identity fragment
    id_start_marker = "# BEGIN GENERATED CATALOGUE IDENTITY"
    id_end_marker = "# END GENERATED CATALOGUE IDENTITY"
    if id_start_marker in template_text and id_end_marker in template_text:
        before = template_text.split(id_start_marker)[0]
        after = template_text.split(id_end_marker)[1]
        template_text = before + identity_fragment + after
    else:
        # Insert right before is_canonical_version() {
        target = "is_canonical_version() {"
        if target not in template_text:
            raise ValueError(f"Target '{target}' not found in {template_path}")
        template_text = template_text.replace(target, f"{identity_fragment}\n\n{target}", 1)

    # Splice catalogue keys in is_whitelisted
    keys_start_marker = "# BEGIN GENERATED CATALOGUE KEYS"
    keys_end_marker = "# END GENERATED CATALOGUE KEYS"
    if keys_start_marker in template_text and keys_end_marker in template_text:
        before = template_text.split(keys_start_marker)[0]
        after = template_text.split(keys_end_marker)[1]
        # Re-attach indentation from before
        template_text = before.rstrip() + "\n" + catalogue_keys + "\n" + after.lstrip("\n")
    else:
        # Replace hand-written keys
        target_handwritten = (
            "    if (p ~ /^schema_version$/) return 1\n"
            "    if (p ~ /^protocol_version$/) return 1\n"
            "    if (p ~ /^version$/) return 1\n"
            "    if (p ~ /^lane$/) return 1\n"
            "    if (p ~ /^created_unix$/) return 1\n"
            "    if (p ~ /^platform_key_id$/) return 1\n"
            "    if (p ~ /^minimum_installer_revision$/) return 1\n"
            "    if (p ~ /^source_commit$/) return 1\n"
            "    if (p ~ /^components$/) return 1"
        )
        if target_handwritten not in template_text:
            raise ValueError(f"Handwritten keys block not found in {template_path}")
        template_text = template_text.replace(target_handwritten, catalogue_keys, 1)

    template_path.write_text(template_text, encoding="utf-8")


if __name__ == "__main__":
    render_all()
    print("Rendered platform.v1.json and spliced install.sh.in successfully.")
