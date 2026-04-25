# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc


def test_no_stray_test_talent_files():
    from think.talent import TALENT_DIR

    stray = sorted(p.name for p in TALENT_DIR.glob("test_*.md"))
    assert stray == [], f"Stray test talent file(s) under {TALENT_DIR}: {stray}"
