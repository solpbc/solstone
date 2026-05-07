# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import sys


def test_importing_categories_does_not_load_av_or_cv2(monkeypatch):
    for mod in (
        "av",
        "cv2",
        "solstone.observe.aruco",
        "solstone.observe.describe",
        "solstone.observe.screen",
    ):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    from solstone.observe.describe import CATEGORIES

    assert isinstance(CATEGORIES, dict)
    assert len(CATEGORIES) > 0
    assert "av" not in sys.modules
    assert "cv2" not in sys.modules
