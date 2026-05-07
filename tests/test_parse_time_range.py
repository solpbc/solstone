# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import importlib
from datetime import datetime


def test_parse_time_range():
    util = importlib.import_module("solstone.think.utils")
    res = util.parse_time_range("July 19 3p-4p")
    assert res is not None
    day, start, end = res
    expected_day = datetime(datetime.now().year, 7, 19).strftime("%Y%m%d")
    assert day == expected_day
    assert start == "150000"
    assert end == "160000"
