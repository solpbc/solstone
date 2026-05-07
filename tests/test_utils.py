# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 sol pbc

import pytest

from solstone.think.utils import parse_duration_seconds


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (30, 30),
        ("45s", 45),
        ("30m", 1800),
        ("1h", 3600),
    ],
)
def test_parse_duration_seconds_valid(spec, expected):
    assert parse_duration_seconds(spec) == expected


@pytest.mark.parametrize(
    "spec",
    [
        0,
        -5,
        "garbage",
        "5x",
        "30 m",
        None,
        [],
    ],
)
def test_parse_duration_seconds_invalid(spec):
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration_seconds(spec)
