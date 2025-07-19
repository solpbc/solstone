import importlib
from datetime import datetime


def test_parse_time_range():
    util = importlib.import_module("think.utils")
    res = util.parse_time_range("July 19 3p-4p")
    assert res is not None
    day, start, end = res
    today = datetime.now().strftime("%Y%m%d")
    assert day == today
    assert start == "150000"
    assert end == "160000"
