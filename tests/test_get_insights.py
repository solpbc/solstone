import importlib
import os


def test_get_insights():
    utils = importlib.import_module("think.utils")
    insights = utils.get_insights()
    assert "flow" in insights
    info = insights["flow"]
    assert os.path.basename(info["path"]) == "flow.txt"
    assert isinstance(info["color"], str)
    assert isinstance(info["mtime"], int)
    assert "title" in info
    assert "occurrences" in info
