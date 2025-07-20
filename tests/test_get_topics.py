import importlib
import os


def test_get_topics():
    utils = importlib.import_module("think.utils")
    topics = utils.get_topics()
    assert "day" in topics
    info = topics["day"]
    assert os.path.basename(info["path"]) == "day.txt"
    assert isinstance(info["color"], str)
    assert isinstance(info["mtime"], int)
    assert "title" in info
