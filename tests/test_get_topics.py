import importlib
import os


def test_get_topics():
    utils = importlib.import_module("think.utils")
    topics = utils.get_topics()
    assert "flow" in topics
    info = topics["flow"]
    assert os.path.basename(info["path"]) == "flow.txt"
    assert isinstance(info["color"], str)
    assert isinstance(info["mtime"], int)
    assert "title" in info
    assert "occurrences" in info
