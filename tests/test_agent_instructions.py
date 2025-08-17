import importlib


def test_agent_instructions_default():
    utils = importlib.import_module("think.utils")
    system, extra, meta = utils.agent_instructions()
    assert system.startswith("You are Sunstone")
    assert "Current Date and Time" in extra
    assert meta.get("title") == "Sunstone Journal Chat"
    assert "description" in meta
