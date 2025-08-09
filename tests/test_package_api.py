import importlib


def test_public_functions(tmp_path):
    think = importlib.import_module("think")
    # Test that core functions are available
    assert hasattr(think, "cluster_day")
    assert hasattr(think, "cluster_range")
    assert hasattr(think, "cluster_scan")
