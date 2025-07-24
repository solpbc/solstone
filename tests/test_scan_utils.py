import importlib

from PIL import Image


def test_cache_roundtrip(tmp_path, monkeypatch):
    scan = importlib.import_module("see.scan")
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    img = Image.new("RGB", (10, 10), "white")
    images = {1: img}
    scan.save_cache(images)
    loaded, ts = scan.load_cache()
    assert 1 in loaded
    hb = tmp_path / "health" / "see.up"
    assert hb.is_file()


def test_censor_border(tmp_path):
    scan = importlib.import_module("see.scan")
    img = Image.new("RGB", (120, 120), "white")
    for x in range(120):
        for y in range(3):
            img.putpixel((x, y), (0, 0, 255))
            img.putpixel((x, 119 - y), (0, 0, 255))
    for y in range(120):
        for x in range(3):
            img.putpixel((x, y), (0, 0, 255))
            img.putpixel((119 - x, y), (0, 0, 255))
    censored = scan.censor_border(img)
    assert censored.getpixel((10, 10)) == (0, 0, 0)
