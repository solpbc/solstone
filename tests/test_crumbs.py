import importlib
import os
from pathlib import Path


def test_crumb_builder(tmp_path):
    mod = importlib.import_module("think.crumbs")
    builder = mod.CrumbBuilder(generator="test")
    file1 = tmp_path / "f1.txt"
    file1.write_text("data")
    builder.add_file(file1)
    builder.add_files([file1])
    builder.add_glob(str(file1))
    builder.add_model("m")
    crumb_path = builder.commit(str(tmp_path / "out.txt"))
    assert Path(crumb_path).is_file()
    data = Path(crumb_path).read_text()
    assert "generator" in data


def test_validate_crumb(tmp_path):
    mod = importlib.import_module("think.crumbs")
    src = tmp_path / "dep.txt"
    src.write_text("x")
    out = tmp_path / "out.txt"
    out.write_text("o")
    mod.CrumbBuilder(generator="test").add_file(src).commit(str(out))

    assert mod.validate_crumb(str(out)) is mod.CrumbState.OK

    src.write_text("changed")
    os.utime(src, (os.path.getmtime(src) + 1, os.path.getmtime(src) + 1))
    assert mod.validate_crumb(str(out)) is mod.CrumbState.STALE

    out.with_suffix(out.suffix + ".crumb").unlink()
    assert mod.validate_crumb(str(out)) is mod.CrumbState.MISSING
