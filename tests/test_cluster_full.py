import importlib
import shutil
from pathlib import Path

FIXTURES = Path("fixtures")


def copy_day(tmp_path: Path) -> Path:
    src = FIXTURES / "journal" / "20240101"
    dest = tmp_path / "20240101"
    shutil.copytree(src, dest)
    return dest


def test_cluster_full(tmp_path):
    mod = importlib.import_module("think.cluster")
    day = copy_day(tmp_path)
    md, count = mod.cluster(str(day))
    assert count == 2
    assert "Audio Transcript" in md
    assert "Screen Activity Summary" in md


def test_cluster_cli(tmp_path, monkeypatch, capsys):
    mod = importlib.import_module("think.cluster")
    day = copy_day(tmp_path)
    monkeypatch.setattr("sys.argv", ["cluster", str(day)])
    mod.main()
    out = capsys.readouterr().out
    assert "Screen Activity Summary" in out
