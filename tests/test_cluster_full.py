import importlib
import shutil
from pathlib import Path

FIXTURES = Path("fixtures")


def copy_day(tmp_path: Path) -> Path:
    src = FIXTURES / "journal" / "20240101"
    dest = tmp_path / "20240101"
    shutil.copytree(src, dest)
    return dest


def test_cluster_full(tmp_path, monkeypatch):
    mod = importlib.import_module("think.cluster")
    copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    md, count = mod.cluster("20240101")
    assert count == 2
    assert "Audio Transcript" in md
    assert "Screen Activity Summary" in md


def test_cluster_cli(tmp_path, monkeypatch, capsys):
    mod = importlib.import_module("think.cluster")
    copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["cluster", "20240101"])
    mod.main()
    out = capsys.readouterr().out
    assert "Screen Activity Summary" in out


def test_cluster_cli_range(tmp_path, monkeypatch, capsys):
    mod = importlib.import_module("think.cluster")
    copy_day(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        ["cluster", "20240101", "--start", "123456", "--length", "1"],
    )
    mod.main()
    out = capsys.readouterr().out
    assert "Screen Activity Summary" in out
