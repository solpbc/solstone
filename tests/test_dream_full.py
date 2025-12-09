import importlib
import shutil
from pathlib import Path

FIXTURES = Path("fixtures")


def copy_journal(tmp_path: Path) -> Path:
    src = FIXTURES / "journal"
    dest = tmp_path / "journal"
    shutil.copytree(src, dest)
    return dest


def test_main_runs(tmp_path, monkeypatch):
    mod = importlib.import_module("think.dream")
    journal = copy_journal(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    called = []
    monkeypatch.setattr(mod, "run_command", lambda cmd, day: called.append(cmd))
    monkeypatch.setattr("think.utils.load_dotenv", lambda: True)
    monkeypatch.setattr(
        "sys.argv", ["think-dream", "--day", "20240101", "--force", "--verbose"]
    )
    mod.main()
    assert any(c[0] == "observe-sense" for c in called)
    assert any(c[0] == "think-insight" for c in called)
    # Verify indexer is called with --rescan (light mode)
    indexer_cmds = [c for c in called if c[0] == "think-indexer"]
    assert len(indexer_cmds) == 1
    assert "--rescan" in indexer_cmds[0]
