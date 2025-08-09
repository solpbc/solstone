import importlib
import shutil
from pathlib import Path

FIXTURES = Path("fixtures")


def copy_journal(tmp_path: Path) -> Path:
    src = FIXTURES / "journal"
    dest = tmp_path / "journal"
    shutil.copytree(src, dest)
    return dest


def test_build_commands(tmp_path):
    mod = importlib.import_module("think.process_day")
    cmds = mod.build_commands("20240101", force=True, verbose=True)
    assert ["see-reduce", "20240101", "--verbose", "--force"] in cmds
    assert any(cmd[0] == "think-ponder" for cmd in cmds)
    assert any(cmd[0] == "think-entity-roll" for cmd in cmds)
    assert cmds[-1][0] == "think-journal-stats"


def test_main_runs(tmp_path, monkeypatch):
    mod = importlib.import_module("think.process_day")
    journal = copy_journal(tmp_path)
    monkeypatch.setenv("JOURNAL_PATH", str(journal))
    called = []
    monkeypatch.setattr(mod, "run_command", lambda cmd: called.append(cmd))
    monkeypatch.setattr("think.utils.load_dotenv", lambda: True)
    monkeypatch.setattr(
        "sys.argv", ["think-process-day", "--day", "20240101", "--force", "--verbose"]
    )
    mod.main()
    assert any(c[0] == "see-reduce" for c in called)
    assert any(c[0] == "think-ponder" for c in called)
    assert any(c[0] == "think-entity-roll" for c in called)
