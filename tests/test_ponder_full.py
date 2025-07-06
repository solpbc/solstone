import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

FIXTURES = Path("fixtures")


def copy_day(tmp_path: Path) -> Path:
    src = FIXTURES / "journal" / "20240101"
    dest = tmp_path / "20240101"
    shutil.copytree(src, dest)
    return dest


def test_ponder_main(tmp_path, monkeypatch):
    mod = importlib.import_module("think.ponder")
    day = copy_day(tmp_path)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("prompt")

    monkeypatch.setattr(
        mod,
        "send_markdown",
        lambda *a, **k: (
            "summary",
            SimpleNamespace(prompt_token_count=1, thoughts_token_count=1, candidates_token_count=1),
        ),
    )
    monkeypatch.setattr(
        mod,
        "send_occurrence",
        lambda *a, **k: [
            {
                "type": "meeting",
                "start": "00:00:00",
                "end": "00:00:00",
                "title": "t",
                "summary": "s",
                "work": True,
                "participants": [],
                "details": "",
            }
        ],
    )
    monkeypatch.setattr(mod, "load_dotenv", lambda: True)
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    monkeypatch.setattr("sys.argv", ["ponder-day", str(day), "-f", str(prompt)])
    mod.main()

    md = day / "ponder_prompt.md"
    js = day / "ponder_prompt.json"
    assert md.read_text() == "summary"
    data = json.loads(js.read_text())
    assert data["day"] == "20240101"
    assert data["occurrences"]
    assert md.with_suffix(md.suffix + ".crumb").is_file()
    assert js.with_suffix(js.suffix + ".crumb").is_file()
