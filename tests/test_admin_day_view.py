import importlib

from think.utils import day_path


def test_admin_day_page(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_path("20240101")

    review = importlib.import_module("convey")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/admin/20240101"):
        html = review.admin_day_page("20240101")
    assert "Sense Repair" in html
    assert "Screen Reduce" in html
    assert "Process Day" in html


def test_admin_day_actions(monkeypatch, tmp_path):
    monkeypatch.setenv("JOURNAL_PATH", str(tmp_path))
    day_path("20240101")

    review = importlib.import_module("convey")
    review.journal_root = str(tmp_path)
    called = []

    import sys

    tr = sys.modules["convey.task_runner"]
    monkeypatch.setattr(tr, "_run_command", lambda cmd, log: called.append(cmd) or 0)
    monkeypatch.setattr(
        "glob.glob",
        lambda pattern: ["prompt1.txt", "prompt2.txt"] if "topics" in pattern else [],
    )

    with review.app.test_request_context(
        "/admin/api/20240101/sense_repair", method="POST"
    ):
        resp = review.admin_sense_repair("20240101")
    assert resp.json["status"] == "ok"
    assert ["observe-sense", "--day", "20240101", "-v"] in called

    called.clear()
    with review.app.test_request_context(
        "/admin/api/20240101/summarize", method="POST"
    ):
        resp = review.admin_summarize("20240101")
    assert resp.json["status"] == "ok"
    assert any(cmd[0] == "think-summarize" for cmd in called)

    called.clear()
    with review.app.test_request_context("/admin/api/20240101/entity", method="POST"):
        resp = review.admin_entity("20240101")
    assert resp.json["status"] == "ok"
    assert ["think-entity-roll", "--day", "20240101", "--force", "--verbose"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/20240101/reduce", method="POST"):
        resp = review.admin_reduce("20240101")
    assert resp.json["status"] == "ok"
    assert ["see-reduce", "20240101", "--verbose"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/20240101/process", method="POST"):
        resp = review.admin_process("20240101")
    assert resp.json["status"] == "ok"
    assert ["think-dream", "--day", "20240101", "--verbose"] in called
