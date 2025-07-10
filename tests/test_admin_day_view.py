import importlib


def test_admin_day_page(tmp_path):
    review = importlib.import_module("dream")
    (tmp_path / "20240101").mkdir()
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/admin/20240101"):
        html = review.admin_day_page("20240101")
    assert "Hear Repair" in html
    assert "See Repair" in html
    assert "Screen Reduce" in html
    assert "Process Day" in html


def test_admin_day_actions(monkeypatch, tmp_path):
    review = importlib.import_module("dream")
    (tmp_path / "20240101").mkdir()
    review.journal_root = str(tmp_path)
    called = []

    import sys

    tr = sys.modules["dream.task_runner"]
    monkeypatch.setattr(tr, "_run_command", lambda cmd, log: called.append(cmd) or 0)
    monkeypatch.setattr(
        "glob.glob", lambda pattern: ["prompt1.txt", "prompt2.txt"] if "ponder" in pattern else []
    )

    with review.app.test_request_context("/admin/api/20240101/repair_hear", method="POST"):
        resp = review.admin_repair_hear("20240101")
    assert resp.json["status"] == "ok"
    assert ["gemini-transcribe", "--repair", "20240101", "-v"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/20240101/repair_see", method="POST"):
        resp = review.admin_repair_see("20240101")
    assert resp.json["status"] == "ok"
    assert ["screen-describe", "--repair", "20240101", "-v"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/20240101/ponder", method="POST"):
        resp = review.admin_ponder("20240101")
    assert resp.json["status"] == "ok"
    assert ["ponder", "20240101", "-f", "prompt1.txt", "-p", "--verbose"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/20240101/entity", method="POST"):
        resp = review.admin_entity("20240101")
    assert resp.json["status"] == "ok"
    assert ["entity-roll", "--day", "20240101", "--force", "--verbose"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/20240101/reduce", method="POST"):
        resp = review.admin_reduce("20240101")
    assert resp.json["status"] == "ok"
    assert ["reduce-screen", "20240101", "--verbose"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/20240101/process", method="POST"):
        resp = review.admin_process("20240101")
    assert resp.json["status"] == "ok"
    assert ["process-day", "--day", "20240101", "--repair", "--verbose"] in called
