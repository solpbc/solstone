import importlib


def test_admin_day_page(tmp_path):
    review = importlib.import_module("dream")
    (tmp_path / "20240101").mkdir()
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/admin/20240101"):
        html = review.admin_day_page("20240101")
    assert "Run Repairs" in html


def test_admin_day_actions(monkeypatch, tmp_path):
    review = importlib.import_module("dream")
    (tmp_path / "20240101").mkdir()
    review.journal_root = str(tmp_path)
    called = []

    monkeypatch.setattr("dream.views.admin._run", lambda cmd: called.append(cmd))
    monkeypatch.setattr(
        "dream.views.admin.entity_roll.process_day",
        lambda day, dirs, force: called.append(["entity", day]),
    )
    monkeypatch.setattr("dream.views.admin.reduce_day", lambda day: called.append(["reduce", day]))
    monkeypatch.setattr(
        "glob.glob", lambda pattern: ["prompt1.txt", "prompt2.txt"] if "ponder" in pattern else []
    )

    with review.app.test_request_context("/admin/api/20240101/repairs", method="POST"):
        resp = review.admin_repair("20240101")
    assert resp.json["status"] == "ok"
    assert ["gemini-transcribe", "--repair", "20240101"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/20240101/ponder", method="POST"):
        resp = review.admin_ponder("20240101")
    assert resp.json["status"] == "ok"
    assert ["ponder", "20240101", "-f", "prompt1.txt", "-p"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/20240101/entity", method="POST"):
        resp = review.admin_entity("20240101")
    assert resp.json["status"] == "ok"
    assert ["entity", "20240101"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/20240101/reduce", method="POST"):
        resp = review.admin_reduce("20240101")
    assert resp.json["status"] == "ok"
    assert ["reduce", "20240101"] in called
