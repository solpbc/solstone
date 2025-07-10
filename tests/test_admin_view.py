import importlib
import json


def test_admin_page(tmp_path):
    review = importlib.import_module("dream")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/admin"):
        html = review.admin_page()
    assert "Admin" in html


def test_admin_page_lists_repairables(tmp_path):
    review = importlib.import_module("dream")
    stats = {
        "days": {
            "20240101": {
                "repair_hear": 1,
                "repair_see": 0,
                "repair_reduce": 0,
                "repair_ponder": 0,
                "repair_entity": 0,
            }
        }
    }
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/admin"):
        html = review.admin_page()
    assert "20240101" in html
    assert "Hear" in html


def test_admin_actions(monkeypatch, tmp_path):
    review = importlib.import_module("dream")
    review.journal_root = str(tmp_path)
    called = []

    import sys

    tr = sys.modules["dream.task_runner"]
    monkeypatch.setattr(tr, "_run_command", lambda cmd, log: called.append(cmd) or 0)

    with review.app.test_request_context("/admin/api/reindex", method="POST"):
        resp = review.reindex()
    assert resp.json["status"] == "ok"
    assert [sys.executable, "-m", "think.indexer", "--rescan"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/summary", method="POST"):
        resp = review.refresh_summary()
    assert resp.json["status"] == "ok"
    assert ["journal-stats"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/reload_entities", method="POST"):
        resp = review.reload_entities_view()
    assert resp.json["status"] == "ok"
    assert [sys.executable, "-m", "think.entities", "--rescan"] in called
