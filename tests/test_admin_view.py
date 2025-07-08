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
    called = {}

    import sys

    tr = sys.modules["dream.task_runner"]
    monkeypatch.setattr(tr, "load_cache", lambda j: {})
    monkeypatch.setattr(tr, "save_cache", lambda j, c: called.setdefault("save", True))
    monkeypatch.setattr(
        tr, "scan_entities", lambda j, c: called.setdefault("entities", True) or True
    )
    monkeypatch.setattr(tr, "scan_ponders", lambda j, c: called.setdefault("ponders", True) or True)
    monkeypatch.setattr(tr, "scan_occurrences", lambda j, c: called.setdefault("occ", True) or True)

    with review.app.test_request_context("/admin/api/reindex", method="POST"):
        resp = review.reindex()
    assert resp.json["status"] == "ok"
    assert called == {"entities": True, "ponders": True, "occ": True, "save": True}

    called.clear()
    monkeypatch.setattr(
        sys.modules["dream.task_runner"].JournalStats,
        "scan",
        lambda self, j: called.setdefault("scan", True),
    )
    monkeypatch.setattr(
        sys.modules["dream.task_runner"].JournalStats,
        "save_markdown",
        lambda self, j: called.setdefault("save", True),
    )
    with review.app.test_request_context("/admin/api/summary", method="POST"):
        resp = review.refresh_summary()
    assert resp.json["status"] == "ok"
    assert called == {"scan": True, "save": True}

    called.clear()
    import sys

    tr = sys.modules["dream.task_runner"]
    monkeypatch.setattr(tr, "reload_entities", lambda: called.setdefault("reload", True))
    with review.app.test_request_context("/admin/api/reload_entities", method="POST"):
        resp = review.reload_entities_view()
    assert resp.json["status"] == "ok"
    assert called == {"reload": True}
