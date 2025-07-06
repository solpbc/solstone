import importlib


def test_admin_page(tmp_path):
    review = importlib.import_module("dream")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/admin"):
        html = review.admin_page()
    assert "Admin" in html


def test_admin_actions(monkeypatch, tmp_path):
    review = importlib.import_module("dream")
    review.journal_root = str(tmp_path)
    called = {}

    monkeypatch.setattr("dream.views.admin.load_cache", lambda j: {})
    monkeypatch.setattr(
        "dream.views.admin.save_cache", lambda j, c: called.setdefault("save", True)
    )
    monkeypatch.setattr(
        "dream.views.admin.scan_entities", lambda j, c: called.setdefault("entities", True) or True
    )
    monkeypatch.setattr(
        "dream.views.admin.scan_ponders", lambda j, c: called.setdefault("ponders", True) or True
    )
    monkeypatch.setattr(
        "dream.views.admin.scan_occurrences", lambda j, c: called.setdefault("occ", True) or True
    )

    with review.app.test_request_context("/admin/api/reindex", method="POST"):
        resp = review.reindex()
    assert resp.json["status"] == "ok"
    assert called == {"entities": True, "ponders": True, "occ": True, "save": True}

    called.clear()
    monkeypatch.setattr(
        "dream.views.admin.JournalStats.scan", lambda self, j: called.setdefault("scan", True)
    )
    monkeypatch.setattr(
        "dream.views.admin.JournalStats.save_markdown",
        lambda self, j: called.setdefault("save", True),
    )
    with review.app.test_request_context("/admin/api/summary", method="POST"):
        resp = review.refresh_summary()
    assert resp.json["status"] == "ok"
    assert called == {"scan": True, "save": True}

    called.clear()
    monkeypatch.setattr(
        "dream.views.admin.reload_entities", lambda: called.setdefault("reload", True)
    )
    with review.app.test_request_context("/admin/api/reload_entities", method="POST"):
        resp = review.reload_entities_view()
    assert resp.json["status"] == "ok"
    assert called == {"reload": True}
