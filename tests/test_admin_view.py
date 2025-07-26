import importlib
import json


def test_admin_page(tmp_path):
    review = importlib.import_module("dream")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/admin"):
        html = review.admin_page()
    assert "Admin" in html


def test_admin_page_shows_log(tmp_path):
    review = importlib.import_module("dream")
    log = tmp_path / "task_log.txt"
    log.write_text("1\tdid it\n")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/admin"):
        html = review.admin_page()
    assert "Task Log" in html


def test_admin_day_page_shows_log(tmp_path):
    review = importlib.import_module("dream")
    day = tmp_path / "20240102"
    day.mkdir()
    (day / "task_log.txt").write_text("2\tdid stuff\n")
    review.journal_root = str(tmp_path)
    with review.app.test_request_context("/admin/20240102"):
        html = review.admin_day_page("20240102")
    assert "Task Log" in html


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
    assert [
        sys.executable,
        "-m",
        "think.indexer",
        "--index",
        "summaries",
        "--rescan",
        "--verbose",
    ] in called
    assert [
        sys.executable,
        "-m",
        "think.indexer",
        "--index",
        "events",
        "--rescan",
        "--verbose",
    ] in called
    assert [
        sys.executable,
        "-m",
        "think.indexer",
        "--index",
        "transcripts",
        "--rescan",
        "--verbose",
    ] in called

    called.clear()
    with review.app.test_request_context("/admin/api/summary", method="POST"):
        resp = review.refresh_summary()
    assert resp.json["status"] == "ok"
    assert ["think-journal-stats", "--verbose"] in called

    called.clear()
    with review.app.test_request_context("/admin/api/reload_entities", method="POST"):
        resp = review.reload_entities_view()
    assert resp.json["status"] == "ok"
    assert [sys.executable, "-m", "think.entities", "--rescan", "--verbose"] in called


def test_task_log_api(monkeypatch, tmp_path):
    review = importlib.import_module("dream")
    log = tmp_path / "task_log.txt"
    log.write_text("10\tfirst\n20\tsecond\n")
    review.journal_root = str(tmp_path)
    monkeypatch.setattr("time.time", lambda: 30)
    with review.app.test_request_context("/admin/api/task_log"):
        resp = review.task_log()
    data = resp.json
    assert data[0]["message"] == "second"
    assert data[0]["since"].endswith("ago")


def test_task_log_api_day(monkeypatch, tmp_path):
    review = importlib.import_module("dream")
    day = tmp_path / "20240101"
    day.mkdir()
    log = day / "task_log.txt"
    log.write_text("60\tdid thing\n")
    review.journal_root = str(tmp_path)
    monkeypatch.setattr("time.time", lambda: 120)
    with review.app.test_request_context("/admin/api/20240101/task_log"):
        resp = review.task_log("20240101")
    data = resp.json
    assert data[0]["message"] == "did thing"
