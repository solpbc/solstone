import importlib
import os


def test_search_topic_api(tmp_path):
    indexer = importlib.import_module("think.indexer")
    review = importlib.import_module("dream")
    search_view = importlib.import_module("dream.views.search")

    journal = tmp_path
    os.environ["JOURNAL_PATH"] = str(journal)
    day = journal / "20240107" / "topics"
    day.mkdir(parents=True)
    (day / "files.md").write_text("This is a test sentence.\n")

    indexer.scan_topics(str(journal))
    review.journal_root = str(journal)

    with review.app.test_request_context("/search/api/topic?q=test"):
        resp = search_view.search_topic_api()

    assert resp.json["total"] == 1
    assert resp.json["results"][0]["topic"] == "files"
