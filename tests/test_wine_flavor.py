"""Wine flavor ingest (and, later, experiment) with Supabase mocked. No real review text."""

import io
import json
from unittest.mock import MagicMock

import pandas as pd

from pipelines.ingest import wine_flavor as wf


def test_lexicon_is_the_notebooks_116_terms():
    version, terms = wf.load_lexicon()
    assert version == 1 and len(terms) == 116 and len(set(terms)) == 116
    assert {"acid", "acidity", "tannin", "fresh", "tar", "brett", "flint"} <= set(terms)
    doc = json.loads(wf.LEXICON_PATH.read_text())
    assert all(t["category"] for t in doc["terms"])


def test_matching_is_substring_and_lowercase_as_in_the_notebook():
    lexicon = ["acid", "acidity", "tar", "cherry", "oak"]
    agg = wf.aggregate(["Bright ACIDITY and a tart cherry finish."], lexicon)
    # "acid" matches inside "acidity"; "tar" matches inside "tart"
    assert set(agg.term_counts) == {"acid", "acidity", "tar", "cherry"}
    assert agg.edge_counts[("acid", "acidity")] == 1


def test_missing_and_single_term_reviews_are_skipped():
    agg = wf.aggregate([None, float("nan"), "cherry only", "cherry and oak"], ["cherry", "oak"])
    assert agg.term_counts == {"cherry": 1, "oak": 1}
    assert agg.edge_counts == {("cherry", "oak"): 1}


def test_first_appearance_order_follows_reviews_then_lexicon():
    agg = wf.aggregate(["oak cherry", "plum oak"], ["cherry", "oak", "plum"])
    assert list(agg.term_counts) == ["cherry", "oak", "plum"]


def test_edge_threshold_and_tables_hold_no_text():
    texts = ["cherry oak"] * 10 + ["cherry plum"] * 9
    agg = wf.aggregate(texts, ["cherry", "oak", "plum"])
    edges = wf.edges_table(agg, min_weight=10)
    assert edges.to_dict("records") == [{"term_a": "cherry", "term_b": "oak", "weight": 10}]
    terms = wf.terms_table(agg)
    assert list(terms.columns) == ["term", "reviews"] and list(edges.columns) == [
        "term_a",
        "term_b",
        "weight",
    ]
    assert dict(zip(terms["term"], terms["reviews"], strict=True)) == {
        "cherry": 19,
        "oak": 10,
        "plum": 9,
    }


def toy_csv(tmp_path):
    df = pd.DataFrame({"country": ["X"] * 12, "description": ["Acidity, tannin and cherry."] * 12})
    path = tmp_path / "toy.csv"
    df.to_csv(path)  # index column, like the Kaggle file
    return path


def mock_client(existing=None, exists=True):
    client = MagicMock()
    datasets = MagicMock()
    datasets.select.return_value.eq.side_effect = lambda col, slug: MagicMock(
        **{"execute.return_value.data": [{"sha256": existing[slug]}] if existing else []}
    )
    client.table.return_value = datasets
    client.storage.from_.return_value.exists.return_value = exists
    return client, datasets


def test_ingest_never_uploads_raw_csv_and_records_licence(tmp_path):
    client, datasets = mock_client()
    assert wf.ingest(client, toy_csv(tmp_path)) == "updated"
    uploaded = [c.args[0] for c in client.storage.from_.return_value.upload.call_args_list]
    assert uploaded == [
        "derived/wine-flavor-terms-v1.parquet",
        "derived/wine-flavor-cooccurrence-v1.parquet",
    ]
    rows = {r["slug"]: r for r in datasets.upsert.call_args.args[0]}
    raw = rows["kaggle-wine-reviews-130k"]
    assert raw["storage_path"] is None and raw["row_count"] == 12
    assert raw["license"].startswith("CC BY-NC-SA 4.0") and "kaggle.com" in raw["source_url"]
    for slug in ("wine-flavor-terms-v1", "wine-flavor-cooccurrence-v1"):
        assert rows[slug]["storage_path"].startswith("datasets/derived/")
        assert "no review text" in rows[slug]["notes"]
    for call in client.storage.from_.return_value.upload.call_args_list:
        table = pd.read_parquet(io.BytesIO(call.args[1]))
        assert "description" not in table.columns
        assert (
            not table.astype(str)
            .apply(lambda s: s.str.contains("cherry.", regex=False))
            .any()
            .any()
        )


def test_ingest_is_idempotent(tmp_path):
    path = toy_csv(tmp_path)
    first, _ = mock_client()
    wf.ingest(first, path)
    shas = {r["slug"]: r["sha256"] for r in first.table.return_value.upsert.call_args.args[0]}
    again, datasets = mock_client(existing=shas)
    assert wf.ingest(again, path) == "unchanged"
    datasets.upsert.assert_not_called()
    again.storage.from_.return_value.upload.assert_not_called()
