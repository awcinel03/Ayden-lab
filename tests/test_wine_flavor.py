"""Wine flavor ingest (and, later, experiment) with Supabase mocked. No real review text."""

import io
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

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


def test_v2_lexicon_keeps_v1_terms_and_has_disjoint_forms():
    _, v1_terms = wf.load_lexicon()
    version, forms = wf.load_forms()
    assert version == 2 and list(forms) == v1_terms
    v1_doc = json.loads(wf.LEXICON_PATH.read_text())
    v2_doc = json.loads(wf.LEXICON_V2_PATH.read_text())
    assert [t["category"] for t in v2_doc["terms"]] == [t["category"] for t in v1_doc["terms"]]
    flat = [f for fs in forms.values() for f in fs]
    assert len(flat) == len(set(flat))
    assert all(term in fs for term, fs in forms.items())
    assert "acidity" not in forms["acid"] and "tart" not in forms["tar"]
    assert "pineapple" not in forms["apple"] and "honeysuckle" not in forms["honey"]


def test_v2_matches_whole_tokens_of_accepted_forms():
    forms = {
        "acid": ["acid", "acids", "acidic"],
        "acidity": ["acidity"],
        "tar": ["tar", "tarry"],
        "apple": ["apple", "apples"],
        "pineapple": ["pineapple"],
        "herbs": ["herbs", "herb"],
    }
    agg = wf.aggregate_tokens(
        ["Bright ACIDITY, a tart edge and pineapple.", "Tarry, with apples and an herb note."],
        forms,
    )
    # "acid" no longer matches inside "acidity", nor "tar" inside "tart", nor "apple"
    # inside "pineapple"; inflections listed as forms ("tarry", "apples", "herb") do match.
    assert agg.term_counts == {"acidity": 1, "pineapple": 1, "tar": 1, "apple": 1, "herbs": 1}
    assert agg.edge_counts == {
        ("acidity", "pineapple"): 1,
        ("apple", "herbs"): 1,
        ("apple", "tar"): 1,
        ("herbs", "tar"): 1,
    }


def test_v2_splits_on_hyphens_and_keeps_lexicon_order():
    agg = wf.aggregate_tokens(["oak-aged; cherry-like"], {"cherry": ["cherry"], "oak": ["oak"]})
    assert list(agg.term_counts) == ["cherry", "oak"]  # lexicon order, not text order


def test_v2_rejects_a_form_listed_twice():
    with pytest.raises(ValueError, match="more than one term"):
        wf.aggregate_tokens([], {"smoke": ["smoke", "smokey"], "smoky": ["smoky", "smokey"]})


def test_ingest_builds_both_versions_with_their_own_slugs(tmp_path):
    client, datasets = mock_client()
    assert wf.ingest(client, toy_csv(tmp_path), versions=(1, 2)) == "updated"
    uploaded = [c.args[0] for c in client.storage.from_.return_value.upload.call_args_list]
    assert uploaded == [
        "derived/wine-flavor-terms-v1.parquet",
        "derived/wine-flavor-cooccurrence-v1.parquet",
        "derived/wine-flavor-terms-v2.parquet",
        "derived/wine-flavor-cooccurrence-v2.parquet",
    ]
    rows = {r["slug"]: r for r in datasets.upsert.call_args.args[0]}
    assert "lexicon v2" in rows["wine-flavor-terms-v2"]["notes"]
    assert "lexicon v1" in rows["wine-flavor-terms-v1"]["notes"]
    v2_terms = pd.read_parquet(
        io.BytesIO(client.storage.from_.return_value.upload.call_args_list[2].args[1])
    )
    # "Acidity, tannin and cherry.": v1 also finds acid (inside acidity); v2 does not.
    assert set(v2_terms["term"]) == {"acidity", "tannin", "cherry"}


def test_v1_outputs_are_unchanged_by_requesting_v2(tmp_path):
    only_v1, _ = mock_client()
    wf.ingest(only_v1, toy_csv(tmp_path))
    both, _ = mock_client()
    wf.ingest(both, toy_csv(tmp_path), versions=(1, 2))
    first = only_v1.storage.from_.return_value.upload.call_args_list
    second = both.storage.from_.return_value.upload.call_args_list[:2]
    assert [c.args[:2] for c in first] == [c.args[:2] for c in second]


def test_match_differences_separates_false_matches_from_added_forms():
    forms = {"tar": ["tar", "tarry"], "herbs": ["herbs", "herb"], "acid": ["acid"]}
    diffs = wf.match_differences(
        ["A tart edge.", "Tarry, herb and herbs.", "One herb.", None], forms
    ).set_index("term")
    assert diffs.loc["tar"].tolist() == [1, 0]  # "tart": substring only
    assert diffs.loc["herbs"].tolist() == [0, 1]  # "herb" alone: form only
    assert diffs.loc["acid"].tolist() == [0, 0]


def test_v2_terms_table_carries_match_differences(tmp_path):
    client, _ = mock_client()
    wf.ingest(client, toy_csv(tmp_path), versions=(2,))
    terms = pd.read_parquet(
        io.BytesIO(client.storage.from_.return_value.upload.call_args_list[0].args[1])
    )
    assert list(terms.columns) == [
        "term",
        "reviews",
        "substring_only_reviews",
        "forms_only_reviews",
    ]
    assert "acid" not in set(terms["term"])  # not a v2 match in "Acidity, tannin and cherry."
    assert (terms[["substring_only_reviews", "forms_only_reviews"]] == 0).all().all()
