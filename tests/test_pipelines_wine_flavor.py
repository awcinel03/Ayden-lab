"""Wine flavor experiment on a toy co-occurrence network, with Supabase mocked."""

import copy
import io
import os
from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import pandas as pd
import pytest

from pipelines.common import network_metrics as nm
from pipelines.experiments.network_science import wine_flavor as wx

RED = ["cherry", "tannin", "plum", "blackberry"]
WHITE = ["apple", "citrus", "lemon", "pear"]


def toy_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two 4-cliques bridged through acid, plus brett with no edges (isolated)."""
    terms = pd.DataFrame(
        {
            "term": ["acid", *RED, *WHITE, "brett"],
            "reviews": [90, 80, 70, 60, 50, 40, 30, 20, 15, 17],
        }
    )
    pairs = [(a, b) for grp in (RED, WHITE) for i, a in enumerate(grp) for b in grp[i + 1 :]]
    pairs += [("acid", "cherry"), ("acid", "apple"), ("acid", "tannin")]
    edges = pd.DataFrame(
        [(a, b, 100 + 10 * i) for i, (a, b) in enumerate(pairs)],
        columns=["term_a", "term_b", "weight"],
    )
    return terms, edges


SMALL = copy.deepcopy(wx.CONFIG)
SMALL["small_world"]["path_sample_size"] = 4
SMALL["robustness"].update(steps=5, random_seeds=[42, 43])
SMALL["louvain_seed_sweep"] = [0, 1]
SMALL["top_pairs"] = 3
SMALL["network_view"]["edge_weight_above"] = 150


def test_build_graph_keeps_isolated_terms_and_weights():
    g = wx.build_graph(*toy_tables())
    assert g.number_of_nodes() == 10 and g.number_of_edges() == 15
    assert g.degree("brett") == 0 and g.nodes["brett"]["reviews"] == 17
    assert list(g.nodes())[0] == "acid"  # terms-table order is node order
    assert g["cherry"]["tannin"]["weight"] == 100


def test_build_graph_rejects_edges_to_unknown_terms():
    terms, edges = toy_tables()
    with pytest.raises(ValueError, match="missing"):
        wx.build_graph(terms[terms["term"] != "pear"], edges)


def test_all_shortest_paths_are_unweighted_and_skip_isolated_nodes():
    g = wx.build_graph(*toy_tables())
    paths = {(p["source"], p["target"]): p for p in wx.all_shortest_paths(g)}
    assert len(paths) == 9 * 8 // 2  # every pair among the 9 connected terms
    assert not any("brett" in key for key in paths)
    assert paths[("cherry", "tannin")]["hops"] == 1
    lemon_plum = paths[("lemon", "plum")]
    assert lemon_plum["hops"] == 4
    steps = lemon_plum["path"].split(" > ")
    assert steps[0] == "lemon" and steps[-1] == "plum" and "acid" in steps


def test_compute_reports_degrees_curves_and_artifacts():
    rows, flavors, cooccurrence, paths = wx.compute(*toy_tables(), SMALL)
    s = {r["metric"]: r["value"] for r in rows if "dims" not in r}
    assert s["nodes"] == 10 and s["edges"] == 15 and s["isolated_nodes"] == 1
    assert s["components"] == 2 and s["min_degree"] == 0 and s["max_degree"] == 4
    assert s["avg_degree"] == pytest.approx(3.0)
    assert s["random_clustering"] == pytest.approx(15 / 45)
    assert s["connected_pairs"] == 36 and s["view_edge_threshold"] == 150
    deg = {r["dims"]["term"]: r["dims"]["degree"] for r in rows if r["metric"] == "term_reviews"}
    assert deg["acid"] == 3 and deg["brett"] == 0
    top = [r for r in rows if r["metric"] == "top_pair"]
    assert [t["value"] for t in top] == [240, 230, 220]
    curves: dict = {}
    for r in rows:
        if r["metric"] == "giant_fraction":
            curves.setdefault((r["dims"]["strategy"], r["dims"].get("seed")), []).append(r["value"])
    assert set(curves) == {("targeted", None), ("random", 42), ("random", 43)}
    assert all(len(c) == 5 for c in curves.values())
    # State before each removal: the first point is the intact giant, 9 of 10 terms.
    assert curves[("targeted", None)][0] == pytest.approx(0.9)
    # Degree-4 ties go to the first in node order: cherry, then apple, which cuts off the
    # white clique (acid now reaches it only through apple).
    assert curves[("targeted", None)][:3] == pytest.approx([0.9, 0.8, 0.4])
    by = {f["term"]: f for f in flavors}
    assert set(by) == {"acid", *RED, *WHITE, "brett"}
    assert {by[t]["community"] for t in RED} != {by[t]["community"] for t in WHITE}
    assert len({by[t]["community"] for t in RED}) == 1
    assert by["brett"]["community"] == max(f["community"] for f in flavors)  # size rank: last
    assert len(cooccurrence) == 15 and len(paths) == 36
    keys = [(r["metric"], tuple(sorted(r.get("dims", {}).items()))) for r in rows]
    assert len(keys) == len(set(keys))


def test_power_law_fit_uses_the_shared_count_above_one_rule():
    rows, *_ = wx.compute(*toy_tables(), SMALL)
    s = {r["metric"]: r["value"] for r in rows if "dims" not in r}
    counts = {r["dims"]["degree"]: r["value"] for r in rows if r["metric"] == "degree_count"}
    assert counts[0] == 1  # the isolated term is counted but kept out of the fit
    assert s["power_law_points"] == sum(1 for c in counts.values() if c > 1)


def _parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def test_run_verifies_all_four_datasets_and_writes_three_artifacts(monkeypatch):
    terms, edges = toy_tables()
    v2_terms = terms.assign(substring_only_reviews=0, forms_only_reviews=1)
    tables = {
        wx.TERMS_V2_SLUG: v2_terms,
        wx.EDGES_V2_SLUG: edges,
        wx.TERMS_SLUG: terms,
        wx.EDGES_SLUG: edges[edges["term_a"] != "acid"],  # v1 differs: no bridge
    }
    blobs = {f"derived/{slug}.parquet": _parquet(t) for slug, t in tables.items()}
    meta = {
        slug: {
            "id": i,
            "sha256": wx.sha256_bytes(blobs[f"derived/{slug}.parquet"]),
            "storage_path": f"datasets/derived/{slug}.parquet",
        }
        for i, slug in enumerate(tables, start=1)
    }
    client = MagicMock()
    eq = client.table.return_value.select.return_value.eq
    eq.side_effect = lambda col, slug: MagicMock(**{"execute.return_value.data": [meta[slug]]})
    client.storage.from_.return_value.download.side_effect = lambda p: blobs[p]
    started, written = {}, []
    monkeypatch.setattr(wx, "sync_experiments", lambda c: {wx.EXPERIMENT_SLUG: 7})
    monkeypatch.setattr(wx, "git_sha", lambda allow_dirty: "abc")
    monkeypatch.setattr(wx, "start_run", lambda c, **kw: started.update(kw) or 11)
    monkeypatch.setattr(wx, "CONFIG", SMALL)
    monkeypatch.setattr(wx, "write_results", lambda c, run_id, rows: written.extend(rows))
    recorded = []
    monkeypatch.setattr(wx, "record_artifact", lambda c, run_id, **kw: recorded.append(kw))
    monkeypatch.setattr(wx, "tracked", lambda c, run_id: MagicMock())
    assert wx.run(client) == 11
    assert started["datasets"] == {m["id"]: m["sha256"] for m in meta.values()}
    assert [a["storage_path"] for a in recorded] == [
        f"wine-flavor-network/11/{n}.parquet" for n in ("flavors", "cooccurrence", "shortest_paths")
    ]
    s = {r["metric"]: r["value"] for r in written if "dims" not in r}
    assert s["edges"] == 15 and s["v1_edges"] == 12  # headline is v2; v1 is the comparison
    assert sum(r["value"] for r in written if r["metric"] == "forms_only_reviews") == 10


def test_compute_comparison_prefixes_v1_and_ranks_communities():
    rows = wx.compute_comparison(*toy_tables(), SMALL)
    assert all(r["metric"].startswith("v1_") for r in rows)
    s = {r["metric"]: r["value"] for r in rows if "dims" not in r}
    assert s["v1_nodes"] == 10 and s["v1_edges"] == 15 and s["v1_isolated_nodes"] == 1
    comm = {r["dims"]["term"]: r["value"] for r in rows if r["metric"] == "v1_term_community"}
    assert comm["brett"] == max(comm.values()) and comm["cherry"] != comm["apple"]
    steps = {
        (r["dims"]["strategy"], r["dims"]["step"])
        for r in rows
        if r["metric"] == "v1_giant_fraction"
    }
    assert steps == set()  # SMALL runs 5 steps, so steps 10 and 25 are not reached
    keys = [(r["metric"], tuple(sorted(r.get("dims", {}).items()))) for r in rows]
    assert len(keys) == len(set(keys))


def test_run_rejects_a_dataset_with_the_wrong_sha(monkeypatch):
    client = MagicMock()
    row = {"id": 1, "sha256": "0" * 64, "storage_path": "datasets/derived/t.parquet"}
    client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [row]
    client.storage.from_.return_value.download.return_value = b"tampered"
    monkeypatch.setattr(wx, "sync_experiments", lambda c: {wx.EXPERIMENT_SLUG: 7})
    monkeypatch.setattr(wx, "git_sha", lambda allow_dirty: "abc")
    monkeypatch.setattr(wx, "start_run", lambda c, **kw: 11)
    monkeypatch.setattr(wx, "tracked", lambda c, run_id: MagicMock())
    with pytest.raises(RuntimeError, match="sha256"):
        wx.run(client)


# Reproduction of the published (v1) figures and the v2 headline on the real Kaggle CSV.
# The CSV is local only (never committed), so these skip unless it is present.
CSV_CANDIDATES = [
    Path(os.environ.get("WINE_CSV", "/nonexistent")),
    Path(__file__).resolve().parent.parent / "data" / "raw" / "winemag-data-130k-v2.csv",
]
CSV = next((p for p in CSV_CANDIDATES if p.is_file()), None)
needs_csv = pytest.mark.skipif(CSV is None, reason="Kaggle wine CSV not available locally")


@pytest.fixture(scope="module")
def real_tables():
    from pipelines.ingest import wine_flavor as wf

    descriptions = pd.read_csv(CSV, index_col=0)["description"]
    _, lexicon = wf.load_lexicon()
    v1 = wf.aggregate(descriptions, lexicon)
    _, forms = wf.load_forms()
    v2 = wf.aggregate_tokens(descriptions, forms)
    return {
        1: (wf.terms_table(v1), wf.edges_table(v1)),
        2: (wf.terms_table(v2), wf.edges_table(v2)),
    }


@needs_csv
def test_v1_reproduces_the_published_figures(real_tables):
    terms, edges = real_tables[1]
    g = wx.build_graph(terms, edges)
    assert (g.number_of_nodes(), g.number_of_edges()) == (116, 4619)
    assert round(2 * g.number_of_edges() / g.number_of_nodes(), 2) == 79.64
    reviews = dict(zip(terms["term"], terms["reviews"], strict=True))
    assert [reviews[t] for t in ("acid", "acidity", "tannin", "cherry")] == [
        38564,
        34121,
        30466,
        26904,
    ]
    assert [g.degree(t) for t in ("acid", "acidity", "fresh", "tar")] == [114, 113, 112, 112]
    assert g.degree("herbs") == 106  # the write-up's 93 is not reproducible
    assert g.degree("brett") == 0 and reviews["brett"] == 17
    assert round(nx.average_clustering(g), 3) == 0.861
    assert round(nx.density(g), 3) == 0.693
    targeted = nm.robustness_curve(g, strategy="targeted", steps=50, seed=0)
    rand = nm.robustness_curve(g, strategy="random", steps=50, seed=42)
    assert [round(targeted[k], 3) for k in (10, 25)] == [0.905, 0.767]
    assert [round(rand[k], 3) for k in (10, 25)] == [0.905, 0.776]
    sizes = nm.louvain(g, seed=42).sizes
    assert sizes == [69, 46, 1]


@needs_csv
def test_v2_headline_is_stable(real_tables):
    terms, edges = real_tables[2]
    g = wx.build_graph(terms, edges)
    assert (g.number_of_nodes(), g.number_of_edges()) == (116, 4470)
    assert g.degree("acidity") == 113 and g.degree("brett") == 0
    comms = nm.louvain(g, seed=42)
    part = comms.partition
    assert comms.sizes == [62, 45, 8, 1]
    assert part["tar"] == part["cherry"] != part["apple"] == part["acid"] == part["acidity"]
