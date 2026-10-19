"""Causal-emergence experiment on toy networks, with Supabase mocked."""

import copy
import gzip
from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

from pipelines.common import markov as mk
from pipelines.experiments.network_science import causal_emergence as ce
from tests.test_pipelines_c_elegans import EDGES, GZ, META_GZ

SMALL = copy.deepcopy(ce.CONFIG)
SMALL["worm"].update(random_groupings=5, null_draws=2)
SMALL["email"].update(random_groupings={"structural_equivalence": 3, "louvain": 3}, null_draws=2)

# Two hubs joined by an edge, each with three one-contact addresses, plus a triangle on B.
STARS = [("A", "B"), ("A", "a1"), ("A", "a2"), ("A", "a3"), ("B", "b1"), ("B", "b2"),
         ("B", "b3"), ("B", "c1"), ("B", "c2"), ("c1", "c2")]  # fmt: skip
STARS_GZ = gzip.compress("".join(f"{u}\t{v}\n{v}\t{u}\n" for u, v in STARS).encode())


def values(rows, metric, **dims):
    return [r["value"] for r in rows if r["metric"] == metric
            and all(r.get("dims", {}).get(k) == v for k, v in dims.items())]  # fmt: skip


def test_random_walk_rows_spread_evenly_and_sinks_stay_empty():
    g = nx.DiGraph([("a", "b"), ("a", "c"), ("b", "a")])
    g.add_node("d")
    w = ce.random_walk(g, ["a", "b", "c", "d"]).toarray()
    assert np.allclose(w, [[0, 0.5, 0.5, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    u = ce.random_walk(nx.Graph([("a", "b"), ("b", "c")]), ["a", "b", "c"]).toarray()
    assert np.allclose(u, [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]])


def test_structural_classes_group_identical_contacts():
    g = nx.Graph(STARS)
    nodes = list(g.nodes())
    classes = [sorted(nodes[i] for i in c) for c in ce.structural_classes(g, nodes)]
    assert ["a1", "a2", "a3"] in classes and ["b1", "b2", "b3"] in classes
    assert ["A"] in classes  # c1 and c2 differ: each lists the other


def test_merging_a_stars_leaves_is_causal_emergence():
    g = nx.star_graph(4)
    w = ce.random_walk(g, list(g.nodes()))
    assert mk.causal_emergence(w, [[0], [1, 2, 3, 4]], np.ones(5)) == pytest.approx(
        1 - mk.effective_information(w)
    )
    assert mk.effective_information(w) < 1


def test_compute_worm_rows():
    g = nx.DiGraph()
    g.add_edges_from((str(u), str(v)) for u, v in EDGES)
    rows = ce.compute_worm(g, {str(i): f"N{i}" for i in range(5)}, SMALL)
    micro = values(rows, "micro_ei", network="worm")[0]
    det, deg = values(rows, "micro_determinism")[0], values(rows, "micro_degeneracy")[0]
    assert det - deg == pytest.approx(micro)
    for grouping in ("roles", "greedy_within_roles", "greedy"):
        ei = values(rows, "grouping_ei", grouping=grouping)[0]
        assert values(rows, "grouping_ce", grouping=grouping)[0] == pytest.approx(ei - micro)
        assert 0 <= values(rows, "random_beaten", grouping=grouping)[0] <= 1
    assert values(rows, "grouping_groups", grouping="roles") == [3]
    assert len(values(rows, "role_transition")) == 9
    assert values(rows, "path_ei", groups=5) == [pytest.approx(micro)]
    assert len(values(rows, "null_ce", grouping="greedy")) == 2
    assert values(rows, "null_determinism") == [pytest.approx(det)] * 2  # degrees are kept


def test_compute_email_rows():
    g = nx.Graph(STARS)
    g.add_edge("x", "x")  # self-loop: dropped
    g.add_edge("y", "z")  # outside the giant component: dropped
    rows = ce.compute_email(g, SMALL)
    assert values(rows, "nodes", network="email") == [10]
    assert values(rows, "grouping_merged", grouping="structural_equivalence") == [6]
    ei = values(rows, "grouping_ei", grouping="structural_equivalence")[0]
    micro = values(rows, "micro_ei", network="email")[0]
    # Not necessarily positive: merging removes degeneracy but also shrinks the network.
    assert values(rows, "grouping_ce", grouping="structural_equivalence")[0] == pytest.approx(
        ei - micro
    )
    assert values(rows, "equivalence_class", rank=1) == [3]
    assert values(rows, "class_size_count", size=3) == [2]
    assert len(values(rows, "null_ce", grouping="louvain")) == 2
    assert len(values(rows, "null_largest_class")) == 2


def test_run_links_all_three_datasets(monkeypatch):
    client = MagicMock()
    blobs = {"snap/e.gz": GZ, "snap/m.gz": META_GZ, "snap/enron.gz": STARS_GZ}
    from pipelines.common.provenance import sha256_bytes

    meta_rows = {
        ce.WORM_SLUG: {"id": 1, "storage_path": "datasets/snap/e.gz"},
        ce.WORM_META_SLUG: {"id": 2, "storage_path": "datasets/snap/m.gz"},
        ce.EMAIL_SLUG: {"id": 3, "storage_path": "datasets/snap/enron.gz"},
    }
    for row in meta_rows.values():
        row["sha256"] = sha256_bytes(blobs[row["storage_path"].split("/", 1)[1]])
    eq = client.table.return_value.select.return_value.eq
    eq.side_effect = lambda col, slug: MagicMock(**{"execute.return_value.data": [meta_rows[slug]]})
    client.storage.from_.return_value.download.side_effect = lambda p: blobs[p]
    started, written = {}, []
    monkeypatch.setattr(ce, "sync_experiments", lambda c: {ce.EXPERIMENT_SLUG: 7})
    monkeypatch.setattr(ce, "git_sha", lambda allow_dirty: "abc")
    monkeypatch.setattr(ce, "start_run", lambda c, **kw: started.update(kw) or 11)
    monkeypatch.setattr(ce, "CONFIG", SMALL)
    monkeypatch.setattr(ce, "write_results", lambda c, run_id, rows: written.extend(rows))
    monkeypatch.setattr(ce, "tracked", lambda c, run_id: MagicMock())
    assert ce.run(client) == 11
    assert started["datasets"] == {r["id"]: r["sha256"] for r in meta_rows.values()}
    assert {r["dims"]["network"] for r in written if "network" in r.get("dims", {})} == {
        "worm",
        "email",
    }


def test_run_rejects_a_corrupted_download(monkeypatch):
    client = MagicMock()
    eq = client.table.return_value.select.return_value.eq
    eq.return_value.execute.return_value.data = [
        {"id": 1, "sha256": "0" * 64, "storage_path": "datasets/snap/e.gz"}
    ]
    client.storage.from_.return_value.download.return_value = GZ
    monkeypatch.setattr(ce, "sync_experiments", lambda c: {ce.EXPERIMENT_SLUG: 7})
    monkeypatch.setattr(ce, "git_sha", lambda allow_dirty: "abc")
    monkeypatch.setattr(ce, "start_run", lambda c, **kw: 11)
    monkeypatch.setattr(ce, "tracked", lambda c, run_id: MagicMock())
    with pytest.raises(RuntimeError, match="sha256"):
        ce.run(client)
