"""Enron experiment on a small graph, with Supabase mocked."""

import copy
import gzip
from unittest.mock import MagicMock

import networkx as nx

from pipelines.experiments.network_science import enron

EDGES = "# comment\n" + "".join(f"{u}\t{v}\n{v}\t{u}\n" for u, v in nx.barbell_graph(6, 0).edges())
GZ = gzip.compress((EDGES + "20\t21\n").encode())

SMALL = copy.deepcopy(enron.CONFIG)
SMALL["small_world"]["path_sample_size"] = 5
SMALL["hub_experiment"]["nodes_removed"] = 2
SMALL["robustness"].update(steps=4, random_seeds=[42, 43])
SMALL["louvain_seed_sweep"] = [0, 1]
SMALL["community_network"]["top_k"] = 2


def test_load_graph_matches_read_edgelist():
    g = enron.load_graph(GZ)
    assert g.number_of_nodes() == 14 and g.number_of_edges() == 32  # 31 barbell + 1 isolated pair
    assert all(isinstance(n, str) for n in g)
    assert list(g)[:2] == ["0", "1"]  # file order


def test_compute_on_small_graph():
    rows, net = enron.compute(enron.load_graph(GZ), SMALL)
    s = {r["metric"]: r["value"] for r in rows if "dims" not in r}
    assert s["nodes"] == 14 and s["components"] == 2 and s["giant_nodes"] == 12
    assert s["max_degree"] == 6
    assert s["communities"] == 2 and s["within_fraction"] == 30 / 31
    targeted = [
        r for r in rows if r["metric"] == "giant_fraction" and r["dims"]["strategy"] == "targeted"
    ]
    assert [r["dims"]["step"] for r in targeted] == [0, 1, 2, 3] and targeted[0]["value"] == 1
    assert {r["dims"]["seed"] for r in rows if r.get("dims", {}).get("strategy") == "random"} == {
        42,
        43,
    }
    assert len([r for r in rows if r["metric"] == "sweep_modularity"]) == 2
    assert sorted(n["size"] for n in net.nodes) == [6, 6]
    assert [e["weight"] for e in net.edges] == [1]
    keys = [(r["metric"], tuple(sorted(r.get("dims", {}).items()))) for r in rows]
    assert len(keys) == len(set(keys))  # unique (metric, dims), as results requires


def test_run_writes_results_and_artifacts(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(enron, "sync_experiments", lambda c: {enron.EXPERIMENT_SLUG: 7})
    monkeypatch.setattr(enron, "git_sha", lambda allow_dirty: "abc")
    monkeypatch.setattr(enron, "start_run", lambda c, **kw: 11)
    monkeypatch.setattr(enron, "CONFIG", SMALL)
    client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
        {"id": 3, "sha256": enron.sha256_bytes(GZ), "storage_path": "datasets/snap/x.gz"}
    ]
    client.storage.from_.return_value.download.return_value = GZ
    written, recorded = [], []
    monkeypatch.setattr(enron, "write_results", lambda c, run_id, rows: written.extend(rows))
    monkeypatch.setattr(enron, "record_artifact", lambda c, run_id, **kw: recorded.append(kw))
    monkeypatch.setattr(enron, "tracked", lambda c, run_id: MagicMock())
    assert enron.run(client) == 11
    assert written
    assert [a["storage_path"] for a in recorded] == [
        "enron-email-network/11/community_nodes.parquet",
        "enron-email-network/11/community_edges.parquet",
    ]
