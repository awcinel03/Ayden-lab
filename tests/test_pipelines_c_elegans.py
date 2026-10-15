"""C. elegans experiment on a toy connectome, with Supabase mocked."""

import copy
import gzip
from unittest.mock import MagicMock

import networkx as nx
import pytest

from pipelines.experiments.network_science import c_elegans as ce

# 0 broadcasts to 1-3; 1-3 feed 4 (integrator); a 1<->2 loop gives 1 and 2 a relay profile.
EDGES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4), (1, 2), (2, 1), (4, 0)]
GZ = gzip.compress(("# comment\n" + "".join(f"{u}\t{v}\n" for u, v in EDGES)).encode())
META_CSV = "node_id,name,posx,posy\n" + "".join(
    f"{i},N{i},{x},{y}\n" for i, (x, y) in enumerate([(0, 0), (1, 1), (1, -1), (-1, 1), (2, 0)])
)
META_GZ = gzip.compress(META_CSV.encode())

SMALL = copy.deepcopy(ce.CONFIG)
SMALL["small_world"]["path_sample_size"] = 3
SMALL["robustness"].update(targeted_steps=3, in_out_steps=2, random_steps=3, random_seeds=[42, 43])
SMALL["louvain_seed_sweep"] = [0]
SMALL["hub_communities"] = 2
SMALL["hub_min_total_degree"] = 4
# Every toy neuron has total degree 4, so the log-log fit has one point and is NaN.
one_point_fit = pytest.mark.filterwarnings("ignore:One or more sample arguments is too small")


@pytest.mark.parametrize(
    ("ins", "outs", "role"),
    [(8, 2, "integrator"), (7, 3, "relay"), (3, 7, "relay"), (2, 8, "broadcaster"), (0, 0, None)],
)
def test_classify_role_thresholds_are_strict(ins, outs, role):
    assert ce.classify_role(ins, outs, high=0.7, low=0.3) == role


def test_toward_centre_uses_mean_position():
    g = nx.DiGraph([("a", "b"), ("b", "a")])
    pos = {"a": (0.0, 0.0), "b": (4.0, 0.0), "c": (2.0, 0.0)}  # c is not in g: ignored
    assert ce.toward_centre(g, pos) == {("a", "b"): False, ("b", "a"): False}  # equidistant
    g.add_edge("a", "c")  # now c counts: centre is x = 2
    flow = ce.toward_centre(g, pos)
    assert flow[("a", "c")] is True and flow[("a", "b")] is False


@one_point_fit
def test_compute_joins_names_and_builds_artifacts():
    rows, neurons, synapses = ce.compute(ce.load_graph(GZ), ce.load_meta(META_GZ), SMALL)
    s = {r["metric"]: r["value"] for r in rows if "dims" not in r}
    assert s["nodes"] == 5 and s["edges"] == 9 and s["weak_components"] == 1
    assert s["max_degree"] == 4 and s["undirected_edges"] == 8
    by = {n["neuron"]: n for n in neurons}
    assert set(by) == {"N0", "N1", "N2", "N3", "N4"}
    assert by["N0"]["role"] == "broadcaster" and by["N4"]["role"] == "integrator"
    assert by["N1"]["role"] == "relay" and (by["N4"]["x"], by["N4"]["y"]) == (2.0, 0.0)
    assert by["N0"]["hub"] and not by["N3"]["hub"]
    assert {(e["source"], e["target"]) for e in synapses} == {(f"N{u}", f"N{v}") for u, v in EDGES}
    steps = {}
    for r in rows:
        if r["metric"] == "giant_fraction":
            key = (r["dims"]["strategy"], r["dims"].get("seed"))
            steps[key] = steps.get(key, 0) + 1
    assert steps == {
        ("targeted", None): 3,
        ("in_degree", None): 2,
        ("out_degree", None): 2,
        ("random", 42): 3,
        ("random", 43): 3,
    }
    hubs = [r for r in rows if r["metric"] == "hub_community"]
    assert [h["dims"]["neuron"] for h in hubs] == ["N0", "N1"]  # degree ties keep file order
    keys = [(r["metric"], tuple(sorted(r.get("dims", {}).items()))) for r in rows]
    assert len(keys) == len(set(keys))


@one_point_fit
def test_run_links_both_datasets_and_writes_artifacts(monkeypatch):
    client = MagicMock()
    blobs = {"datasets/snap/e.gz": GZ, "datasets/snap/m.gz": META_GZ}
    meta_rows = {
        ce.EDGES_SLUG: {
            "id": 1,
            "sha256": ce.sha256_bytes(GZ),
            "storage_path": "datasets/snap/e.gz",
        },
        ce.META_SLUG: {
            "id": 2,
            "sha256": ce.sha256_bytes(META_GZ),
            "storage_path": "datasets/snap/m.gz",
        },
    }
    eq = client.table.return_value.select.return_value.eq
    eq.side_effect = lambda col, slug: MagicMock(**{"execute.return_value.data": [meta_rows[slug]]})
    client.storage.from_.return_value.download.side_effect = lambda p: blobs[f"datasets/{p}"]
    started = {}
    monkeypatch.setattr(ce, "sync_experiments", lambda c: {ce.EXPERIMENT_SLUG: 7})
    monkeypatch.setattr(ce, "git_sha", lambda allow_dirty: "abc")
    monkeypatch.setattr(ce, "start_run", lambda c, **kw: started.update(kw) or 11)
    monkeypatch.setattr(ce, "CONFIG", SMALL)
    monkeypatch.setattr(ce, "write_results", lambda c, run_id, rows: None)
    recorded = []
    monkeypatch.setattr(ce, "record_artifact", lambda c, run_id, **kw: recorded.append(kw))
    monkeypatch.setattr(ce, "tracked", lambda c, run_id: MagicMock())
    assert ce.run(client) == 11
    assert started["datasets"] == {
        1: meta_rows[ce.EDGES_SLUG]["sha256"],
        2: meta_rows[ce.META_SLUG]["sha256"],
    }
    assert [a["storage_path"] for a in recorded] == [
        "c-elegans-connectome/11/neurons.parquet",
        "c-elegans-connectome/11/synapses.parquet",
    ]
