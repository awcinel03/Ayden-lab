import math

import networkx as nx
import numpy as np
import pytest

from pipelines.common import network_metrics as nm


def two_cliques() -> nx.Graph:
    """Two K5s (nodes 0-4 and 5-9) joined by the single edge 4-5."""
    g = nx.disjoint_union(nx.complete_graph(5), nx.complete_graph(5))
    g.add_edge(4, 5)
    return g


def test_giant_component_keeps_node_order():
    g = nx.Graph([(3, 1), (1, 2), (9, 8)])
    assert list(nm.giant_component(g).nodes()) == [3, 1, 2]


def test_giant_component_directed_is_weak():
    g = nx.DiGraph([("a", "b"), ("c", "b"), ("d", "e")])
    assert set(nm.giant_component(g)) == {"a", "b", "c"}
    assert nm.giant_size(g) == 3


def test_degree_counts():
    star = nx.star_graph(4)
    assert nm.degree_counts(star).tolist() == [0, 4, 0, 0, 1]
    d = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    assert nm.degree_counts(d, "in").tolist() == [1, 1, 1]
    assert nm.degree_counts(d, "out").tolist() == [1, 1, 1]
    with pytest.raises(ValueError):
        nm.degree_counts(star, "in")


def test_top_degree_nodes_ties_keep_node_order():
    g = nx.Graph([(0, 1), (2, 3), (2, 4)])
    assert nm.top_degree_nodes(g, 3) == [(2, 2), (0, 1), (1, 1)]


def test_power_law_fit_recovers_exact_slope_and_drops_singletons():
    counts = np.zeros(20, dtype=int)
    counts[[1, 2, 4, 8]] = [4096, 1024, 256, 64]
    counts[16] = 1  # singleton: excluded from the fit
    fit = nm.power_law_fit(counts)
    assert fit.slope == pytest.approx(-2)
    assert fit.r2 == pytest.approx(1)
    assert fit.n_points == 4


def test_power_law_fit_rejects_degree_zero():
    with pytest.raises(ValueError):
        nm.power_law_fit(np.array([2, 5, 3]))


def test_small_world_complete_graph():
    sw = nm.small_world(nx.complete_graph(5), sample_size=None, seed=0)
    assert sw.avg_clustering == 1
    assert sw.avg_path == pytest.approx(4 / 5)  # self-distance 0 is included
    assert sw.random_clustering == 1
    assert sw.random_path == pytest.approx(1)
    assert sw.clustering_ratio == 1


def test_small_world_path_graph_exact_and_sampled():
    g = nx.path_graph(4)
    sw = nm.small_world(g, sample_size=None, seed=0)
    assert sw.avg_path == pytest.approx(20 / 16)
    assert sw.avg_clustering == 0
    p = 3 / 6
    assert sw.random_path == pytest.approx(math.log(4) / math.log(4 * p))
    a = nm.small_world(g, sample_size=2, seed=7)
    b = nm.small_world(g, sample_size=2, seed=7)
    assert a == b and a.path_sources == 2


def test_targeted_attack_destroys_star_random_barely_does():
    star = nx.star_graph(5)  # centre 0, six nodes
    assert nm.robustness_curve(star, strategy="targeted", steps=3, seed=0) == [1, 1 / 6, 1 / 6]
    rand = nm.robustness_curve(star, strategy="random", steps=2, seed=1)
    assert rand[0] == 1
    assert rand == nm.robustness_curve(star, strategy="random", steps=2, seed=1)


def test_robustness_curve_stops_on_empty_graph():
    curve = nm.robustness_curve(nx.path_graph(2), strategy="targeted", steps=5, seed=0)
    assert curve == [1, 0.5, 0]


def test_robustness_directed_out_degree():
    g = nx.DiGraph([("a", "h"), ("b", "h"), ("h", "c")])  # h: in-degree 2, out-degree 1
    assert nm.robustness_curve(g, strategy="in_degree", steps=2, seed=0) == [1, 0.25]
    assert nm.robustness_curve(g, strategy="out_degree", steps=2, seed=0) == [1, 0.75]


def test_robustness_trials_one_curve_per_seed():
    trials = nm.robustness_trials(nx.star_graph(5), steps=2, seeds=[1, 2])
    assert list(trials) == [1, 2] and all(len(c) == 2 for c in trials.values())


def test_removal_loss():
    assert nm.removal_loss(nx.star_graph(5), [0]) == 5
    assert nm.removal_loss(nx.star_graph(5), [3]) == 1


def test_louvain_two_cliques():
    c = nm.louvain(two_cliques(), seed=42)
    assert c.sizes == [5, 5]
    assert c.within_fraction == pytest.approx(20 / 21)
    assert c.modularity == pytest.approx(19 / 42)
    assert nm.louvain(two_cliques(), seed=42).partition == c.partition


def test_louvain_directed_uses_undirected_projection():
    c = nm.louvain(nx.DiGraph(two_cliques()), seed=42)
    assert c.sizes == [5, 5]


def test_community_network_and_pairs():
    g = two_cliques()
    partition = nm.louvain(g, seed=42).partition
    a, b = partition[4], partition[5]
    assert nm.cross_community_pairs(g, partition, top_n=10) == [(a, b, 1)]
    net = nm.community_network(g, partition, top_k=1, layout_seed=42, layout_k=15)
    assert len(net.nodes) == 1 and net.edges == []
    net = nm.community_network(g, partition, top_k=2, layout_seed=42, layout_k=15)
    assert sorted(n["size"] for n in net.nodes) == [5, 5]
    assert [e["weight"] for e in net.edges] == [1]
    assert net == nm.community_network(g, partition, top_k=2, layout_seed=42, layout_k=15)


def test_small_world_directed_baseline_uses_directed_density():
    g = nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)])  # n=3, m=4
    sw = nm.small_world(g, sample_size=None, seed=0)
    p = 4 / (3 * 2)
    assert sw.random_clustering == pytest.approx(p)
    assert sw.random_path == pytest.approx(math.log(3) / math.log(3 * p))
    assert sw.avg_clustering == pytest.approx(nx.average_clustering(g))
    # BFS follows direction: distance sums from 0, 1, 2 are 2, 3 (1->2->0), 3 (2->0->1)
    assert sw.avg_path == pytest.approx(8 / 9)
