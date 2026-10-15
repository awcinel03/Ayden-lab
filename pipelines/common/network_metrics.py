"""Network-science metrics shared by the network entries (Enron, C. elegans, wine).

Pure functions on NetworkX graphs, ported from the original notebooks without
changing the method. Every stochastic function takes an explicit ``seed`` and
draws from ``random.Random(seed)``, which yields the same sequence as the
notebooks' ``random.seed(seed)`` followed by module-level ``random`` calls.

Directed graphs: components are weakly connected, degree can be in/out/total,
clustering and BFS follow edge direction, and Louvain runs on the undirected
projection.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import community as community_louvain
import networkx as nx
import numpy as np
from scipy import stats

DegreeKind = Literal["total", "in", "out"]
Strategy = Literal["targeted", "random", "in_degree", "out_degree"]


def _components(g: nx.Graph) -> list[set[Hashable]]:
    """Connected components (weakly connected for directed graphs)."""
    if g.is_directed():
        return list(nx.weakly_connected_components(g))
    return list(nx.connected_components(g))


def giant_size(g: nx.Graph) -> int:
    """Number of nodes in the largest (weakly) connected component; 0 for an empty graph."""
    return max((len(c) for c in _components(g)), default=0)


def giant_component(g: nx.Graph) -> nx.Graph:
    """Copy of the largest (weakly) connected component, keeping ``g``'s node order."""
    comps = sorted(_components(g), key=len, reverse=True)
    return g.subgraph(comps[0]).copy()


def degree_counts(g: nx.Graph, kind: DegreeKind = "total") -> np.ndarray:
    """``np.bincount`` of node degrees: element ``k`` is the number of nodes of degree ``k``."""
    if kind == "total":
        view = g.degree
    elif g.is_directed():
        view = g.in_degree if kind == "in" else g.out_degree
    else:
        raise ValueError(f"{kind}-degree needs a directed graph")
    return np.bincount([d for _, d in view()])


def top_degree_nodes(g: nx.Graph, k: int) -> list[tuple[Hashable, int]]:
    """The ``k`` highest-degree ``(node, degree)`` pairs; ties keep node order (stable sort)."""
    return sorted(g.degree(), key=lambda x: x[1], reverse=True)[:k]


@dataclass(frozen=True)
class PowerLawFit:
    """Least-squares line through the log10-log10 degree distribution."""

    slope: float
    intercept: float
    r2: float
    n_points: int


def power_law_fit(counts: np.ndarray) -> PowerLawFit:
    """Fit ``log10(count) ~ log10(degree)`` over degrees whose count is greater than 1.

    Singleton-filtered, as in the notebooks: degrees seen exactly once are dropped.
    Raises ``ValueError`` if degree 0 would enter the fit (its log is undefined).
    """
    degrees = np.arange(len(counts))
    mask = counts > 1
    if mask[0]:
        raise ValueError("degree 0 has count > 1; log10(0) is undefined")
    fit = stats.linregress(np.log10(degrees[mask]), np.log10(counts[mask]))
    return PowerLawFit(
        slope=float(fit.slope),
        intercept=float(fit.intercept),
        r2=float(fit.rvalue**2),
        n_points=int(mask.sum()),
    )


@dataclass(frozen=True)
class SmallWorld:
    """Clustering and path length against an Erdős–Rényi baseline with the same n and m."""

    avg_clustering: float
    avg_path: float
    random_clustering: float
    random_path: float
    clustering_ratio: float
    path_sources: int


def small_world(g: nx.Graph, *, sample_size: int | None, seed: int) -> SmallWorld:
    """Small-world metrics of ``g`` (normally its giant component).

    Average path length is the mean of every BFS distance from ``sample_size``
    sources drawn with ``random.Random(seed).sample(list(g.nodes()), …)``,
    including each source's zero distance to itself (as the notebooks compute it).
    ``sample_size=None`` uses every node as a source. The baseline density is
    ``p = m / (n(n-1)/2)``, or ``m / (n(n-1))`` for a directed graph; random
    clustering is ``p`` and random path ``ln n / ln(np)``.
    """
    nodes = list(g.nodes())
    sources = nodes if sample_size is None else random.Random(seed).sample(nodes, sample_size)
    total = count = 0
    for node in sources:
        lengths = nx.single_source_shortest_path_length(g, node)
        total += sum(lengths.values())
        count += len(lengths)
    avg_clustering = nx.average_clustering(g)
    n, m = g.number_of_nodes(), g.number_of_edges()
    p = m / (n * (n - 1)) if g.is_directed() else m / (n * (n - 1) / 2)
    return SmallWorld(
        avg_clustering=avg_clustering,
        avg_path=total / count,
        random_clustering=p,
        random_path=math.log(n) / math.log(n * p),
        clustering_ratio=avg_clustering / p,
        path_sources=len(sources),
    )


def robustness_curve(g: nx.Graph, *, strategy: Strategy, steps: int, seed: int) -> list[float]:
    """Giant-component fraction before each of ``steps`` single-node removals.

    Element ``i`` is the giant size after ``i`` removals divided by ``g``'s node
    count, so the last element reflects ``steps - 1`` removals (notebook
    indexing). ``targeted`` removes the current highest-degree node (first in
    node order on ties); ``in_degree``/``out_degree`` likewise for directed
    graphs; ``random`` removes ``random.Random(seed).choice(list(nodes))``.
    """
    rng = random.Random(seed)
    h = g.copy()
    n = h.number_of_nodes()
    curve: list[float] = []
    for _ in range(steps):
        curve.append(giant_size(h) / n)
        if h.number_of_nodes() == 0:
            break
        if strategy == "random":
            target = rng.choice(list(h.nodes()))
        else:
            view = getattr(h, {"targeted": "degree"}.get(strategy, strategy))
            target = max(view(), key=lambda x: x[1])[0]
        h.remove_node(target)
    return curve


def robustness_trials(g: nx.Graph, *, steps: int, seeds: Sequence[int]) -> dict[int, list[float]]:
    """Random-removal curves, one per seed."""
    return {s: robustness_curve(g, strategy="random", steps=steps, seed=s) for s in seeds}


def removal_loss(g: nx.Graph, nodes: Iterable[Hashable]) -> int:
    """Nodes lost from the giant component when ``nodes`` are removed together."""
    h = g.copy()
    h.remove_nodes_from(nodes)
    return giant_size(g) - giant_size(h)


@dataclass(frozen=True)
class Communities:
    """A Louvain partition and its summary statistics."""

    partition: dict[Hashable, int]
    sizes: list[int]  # descending
    modularity: float
    within_fraction: float  # share of edges whose endpoints share a community


def louvain(g: nx.Graph, *, seed: int) -> Communities:
    """Louvain communities via ``python-louvain`` (undirected projection if directed)."""
    u = g.to_undirected() if g.is_directed() else g
    partition = community_louvain.best_partition(u, random_state=seed)
    within = sum(1 for a, b in u.edges() if partition[a] == partition[b])
    return Communities(
        partition=partition,
        sizes=sorted(Counter(partition.values()).values(), reverse=True),
        modularity=community_louvain.modularity(partition, u),
        within_fraction=within / u.number_of_edges(),
    )


def _community_graph(g: nx.Graph, partition: dict[Hashable, int]) -> nx.Graph:
    """One node per community (attribute ``size``); edge weight = edges between them."""
    sizes = Counter(partition.values())
    cg = nx.Graph()
    for comm, size in sizes.items():
        cg.add_node(comm, size=size)
    for a, b in g.edges():
        ca, cb = partition[a], partition[b]
        if ca != cb:
            if cg.has_edge(ca, cb):
                cg[ca][cb]["weight"] += 1
            else:
                cg.add_edge(ca, cb, weight=1)
    return cg


def cross_community_pairs(
    g: nx.Graph, partition: dict[Hashable, int], *, top_n: int
) -> list[tuple[int, int, int]]:
    """The ``top_n`` community pairs with the most edges between them, as ``(a, b, edges)``."""
    cg = _community_graph(g, partition)
    pairs = [(a, b, d["weight"]) for a, b, d in cg.edges(data=True)]
    return sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]


@dataclass(frozen=True)
class CommunityNetwork:
    """Laid-out graph of the largest communities, ready to plot."""

    nodes: list[dict[str, float]]  # community, size, x, y
    edges: list[dict[str, int]]  # source, target, weight


def community_network(
    g: nx.Graph,
    partition: dict[Hashable, int],
    *,
    top_k: int,
    layout_seed: int,
    layout_k: float,
) -> CommunityNetwork:
    """The ``top_k`` largest communities with a seeded ``spring_layout(k=layout_k)``."""
    cg = _community_graph(g, partition)
    sizes = Counter(partition.values())
    top = [c for c, _ in sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    sub = cg.subgraph(top).copy()
    pos = nx.spring_layout(sub, k=layout_k, seed=layout_seed)
    return CommunityNetwork(
        nodes=[
            {"community": c, "size": sizes[c], "x": float(pos[c][0]), "y": float(pos[c][1])}
            for c in sub.nodes()
        ],
        edges=[
            {"source": a, "target": b, "weight": d["weight"]} for a, b, d in sub.edges(data=True)
        ],
    )
