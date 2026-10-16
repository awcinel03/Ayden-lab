"""Wine flavor co-occurrence network: frequencies, degree, small world, robustness, communities.

Run: python -m pipelines.experiments.network_science.wine_flavor [--allow-dirty]

A faithful port of the original Colab notebook (Chapter 3 of "Conspiracies,
Connectomes, and Cabernet"). Loads the two derived aggregates written by
``pipelines.ingest.wine_flavor`` (term review counts and the co-occurrence edge
list; no review text) from Storage, verifying both sha256s. Nodes come from the
terms table (every term found in a review, so a term whose pairs all fall below
the weight threshold is an isolated node, as in the notebook) and edges from the
edge table. Writes long-format results plus three small Parquet artifacts
(flavors with a layout, the edge list, all-pairs shortest paths) and is skipped
by fingerprint when nothing changed.
"""

from __future__ import annotations

import argparse
import io
import logging
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from pipelines.common import network_metrics as nm
from pipelines.common.client import get_client
from pipelines.common.provenance import git_sha, sha256_bytes
from pipelines.common.runs import record_artifact, start_run, tracked, write_results
from pipelines.common.storage import upload
from pipelines.common.sync import sync_experiments
from pipelines.ingest.wine_flavor import EDGES_SLUG, TERMS_SLUG
from supabase import Client

log = logging.getLogger(__name__)

EXPERIMENT_SLUG = "wine-flavor-network"
ARTIFACT_BUCKET = "artifacts"
CONFIG: dict[str, Any] = {
    "version": 1,
    "datasets": [TERMS_SLUG, EDGES_SLUG],
    "graph": (
        "undirected nx.Graph; nodes = terms table in row order (first appearance in the "
        "reviews, lexicon order within a review), attribute reviews; edges = edge table, "
        "attribute weight (reviews containing both terms, >= 10)"
    ),
    "degree": "unweighted; histogram 20 bins",
    # The notebook kept count > 0, which includes degree 0 (the isolated brett), so its
    # log10 fit was NaN. The lab's shared rule (count > 1) is used for every entry.
    "power_law_fit": "np.bincount of degrees; keep count > 1; linregress on log10-log10",
    "small_world": {
        "on": "full graph (as the notebook)",
        "clustering": "unweighted",
        "baseline_p": "m / (n(n-1)/2)",
        "path_sample_size": 50,
        "path_seed": 42,
    },
    "robustness": {
        "on": "full graph; fraction of all nodes",
        "steps": 50,  # notebook cell 6; state recorded before each removal
        "random_seeds": list(range(42, 52)),  # 42 as cell 6; 43-51 add trials
    },
    "louvain": {"package": "python-louvain", "weighted": True, "seed": 42},
    "louvain_seed_sweep": list(range(20)),  # diagnostic spread only; never used to pick a seed
    "top_pairs": 15,
    "network_view": {
        "edge_weight_above": 3000,  # notebook cell 5
        "layout": "spring_layout on the thresholded graph",
        "k": 3,
        "seed": 42,
    },
    "shortest_paths": "unweighted hops; one BFS path per unordered pair (nx.shortest_path)",
}


def build_graph(terms: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    """Graph with a node per term (``reviews`` attribute) and weighted co-occurrence edges."""
    g = nx.Graph()
    for term, reviews in zip(terms["term"], terms["reviews"], strict=True):
        g.add_node(str(term), reviews=int(reviews))
    for a, b, w in zip(edges["term_a"], edges["term_b"], edges["weight"], strict=True):
        if a not in g or b not in g:
            raise ValueError(f"edge {a}-{b} has a term missing from the terms table")
        g.add_edge(str(a), str(b), weight=int(w))
    return g


def network_view(g: nx.Graph, *, threshold: int, k: float, seed: int) -> dict[str, tuple]:
    """Notebook cell 5's layout: ``spring_layout(k, seed)`` of all nodes plus only the
    edges heavier than ``threshold``."""
    view = nx.Graph()
    view.add_nodes_from(g.nodes())
    view.add_edges_from((a, b) for a, b, w in g.edges(data="weight") if w > threshold)
    return nx.spring_layout(view, k=k, seed=seed)


def all_shortest_paths(g: nx.Graph) -> list[dict[str, Any]]:
    """One unweighted shortest path per unordered pair of connected nodes.

    Pairs are keyed with ``source < target`` (alphabetical); ``path`` is the
    node sequence from source to target joined by ``" > "``; ``hops`` is its length.
    """
    out = []
    nodes = sorted(g.nodes())
    for i, s in enumerate(nodes):
        paths = nx.single_source_shortest_path(g, s)
        for t in nodes[i + 1 :]:
            if t in paths:
                out.append(
                    {
                        "source": s,
                        "target": t,
                        "hops": len(paths[t]) - 1,
                        "path": " > ".join(paths[t]),
                    }
                )
    return out


def compute(
    terms: pd.DataFrame, edges: pd.DataFrame, config: dict[str, Any] = CONFIG
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Result rows plus flavor, co-occurrence and shortest-path records for the artifacts."""
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: float, unit: str | None = None, **dims: Any) -> None:
        row: dict[str, Any] = {"metric": metric, "value": value, "unit": unit}
        if dims:
            row["dims"] = dims
        rows.append(row)

    g = build_graph(terms, edges)
    n, m = g.number_of_nodes(), g.number_of_edges()
    degrees = dict(g.degree())
    add("nodes", n, "terms")
    add("edges", m, "pairs")
    add("avg_degree", 2 * m / n)
    add("density", nx.density(g))
    add("max_degree", max(degrees.values()))
    add("min_degree", min(degrees.values()))
    add("isolated_nodes", nx.number_of_isolates(g), "terms")
    add("components", nx.number_connected_components(g))
    add("min_edge_weight", int(edges["weight"].min()) if len(edges) else float("nan"), "reviews")
    for term, reviews in nx.get_node_attributes(g, "reviews").items():
        add("term_reviews", reviews, "reviews", term=term, degree=degrees[term])
    top = sorted(g.edges(data="weight"), key=lambda e: e[2], reverse=True)[: config["top_pairs"]]
    for rank, (a, b, w) in enumerate(top, start=1):
        add("top_pair", w, "reviews", rank=rank, term_a=a, term_b=b)

    counts = nm.degree_counts(g)
    for degree, count in enumerate(counts.tolist()):
        if count:
            add("degree_count", count, "terms", degree=degree)
    fit = nm.power_law_fit(counts)
    add("power_law_slope", fit.slope)
    add("power_law_intercept", fit.intercept)
    add("power_law_r2", fit.r2)
    add("power_law_points", fit.n_points)

    sw_cfg = config["small_world"]
    sw = nm.small_world(g, sample_size=sw_cfg["path_sample_size"], seed=sw_cfg["path_seed"])
    add("avg_clustering", sw.avg_clustering)
    add("avg_path_sampled", sw.avg_path, "hops")
    add("random_clustering", sw.random_clustering)
    add("random_path", sw.random_path, "hops")
    add("clustering_ratio", sw.clustering_ratio, "x")

    rob = config["robustness"]
    curves = {
        ("targeted", None): nm.robustness_curve(g, strategy="targeted", steps=rob["steps"], seed=0)
    }
    trials = nm.robustness_trials(g, steps=rob["steps"], seeds=rob["random_seeds"])
    curves |= {("random", seed): curve for seed, curve in trials.items()}
    for (strategy, seed), curve in curves.items():
        dims = {"strategy": strategy} | ({"seed": seed} if seed is not None else {})
        for step, frac in enumerate(curve):
            add("giant_fraction", frac, "fraction", **dims, step=step)

    comms = nm.louvain(g, seed=config["louvain"]["seed"])
    add("communities", len(comms.sizes))
    add("modularity", comms.modularity)
    add("within_fraction", comms.within_fraction, "fraction")
    for rank, size in enumerate(comms.sizes, start=1):
        add("community_size", size, "terms", rank=rank)
    for seed in config["louvain_seed_sweep"]:
        s = nm.louvain(g, seed=seed)
        add("sweep_communities", len(s.sizes), seed=seed)
        add("sweep_modularity", s.modularity, seed=seed)
        add("sweep_within_fraction", s.within_fraction, "fraction", seed=seed)
        add("sweep_largest_community", s.sizes[0], "terms", seed=seed)
        add("sweep_second_community", s.sizes[1] if len(s.sizes) > 1 else 0, "terms", seed=seed)

    # Communities are relabelled by size rank (1 = largest) so labels are stable to read.
    by_size = Counter(comms.partition.values()).most_common()
    rank_of = {c: r for r, (c, _) in enumerate(by_size, start=1)}
    view_cfg = config["network_view"]
    pos = network_view(
        g, threshold=view_cfg["edge_weight_above"], k=view_cfg["k"], seed=view_cfg["seed"]
    )
    add("view_edge_threshold", view_cfg["edge_weight_above"], "reviews")
    flavors = [
        {
            "term": t,
            "reviews": g.nodes[t]["reviews"],
            "degree": degrees[t],
            "community": rank_of[comms.partition[t]],
            "x": float(pos[t][0]),
            "y": float(pos[t][1]),
        }
        for t in g.nodes()
    ]
    cooccurrence = [{"term_a": a, "term_b": b, "weight": w} for a, b, w in g.edges(data="weight")]
    paths = all_shortest_paths(g)
    hops = Counter(p["hops"] for p in paths)
    for h, c in sorted(hops.items()):
        add("path_hops_count", c, "pairs", hops=h)
    add("connected_pairs", len(paths), "pairs")
    add(
        "avg_path_all_pairs",
        float(np.mean([p["hops"] for p in paths])) if paths else float("nan"),
        "hops",
    )
    return rows, flavors, cooccurrence, paths


def _parquet(records: list[dict[str, Any]]) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(records).to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


def _download(client: Client, dataset: dict[str, Any]) -> bytes:
    bucket, path = dataset["storage_path"].split("/", 1)
    data = client.storage.from_(bucket).download(path)
    if sha256_bytes(data) != dataset["sha256"]:
        raise RuntimeError(f"{path} does not match its recorded sha256")
    return data


def run(client: Client, *, allow_dirty: bool = False) -> int | None:
    """Execute the experiment; returns the run id, or ``None`` if skipped."""
    experiment_id = sync_experiments(client)[EXPERIMENT_SLUG]
    datasets = {}
    for slug in (TERMS_SLUG, EDGES_SLUG):
        rows = (
            client.table("datasets")
            .select("id, sha256, storage_path")
            .eq("slug", slug)
            .execute()
            .data
        )
        if not rows:
            raise RuntimeError(f"Dataset {slug} missing; run pipelines.ingest.wine_flavor first")
        datasets[slug] = rows[0]
    run_id = start_run(
        client,
        experiment_id=experiment_id,
        config=CONFIG,
        git_sha=git_sha(allow_dirty=allow_dirty),
        datasets={d["id"]: d["sha256"] for d in datasets.values()},
    )
    if run_id is None:
        return None
    with tracked(client, run_id):
        terms = pd.read_parquet(io.BytesIO(_download(client, datasets[TERMS_SLUG])))
        edges = pd.read_parquet(io.BytesIO(_download(client, datasets[EDGES_SLUG])))
        results, flavors, cooccurrence, paths = compute(terms, edges, CONFIG)
        write_results(client, run_id, results)
        for name, records in (
            ("flavors", flavors),
            ("cooccurrence", cooccurrence),
            ("shortest_paths", paths),
        ):
            blob = _parquet(records)
            storage_path = f"{EXPERIMENT_SLUG}/{run_id}/{name}.parquet"
            upload(client, ARTIFACT_BUCKET, storage_path, blob, "application/vnd.apache.parquet")
            record_artifact(
                client,
                run_id,
                kind="table",
                storage_path=storage_path,
                data=blob,
                content_type="application/vnd.apache.parquet",
            )
    return run_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-dirty", action="store_true", help="record a run from uncommitted code"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_id = run(get_client(), allow_dirty=args.allow_dirty)
    log.info(
        "skipped: identical run already succeeded" if run_id is None else f"run {run_id} succeeded"
    )


if __name__ == "__main__":
    main()
