"""Basic statistics and degree distribution of the email-Eu-core graph.

Run: python -m pipelines.experiments.network_science.graph_stats [--allow-dirty]

Loads the dataset from Storage (verifying its sha256), computes the statistics,
and writes one run with long-format results. Re-running on the same commit with
the same data and config is skipped by fingerprint.
"""

from __future__ import annotations

import argparse
import gzip
import logging
from collections import Counter
from typing import Any

import networkx as nx

from pipelines.common.client import get_client
from pipelines.common.provenance import git_sha, sha256_bytes
from pipelines.common.runs import start_run, tracked, write_results
from pipelines.common.sync import sync_experiments
from supabase import Client

log = logging.getLogger(__name__)

EXPERIMENT_SLUG = "email-eu-core-stats"
DATASET_SLUG = "snap-email-eu-core"
CONFIG: dict[str, Any] = {
    "version": 1,
    "dataset": DATASET_SLUG,
    "graph": "directed; self-loops kept for degree, dropped for clustering",
    "clustering": "undirected simple projection",
}


def load_graph(gz_data: bytes) -> nx.DiGraph:
    """Parse a gzipped ``u v`` edge list into a directed graph."""
    g = nx.DiGraph()
    for line in gzip.decompress(gz_data).decode().splitlines():
        if line.strip() and not line.startswith("#"):
            u, v = line.split()[:2]
            g.add_edge(int(u), int(v))
    return g


def compute(g: nx.DiGraph) -> list[dict[str, Any]]:
    """Scalar statistics plus in/out/total degree distributions, as result rows."""
    n = g.number_of_nodes()
    wccs = list(nx.weakly_connected_components(g))
    sccs = list(nx.strongly_connected_components(g))
    simple = nx.Graph(g)
    simple.remove_edges_from(nx.selfloop_edges(simple))
    degrees = {"in": dict(g.in_degree()), "out": dict(g.out_degree()), "total": dict(g.degree())}

    rows: list[dict[str, Any]] = [
        {"metric": "nodes", "value": n, "unit": "nodes"},
        {"metric": "edges", "value": g.number_of_edges(), "unit": "edges"},
        {"metric": "self_loops", "value": nx.number_of_selfloops(g), "unit": "edges"},
        {"metric": "density", "value": nx.density(g)},
        {"metric": "reciprocity", "value": nx.reciprocity(g), "unit": "fraction"},
        {"metric": "mean_in_degree", "value": g.number_of_edges() / n},
        {"metric": "weak_components", "value": len(wccs)},
        {"metric": "strong_components", "value": len(sccs)},
        {"metric": "largest_wcc_fraction", "value": max(map(len, wccs)) / n, "unit": "fraction"},
        {"metric": "largest_scc_fraction", "value": max(map(len, sccs)) / n, "unit": "fraction"},
        {"metric": "average_clustering", "value": nx.average_clustering(simple)},
        {"metric": "transitivity", "value": nx.transitivity(simple)},
    ]
    for kind, by_node in degrees.items():
        rows.append({"metric": f"max_{kind}_degree", "value": max(by_node.values())})
        for degree, count in sorted(Counter(by_node.values()).items()):
            rows.append(
                {"metric": "degree_count", "value": count, "dims": {"kind": kind, "degree": degree}}
            )
    return rows


def run(client: Client, *, allow_dirty: bool = False) -> int | None:
    """Execute the experiment; returns the run id, or ``None`` if skipped."""
    experiment_id = sync_experiments(client)[EXPERIMENT_SLUG]
    rows = (
        client.table("datasets")
        .select("id, sha256, storage_path")
        .eq("slug", DATASET_SLUG)
        .execute()
        .data
    )
    if not rows:
        raise RuntimeError(f"Dataset {DATASET_SLUG} missing; run pipelines.ingest.snap first")
    dataset = rows[0]
    run_id = start_run(
        client,
        experiment_id=experiment_id,
        config=CONFIG,
        git_sha=git_sha(allow_dirty=allow_dirty),
        datasets={dataset["id"]: dataset["sha256"]},
    )
    if run_id is None:
        return None
    with tracked(client, run_id):
        bucket, path = dataset["storage_path"].split("/", 1)
        data = client.storage.from_(bucket).download(path)
        if sha256_bytes(data) != dataset["sha256"]:
            raise RuntimeError("Downloaded dataset does not match its recorded sha256")
        write_results(client, run_id, compute(load_graph(data)))
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
