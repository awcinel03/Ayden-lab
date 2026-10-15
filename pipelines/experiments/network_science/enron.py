"""Enron email network: degree distribution, small world, robustness and communities.

Run: python -m pipelines.experiments.network_science.enron [--allow-dirty]

A faithful port of the original Colab notebook (Chapter 1 of "Conspiracies,
Connectomes, and Cabernet"). Loads SNAP email-Enron from Storage (verifying its
sha256), computes every metric the page plots, writes long-format results plus
two small Parquet artifacts (the laid-out top-30 community network), and is
skipped by fingerprint when nothing changed.
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import random
import statistics
from collections import Counter
from typing import Any

import networkx as nx
import pandas as pd

from pipelines.common import network_metrics as nm
from pipelines.common.client import get_client
from pipelines.common.provenance import git_sha, sha256_bytes
from pipelines.common.runs import record_artifact, start_run, tracked, write_results
from pipelines.common.storage import upload
from pipelines.common.sync import sync_experiments
from supabase import Client

log = logging.getLogger(__name__)

EXPERIMENT_SLUG = "enron-email-network"
DATASET_SLUG = "snap-email-enron"
ARTIFACT_BUCKET = "artifacts"
CONFIG: dict[str, Any] = {
    "version": 1,
    "dataset": DATASET_SLUG,
    "graph": "undirected nx.Graph from the edge list, string node IDs in file order",
    "power_law_fit": "np.bincount of degrees; keep count > 1; linregress on log10-log10",
    "small_world": {"on": "giant component", "path_sample_size": 500, "path_seed": 42},
    "hub_experiment": {"nodes_removed": 5, "random_seed": 42, "on": "full graph"},
    "robustness": {
        "on": "giant component",
        "steps": 100,
        "targeted": "highest current degree",
        "random_seeds": list(range(42, 52)),  # 42 reproduces the notebook; 43-51 add trials
    },
    "louvain": {"package": "python-louvain", "on": "giant component", "seed": 42},
    "louvain_seed_sweep": list(range(20)),  # diagnostic spread only; never used to pick a seed
    "community_network": {"top_k": 30, "layout": "spring_layout", "k": 15, "seed": 42},
    "cross_community_pairs": 10,
}


def load_graph(gz_data: bytes) -> nx.Graph:
    """Parse the gzipped edge list exactly as the notebook's ``nx.read_edgelist`` call did."""
    lines = gzip.decompress(gz_data).decode().splitlines()
    return nx.parse_edgelist(lines, comments="#", create_using=nx.Graph())


def compute(
    g: nx.Graph, config: dict[str, Any] = CONFIG
) -> tuple[list[dict[str, Any]], nm.CommunityNetwork]:
    """Every metric for the page, as result rows, plus the laid-out community network."""
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: float, unit: str | None = None, **dims: Any) -> None:
        row: dict[str, Any] = {"metric": metric, "value": value, "unit": unit}
        if dims:
            row["dims"] = dims
        rows.append(row)

    n = g.number_of_nodes()
    giant = nm.giant_component(g)
    counts = nm.degree_counts(g)
    add("nodes", n, "nodes")
    add("edges", g.number_of_edges(), "edges")
    add("components", nx.number_connected_components(g))
    add("giant_nodes", giant.number_of_nodes(), "nodes")
    add("avg_degree", 2 * g.number_of_edges() / n)
    add("max_degree", len(counts) - 1)
    for degree, count in enumerate(counts.tolist()):
        if count:
            add("degree_count", count, "nodes", degree=degree)

    fit = nm.power_law_fit(counts)
    add("power_law_slope", fit.slope)
    add("power_law_intercept", fit.intercept)
    add("power_law_r2", fit.r2)
    add("power_law_points", fit.n_points)

    sw_cfg = config["small_world"]
    sw = nm.small_world(giant, sample_size=sw_cfg["path_sample_size"], seed=sw_cfg["path_seed"])
    add("avg_clustering", sw.avg_clustering)
    add("avg_path_sampled", sw.avg_path, "hops")
    add("random_clustering", sw.random_clustering)
    add("random_path", sw.random_path, "hops")
    add("clustering_ratio", sw.clustering_ratio, "x")

    hub_cfg = config["hub_experiment"]
    k = hub_cfg["nodes_removed"]
    hubs = [node for node, _ in nm.top_degree_nodes(g, k)]
    randoms = random.Random(hub_cfg["random_seed"]).sample(list(g.nodes()), k)
    hub_loss, random_loss = nm.removal_loss(g, hubs), nm.removal_loss(g, randoms)
    add("hub_removal_loss", hub_loss, "nodes")
    add("random_removal_loss", random_loss, "nodes")
    add("damage_ratio", hub_loss / random_loss if random_loss else float("nan"), "x")

    rob = config["robustness"]
    targeted = nm.robustness_curve(giant, strategy="targeted", steps=rob["steps"], seed=0)
    for step, frac in enumerate(targeted):
        add("giant_fraction", frac, "fraction", strategy="targeted", step=step)
    trials = nm.robustness_trials(giant, steps=rob["steps"], seeds=rob["random_seeds"])
    for seed, curve in trials.items():
        for step, frac in enumerate(curve):
            add("giant_fraction", frac, "fraction", strategy="random", seed=seed, step=step)

    lv = config["louvain"]
    comms = nm.louvain(giant, seed=lv["seed"])
    add("communities", len(comms.sizes))
    add("modularity", comms.modularity)
    add("within_fraction", comms.within_fraction, "fraction")
    add("median_community_size", statistics.median(comms.sizes), "nodes")
    for rank, size in enumerate(comms.sizes, start=1):
        add("community_size", size, "nodes", rank=rank)
    sizes = Counter(comms.partition.values())
    for rank, (a, b, w) in enumerate(
        nm.cross_community_pairs(giant, comms.partition, top_n=config["cross_community_pairs"]),
        start=1,
    ):
        add(
            "cross_community_edges",
            w,
            "edges",
            rank=rank,
            a=a,
            b=b,
            a_size=sizes[a],
            b_size=sizes[b],
        )

    for seed in config["louvain_seed_sweep"]:
        s = nm.louvain(giant, seed=seed)
        add("sweep_communities", len(s.sizes), seed=seed)
        add("sweep_modularity", s.modularity, seed=seed)
        add("sweep_within_fraction", s.within_fraction, "fraction", seed=seed)

    net_cfg = config["community_network"]
    network = nm.community_network(
        giant,
        comms.partition,
        top_k=net_cfg["top_k"],
        layout_seed=net_cfg["seed"],
        layout_k=net_cfg["k"],
    )
    return rows, network


def _parquet(records: list[dict[str, Any]]) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(records).to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


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
        results, network = compute(load_graph(data), CONFIG)
        write_results(client, run_id, results)
        for name, records in (
            ("community_nodes", network.nodes),
            ("community_edges", network.edges),
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
