"""C. elegans frontal connectome: degree, neuron roles, small world, robustness, communities.

Run: python -m pipelines.experiments.network_science.c_elegans [--allow-dirty]

A faithful port of the original Colab notebook (Chapter 2 of "Conspiracies,
Connectomes, and Cabernet"). Loads the SNAP edge list and neuron metadata from
Storage (verifying both sha256s), computes every metric the page plots, writes
long-format results plus two small Parquet artifacts (neurons with anatomical
positions, synapses), and is skipped by fingerprint when nothing changed.
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import statistics
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
from supabase import Client

log = logging.getLogger(__name__)

EXPERIMENT_SLUG = "c-elegans-connectome"
EDGES_SLUG = "snap-c-elegans-frontal"
META_SLUG = "snap-c-elegans-frontal-meta"
ARTIFACT_BUCKET = "artifacts"
CONFIG: dict[str, Any] = {
    "version": 1,
    "datasets": [EDGES_SLUG, META_SLUG],
    "graph": "directed nx.DiGraph from the edge list, string node IDs in file order",
    "degree": "total = in + out; histogram 20 bins",
    "power_law_fit": "np.bincount of total degrees; keep count > 1; linregress on log10-log10",
    "roles": {"ratio": "in / (in + out)", "integrator_above": 0.7, "broadcaster_below": 0.3},
    "hub_min_total_degree": 21,  # notebook labels neurons with total degree > 20
    "small_world": {
        "on": "weakly connected giant",
        "clustering": "directed",
        "baseline_p": "m / (n(n-1))",
        "path_sample_size": 50,
        "path_seed": 42,
    },
    "robustness": {
        "on": "full graph (weakly connected)",
        "targeted_steps": 40,  # notebook cell 6: by total degree
        "in_out_steps": 30,  # notebook cell 3: by in-degree and by out-degree
        "random_steps": 40,
        "random_seeds": list(range(42, 52)),  # 42 reproduces the notebook; 43-51 add trials
    },
    "signal_flow": "edge toward centre if target is closer than source to mean position",
    "louvain": {"package": "python-louvain", "on": "undirected projection", "seed": 42},
    "louvain_seed_sweep": list(range(20)),  # diagnostic spread only; never used to pick a seed
    "hub_communities": 8,
}


def load_graph(gz_data: bytes) -> nx.DiGraph:
    """Parse the gzipped edge list exactly as the notebook's ``nx.read_edgelist`` call did."""
    lines = gzip.decompress(gz_data).decode().splitlines()
    return nx.parse_edgelist(lines, comments="#", create_using=nx.DiGraph())


def load_meta(gz_data: bytes) -> pd.DataFrame:
    """Neuron metadata (``node_id``, ``name``, ``posx``, ``posy``) from the gzipped CSV."""
    return pd.read_csv(io.BytesIO(gzip.decompress(gz_data)))


def classify_role(in_degree: int, out_degree: int, *, high: float, low: float) -> str | None:
    """Neuron role by ``in / (in + out)``: above ``high`` integrator, below ``low``
    broadcaster, otherwise relay; ``None`` for an isolated neuron."""
    total = in_degree + out_degree
    if total == 0:
        return None
    ratio = in_degree / total
    if ratio > high:
        return "integrator"
    if ratio < low:
        return "broadcaster"
    return "relay"


def toward_centre(
    g: nx.DiGraph, pos: dict[str, tuple[float, float]]
) -> dict[tuple[str, str], bool]:
    """For each edge with both ends positioned: is the target closer than the source to the
    mean position of the graph's positioned nodes?"""
    placed = [n for n in g.nodes() if n in pos]
    cx = np.mean([pos[n][0] for n in placed])
    cy = np.mean([pos[n][1] for n in placed])

    def dist(n: str) -> float:
        return float(np.sqrt((pos[n][0] - cx) ** 2 + (pos[n][1] - cy) ** 2))

    return {(u, v): dist(v) < dist(u) for u, v in g.edges() if u in pos and v in pos}


def compute(
    g: nx.DiGraph, meta: pd.DataFrame, config: dict[str, Any] = CONFIG
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Result rows plus neuron and synapse records for the page's artifacts."""
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: float, unit: str | None = None, **dims: Any) -> None:
        row: dict[str, Any] = {"metric": metric, "value": value, "unit": unit}
        if dims:
            row["dims"] = dims
        rows.append(row)

    names = dict(zip(meta["node_id"].astype(str), meta["name"], strict=True))
    pos = {
        str(int(r.node_id)): (float(r.posx), float(r.posy))
        for r in meta.itertuples()
        if str(int(r.node_id)) in g
    }

    def name(n: str) -> str:
        return names.get(n, f"Node {n}")

    ins, outs = dict(g.in_degree()), dict(g.out_degree())
    totals = [ins[n] + outs[n] for n in g.nodes()]
    add("nodes", g.number_of_nodes(), "neurons")
    add("edges", g.number_of_edges(), "synapses")
    add("weak_components", nx.number_weakly_connected_components(g))
    add("mean_degree", float(np.mean(totals)))
    add("median_degree", float(np.median(totals)))
    add("max_degree", max(totals))
    counts = nm.degree_counts(g)
    for degree, count in enumerate(counts.tolist()):
        if count:
            add("degree_count", count, "neurons", degree=degree)
    fit = nm.power_law_fit(counts)
    add("power_law_slope", fit.slope)
    add("power_law_intercept", fit.intercept)
    add("power_law_r2", fit.r2)
    add("power_law_points", fit.n_points)

    roles_cfg = config["roles"]
    roles = {
        n: classify_role(
            ins[n], outs[n], high=roles_cfg["integrator_above"], low=roles_cfg["broadcaster_below"]
        )
        for n in g.nodes()
    }
    for role, count in Counter(roles.values()).items():
        add("role_count", count, "neurons", role=role)

    sw_cfg = config["small_world"]
    giant = nm.giant_component(g)
    sw = nm.small_world(giant, sample_size=sw_cfg["path_sample_size"], seed=sw_cfg["path_seed"])
    add("avg_clustering", sw.avg_clustering)
    add("avg_path_sampled", sw.avg_path, "hops")
    add("random_clustering", sw.random_clustering)
    add("random_path", sw.random_path, "hops")
    add("clustering_ratio", sw.clustering_ratio, "x")

    # Fractions are of g's node count, which equals the original weak giant (g is connected).
    rob = config["robustness"]
    curves = {
        ("targeted", None): nm.robustness_curve(
            g, strategy="targeted", steps=rob["targeted_steps"], seed=0
        ),
        ("in_degree", None): nm.robustness_curve(
            g, strategy="in_degree", steps=rob["in_out_steps"], seed=0
        ),
        ("out_degree", None): nm.robustness_curve(
            g, strategy="out_degree", steps=rob["in_out_steps"], seed=0
        ),
    }
    trials = nm.robustness_trials(g, steps=rob["random_steps"], seeds=rob["random_seeds"])
    curves |= {("random", seed): curve for seed, curve in trials.items()}
    for (strategy, seed), curve in curves.items():
        dims = {"strategy": strategy} | ({"seed": seed} if seed is not None else {})
        for step, frac in enumerate(curve):
            add("giant_fraction", frac, "fraction", **dims, step=step)

    flow = toward_centre(g, pos)
    add("signal_toward_centre", sum(flow.values()), "synapses")
    add("signal_away_from_centre", len(flow) - sum(flow.values()), "synapses")

    lv = config["louvain"]
    comms = nm.louvain(g, seed=lv["seed"])
    sizes = Counter(comms.partition.values())
    add("undirected_edges", g.to_undirected().number_of_edges(), "edges")
    add("communities", len(comms.sizes))
    add("modularity", comms.modularity)
    add("within_fraction", comms.within_fraction, "fraction")
    add("median_community_size", statistics.median(comms.sizes), "neurons")
    for rank, size in enumerate(comms.sizes, start=1):
        add("community_size", size, "neurons", rank=rank)
    for rank, (n, degree) in enumerate(nm.top_degree_nodes(g, config["hub_communities"]), start=1):
        c = comms.partition[n]
        add(
            "hub_community",
            degree,
            "degree",
            rank=rank,
            neuron=name(n),
            community=c,
            community_size=sizes[c],
        )
    for seed in config["louvain_seed_sweep"]:
        s = nm.louvain(g, seed=seed)
        add("sweep_communities", len(s.sizes), seed=seed)
        add("sweep_modularity", s.modularity, seed=seed)
        add("sweep_within_fraction", s.within_fraction, "fraction", seed=seed)

    hub_min = config["hub_min_total_degree"]
    neurons = [
        {
            "neuron": name(n),
            "x": pos[n][0] if n in pos else None,
            "y": pos[n][1] if n in pos else None,
            "in_degree": ins[n],
            "out_degree": outs[n],
            "total_degree": ins[n] + outs[n],
            "role": roles[n],
            "community": comms.partition[n],
            "hub": ins[n] + outs[n] >= hub_min,
        }
        for n in g.nodes()
    ]
    synapses = [
        {"source": name(u), "target": name(v), "toward_centre": flow.get((u, v))}
        for u, v in g.edges()
    ]
    return rows, neurons, synapses


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
    for slug in (EDGES_SLUG, META_SLUG):
        rows = (
            client.table("datasets")
            .select("id, sha256, storage_path")
            .eq("slug", slug)
            .execute()
            .data
        )
        if not rows:
            raise RuntimeError(f"Dataset {slug} missing; run pipelines.ingest.snap first")
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
        g = load_graph(_download(client, datasets[EDGES_SLUG]))
        meta = load_meta(_download(client, datasets[META_SLUG]))
        results, neurons, synapses = compute(g, meta, CONFIG)
        write_results(client, run_id, results)
        for name, records in (("neurons", neurons), ("synapses", synapses)):
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
