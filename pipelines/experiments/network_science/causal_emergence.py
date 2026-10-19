"""Causal emergence in the worm, in email and in Arsenal's passing: do macro-nodes win?

Run: python -m pipelines.experiments.network_science.causal_emergence [--allow-dirty]

Treats each network as a random walk (row i spreads evenly over i's
out-neighbours) and measures its effective information (Klein & Hoel 2020) at
the micro scale and after merging nodes into macro-nodes:

- C. elegans frontal connectome (directed): the three degree roles
  (integrator, relay, broadcaster), a greedy search restricted to merges
  within a role, and an unrestricted greedy search;
- Enron email (undirected giant component): structurally equivalent
  addresses (identical contacts) and Louvain communities;
- Arsenal 2003/04 (weighted passing chains of each match's starting XI, as in
  the football entry): positional roles and lines, and a greedy search.

Every grouping is compared with random groupings of the same sizes, and each
network with degree-preserving rewired versions of itself (for Arsenal, the
football entry's configuration null, which keeps passes made and received). Reads the SNAP
datasets already in Storage (verifying sha256s) and is skipped by fingerprint
when nothing changed.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pipelines.common import markov as mk
from pipelines.common import network_metrics as nm
from pipelines.common.client import get_client
from pipelines.common.provenance import git_sha, sha256_bytes
from pipelines.common.runs import start_run, tracked, write_results
from pipelines.common.sync import sync_experiments
from pipelines.experiments.network_science import c_elegans, enron
from pipelines.experiments.network_science import football as fb
from supabase import Client

log = logging.getLogger(__name__)

EXPERIMENT_SLUG = "causal-emergence-worm-vs-email"
WORM_SLUG = c_elegans.EDGES_SLUG
WORM_META_SLUG = c_elegans.META_SLUG
EMAIL_SLUG = enron.DATASET_SLUG
ROLES = ("integrator", "relay", "broadcaster")
CONFIG: dict[str, Any] = {
    "version": 1,
    "datasets": [WORM_SLUG, WORM_META_SLUG, EMAIL_SLUG, *fb.CONFIG["datasets"]],
    "chain": "random walk: row i spreads evenly over i's out-neighbours; edges unweighted",
    "ei": "Klein & Hoel 2020: H(<W_i>) - <H(W_i)> over nodes with outputs; uniform intervention",
    "effectiveness": "EI / log2(nodes with outputs)",
    "macro_nodes": "mu|pi with uniform weights (the intervention distribution)",
    "seed": 42,
    "worm": {
        "graph": "directed, as the C. elegans entry loads it",
        "roles": c_elegans.CONFIG["roles"],
        "sinks": "neurons with no outgoing synapse are left out of the intervention",
        "random_groupings": 1000,
        "null": "nx.directed_edge_swap, nswap = 4 x edges, seed = draw",
        "null_draws": 20,
    },
    "email": {
        "graph": "undirected giant component, as the Enron entry loads it, self-loops dropped",
        "structural_equivalence": "addresses with identical neighbour sets",
        "louvain": {"package": "python-louvain", "seed": 42},
        "random_groupings": {"structural_equivalence": 20, "louvain": 100},
        "null": "nx.double_edge_swap, nswap = 2 x edges, seed = draw",
        "null_draws": 10,
    },
    "football": {
        "team": fb.CONFIG["team"],
        "window": fb.CONFIG["window"],
        "chain": "row i = player i's completed passes to each teammate, as shares",
        "groupings": "roles (side x line) and lines from starting positions, as the football "
        "entry; greedy search with any merge",
        "random_groupings": 100,
        "null": "markov.configuration_counts (passes made and received kept, no self-passes)",
        "null_draws": 500,
        "null_greedy_draws": 100,
        "summary": "season values are medians over matches",
    },
}

Row = dict[str, Any]
Adder = Callable[..., None]


def adder(rows: list[Row]) -> Adder:
    """``add(metric, value, unit=None, **dims)`` appending long-format rows to ``rows``."""

    def add(metric: str, value: float, unit: str | None = None, **dims: Any) -> None:
        row: Row = {"metric": metric, "value": float(value), "unit": unit}
        if dims:
            row["dims"] = dims
        rows.append(row)

    return add


def random_walk(g: nx.Graph, nodes: Sequence[Hashable]) -> sp.csr_matrix:
    """Row-stochastic sparse matrix: each node's walk spreads evenly over its out-neighbours."""
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in g.edges()]
    if not g.is_directed():
        edges += [(v, u) for u, v in edges if u != v]
    rows, cols = zip(*edges, strict=True) if edges else ((), ())
    a = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes),) * 2)
    out = np.asarray(a.sum(axis=1)).ravel()
    return sp.csr_matrix(sp.diags(np.divide(1, out, out=np.zeros_like(out), where=out > 0)) @ a)


def structural_classes(g: nx.Graph, nodes: Sequence[Hashable]) -> list[list[int]]:
    """Groups of node indices with identical neighbour sets, in first-seen order."""
    classes: dict[frozenset[Hashable], list[int]] = defaultdict(list)
    for i, v in enumerate(nodes):
        classes[frozenset(g.neighbors(v))].append(i)
    return list(classes.values())


def micro_rows(add: Adder, w: mk.Matrix, network: str, prefix: str = "micro", **dims: Any) -> float:
    """EI, effectiveness, determinism and degeneracy of ``w``; returns its EI."""
    ei = mk.effective_information(w)
    det, deg = mk.determinism_degeneracy(w)
    add(f"{prefix}_ei", ei, "bits", network=network, **dims)
    add(f"{prefix}_effectiveness", mk.effectiveness(w), network=network, **dims)
    add(f"{prefix}_determinism", det, "bits", network=network, **dims)
    add(f"{prefix}_degeneracy", deg, "bits", network=network, **dims)
    return ei


def grouping_rows(
    add: Adder,
    w: mk.Matrix,
    partition: Sequence[Sequence[int]],
    micro_ei: float,
    *,
    network: str,
    grouping: str,
    draws: int,
    rng: random.Random,
) -> float:
    """Macro EI, CE and effectiveness of ``partition``, plus random same-size groupings."""
    weights = np.ones(w.shape[0])
    macro = mk.macro_matrix(w, partition, weights)
    ei = mk.effective_information(macro)
    dims = {"network": network, "grouping": grouping}
    add("grouping_ei", ei, "bits", **dims)
    add("grouping_ce", ei - micro_ei, "bits", **dims)
    add("grouping_effectiveness", mk.effectiveness(macro), **dims)
    add("grouping_groups", len(partition), "macro-nodes", **dims)
    add("grouping_merged", sum(len(p) for p in partition if len(p) > 1), "nodes", **dims)
    rand = np.array(
        [
            mk.effective_information(
                mk.macro_matrix(w, mk.random_partition_like(partition, w.shape[0], rng), weights)
            )
            for _ in range(draws)
        ]
    )
    add("random_ei_mean", rand.mean(), "bits", **dims)
    add("random_ei_lo", np.percentile(rand, 2.5), "bits", **dims)
    add("random_ei_hi", np.percentile(rand, 97.5), "bits", **dims)
    add("random_beaten", float(np.mean(rand < ei)), "fraction", **dims)
    return ei


def compute_worm(
    g: nx.DiGraph, names: dict[str, str], config: dict[str, Any] = CONFIG
) -> list[Row]:
    """Rows for the C. elegans connectome: roles, greedy macro-nodes and rewired nulls."""
    cfg, rows = config["worm"], []
    add = adder(rows)
    rng = random.Random(config["seed"])
    nodes = list(g.nodes())
    w = random_walk(g, nodes).toarray()
    roles = [
        c_elegans.classify_role(
            g.in_degree(v),
            g.out_degree(v),
            high=cfg["roles"]["integrator_above"],
            low=cfg["roles"]["broadcaster_below"],
        )
        for v in nodes
    ]
    role_part = [[i for i, r in enumerate(roles) if r == role] for role in ROLES]
    role_part = [p for p in role_part if p]
    add("nodes", len(nodes), "neurons", network="worm")
    add("edges", g.number_of_edges(), "synapses", network="worm")
    micro = micro_rows(add, w, "worm")

    draws = cfg["random_groupings"]
    grouping_rows(add, w, role_part, micro, network="worm", grouping="roles", draws=draws, rng=rng)
    macro = mk.macro_matrix(w, role_part, np.ones(len(nodes)))
    present = [r for r in ROLES if r in roles]
    for a, src in enumerate(present):
        for b, dst in enumerate(present):
            add("role_transition", macro[a, b], "probability", network="worm", source=src,
                target=dst)  # fmt: skip

    path = mk.greedy_path(w, np.ones(len(nodes)), within=roles)
    for groups, ei in path:
        m = mk.macro_matrix(w, groups, np.ones(len(nodes)))
        add("path_ei", ei, "bits", network="worm", groups=len(groups))
        add("path_effectiveness", mk.effectiveness(m), network="worm", groups=len(groups))
    best_within = max(path, key=lambda p: (p[1], len(p[0])))[0]
    grouping_rows(
        add, w, best_within, micro, network="worm", grouping="greedy_within_roles", draws=draws,
        rng=rng,
    )  # fmt: skip
    greedy, _ = mk.greedy_partition(w, np.ones(len(nodes)))
    grouping_rows(add, w, greedy, micro, network="worm", grouping="greedy", draws=draws, rng=rng)
    merged = sorted((p for p in greedy if len(p) > 1), key=len, reverse=True)
    for rank, members in enumerate(merged, start=1):
        counts = Counter(roles[i] for i in members)
        add(
            "macro_node",
            len(members),
            "neurons",
            network="worm",
            rank=rank,
            members=", ".join(sorted(names.get(nodes[i], nodes[i]) for i in members)),
            **{role: counts.get(role, 0) for role in ROLES},
        )

    for draw in range(cfg["null_draws"]):
        h = g.copy()
        nx.directed_edge_swap(h, nswap=4 * h.number_of_edges(), max_tries=10**7, seed=draw)
        wn = random_walk(h, nodes).toarray()
        e0 = micro_rows(add, wn, "worm", prefix="null", draw=draw)
        _, best = mk.greedy_partition(wn, np.ones(len(nodes)))
        add("null_ce", best - e0, "bits", network="worm", grouping="greedy", draw=draw)
        roles_ei = mk.effective_information(mk.macro_matrix(wn, role_part, np.ones(len(nodes))))
        add("null_ce", roles_ei - e0, "bits", network="worm", grouping="roles", draw=draw)
    return rows


def compute_email(g: nx.Graph, config: dict[str, Any] = CONFIG) -> list[Row]:
    """Rows for the Enron giant component: structural equivalence, Louvain and rewired nulls."""
    cfg, rows = config["email"], []
    add = adder(rows)
    rng = random.Random(config["seed"])
    g = nx.Graph(g)
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    giant = nm.giant_component(g)
    nodes = list(giant.nodes())
    w = random_walk(giant, nodes)
    add("nodes", len(nodes), "addresses", network="email")
    add("edges", giant.number_of_edges(), "pairs", network="email")
    micro = micro_rows(add, w, "email")

    classes = structural_classes(giant, nodes)
    draws = cfg["random_groupings"]
    grouping_rows(
        add, w, classes, micro, network="email", grouping="structural_equivalence",
        draws=draws["structural_equivalence"], rng=rng,
    )  # fmt: skip
    for size, count in sorted(Counter(len(c) for c in classes if len(c) > 1).items()):
        add("class_size_count", count, "classes", network="email", size=size)
    biggest = sorted((c for c in classes if len(c) > 1), key=len, reverse=True)[:10]
    for rank, members in enumerate(biggest, start=1):
        contacts = giant.degree(nodes[members[0]])
        add("equivalence_class", len(members), "addresses", network="email", rank=rank,
            contacts=contacts)  # fmt: skip

    louvain = nm.louvain(giant, seed=cfg["louvain"]["seed"])
    communities = mk.labels_to_partition([louvain.partition[v] for v in nodes])
    grouping_rows(
        add, w, communities, micro, network="email", grouping="louvain",
        draws=draws["louvain"], rng=rng,
    )  # fmt: skip

    for draw in range(cfg["null_draws"]):
        h = giant.copy()
        nx.double_edge_swap(h, nswap=2 * h.number_of_edges(), max_tries=10**8, seed=draw)
        wn = random_walk(h, nodes)
        e0 = micro_rows(add, wn, "email", prefix="null", draw=draw)
        null_classes = structural_classes(h, nodes)
        eq = mk.effective_information(mk.macro_matrix(wn, null_classes, np.ones(len(nodes))))
        add("null_ce", eq - e0, "bits", network="email", grouping="structural_equivalence",
            draw=draw)  # fmt: skip
        add("null_merged", sum(len(c) for c in null_classes if len(c) > 1), "nodes",
            network="email", grouping="structural_equivalence", draw=draw)  # fmt: skip
        add("null_largest_class", max(len(c) for c in null_classes), "addresses",
            network="email", draw=draw)  # fmt: skip
        lv = nm.louvain(h, seed=cfg["louvain"]["seed"])
        part = mk.labels_to_partition([lv.partition[v] for v in nodes])
        lv_ei = mk.effective_information(mk.macro_matrix(wn, part, np.ones(len(nodes))))
        add("null_ce", lv_ei - e0, "bits", network="email", grouping="louvain", draw=draw)
    return rows


def compute_football(
    matches: list[dict[str, Any]],
    events: dict[int, list[dict[str, Any]]],
    config: dict[str, Any] = CONFIG,
) -> list[Row]:
    """Rows for Arsenal's starting-XI passing chains: per match and season medians."""
    cfg, rows = config["football"], []
    add = adder(rows)
    per_match: dict[str, list[float]] = defaultdict(list)
    for m in sorted(matches, key=lambda m: m["match_date"]):
        match = m["match_id"]
        _, positions, counts, _ = fb.window_counts(events[match], cfg["team"])
        w = mk.transition_matrix(counts)
        ones = np.ones(len(w))
        ei = mk.effective_information(w)
        det, deg = mk.determinism_degeneracy(w)
        rng = random.Random(config["seed"] + match)
        null = []
        for _ in range(cfg["null_draws"]):
            wn = mk.transition_matrix(mk.configuration_counts(counts, rng))
            null.append((mk.effective_information(wn), *mk.determinism_degeneracy(wn)))
        null_arr = np.array(null)
        null_greedy = [
            mk.greedy_partition(wn, ones)[1] - mk.effective_information(wn)
            for wn in (
                mk.transition_matrix(mk.configuration_counts(counts, rng))
                for _ in range(cfg["null_greedy_draws"])
            )
        ]
        values = {
            "passes": float(counts.sum()),
            "ei": ei,
            "effectiveness": mk.effectiveness(w),
            "determinism": det,
            "degeneracy": deg,
            "null_ei": null_arr[:, 0].mean(),
            "null_determinism": null_arr[:, 1].mean(),
            "null_degeneracy": null_arr[:, 2].mean(),
            "degeneracy_pct": float(np.mean(null_arr[:, 2] < deg)),
            "ei_pct": float(np.mean(null_arr[:, 0] < ei)),
            "null_ce_greedy": float(np.mean(null_greedy)),
        }
        partitions = {
            "roles": mk.labels_to_partition([fb.role_of(p) for p in positions]),
            "lines": mk.labels_to_partition([fb.line_of(p) for p in positions]),
            "greedy": mk.greedy_partition(w, ones)[0],
        }
        for name, part in partitions.items():
            macro = mk.macro_matrix(w, part, ones)
            values[f"{name}_ei"] = mk.effective_information(macro)
            values[f"{name}_ce"] = values[f"{name}_ei"] - ei
            values[f"{name}_effectiveness"] = mk.effectiveness(macro)
            values[f"{name}_groups"] = float(len(part))
        rand = [
            mk.effective_information(
                mk.macro_matrix(w, mk.random_partition_like(partitions["roles"], len(w), rng), ones)
            )
            for _ in range(cfg["random_groupings"])
        ]
        values["roles_random_ei"] = float(np.mean(rand))
        values["roles_beaten"] = float(np.mean(np.array(rand) < values["roles_ei"]))
        for key, v in values.items():
            add(f"match_{key}", v, network="football", match=match)
            per_match[key].append(v)

    med = {k: float(np.median(v)) for k, v in per_match.items()}
    add("nodes", 11, "players", network="football")
    add("matches", len(per_match["ei"]), "matches", network="football")
    add("median_passes", med["passes"], "passes", network="football")
    for key in ("ei", "effectiveness", "determinism", "degeneracy"):
        add(f"micro_{key}", med[key], "bits" if key != "effectiveness" else None,
            network="football")  # fmt: skip
        if key != "effectiveness":
            add(f"null_median_{key}", med[f"null_{key}"], "bits", network="football")
    for name in ("roles", "lines", "greedy"):
        dims = {"network": "football", "grouping": name}
        add("grouping_ei", med[f"{name}_ei"], "bits", **dims)
        add("grouping_ce", med[f"{name}_ce"], "bits", **dims)
        add("grouping_effectiveness", med[f"{name}_effectiveness"], **dims)
        add("grouping_groups", med[f"{name}_groups"], "macro-nodes", **dims)
    roles = {"network": "football", "grouping": "roles"}
    add("random_ei_mean", med["roles_random_ei"], "bits", **roles)
    add("random_beaten", med["roles_beaten"], "fraction", **roles)
    return rows


def _dataset(client: Client, slug: str) -> dict[str, Any]:
    rows = (
        client.table("datasets").select("id, sha256, storage_path").eq("slug", slug).execute().data
    )
    if not rows:
        raise RuntimeError(f"Dataset {slug} missing; run pipelines.ingest.snap first")
    return rows[0]


def _download(client: Client, dataset: dict[str, Any]) -> bytes:
    bucket, path = dataset["storage_path"].split("/", 1)
    data = client.storage.from_(bucket).download(path)
    if sha256_bytes(data) != dataset["sha256"]:
        raise RuntimeError(f"{path} does not match its recorded sha256")
    return data


def run(client: Client, *, allow_dirty: bool = False) -> int | None:
    """Execute the experiment; returns the run id, or ``None`` if skipped."""
    experiment_id = sync_experiments(client)[EXPERIMENT_SLUG]
    datasets = {slug: _dataset(client, slug) for slug in CONFIG["datasets"]}
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
        worm = c_elegans.load_graph(_download(client, datasets[WORM_SLUG]))
        meta: pd.DataFrame = c_elegans.load_meta(_download(client, datasets[WORM_META_SLUG]))
        names = dict(zip(meta["node_id"].astype(str), meta["name"], strict=True))
        log.info("worm: roles, greedy search and %d nulls", CONFIG["worm"]["null_draws"])
        results = compute_worm(worm, names, CONFIG)
        log.info("email: macro-nodes and %d nulls", CONFIG["email"]["null_draws"])
        email = enron.load_graph(_download(client, datasets[EMAIL_SLUG]))
        results += compute_email(email, CONFIG)
        log.info("football: %d nulls per match", CONFIG["football"]["null_draws"])
        matches, events, _ = fb.load_season(client, datasets)
        results += compute_football(matches, events, CONFIG)
        write_results(client, run_id, results)
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
