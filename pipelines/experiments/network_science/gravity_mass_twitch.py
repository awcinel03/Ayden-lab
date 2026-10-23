"""Follow-up D: the gravity-mass walk on a real social network.

Twitch's English-language mutual-friendship network (Rozemberczki, Allen & Sarkar, MUSAE,
J. Complex Networks 2021; collected from Twitch's public API in May 2018): 7,126 streamers,
35,324 friendships. Gravity is a streamer's (log) view count, with account age and degree as
comparisons, and the same view counts shuffled across streamers as the control.

For each field this repeats the earlier analyses on one network: the hitting-time curve and
best mass, trap climbs and condensation (follow-up A), mass sorting (B) and coarse-graining
into flow basins (C). Everything is sparse, since n is past the dense limit.

Run: python -m pipelines.experiments.network_science.gravity_mass_twitch   (about 15 min)
Downloads the two source files to data/raw/twitch/ (gitignored) if missing, checks their
sha256, and writes lab/gravity_mass/results/twitch_results.json with config and git SHA.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from lab.gravity_mass.model import (
    LAMS,
    RESULTS_PATH,
    deep_basins,
    flow_basins,
    hitting_and_occupancy,
    kernel_sparse,
    participation_ratio,
    standardize,
    stationary_closed_form,
    target_of,
)
from pipelines.common import markov as mk
from pipelines.common.provenance import git_sha, sha256_file
from pipelines.experiments.network_science.gravity_mass_sorting import MASSES, round_floats

OUT = RESULTS_PATH.parent / "twitch_results.json"
RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "twitch"
USER_AGENT = "aydens-lab-ingest/0.1 (+https://github.com/awcinel03/Ayden-lab)"
BASE = "https://raw.githubusercontent.com/benedekrozemberczki/MUSAE/master/input"
FILES = {
    "ENGB_edges.csv": (
        f"{BASE}/edges/ENGB_edges.csv",
        "5ea4d7f7267454e9523577eb2a9b4dcb7a22dd660a548240f01b815d08e0e3cc",
    ),
    "ENGB_target.csv": (
        f"{BASE}/target/ENGB_target.csv",
        "e9d51399878216d2dd662b0a152b602cccde7b5cbb8452cb91d9379f64114443",
    ),
}
SOURCE = {
    "name": "Twitch Social Networks, English (ENGB)",
    "source": "MUSAE repository (also on SNAP as twitch-social-networks)",
    "source_url": "https://github.com/benedekrozemberczki/MUSAE",
    "license": "GPL-3.0 (repository licence; SNAP states none)",
    "citation": (
        "B. Rozemberczki, C. Allen and R. Sarkar. Multi-scale attributed node embedding. "
        "Journal of Complex Networks 9(2), cnab014, 2021."
    ),
    "collected": "Twitch public API, May 2018",
}
CONFIG = {
    "version": 1,
    "lams": [float(x) for x in LAMS],
    "coarse_lams": [float(x) for x in np.arange(0, 8.01, 1.0)],
    "masses": MASSES,
    "n_shuffles": 5,
    "n_null": 5,
    "fields": ["log views", "log account age", "log degree", "shuffled log views"],
    "target": "top 10% of g",
}


def fetch() -> None:
    """Download the source files once, politely, and verify their checksums."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in FILES.items():
        path = RAW / name
        if not path.exists():
            r = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
            r.raise_for_status()
            path.write_bytes(r.content)
        if sha256_file(path) != sha:
            raise RuntimeError(f"{name}: sha256 mismatch; the source changed")


def load() -> tuple[sp.csr_matrix, pd.DataFrame]:
    """Symmetric adjacency and the per-streamer table, both indexed by new_id."""
    E = pd.read_csv(RAW / "ENGB_edges.csv")
    T = pd.read_csv(RAW / "ENGB_target.csv").sort_values("new_id").reset_index(drop=True)
    n = len(T)
    A = sp.coo_matrix((np.ones(len(E)), (E["from"], E["to"])), shape=(n, n)).tocsr()
    return ((A + A.T) > 0).astype(float).tocsr(), T


def barrier_climbs_sparse(A: sp.csr_matrix, g: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Floor-to-rim climb of every trap: the sparse twin of `barrier_prediction`.

    Edge depth is g_u + g_v. Edges are added deepest first with a union-find in which T is
    one super-node; a node's rim is the edge depth at which its cluster joins T. Its floor
    is its deepest edge, and its climb is floor minus rim. Returns climbs of the traps
    (local wells outside T).
    """
    n = len(g)
    U = sp.triu(A, k=1).tocoo()
    iu, ju = U.row, U.col
    e = g[iu] + g[ju]
    parent = np.arange(n + 1)
    SUPER = n
    parent[np.flatnonzero(T)] = SUPER

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rim = np.full(n, np.nan)
    members = {i: [i] for i in range(n) if not T[i]}
    for idx in np.argsort(-e):
        a, b = find(iu[idx]), find(ju[idx])
        if a == b:
            continue
        if b == SUPER:
            a, b = b, a
        if a == SUPER:
            for v in members.pop(b):
                rim[v] = e[idx]
            parent[b] = SUPER
        else:
            members[a].extend(members.pop(b))
            parent[b] = a
    nbr_max = np.maximum.reduceat(g[A.indices], A.indptr[:-1])
    trap = (g > nbr_max) & ~T
    return (nbr_max + g - rim)[trap]


def entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-300, 1.0)
    return -(p * np.log2(p)).sum(-1)


def field_stats(A: sp.csr_matrix, g: np.ndarray, seed: int) -> dict:
    T = target_of(g)
    Q = ~T
    nbr_max = np.maximum.reduceat(g[A.indices], A.indptr[:-1])
    # best mass and condensation (follow-up A), with flow basins and T absorbing
    lab = flow_basins(A, g, T)[Q]
    curve = []
    for lam in LAMS:
        h, o = hitting_and_occupancy(kernel_sparse(A, g, lam), T)
        y2 = participation_ratio(np.bincount(lab, weights=o))
        curve.append(dict(lam=float(lam), h_mean=float(h.mean()), y2=y2))
    hm = np.array([c["h_mean"] for c in curve])
    i_star = int(np.argmin(hm))
    climbs = barrier_climbs_sparse(A, g, T)
    # mass sorting (follow-up B)
    blab = deep_basins(A, T)
    k = int(blab.max() + 1)
    onehot = sp.csr_matrix((np.ones(T.sum()), (np.arange(T.sum()), blab)), shape=(T.sum(), k))
    gT = g[T]
    deepest = blab[np.argmax(gT)]
    D, per_mass = [], []
    for lam in MASSES:
        P = kernel_sparse(A, g, lam)
        lu = splu((sp.identity(int(Q.sum()), format="csc") - P[Q][:, Q]).tocsc())
        B = lu.solve(P[Q][:, T].toarray())  # absorption probabilities into each target node
        Dm = np.asarray(onehot.T @ B.T).T
        D.append(Dm)
        per_mass.append(
            dict(
                lam=lam,
                dest_depth=float((B @ gT).mean()),
                deepest_basin=float(Dm[:, deepest].mean()),
            )
        )
    D = np.array(D)
    dest_mi = float((entropy_bits(D.mean(0)) - entropy_bits(D).mean(0)).mean())
    pis = np.array([stationary_closed_form(A, g, lam) for lam in MASSES])
    settle_mi = float(entropy_bits(pis.mean(0)) - entropy_bits(pis).mean())
    h_m = np.log2(len(MASSES))
    # coarse-graining (follow-up C)
    part = mk.labels_to_partition(flow_basins(A, g, np.zeros(len(g), bool)))
    rng = random.Random(seed)
    nulls = [mk.random_partition_like(part, len(g), rng) for _ in range(CONFIG["n_null"])]
    coarse = []
    for lam in CONFIG["coarse_lams"]:
        P = kernel_sparse(A, g, lam)
        pi = stationary_closed_form(A, g, lam)
        null_ei = [mk.effective_information(mk.macro_matrix(P, q, pi)) for q in nulls]
        coarse.append(
            dict(
                lam=lam,
                micro_ei=mk.effective_information(P),
                macro_ei=mk.effective_information(mk.macro_matrix(P, part, pi)),
                null_ei=float(np.mean(null_ei)),
            )
        )
    return dict(
        trap_share=float(((g > nbr_max) & ~T).mean()),
        lam_star=float(LAMS[i_star]),
        censored=bool(i_star == len(LAMS) - 1),
        curve=curve,
        n_traps=int(len(climbs)),
        climbs=[round(float(c), 4) for c in climbs],
        n_basins=k,
        dest_sorting=dest_mi / h_m,
        settle_sorting=settle_mi / h_m,
        tv_0_8=float(0.5 * np.abs(D[-1] - D[0]).sum(-1).mean()),
        per_mass=per_mass,
        n_flow_basins=len(part),
        coarse=coarse,
    )


def make_fields(A: sp.csr_matrix, T: pd.DataFrame) -> list[tuple[str, int, np.ndarray]]:
    """(name, seed, g) for the real attributes, degree, and the shuffled-views control."""
    views = standardize(np.log(T["views"].to_numpy(float)))
    out = [
        ("log views", 0, views),
        ("log account age", 0, standardize(np.log(T["days"].to_numpy(float)))),
        ("log degree", 0, standardize(np.log(np.diff(A.indptr).astype(float)))),
    ]
    for s in range(CONFIG["n_shuffles"]):
        out.append(("shuffled log views", s, np.random.default_rng(s).permutation(views)))
    return out


def main():
    fetch()
    A, T = load()
    k = np.diff(A.indptr).astype(float)
    out = {
        "config": CONFIG,
        "git_sha": git_sha(),
        "source": SOURCE,
        "sha256": {name: sha for name, (_, sha) in FILES.items()},
        "network": {
            "n": int(A.shape[0]),
            "edges": int(A.nnz // 2),
            "mean_degree": float(k.mean()),
            "corr_log_views_log_degree": float(np.corrcoef(np.log(T["views"]), np.log(k))[0, 1]),
            "corr_log_age_log_degree": float(np.corrcoef(np.log(T["days"]), np.log(k))[0, 1]),
        },
        "fields": [],
    }
    for name, seed, g in make_fields(A, T):
        t0 = time.time()
        out["fields"].append(round_floats(dict(field=name, seed=seed, **field_stats(A, g, seed))))
        print(f"{name} (seed {seed}): {time.time() - t0:.0f}s", flush=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
