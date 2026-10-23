"""Follow-up A: is there a glass-like transition in mass?

Trap model (Bouchaud 1992; Baronchelli, Barrat & Pastor-Satorras 2009): escaping a well with
climb G takes about e^{lam G} steps. If climbs have an exponential tail with scale G0, the
average escape time diverges at lam_c = 1/G0, and above it the time before arrival condenses
into a few deepest wells. Order parameter: the participation ratio Y2 of the time spent in
each well basin. Below lam_c, Y2 -> 0 as n grows; above it, Y2 stays finite (trap-model
prediction: Y2 -> 1 - lam_c/lam).

For each field shape and network size this records the climbs of every trap (local wells
outside T), and over a lam grid the basin and node participation ratios and the mean and
median hitting time.

Run: python -m pipelines.experiments.network_science.gravity_mass_glass   (about 30 min)
Writes lab/gravity_mass/results/glass_results.json with its config and git SHA (refuses a
dirty tree). No Supabase run, as for the step 3 file.
"""

from __future__ import annotations

import json
import time

import networkx as nx
import numpy as np
import scipy.sparse as sp

from lab.gravity_mass.model import (
    RESULTS_PATH,
    hitting_and_occupancy,
    kernel_sparse,
    participation_ratio,
    standardize,
    target_of,
    well_basins,
)
from pipelines.common.provenance import git_sha
from pipelines.experiments.network_science.gravity_mass import barrier_prediction

OUT = RESULTS_PATH.parent / "glass_results.json"
CONFIG = {
    "version": 1,
    "graph": "ER, 3n edges (mean degree 6), giant component, graph seed = field seed",
    "sizes": [500, 1000, 2000, 4000],
    "seeds": {"500": 24, "1000": 24, "2000": 24, "4000": 12},
    "lams": [float(x) for x in np.arange(0, 10.01, 0.5)],
    "fields": ["gauss", "exp_deep", "exp_shallow", "laplace"],
    "target": "top 10% of g",
}
FIELD_LABELS = {
    "gauss": "Gaussian",
    "exp_deep": "Exponential, long tail toward deep wells",
    "exp_shallow": "Exponential, long tail toward shallow ground",
    "laplace": "Laplace (exponential tails on both sides)",
}


def er_graph(n: int, seed: int) -> sp.csr_matrix:
    G = nx.gnm_random_graph(n, 3 * n, seed=seed)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.to_scipy_sparse_array(G, format="csr").astype(float)


def make_field(kind: str, n: int, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    x = {
        "gauss": lambda: r.standard_normal(n),
        "exp_deep": lambda: r.exponential(size=n),
        "exp_shallow": lambda: -r.exponential(size=n),
        "laplace": lambda: r.laplace(size=n),
    }[kind]()
    return standardize(x)


def trap_climbs(A: sp.csr_matrix, g: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Floor-to-rim climb of every local well outside T (experiment B's barrier, per trap)."""
    Ad = A.toarray()
    _, depth = barrier_prediction(Ad, g, T)
    nbr_max = np.where(Ad > 0, g[None, :], -np.inf).max(axis=1)
    return depth[(g > nbr_max) & ~T]


def run_one(kind: str, n: int, seed: int, lams: list[float]) -> dict:
    A = er_graph(n, seed)
    g = make_field(kind, A.shape[0], seed)
    T = target_of(g)
    lab = well_basins(A, g, T)[~T]
    rows = []
    for lam in lams:
        h, o = hitting_and_occupancy(kernel_sparse(A, g, lam), T)
        rows.append(
            dict(
                lam=lam,
                y2_basin=participation_ratio(np.bincount(lab, weights=o)),
                y2_node=participation_ratio(o),
                h_mean=float(h.mean()),
                h_median=float(np.median(h)),
            )
        )
    return dict(
        field=kind,
        n=int(A.shape[0]),
        n_nominal=n,
        seed=seed,
        n_basins=int(len(np.unique(lab))),
        climbs=[round(float(c), 4) for c in trap_climbs(A, g, T)],
        rows=rows,
    )


def main():
    out = {"config": CONFIG, "git_sha": git_sha(), "field_labels": FIELD_LABELS, "runs": []}
    for kind in CONFIG["fields"]:
        for n in CONFIG["sizes"]:
            t0 = time.time()
            for seed in range(CONFIG["seeds"][str(n)]):
                out["runs"].append(run_one(kind, n, seed, CONFIG["lams"]))
            print(f"{kind} n={n}: {time.time() - t0:.0f}s", flush=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
