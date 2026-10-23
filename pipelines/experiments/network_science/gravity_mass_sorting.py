"""Follow-up B: do messages sort by mass?

Messages of several masses leave the same start. The destination T splits into deep basins
(connected clusters of target nodes), and absorption probabilities give, exactly, which basin
each mass reaches first. Destination sorting is the mutual information between mass and
basin given the start, divided by the entropy of mass (0 = mass tells you nothing about
where a message goes, 1 = mass decides it). Settling sorting is the same quantity for
long-run position, from the closed-form stationary distribution.

Also records, per mass, how deep the destination node is, how far it is in hops, and how
often it is the deepest basin, and the total-variation distance between each mass's
destinations and a massless message's, over the full lam grid.

Run: python -m pipelines.experiments.network_science.gravity_mass_sorting   (about 6 min)
Writes lab/gravity_mass/results/sorting_results.json with its config and git SHA.
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.sparse.csgraph import shortest_path

from lab.gravity_mass.model import (
    LAMS,
    RESULTS_PATH,
    absorption_probabilities,
    deep_basins,
    heat_field,
    kernel,
    knn_backbone,
    load_mouse,
    local_max_fraction,
    rho_field,
    standardize,
    stationary_closed_form,
    target_of,
)
from pipelines.common.provenance import git_sha
from pipelines.experiments.network_science.gravity_mass import make_graphs

OUT = RESULTS_PATH.parent / "sorting_results.json"
MASSES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
CONFIG = {
    "version": 1,
    "graphs": "BA (m = 3) and ER (3n edges), n = 1000, seed 1, as in step 3; mouse backbone k = 6",
    "masses": MASSES,
    "lams": [float(x) for x in LAMS],
    "fields": {"rho": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "heat": [0.25, 0.5, 1.0, 2.0]},
    "seeds": 8,
    "target": "top 10% of g; deep basins = connected clusters of target nodes",
}


def entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-300, 1.0)
    return -(p * np.log2(p)).sum(-1)


def sorting_stats(A: np.ndarray, g: np.ndarray, hops: np.ndarray) -> dict:
    T = target_of(g)
    Q = ~T
    lab = deep_basins(A, T)
    k = int(lab.max() + 1)
    onehot = np.eye(k)[lab]  # target node -> basin
    gT = g[T]
    deepest = lab[np.argmax(gT)]
    hopsQT = hops[np.ix_(Q, T)]
    basin_p, per_mass = {}, []
    for lam in LAMS:
        B = absorption_probabilities(kernel(A, g, lam), T)
        basin_p[float(lam)] = B @ onehot
        if float(lam) in MASSES:
            per_mass.append(
                dict(
                    lam=float(lam),
                    dest_depth=float((B @ gT).mean()),
                    dest_hops=float((B * hopsQT).sum(1).mean()),
                    deepest_basin=float(basin_p[float(lam)][:, deepest].mean()),
                )
            )
    D = np.array([basin_p[m] for m in MASSES])  # mass x start x basin
    dest_mi = float((entropy_bits(D.mean(0)) - entropy_bits(D).mean(0)).mean())
    pis = np.array([stationary_closed_form(A, g, m) for m in MASSES])
    settle_mi = float(entropy_bits(pis.mean(0)) - entropy_bits(pis).mean())
    h_m = np.log2(len(MASSES))
    tv = [
        [float(0.5 * np.abs(D[i] - D[j]).sum(-1).mean()) for j in range(len(MASSES))]
        for i in range(len(MASSES))
    ]
    tv_curve = [
        float(0.5 * np.abs(basin_p[float(lam)] - basin_p[0.0]).sum(-1).mean()) for lam in LAMS
    ]
    return dict(
        n_basins=k,
        trap_share=float(local_max_fraction(A, g, T)),
        dest_sorting=dest_mi / h_m,
        settle_sorting=settle_mi / h_m,
        tv=tv,
        tv_curve=tv_curve,
        per_mass=per_mass,
    )


def round_floats(x, digits: int = 5):
    """Round every float in a nested structure to `digits` significant figures."""
    if isinstance(x, float):
        return float(f"{x:.{digits}g}")
    if isinstance(x, dict):
        return {k: round_floats(v, digits) for k, v in x.items()}
    if isinstance(x, list):
        return [round_floats(v, digits) for v in x]
    return x


def main():
    out = {"config": CONFIG, "git_sha": git_sha(), "runs": [], "mouse": []}
    for gname, A in make_graphs().items():
        hops = shortest_path(A > 0, unweighted=True)
        evals, evecs = np.linalg.eigh(np.diag(A.sum(1)) - A)
        for kind, pars in CONFIG["fields"].items():
            t0 = time.time()
            for par in pars:
                for seed in range(CONFIG["seeds"]):
                    g = (
                        rho_field(A, par, seed)
                        if kind == "rho"
                        else heat_field(evals, evecs, par, seed)
                    )
                    row = dict(
                        graph=gname, kind=kind, par=par, seed=seed, **sorting_stats(A, g, hops)
                    )
                    out["runs"].append(round_floats(row))
            print(f"{gname} {kind}: {time.time() - t0:.0f}s", flush=True)
    A_full, vol, fa = load_mouse()
    A = knn_backbone(A_full, 6)
    hops = shortest_path(A > 0, unweighted=True)
    fields = {
        "log volume": standardize(np.log(vol)),
        "FA": standardize(fa),
        "random (rho=0)": standardize(np.random.default_rng(0).standard_normal(len(A))),
    }
    for name, g in fields.items():
        out["mouse"].append(round_floats(dict(field=name, **sorting_stats(A, g, hops))))
    print("mouse done", flush=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
