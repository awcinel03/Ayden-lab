"""Follow-up C: does coarse-graining make heavy messages' walk more informative?

Groups nodes into flow basins (each node with the pair of mutual deepest neighbours it ends
on when every step goes to the deepest neighbour), builds the macro-scale walk with the
mu|pi construction weighted by the closed-form pi(lam), and compares its effective
information with the micro walk's and with random groupings of the same sizes. The grouping
depends on the landscape only, so the same macro-nodes are scored at every mass.

Run: python -m pipelines.experiments.network_science.gravity_mass_coarse   (about 6 min)
Writes lab/gravity_mass/results/coarse_results.json with its config and git SHA.
"""

from __future__ import annotations

import json
import random
import time

import numpy as np

from lab.gravity_mass.model import (
    RESULTS_PATH,
    flow_basins,
    heat_field,
    kernel,
    knn_backbone,
    load_mouse,
    rho_field,
    standardize,
    stationary_closed_form,
)
from pipelines.common import markov as mk
from pipelines.common.provenance import git_sha
from pipelines.experiments.network_science.gravity_mass import make_graphs
from pipelines.experiments.network_science.gravity_mass_sorting import round_floats

OUT = RESULTS_PATH.parent / "coarse_results.json"
CONFIG = {
    "version": 1,
    "graphs": "BA (m = 3) and ER (3n edges), n = 1000, seed 1, as in step 3; mouse backbone k = 6",
    "lams": [float(x) for x in np.arange(0, 8.01, 0.5)],
    "fields": {"rho": [0.0, 0.25, 0.5, 0.75, 1.0], "heat": [0.5, 2.0]},
    "seeds": 5,
    "n_null": 10,
    "partition": "flow basins of the deepest-neighbour map, no absorbing target",
    "macro": "markov.macro_matrix weighted by the closed-form pi(lam)",
}


def coarse_stats(A: np.ndarray, g: np.ndarray, seed: int) -> dict:
    part = mk.labels_to_partition(flow_basins(A, g, np.zeros(len(g), bool)))
    rng = random.Random(seed)
    nulls = [mk.random_partition_like(part, len(g), rng) for _ in range(CONFIG["n_null"])]
    rows = []
    for lam in CONFIG["lams"]:
        P = kernel(A, g, lam)
        pi = stationary_closed_form(A, g, lam)
        M = np.asarray(mk.macro_matrix(P, part, pi))
        null_ei = [mk.effective_information(mk.macro_matrix(P, q, pi)) for q in nulls]
        det, deg = mk.determinism_degeneracy(P)
        mdet, mdeg = mk.determinism_degeneracy(M)
        rows.append(
            dict(
                lam=lam,
                micro_ei=mk.effective_information(P),
                micro_det=det,
                micro_deg=deg,
                macro_ei=mk.effective_information(M),
                macro_det=mdet,
                macro_deg=mdeg,
                self_loop=float(M.diagonal().mean()),
                null_ei=float(np.mean(null_ei)),
                null_sd=float(np.std(null_ei)),
            )
        )
    return dict(n=len(g), k=len(part), rows=rows)


def main():
    out = {"config": CONFIG, "git_sha": git_sha(), "runs": [], "mouse": []}
    for gname, A in make_graphs().items():
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
                        graph=gname, kind=kind, par=par, seed=seed, **coarse_stats(A, g, seed)
                    )
                    out["runs"].append(round_floats(row))
            print(f"{gname} {kind}: {time.time() - t0:.0f}s", flush=True)
    A_full, vol, fa = load_mouse()
    A = knn_backbone(A_full, 6)
    fields = {
        "log volume": standardize(np.log(vol)),
        "FA": standardize(fa),
        "random (rho=0)": standardize(np.random.default_rng(0).standard_normal(len(A))),
    }
    for name, g in fields.items():
        out["mouse"].append(round_floats(dict(field=name, **coarse_stats(A, g, 0))))
    print("mouse done", flush=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
