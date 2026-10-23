"""Step 3 experiments for the gravity-mass walk.

A. lambda* vs landscape roughness (rho dial + smooth fields), BA and ER
B. Arrhenius test: large-lambda slope of log MFPT vs predicted barrier
C. Push contagion statistics + redundancy (wasted pushes)
D. Effective distance vs contagion arrival times
E. Effective information vs lambda
F. Real network: mouse DTI connectome with region volume as gravity

Run: python -m pipelines.experiments.network_science.gravity_mass   (about 20 min)
Overwrites lab/gravity_mass/results/step3_results.json, which the page reads as-is.
Unlike the other experiments this one writes no Supabase run; see docs/DECISIONS.md.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from lab.gravity_mass.model import (
    LAMS,
    RESULTS_PATH,
    effective_information,
    heat_field,
    kernel,
    knn_backbone,
    load_mouse,
    local_max_fraction,
    mfpt_curve,
    mfpt_to_set,
    rho_field,
    standardize,
    target_of,
)

rng = np.random.default_rng(42)


def make_graphs(n: int = 1000) -> dict[str, np.ndarray]:
    """Giant components of BA(n, m=3) and ER(n, 3n edges), seed 1, as dense adjacency."""
    out = {}
    for name, G in [
        ("BA", nx.barabasi_albert_graph(n, 3, seed=1)),
        ("ER", nx.gnm_random_graph(n, 3 * n, seed=1)),
    ]:
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        out[name] = nx.to_numpy_array(G)
    return out


# ---------- A. lambda* vs roughness ----------
def experiment_A(graphs, n_seeds=8):
    rows = []
    for gname, A in graphs.items():
        L = np.diag(A.sum(1)) - A
        evals, evecs = np.linalg.eigh(L)
        specs = [("rho", r) for r in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)] + [
            ("heat", t) for t in (0.25, 0.5, 1.0, 2.0)
        ]
        for kind, par in specs:
            for seed in range(n_seeds):
                g = (
                    rho_field(A, par, seed)
                    if kind == "rho"
                    else heat_field(evals, evecs, par, seed)
                )
                h = mfpt_curve(A, g)
                i = int(np.argmin(h))
                rows.append(
                    dict(
                        graph=gname,
                        kind=kind,
                        par=par,
                        seed=seed,
                        roughness=float(local_max_fraction(A, g, target_of(g))),
                        lam_star=float(LAMS[i]),
                        censored=bool(i == len(LAMS) - 1),
                        h0=float(h[0]),
                        hmin=float(h[i]),
                        h8=float(h[-1]),
                    )
                )
            sub = [r for r in rows if r["graph"] == gname and r["kind"] == kind and r["par"] == par]
            print(
                f"A {gname} {kind}={par}: rough={np.mean([r['roughness'] for r in sub]):.3f} "
                f"lam*={np.mean([r['lam_star'] for r in sub]):.2f} "
                f"censored={sum(r['censored'] for r in sub)}/{len(sub)}",
                flush=True,
            )
    return rows


# ---------- B. Arrhenius test ----------
def barrier_prediction(A, g, T):
    """Edge landscape e_uv = g_u + g_v. Trap depth D_j = top edge at j minus the
    maximin edge value on paths from j to T. Returns Gamma* = max_j D_j over j not in T."""
    n = len(g)
    iu, ju = np.nonzero(np.triu(A))
    e = g[iu] + g[ju]
    order = np.argsort(-e)
    parent = np.arange(n + 1)  # node n = merged target super-node

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    SUPER = n
    for t in np.flatnonzero(T):
        parent[t] = SUPER
    phi = np.full(n, np.nan)
    members = {i: [i] for i in range(n) if not T[i]}
    members[SUPER] = []
    for idx in order:
        a, b = find(iu[idx]), find(ju[idx])
        if a == b:
            continue
        if b == SUPER:
            a, b = b, a
        if a == SUPER:
            for v in members.pop(b):
                phi[v] = e[idx]
            parent[b] = SUPER
        else:
            members[a].extend(members.pop(b))
            parent[b] = a
    top_edge = np.where(A > 0, g[None, :], -np.inf).max(1) + g
    depth = np.where(T, 0.0, top_edge - phi)
    return float(np.nanmax(depth)), depth


def experiment_B(graphs, n_seeds=6, lam_hi=np.arange(6, 12.01, 1.0)):  # noqa: B008 (read-only)
    rows = []
    for gname, A in graphs.items():
        for seed in range(n_seeds):
            g = rho_field(A, 0.0, seed)
            T = target_of(g)
            gamma, depth = barrier_prediction(A, g, T)
            h = np.array([mfpt_to_set(kernel(A, g, lam), T) for lam in lam_hi])
            slope = float(np.polyfit(lam_hi[-3:], np.log(h[-3:]), 1)[0])
            rows.append(
                dict(graph=gname, seed=seed, gamma_pred=gamma, slope=slope, h=[float(x) for x in h])
            )
            print(
                f"B {gname} seed {seed}: predicted Gamma* = {gamma:.3f}, "
                f"fitted slope = {slope:.3f}",
                flush=True,
            )
    return rows


# ---------- C/D. push contagion ----------
def csr_sampler(P):
    S = sp.csr_matrix(P)
    S.eliminate_zeros()
    cum = np.empty_like(S.data)
    for i in range(S.shape[0]):
        a, b = S.indptr[i], S.indptr[i + 1]
        c = np.cumsum(S.data[a:b])
        cum[a:b] = i + c / c[-1]
    cum[S.indptr[1:] - 1] = np.arange(1, S.shape[0] + 1) - 1e-12  # guard last bin
    return cum, S.indices


def push_run(cum, idx, n, T, seed_node, frac=0.5, max_steps=5000):
    informed = np.zeros(n, bool)
    informed[seed_node] = True
    arrival = np.full(n, np.inf)
    arrival[seed_node] = 0
    need = int(np.ceil(frac * T.sum()))
    pushes = wasted = 0
    t_half = None
    for t in range(1, max_steps + 1):
        src = np.flatnonzero(informed)
        pos = np.searchsorted(cum, src + rng.random(len(src)))
        dst = idx[pos]
        new = dst[~informed[dst]]
        if t_half is None:  # count redundancy only until half the target is informed
            pushes += len(dst)
            wasted += len(dst) - len(np.unique(new))
        informed[new] = True
        arrival[new] = np.minimum(arrival[new], t)
        if t_half is None and (informed & T).sum() >= need:
            t_half = t
        if informed.all() or (t_half is not None and t >= 3 * t_half and t > 200):
            break
    return t_half if t_half is not None else max_steps, wasted / pushes, arrival


def experiment_C(graphs, lams=(0, 0.5, 1, 1.5, 2, 3, 4), n_seeds=5, n_runs=60):
    rows = []
    for gname, A in graphs.items():
        n = len(A)
        for seed in range(n_seeds):
            g = rho_field(A, 0.0, seed)
            T = target_of(g)
            starts = np.random.default_rng(seed).choice(np.flatnonzero(~T), n_runs)
            walker = {lam: mfpt_to_set(kernel(A, g, lam), T) for lam in lams}
            for lam in lams:
                cum, idx = csr_sampler(kernel(A, g, lam))
                res = [push_run(cum, idx, n, T, s)[:2] for s in starts]
                rows.append(
                    dict(
                        graph=gname,
                        seed=seed,
                        lam=lam,
                        t_half=float(np.mean([r[0] for r in res])),
                        wasted=float(np.mean([r[1] for r in res])),
                        walker_mfpt=float(walker[lam]),
                    )
                )
        for lam in lams:
            sub = [r for r in rows if r["graph"] == gname and r["lam"] == lam]
            print(
                f"C {gname} lam={lam}: t_half={np.mean([r['t_half'] for r in sub]):.1f} "
                f"wasted={np.mean([r['wasted'] for r in sub]):.3f}",
                flush=True,
            )
    return rows


def experiment_D(graphs, lams=(0, 1, 2, 4), n_runs=200):
    rows = []
    for gname, A in graphs.items():
        n = len(A)
        g = rho_field(A, 0.0, 0)
        T = target_of(g)
        s0 = int(np.flatnonzero(~T)[0])
        hops = dijkstra(sp.csr_matrix(A > 0), indices=s0, unweighted=True)
        for lam in lams:
            P = kernel(A, g, lam)
            cum, idx = csr_sampler(P)
            arr = np.array(
                [push_run(cum, idx, n, T, s0, frac=1.01, max_steps=400)[2] for _ in range(n_runs)]
            )
            arr = np.where(np.isinf(arr), np.nan, arr)
            reach = np.mean(~np.isnan(arr), 0)
            mean_arr = np.nanmean(arr, 0)
            with np.errstate(divide="ignore"):
                W = np.where(P > 0, 1 - np.log(P), 0)
            deff = dijkstra(sp.csr_matrix(W), indices=s0)
            ok = (reach > 0.9) & (np.arange(n) != s0)
            rows.append(
                dict(
                    graph=gname,
                    lam=lam,
                    n_nodes=int(ok.sum()),
                    r_eff=float(np.corrcoef(deff[ok], mean_arr[ok])[0, 1]),
                    r_hop=float(np.corrcoef(hops[ok], mean_arr[ok])[0, 1]),
                )
            )
            print(
                f"D {gname} lam={lam}: r(eff dist, arrival)={rows[-1]['r_eff']:.3f} "
                f"r(hops, arrival)={rows[-1]['r_hop']:.3f} on {ok.sum()} nodes",
                flush=True,
            )
    return rows


# ---------- E. effective information ----------
def experiment_E(graphs, lams=(0, 0.5, 1, 2, 3, 4, 6, 8), n_seeds=5):
    rows = []
    for gname, A in graphs.items():
        for rho in (0.0, 0.5, 1.0):
            for seed in range(n_seeds):
                g = rho_field(A, rho, seed)
                for lam in lams:
                    rows.append(
                        dict(
                            graph=gname,
                            rho=rho,
                            seed=seed,
                            lam=lam,
                            ei=float(effective_information(kernel(A, g, lam))),
                        )
                    )
            m = [
                np.mean(
                    [
                        r["ei"]
                        for r in rows
                        if r["graph"] == gname and r["rho"] == rho and r["lam"] == lam
                    ]
                )
                for lam in lams
            ]
            print(f"E {gname} rho={rho}: " + " ".join(f"{x:.2f}" for x in m), flush=True)
    return rows


# ---------- F. mouse connectome ----------
def experiment_F(lams=LAMS, k=6):
    A_full, vol, fa = load_mouse()
    # The raw DTI graph is 66% dense: every node touches the target set, so there are
    # no traps and MFPT -> 1 trivially. Use a k-strongest-edges backbone instead.
    A = knn_backbone(A_full, k)
    out = {
        "n": int(len(A)),
        "k": k,
        "density_raw": float((A_full > 0).mean()),
        "density": float((A > 0).mean()),
        "lams": [float(lam) for lam in lams],
        "fields": {},
    }
    s = A.sum(1)
    fields = {
        "log volume": standardize(np.log(vol)),
        "FA": standardize(fa),
        "random (rho=0)": standardize(np.random.default_rng(0).standard_normal(len(A))),
    }
    for name, g in fields.items():
        T = target_of(g)
        h = mfpt_curve(A, g, lams)
        i = int(np.argmin(h))
        out["fields"][name] = dict(
            corr_log_strength=float(np.corrcoef(g, np.log(s))[0, 1]),
            roughness=float(local_max_fraction(A, g, T)),
            mfpt=[float(x) for x in h],
            lam_star=float(lams[i]),
            ei=[float(effective_information(kernel(A, g, lam))) for lam in (0, 1, 2, 4, 8)],
        )
        f = out["fields"][name]
        print(
            f"F {name}: corr(g, log strength)={f['corr_log_strength']:.2f} "
            f"rough={f['roughness']:.3f} lam*={f['lam_star']} "
            f"h0={h[0]:.1f} hmin={h[i]:.1f} h8={h[-1]:.1f}",
            flush=True,
        )
    return out


def main():
    graphs = make_graphs()
    res = {"lams": [float(lam) for lam in LAMS]}
    res["F"] = experiment_F()
    res["B"] = experiment_B(graphs)
    res["E"] = experiment_E(graphs)
    res["D"] = experiment_D(graphs)
    res["C"] = experiment_C(graphs)
    res["A"] = experiment_A(graphs)
    RESULTS_PATH.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
