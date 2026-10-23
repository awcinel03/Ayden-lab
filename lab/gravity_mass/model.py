"""Gravity-mass walk: the model and its step 1 numerical checks.

P_ij = A_ij exp(lam g_j) / sum_k A_ik exp(lam g_k), lam = beta * m.
Single source of truth for the math: the page and the offline experiments
(``pipelines/experiments/network_science/gravity_mass.py``) both import it.

Run the step 1 checks: python -m lab.gravity_mass.model   (about 25 s)
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.special import logsumexp

DATA_DIR = Path(__file__).parent / "data"
RESULTS_PATH = Path(__file__).parent / "results" / "step3_results.json"
Q_TARGET = 0.10
LAMS = np.round(np.arange(0, 8.01, 0.25), 2)

rng = np.random.default_rng(0)


def standardize(x: np.ndarray) -> np.ndarray:
    """Shift and scale to mean 0, variance 1."""
    return (x - x.mean()) / x.std()


def kernel(A: np.ndarray, g: np.ndarray, lam: float) -> np.ndarray:
    """Dense transition matrix P(lam), computed with log-sum-exp for large lam."""
    logits = np.where(A > 0, lam * g[None, :] + np.log(np.where(A > 0, A, 1.0)), -np.inf)
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def stationary_closed_form(A: np.ndarray, g: np.ndarray, lam: float) -> np.ndarray:
    """pi_i ∝ e^{lam g_i} sum_k A_ik e^{lam g_k}; valid for undirected (symmetric) A."""
    # shift g so exp() stays finite; pi is invariant to the shift
    w = np.exp(lam * (g - g.max()))
    pi = w * (A @ w)
    return pi / pi.sum()


def stationary_numeric(P: np.ndarray) -> np.ndarray:
    """Stationary distribution as the leading left eigenvector of P."""
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def mfpt_to_set(P: np.ndarray, target: np.ndarray) -> float:
    """Mean first-passage time to `target`, averaged uniformly over start nodes outside it."""
    Q = ~target
    PQ = P[np.ix_(Q, Q)]
    h = np.linalg.solve(np.eye(Q.sum()) - PQ, np.ones(Q.sum()))
    return h.mean()


def mfpt_monte_carlo(
    P: np.ndarray, target: np.ndarray, n_walks: int = 4000, max_steps: int = 10**6
) -> tuple[float, float]:
    """Monte Carlo estimate of `mfpt_to_set`: (mean, standard error) over `n_walks` walks."""
    cum = np.cumsum(P, axis=1)
    starts = rng.choice(np.flatnonzero(~target), n_walks)
    times = np.empty(n_walks)
    for w, s in enumerate(starts):
        x, t = s, 0
        while not target[x] and t < max_steps:
            x = min(np.searchsorted(cum[x], rng.random()), len(target) - 1)
            t += 1
        times[w] = t
    return times.mean(), times.std() / np.sqrt(n_walks)


def entropy_rate(P: np.ndarray, pi: np.ndarray) -> float:
    """Entropy rate of the chain in nats: -sum_i pi_i sum_j P_ij log P_ij."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    return -(pi[:, None] * P * logP).sum()


def spectral_gap(P: np.ndarray, pi: np.ndarray) -> float:
    """1 minus the second-largest eigenvalue of the symmetrized (reversible) chain."""
    s = np.sqrt(pi)
    S = (s[:, None] * P) / s[None, :]
    S = (S + S.T) / 2
    ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    return 1 - ev[1]


def local_max_fraction(A: np.ndarray, g: np.ndarray, target: np.ndarray) -> float:
    """Trap share: fraction of nodes that are local wells (strict local maxima of g) outside T."""
    nbr_max = np.where(A > 0, g[None, :], -np.inf).max(axis=1)
    is_max = g > nbr_max
    return (is_max & ~target).sum() / len(g)


def fields(
    G: nx.Graph,
    A: np.ndarray,
    rho_values: tuple[float, ...] = (1.0, 0.5, 0.0),
    tau: float = 5.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Step 1 gravity fields: the rho dial at `rho_values` plus a (I + tau L)^-1 smoothing."""
    r = np.random.default_rng(seed)
    k = A.sum(1)
    logk = standardize(np.log(k))
    xi = r.standard_normal(len(k))
    out = {f"rho={rho:g}": standardize(rho * logk + np.sqrt(1 - rho**2) * xi) for rho in rho_values}
    L = np.diag(k) - A
    out[f"smooth(tau={tau:g})"] = standardize(np.linalg.solve(np.eye(len(k)) + tau * L, xi))
    return out


def push_contagion_time(
    P: np.ndarray, target: np.ndarray, frac: float = 0.5, n_runs: int = 40, max_steps: int = 5000
) -> float:
    """Push SI: each informed node forwards to one neighbor drawn from P per step.
    Returns mean steps until `frac` of the target set is informed."""
    n = P.shape[0]
    cum = np.cumsum(P, axis=1)
    need = int(np.ceil(frac * target.sum()))
    times = []
    for _ in range(n_runs):
        informed = np.zeros(n, bool)
        informed[rng.choice(np.flatnonzero(~target))] = True
        for t in range(1, max_steps + 1):  # noqa: B007 (t is read after the loop)
            src = np.flatnonzero(informed)
            u = rng.random(len(src))
            dst = np.minimum((cum[src] < u[:, None]).sum(1), n - 1)
            informed[dst] = True
            if (informed & target).sum() >= need:
                break
        times.append(t)
    return float(np.mean(times))


# ---------- shared with step 3 (moved from step3_experiments.py) ----------
def rho_field(A: np.ndarray, rho: float, seed: int) -> np.ndarray:
    """g = z(rho z(log k) + sqrt(1 - rho^2) xi); k is the unweighted degree if A is weighted."""
    k = A.sum(1) if A.max() <= 1 else (A > 0).sum(1)
    xi = np.random.default_rng(seed).standard_normal(len(k))
    return standardize(rho * standardize(np.log(k)) + np.sqrt(1 - rho**2) * xi)


def heat_field(evals: np.ndarray, evecs: np.ndarray, t: float, seed: int) -> np.ndarray:
    """g = z(e^{-tL} xi), given the Laplacian eigendecomposition (evals, evecs)."""
    xi = np.random.default_rng(seed).standard_normal(len(evals))
    return standardize(evecs @ (np.exp(-t * evals) * (evecs.T @ xi)))


def target_of(g: np.ndarray) -> np.ndarray:
    """Target set T: the top `Q_TARGET` share of nodes by gravity."""
    return g >= np.quantile(g, 1 - Q_TARGET)


def mfpt_curve(A: np.ndarray, g: np.ndarray, lams: np.ndarray = LAMS) -> np.ndarray:
    """Hitting time to T at each lam in `lams`."""
    T = target_of(g)
    return np.array([mfpt_to_set(kernel(A, g, lam), T) for lam in lams])


def effective_information(P: np.ndarray) -> float:
    """Klein-Hoel EI in bits: H(mean row) - mean H(row)."""

    def H(p):
        p = p[p > 0]
        return -(p * np.log2(p)).sum()

    return H(P.mean(0)) - np.mean([H(r) for r in P])


def load_mouse(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mouse DTI connectome N54776: (weighted adjacency, volume_mm3, FA), 332 regions.

    The node-to-ROI mapping is inferred, not documented (see ``data/README.md``).
    """
    E = np.loadtxt(data_dir / "sub-54776_ses-1_dti.edgelist")
    n = 332
    A = np.zeros((n, n))
    A[E[:, 0].astype(int), E[:, 1].astype(int)] = E[:, 2]
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    feats = np.genfromtxt(data_dir / "54776.csv", delimiter=",", skip_header=3)
    feats = feats[feats[:, 0] > 0]
    feats = feats[np.argsort(feats[:, 0])]
    return A, feats[:, 2], feats[:, 8]  # adjacency, volume_mm3, FA


def knn_backbone(A: np.ndarray, k: int) -> np.ndarray:
    """Keep each node's k strongest connections (symmetrized), preserving weights."""
    keep = np.zeros_like(A, bool)
    top = np.argsort(-A, axis=1)[:, :k]
    keep[np.repeat(np.arange(len(A)), k), top.ravel()] = True
    keep |= keep.T
    return np.where(keep, A, 0.0)


# ---------- sparse versions, for networks beyond the dense limit (n > ~2,000) ----------
def kernel_sparse(A: sp.spmatrix, g: np.ndarray, lam: float) -> sp.csr_matrix:
    """Sparse P(lam); same values as `kernel`, computed row by row with log-sum-exp."""
    A = sp.csr_matrix(A)
    A.sort_indices()
    counts = np.diff(A.indptr)
    logits = lam * g[A.indices] + np.log(A.data)
    shift = np.repeat(np.maximum.reduceat(logits, A.indptr[:-1]), counts)
    w = np.exp(logits - shift)
    w /= np.repeat(np.add.reduceat(w, A.indptr[:-1]), counts)
    return sp.csr_matrix((w, A.indices.copy(), A.indptr.copy()), shape=A.shape)


def hitting_and_occupancy(P: sp.spmatrix, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-start hitting times to `target` and mean occupancy before arrival.

    With Q the nodes outside `target` and N = (I - P_QQ)^-1, returns h = N 1 (steps to reach
    the target from each start in Q) and o = 1^T N / |Q| (expected steps spent at each node
    of Q, averaged over uniform starts in Q). Both are indexed over Q; o.sum() == h.mean().
    """
    Q = ~target
    P = sp.csr_matrix(P)
    lu = splu((sp.identity(int(Q.sum()), format="csc") - P[Q][:, Q]).tocsc())
    ones = np.ones(int(Q.sum()))
    return lu.solve(ones), lu.solve(ones / Q.sum(), trans="T")


def well_basins(A: sp.spmatrix, g: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Label each node with the local well it drains to by always stepping to its deepest
    neighbour while that neighbour is deeper. Nodes in `target` label themselves."""
    A = sp.csr_matrix(A)
    nxt = np.arange(len(g))
    for i in np.flatnonzero(~target):
        nb = A.indices[A.indptr[i] : A.indptr[i + 1]]
        j = nb[np.argmax(g[nb])]
        if g[j] > g[i]:
            nxt[i] = j
    while not np.array_equal(nxt, nxt[nxt]):
        nxt = nxt[nxt]
    return nxt


def participation_ratio(w: np.ndarray) -> float:
    """Y2 = sum of squared shares: about 1/(number of equal parts), 1 if one part has all."""
    w = np.asarray(w, float) / np.sum(w)
    return float((w**2).sum())


# ---------- destinations, for mass sorting ----------
def deep_basins(A: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Split `target` into deep basins: connected clusters of target nodes. Returns a basin
    label 0..k-1 for each target node, in node order."""
    _, labels = connected_components(sp.csr_matrix(A[np.ix_(target, target)] > 0), directed=False)
    return labels


def absorption_probabilities(P: np.ndarray, target: np.ndarray) -> np.ndarray:
    """B[s, t]: probability that a walk from start s (outside `target`) first enters `target`
    at node t. Rows run over the nodes outside `target`, columns over target nodes; rows sum
    to 1."""
    Q = ~target
    return np.linalg.solve(np.eye(Q.sum()) - P[np.ix_(Q, Q)], P[np.ix_(Q, target)])


def main() -> None:
    """Run the step 1 checks on BA and ER graphs at n = 1000 and print the results."""
    n, q = 1000, 0.10
    graphs = {
        "BA(n=1000, m=3)": nx.barabasi_albert_graph(n, 3, seed=1),
        "ER(n=1000, <k>=6)": nx.gnm_random_graph(n, 3 * n, seed=1),
    }
    lams = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6]

    for gname, G in graphs.items():
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        A = nx.to_numpy_array(G)
        k = A.sum(1)
        print(f"\n=== {gname}: n={len(k)}, <k>={k.mean():.2f}")

        # 1. closed-form stationary distribution
        g = standardize(rng.standard_normal(len(k)))
        for lam in (0.0, 1.0, 3.0):
            P = kernel(A, g, lam)
            err = np.abs(stationary_closed_form(A, g, lam) - stationary_numeric(P)).max()
            print(f"[pi check] lam={lam}: max |closed-form - numeric| = {err:.2e}")

        # 2. MERW special case
        vals, vecs = np.linalg.eigh(A)
        psi = np.abs(vecs[:, -1])
        P_merw = A * psi[None, :] / (vals[-1] * psi[:, None])
        err = np.abs(kernel(A, np.log(psi), 1.0) - P_merw).max()
        print(f"[MERW check] max |P(g=log psi, lam=1) - P_MERW| = {err:.2e}")

        # 3. degree-biased special case
        a = 1.7
        P_db = A * k[None, :] ** a
        P_db /= P_db.sum(1, keepdims=True)
        err = np.abs(kernel(A, np.log(k), a) - P_db).max()
        print(f"[degree-biased check] max diff = {err:.2e}")

        # 4. MFPT to top-q gravity nodes vs lam, per field (averaged over 5 field seeds)
        print(f"\nMFPT to top {q:.0%} gravity nodes (mean over 5 field seeds)")
        header = (
            "field".ljust(18) + "lmax%".rjust(7) + "".join(f"{lam:>9g}" for lam in lams) + "   lam*"
        )
        print(header)
        for fname in fields(G, A):
            rows, lmf = [], []
            for seed in range(5):
                gf = fields(G, A, seed=seed)[fname]
                target = gf >= np.quantile(gf, 1 - q)
                lmf.append(local_max_fraction(A, gf, target))
                rows.append([mfpt_to_set(kernel(A, gf, lam), target) for lam in lams])
            h = np.mean(rows, 0)
            print(
                fname.ljust(18)
                + f"{100 * np.mean(lmf):7.1f}"
                + "".join(f"{x:9.1f}" for x in h)
                + f"   {lams[int(np.argmin(h))]}"
            )

        # 5. Monte Carlo check of the linear solve
        gf = fields(G, A, seed=0)["rho=0"]
        target = gf >= np.quantile(gf, 1 - q)
        for lam in (0.0, 2.0):
            P = kernel(A, gf, lam)
            mc, se = mfpt_monte_carlo(P, target)
            exact = mfpt_to_set(P, target)
            print(f"[MC check] rho=0, lam={lam}: exact {exact:.2f}, MC {mc:.2f} +/- {se:.2f}")

        # 6. entropy rate and spectral gap
        print("\nlam, entropy rate (nats), spectral gap   [rho=0, seed 0]")
        for lam in (0, 1, 2, 4, 6):
            P = kernel(A, gf, lam)
            pi = stationary_closed_form(A, gf, lam)
            print(f"{lam:>4} {entropy_rate(P, pi):10.3f} {spectral_gap(P, pi):12.2e}")

        # 7. push contagion
        print("\nPush contagion: steps to inform 50% of target  [rho=0, seed 0]")
        for lam in (0, 1, 2, 4):
            print(f"lam={lam}: {push_contagion_time(kernel(A, gf, lam), target):.1f}")


if __name__ == "__main__":
    main()
