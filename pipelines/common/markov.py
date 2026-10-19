"""Markov chains on weighted directed networks: effective information and coarse-graining.

Effective information (EI) follows Klein & Hoel, "The emergence of informative
higher scales in complex networks" (Complexity, 2020): for a network whose rows
are out-going transition probabilities W_i, intervene uniformly on every node
with outputs and measure

    EI = H(<W_i>) - <H(W_i)>          (bits)

the entropy of the average out-distribution (how spread out the effects of
interventions are) minus the average entropy of each node's out-distribution
(how noisy each node's own effect is). A macro-scale network groups nodes into
macro-nodes; *causal emergence* is EI(macro) - EI(micro) > 0.

Macro-nodes use the "mu|pi" construction: a macro-node's out-distribution is
the average of its members' out-distributions weighted by ``weights`` (here the
empirical stationary distribution, i.e. how often each node is the sender).
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence

import numpy as np
import scipy.sparse as sp

Partition = Sequence[Sequence[int]]
Matrix = np.ndarray | sp.sparray | sp.spmatrix


def transition_matrix(counts: np.ndarray) -> np.ndarray:
    """Row-normalise a count matrix; rows with no counts stay all-zero."""
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def _entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits along the last axis (0 log 0 = 0)."""
    logs = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logs).sum(axis=-1)


def _ei_parts(w: Matrix) -> tuple[float, float, int]:
    """``(H(<W_i>), <H(W_i)>, n)`` over the ``n`` rows that have outputs; dense or sparse."""
    if sp.issparse(w):
        w = sp.csr_matrix(w, dtype=float)
        w.eliminate_zeros()
        has = np.diff(w.indptr) > 0
        n = int(has.sum())
        if n == 0:
            return 0.0, 0.0, 0
        mean_row = np.asarray(w[has].sum(axis=0)).ravel() / n
        terms = -w.data * np.log2(w.data)
        row_h = np.add.reduceat(terms, w.indptr[:-1][has])
        return float(_entropy(mean_row)), float(row_h.mean()), n
    w = np.asarray(w, dtype=float)
    rows = w[w.sum(axis=1) > 0]
    if len(rows) == 0:
        return 0.0, 0.0, 0
    return float(_entropy(rows.mean(axis=0))), float(_entropy(rows).mean()), len(rows)


def effective_information(w: Matrix) -> float:
    """EI in bits of a row-stochastic matrix (dense or sparse), over the rows that have outputs."""
    h_mean, mean_h, _ = _ei_parts(w)
    return h_mean - mean_h


def effectiveness(w: Matrix) -> float:
    """EI divided by its maximum, log2 of the number of nodes with outputs."""
    h_mean, mean_h, n = _ei_parts(w)
    return (h_mean - mean_h) / np.log2(n) if n > 1 else 0.0


def determinism_degeneracy(w: Matrix) -> tuple[float, float]:
    """Klein & Hoel's two terms in bits, with ``EI = determinism - degeneracy``.

    Determinism is ``log2 n - <H(W_i)>`` (how sharply each node's outputs are
    focused); degeneracy is ``log2 n - H(<W_i>)`` (how much different nodes
    send to the same places), both over the ``n`` nodes with outputs.
    """
    h_mean, mean_h, n = _ei_parts(w)
    top = float(np.log2(n)) if n else 0.0
    return top - mean_h, top - h_mean


def macro_matrix(w: Matrix, partition: Partition, weights: np.ndarray) -> Matrix:
    """Macro-scale transition matrix for ``partition`` (groups of node indices).

    Row mu is the ``weights``-weighted average of its members' rows (uniform if
    the group's weights are all zero), summed over each target group's columns;
    flows inside a group become self-loops. Sparse in, sparse out.
    """
    k, n = len(partition), w.shape[0]
    members = np.fromiter((i for g in partition for i in g), dtype=int)
    group = np.repeat(np.arange(k), [len(g) for g in partition])
    wts = np.asarray(weights, dtype=float)[members]
    empty = np.bincount(group, weights=wts, minlength=k) <= 0
    wts = np.where(empty[group], 1.0, wts)  # e.g. end states that never pass
    share = wts / np.bincount(group, weights=wts, minlength=k)[group]
    s = sp.csr_matrix((np.ones(len(members)), (members, group)), shape=(n, k))
    p = sp.csr_matrix((share, (group, members)), shape=(k, n))
    out = p @ w @ s
    return sp.csr_matrix(out) if sp.issparse(w) else np.asarray(out)


def causal_emergence(w: Matrix, partition: Partition, weights: np.ndarray) -> float:
    """EI(macro) - EI(micro) in bits; positive means the macro scale is more informative."""
    return effective_information(macro_matrix(w, partition, weights)) - effective_information(w)


def with_end_states(w: np.ndarray, ends: Sequence[int], *, absorbing: bool = True) -> np.ndarray:
    """Copy of ``w`` with the ``ends`` rows made absorbing (self-loop 1) or empty (sinks).

    Absorbing end states are deterministic, so they add to EI; as sinks they
    have no outputs and are left out of the intervention distribution.
    """
    out = np.array(w, dtype=float)
    for i in ends:
        out[i] = 0.0
        if absorbing:
            out[i, i] = 1.0
    return out


def _merged(m: np.ndarray, size: np.ndarray, count: np.ndarray, a: int, b: int) -> np.ndarray:
    """Macro matrix after merging groups ``a`` and ``b`` (appended last), from the current one.

    Equals ``macro_matrix`` on the merged partition: rows combine by group
    weight (by member count when both groups' weights are zero) and the two
    columns add.
    """
    wa, wb = (size[a], size[b]) if size[a] + size[b] > 0 else (count[a], count[b])
    keep = [i for i in range(len(m)) if i not in (a, b)]
    rows = np.vstack([m[keep], (wa * m[a] + wb * m[b]) / (wa + wb)])
    return np.column_stack([rows[:, keep], rows[:, a] + rows[:, b]])


def greedy_path(
    w: np.ndarray,
    weights: np.ndarray,
    mergeable: Sequence[int] | None = None,
    within: Sequence[Hashable] | None = None,
) -> list[tuple[list[list[int]], float]]:
    """Every partition an agglomerative EI search visits, from singletons, with its macro EI.

    Each step merges the pair of groups whose merge gives the highest macro EI.
    Only groups made of ``mergeable`` nodes (default: all) may merge, and with
    ``within`` (one label per node) only groups sharing a label. Ends when no
    pair may merge.
    """
    allowed = set(range(len(w))) if mergeable is None else set(mergeable)
    weights = np.asarray(weights, dtype=float)
    groups = [[i] for i in range(len(w))]
    m = np.asarray(macro_matrix(w, groups, weights))
    size, count = weights.copy(), np.ones(len(w))
    path = [([list(g) for g in groups], effective_information(m))]
    while len(groups) > 1:
        best = None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if not allowed.issuperset(groups[a] + groups[b]):
                    continue
                if within is not None and within[groups[a][0]] != within[groups[b][0]]:
                    continue
                candidate = _merged(m, size, count, a, b)
                ei = effective_information(candidate)
                if best is None or ei > best[0]:
                    best = (ei, a, b, candidate)
        if best is None:
            break
        ei, a, b, m = best
        keep = [i for i in range(len(groups)) if i not in (a, b)]
        groups = [groups[i] for i in keep] + [groups[a] + groups[b]]
        size = np.append(size[keep], size[a] + size[b])
        count = np.append(count[keep], count[a] + count[b])
        path.append(([list(g) for g in groups], ei))
    return path


def greedy_partition(
    w: np.ndarray,
    weights: np.ndarray,
    mergeable: Sequence[int] | None = None,
    within: Sequence[Hashable] | None = None,
) -> tuple[list[list[int]], float]:
    """Agglomerative search for a high-EI partition (see ``greedy_path``).

    Returns the best partition seen with its EI. Greedy, so a lower bound on
    the best possible macro EI.
    """
    path = greedy_path(w, weights, mergeable, within)
    best_groups, best_ei = path[0]
    for groups, ei in path[1:]:
        if ei > best_ei + 1e-12:
            best_groups, best_ei = groups, ei
    return sorted(sorted(g) for g in best_groups), best_ei


def random_partition_like(partition: Partition, n: int, rng: random.Random) -> list[list[int]]:
    """A random partition of ``range(n)`` with the same group sizes as ``partition``."""
    nodes = list(range(n))
    rng.shuffle(nodes)
    out, start = [], 0
    for group in partition:
        out.append(nodes[start : start + len(group)])
        start += len(group)
    return out


def configuration_counts(
    counts: np.ndarray, rng: random.Random, max_tries: int = 1000
) -> np.ndarray:
    """Random count matrix keeping every node's out- and in-totals, with no self-loops.

    Pairs out-stubs with a random permutation of in-stubs; self-pairs are
    re-drawn by swapping with a random other pair (falls back to a fresh
    shuffle up to ``max_tries`` times).
    """
    counts = np.asarray(counts, dtype=int)
    outs = np.repeat(np.arange(len(counts)), counts.sum(axis=1))
    ins = np.repeat(np.arange(len(counts)), counts.sum(axis=0))
    for _ in range(max_tries):
        perm = ins.copy()
        rng.shuffle(perm)  # type: ignore[arg-type]
        for i in np.flatnonzero(outs == perm):
            for _ in range(50):
                j = rng.randrange(len(perm))
                if outs[i] != perm[j] and outs[j] != perm[i]:
                    perm[i], perm[j] = perm[j], perm[i]
                    break
        if not (outs == perm).any():
            out = np.zeros_like(counts)
            np.add.at(out, (outs, perm), 1)
            return out
    raise RuntimeError("could not draw a self-loop-free configuration")


def labels_to_partition(labels: Sequence[Hashable]) -> list[list[int]]:
    """Group node indices by label, groups in first-appearance order."""
    groups: dict[Hashable, list[int]] = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)
    return list(groups.values())
