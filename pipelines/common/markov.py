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

Partition = Sequence[Sequence[int]]


def transition_matrix(counts: np.ndarray) -> np.ndarray:
    """Row-normalise a count matrix; rows with no counts stay all-zero."""
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def _entropy(p: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits along the last axis (0 log 0 = 0)."""
    logs = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -(p * logs).sum(axis=-1)


def effective_information(w: np.ndarray) -> float:
    """EI in bits of a row-stochastic matrix, over the rows that have outputs."""
    w = np.asarray(w, dtype=float)
    rows = w[w.sum(axis=1) > 0]
    if len(rows) == 0:
        return 0.0
    return float(_entropy(rows.mean(axis=0)) - _entropy(rows).mean())


def effectiveness(w: np.ndarray) -> float:
    """EI divided by its maximum, log2 of the number of nodes with outputs."""
    n = int((w.sum(axis=1) > 0).sum())
    return effective_information(w) / np.log2(n) if n > 1 else 0.0


def macro_matrix(w: np.ndarray, partition: Partition, weights: np.ndarray) -> np.ndarray:
    """Macro-scale transition matrix for ``partition`` (groups of node indices).

    Row mu is the ``weights``-weighted average of its members' rows (uniform if
    the group's weights are all zero), summed over each target group's columns;
    flows inside a group become self-loops.
    """
    k = len(partition)
    out = np.zeros((k, k))
    for a, group in enumerate(partition):
        g = list(group)
        wts = weights[g]
        if wts.sum() <= 0:  # e.g. end states that never pass: average members uniformly
            wts = np.ones(len(g))
        row = (wts[:, None] * w[g]).sum(axis=0) / wts.sum()
        for b, target in enumerate(partition):
            out[a, b] = row[list(target)].sum()
    return out


def causal_emergence(w: np.ndarray, partition: Partition, weights: np.ndarray) -> float:
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


def greedy_partition(
    w: np.ndarray, weights: np.ndarray, mergeable: Sequence[int] | None = None
) -> tuple[list[list[int]], float]:
    """Agglomerative search for a high-EI partition.

    From singletons, repeatedly merge the pair of groups whose merge gives the
    highest macro EI, and return the best partition seen with its EI. Only
    groups made of ``mergeable`` nodes (default: all) may merge. Greedy, so a
    lower bound on the best possible macro EI.
    """
    allowed = set(range(len(w))) if mergeable is None else set(mergeable)
    groups = [[i] for i in range(len(w))]
    best_groups = [list(g) for g in groups]
    best_ei = effective_information(macro_matrix(w, groups, weights))
    while len(groups) > 1:
        candidates = []
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if not allowed.issuperset(groups[a] + groups[b]):
                    continue
                merged = [g for i, g in enumerate(groups) if i not in (a, b)]
                merged.append(groups[a] + groups[b])
                candidates.append((effective_information(macro_matrix(w, merged, weights)), merged))
        if not candidates:
            break
        ei, groups = max(candidates, key=lambda c: c[0])
        if ei > best_ei + 1e-12:
            best_ei, best_groups = ei, [sorted(g) for g in groups]
    return sorted(best_groups), best_ei


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
