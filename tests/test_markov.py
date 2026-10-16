"""Effective information and coarse-graining on small known chains."""

import random

import numpy as np
import pytest

from pipelines.common import markov as mk


def test_transition_matrix_keeps_empty_rows_zero():
    w = mk.transition_matrix(np.array([[0, 2, 2], [0, 0, 0], [1, 0, 0]]))
    assert np.allclose(w, [[0, 0.5, 0.5], [0, 0, 0], [1, 0, 0]])


def test_ei_of_a_permutation_is_log2_n_and_of_uniform_noise_is_zero():
    assert mk.effective_information(np.eye(4)[[1, 2, 3, 0]]) == pytest.approx(2.0)
    assert mk.effective_information(np.full((4, 4), 0.25)) == pytest.approx(0.0)
    assert mk.effectiveness(np.eye(4)[[1, 2, 3, 0]]) == pytest.approx(1.0)


def test_ei_ignores_nodes_without_outputs():
    w = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    assert mk.effective_information(w) == pytest.approx(1.0)


def test_causal_emergence_in_the_klein_hoel_example():
    third = 1 / 3
    w = np.array([[third] * 3 + [0]] * 3 + [[0, 0, 0, 1]])
    assert mk.effective_information(w) == pytest.approx(0.8113, abs=1e-4)
    assert mk.causal_emergence(w, [[0, 1, 2], [3]], np.ones(4)) == pytest.approx(0.1887, abs=1e-4)


def test_macro_rows_are_weighted_averages_and_sum_to_one():
    w = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    macro = mk.macro_matrix(w, [[0, 1], [2]], np.array([3.0, 1.0, 1.0]))
    assert np.allclose(macro, [[0.75, 0.25], [1.0, 0.0]])
    assert np.allclose(macro.sum(axis=1), 1)


def test_greedy_finds_the_obvious_merge():
    third = 1 / 3
    w = np.array([[third] * 3 + [0]] * 3 + [[0, 0, 0, 1]])
    groups, ei = mk.greedy_partition(w, np.ones(4))
    assert groups == [[0, 1, 2], [3]] and ei == pytest.approx(1.0)


def test_greedy_keeps_singletons_when_nothing_helps():
    w = np.eye(3)[[1, 2, 0]]
    assert mk.greedy_partition(w, np.ones(3))[0] == [[0], [1], [2]]


def test_random_partition_keeps_group_sizes():
    p = mk.random_partition_like([[0, 1, 2], [3], [4, 5]], 6, random.Random(1))
    assert sorted(len(g) for g in p) == [1, 2, 3] and sorted(sum(p, [])) == list(range(6))


def test_configuration_keeps_totals_and_avoids_self_loops():
    counts = np.array([[0, 5, 1, 0], [2, 0, 3, 1], [0, 4, 0, 2], [1, 0, 3, 0]])
    rng = random.Random(3)
    for _ in range(20):
        out = mk.configuration_counts(counts, rng)
        assert (out.sum(axis=1) == counts.sum(axis=1)).all()
        assert (out.sum(axis=0) == counts.sum(axis=0)).all()
        assert np.trace(out) == 0


def test_labels_to_partition_groups_in_first_seen_order():
    assert mk.labels_to_partition(["b", "a", "b", "c"]) == [[0, 2], [1], [3]]
