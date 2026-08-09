"""Deterministic tests for the dependency-free influence path."""

import numpy as np
import scipy.sparse as sp

from graphem_rapids.influence import (
    degree_discount_seed_selection,
    estimate_independent_cascade,
)


def _undirected_csr(n_vertices, edges):
    edges = np.asarray(edges, dtype=np.int64)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_vertices, n_vertices),
    )


def test_ic_probability_extremes_are_exact():
    path = _undirected_csr(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])

    blocked = estimate_independent_cascade(
        path, [0], p=0.0, n_simulations=8, random_seed=3, backend="cpu"
    )
    certain = estimate_independent_cascade(
        path, [0], p=1.0, n_simulations=8, random_seed=3, backend="cpu"
    )

    assert blocked.mean == 1.0
    assert blocked.minimum == blocked.maximum == 1
    assert certain.mean == 6.0
    assert certain.minimum == certain.maximum == 6


def test_ic_trials_are_reproducible():
    graph = _undirected_csr(7, [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)])
    first = estimate_independent_cascade(
        graph, [0], p=0.35, n_simulations=64, random_seed=91, backend="cpu"
    )
    second = estimate_independent_cascade(
        graph, [0], p=0.35, n_simulations=64, random_seed=91, backend="cpu"
    )
    assert first == second
    assert first.trials == 64
    assert len(first.samples) == first.trials
    assert first.minimum <= first.mean <= first.maximum


def test_degree_discount_selects_star_hub_first():
    star = _undirected_csr(9, [(0, node) for node in range(1, 9)])
    seeds = degree_discount_seed_selection(star, k=3, p=0.1)
    assert seeds[0] == 0
    assert len(seeds) == len(set(seeds)) == 3


def test_degree_discount_candidate_pool_is_validated():
    graph = _undirected_csr(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    try:
        degree_discount_seed_selection(graph, k=3, candidate_pool_size=2)
    except ValueError as exc:
        assert "at least k" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("expected candidate pool validation")
