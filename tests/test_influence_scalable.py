"""Deterministic tests for the dependency-free influence path."""

# pylint: disable=broad-exception-caught,missing-class-docstring
# pylint: disable=missing-function-docstring,use-implicit-booleaness-not-comparison

import numpy as np
import pytest
import scipy.sparse as sp

import graphem_rapids.influence as influence_module
from graphem_rapids.influence import (
    _IC_KERNEL_SOURCE,
    _queue_workspace_bytes,
    degree_discount_seed_selection,
    estimate_independent_cascade,
    graphem_seed_selection,
    warm_independent_cascade_kernels,
)


try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse

    CUPY_GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - environment-dependent import/driver path
    cp = None
    cpx_sparse = None
    CUPY_GPU_AVAILABLE = False


requires_cupy_gpu = pytest.mark.skipif(
    not CUPY_GPU_AVAILABLE, reason="CuPy CUDA device is unavailable"
)


def _undirected_csr(n_vertices, edges):
    edges = np.asarray(edges, dtype=np.int64)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    return sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_vertices, n_vertices),
    )


def _directed_csr(n_vertices, edges):
    edges = np.asarray(edges, dtype=np.int64)
    return sp.csr_matrix(
        (
            np.ones(edges.shape[0], dtype=np.float32),
            (edges[:, 0], edges[:, 1]),
        ),
        shape=(n_vertices, n_vertices),
    )


def _device_csr(adjacency, index_dtype=None):
    index_dtype = cp.int32 if index_dtype is None else index_dtype
    return cpx_sparse.csr_matrix(
        (
            cp.asarray(adjacency.data),
            cp.asarray(adjacency.indices, dtype=index_dtype),
            cp.asarray(adjacency.indptr, dtype=index_dtype),
        ),
        shape=adjacency.shape,
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


def test_ic_common_random_worlds_are_monotone_for_seed_superset():
    graph = _directed_csr(
        10,
        [
            (0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5),
            (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9),
        ],
    )
    smaller = estimate_independent_cascade(
        graph, [0], p=0.27, n_simulations=127, random_seed=17, backend="cpu"
    )
    larger = estimate_independent_cascade(
        graph, [0, 6], p=0.27, n_simulations=127, random_seed=17, backend="cpu"
    )
    assert np.all(np.asarray(larger.samples) >= np.asarray(smaller.samples))


def test_ic_canonicalization_ignores_storage_order_duplicates_and_zeros():
    clean = _directed_csr(4, [(0, 1), (0, 2), (1, 2), (2, 3)])
    noncanonical = sp.csr_matrix(
        (
            np.asarray([1, 1, 0, 1, 1, 1], dtype=np.float32),
            np.asarray([2, 1, 3, 2, 2, 3], dtype=np.int32),
            np.asarray([0, 3, 5, 6, 6], dtype=np.int32),
        ),
        shape=(4, 4),
    )
    first = estimate_independent_cascade(
        clean, [0], p=0.31, n_simulations=97, random_seed=5, backend="cpu"
    )
    second = estimate_independent_cascade(
        noncanonical, [0], p=0.31, n_simulations=97, random_seed=5, backend="cpu"
    )
    assert first == second


def test_ic_empty_seeds_and_probability_zero_are_exact():
    graph = _undirected_csr(6, [(0, 1), (1, 2), (2, 3)])
    empty = estimate_independent_cascade(
        graph, [], p=0.5, n_simulations=11, random_seed=2, backend="cpu"
    )
    blocked = estimate_independent_cascade(
        graph, [0, 4], p=0.0, n_simulations=11, random_seed=2, backend="cpu"
    )
    assert empty.samples == (0,) * 11
    assert blocked.samples == (2,) * 11


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("random_seed", -1, "random_seed"),
        ("random_seed", 2**64, "random_seed"),
        ("random_seed", True, "random_seed"),
        ("n_simulations", 1.5, "n_simulations"),
        ("n_simulations", True, "n_simulations"),
    ],
)
def test_ic_integer_inputs_are_validated(keyword, value, message):
    graph = _directed_csr(2, [(0, 1)])
    arguments = {"n_simulations": 3, "random_seed": 0}
    arguments[keyword] = value
    with pytest.raises(ValueError, match=message):
        estimate_independent_cascade(graph, [0], backend="cpu", **arguments)


@pytest.mark.parametrize(
    "seeds",
    [
        [0.0],
        [0.9],
        [True],
        ["0"],
        np.asarray([0], dtype=np.float32),
        np.asarray([False], dtype=np.bool_),
    ],
)
def test_ic_seed_ids_are_validated_before_integer_conversion(seeds):
    graph = _directed_csr(2, [(0, 1)])
    with pytest.raises(ValueError, match="integer node ids"):
        estimate_independent_cascade(
            graph, seeds, p=0.0, n_simulations=2, backend="cpu"
        )


@pytest.mark.parametrize("seeds", [[], [0]])
@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_ic_batch_size_is_validated_before_fast_paths(seeds, batch_size):
    graph = _directed_csr(2, [(0, 1)])
    with pytest.raises(ValueError, match="batch_size"):
        estimate_independent_cascade(
            graph,
            seeds,
            p=0.0,
            n_simulations=2,
            backend="cpu",
            batch_size=batch_size,
        )


@pytest.mark.parametrize("bad_target", [-1, 4])
def test_ic_rejects_invalid_host_csr_indices(bad_target):
    graph = _directed_csr(4, [(0, 1), (1, 2)])
    graph.indices[0] = bad_target
    with pytest.raises(ValueError, match="invalid CSR"):
        estimate_independent_cascade(
            graph, [0], p=0.2, n_simulations=2, backend="cpu"
        )


def test_ic_rejects_rectangular_sparse_graph_before_evaluation():
    graph = sp.csr_matrix((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="square"):
        estimate_independent_cascade(
            graph, [0], p=0.0, n_simulations=2, backend="cpu"
        )


def test_ic_rejects_malformed_host_indptr():
    graph = _directed_csr(4, [(0, 1), (1, 2)])
    graph.indptr[2] = graph.indptr[1] - 1
    with pytest.raises(ValueError, match="invalid CSR"):
        estimate_independent_cascade(
            graph, [0], p=0.2, n_simulations=2, backend="cpu"
        )


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_ic_available_memory_bytes_is_validated(value):
    graph = _directed_csr(2, [(0, 1)])
    with pytest.raises(ValueError, match="available_memory_bytes"):
        estimate_independent_cascade(
            graph,
            [0],
            p=0.2,
            n_simulations=2,
            backend="cpu",
            available_memory_bytes=value,
        )


def test_ic_trusted_fast_path_requires_actual_csr():
    graph = _directed_csr(3, [(0, 1), (1, 2)]).tocsc()
    with pytest.raises(ValueError, match="requires CSR"):
        estimate_independent_cascade(
            graph,
            [0],
            p=0.2,
            n_simulations=2,
            backend="cpu",
            assume_validated_csr=True,
        )


def test_graphem_selection_does_not_run_layout_by_default():
    class Embedder:
        n = 5

        def __init__(self):
            self.layout_calls = []

        def run_layout(self, num_iterations):
            self.layout_calls.append(num_iterations)

        @staticmethod
        def topk_nodes(k):
            return list(range(k))

    embedder = Embedder()
    assert graphem_seed_selection(embedder, 2) == [0, 1]
    assert embedder.layout_calls == []
    assert graphem_seed_selection(embedder, 2, num_iterations=3) == [0, 1]
    assert embedder.layout_calls == [3]
    with pytest.raises(ValueError, match="number of vertices"):
        graphem_seed_selection(embedder, 6)
    for invalid_k in (1.5, True, "1"):
        with pytest.raises(ValueError, match="k must be"):
            graphem_seed_selection(embedder, invalid_k)
    for invalid_pool in (1.5, True, "2"):
        with pytest.raises(ValueError, match="candidate_pool_size"):
            graphem_seed_selection(
                embedder,
                1,
                diversity=0.5,
                candidate_pool_size=invalid_pool,
            )
    with pytest.raises(ValueError, match="num_iterations"):
        graphem_seed_selection(embedder, 1, num_iterations=True)


def test_cuda_source_uses_compact_typed_queue_and_target_workspace_is_bounded():
    assert "atomicOr(word, bit)" in _IC_KERNEL_SOURCE
    assert "ic_expand_i32_q32" in _IC_KERNEL_SOURCE
    assert "ic_expand_i64_q32" in _IC_KERNEL_SOURCE
    assert "ic_expand_i32_q64" in _IC_KERNEL_SOURCE
    assert "ic_expand_i64_q64" in _IC_KERNEL_SOURCE
    assert "unsigned char* frontier" not in _IC_KERNEL_SOURCE
    assert _queue_workspace_bytes(10_000_000, 128) < 5 * 2**30


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


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("k", 1.5),
        ("k", True),
        ("k", "1"),
        ("candidate_pool_size", 1.5),
        ("candidate_pool_size", True),
        ("candidate_pool_size", "2"),
    ],
)
def test_degree_discount_integer_inputs_are_strict(keyword, value):
    graph = _undirected_csr(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    arguments = {"k": 1, "candidate_pool_size": None}
    arguments[keyword] = value
    with pytest.raises(ValueError, match=keyword):
        degree_discount_seed_selection(graph, **arguments)


@requires_cupy_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("probability", [0.0, 0.13, 1.0])
def test_ic_cupy_samples_exactly_match_cpu(probability):
    graph = _directed_csr(
        9,
        [
            (0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5),
            (4, 5), (5, 6), (5, 7), (6, 8), (7, 8),
        ],
    )
    cpu_result = estimate_independent_cascade(
        graph, [0, 4], p=probability, n_simulations=37,
        random_seed=23, backend="cpu",
    )
    gpu_result = estimate_independent_cascade(
        _device_csr(graph), [0, 4], p=probability, n_simulations=37,
        random_seed=23, backend="cupy", batch_size=7,
    )
    assert gpu_result == cpu_result


@requires_cupy_gpu
@pytest.mark.gpu
def test_ic_cupy_is_batch_invariant_repeatable_and_monotone():
    graph = _device_csr(
        _undirected_csr(
            12,
            [
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5),
                (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11),
            ],
        )
    )
    results = [
        estimate_independent_cascade(
            graph, [0], p=0.29, n_simulations=41, random_seed=101,
            backend="cupy", batch_size=batch_size,
        )
        for batch_size in (1, 7, 41)
    ]
    repeated = estimate_independent_cascade(
        graph, [0], p=0.29, n_simulations=41, random_seed=101,
        backend="cupy", batch_size=7,
    )
    assert results[0] == results[1] == results[2] == repeated
    superset = estimate_independent_cascade(
        graph, [0, 8], p=0.29, n_simulations=41, random_seed=101,
        backend="cupy", batch_size=6,
    )
    assert np.all(np.asarray(superset.samples) >= np.asarray(results[0].samples))


@requires_cupy_gpu
@pytest.mark.gpu
def test_ic_cupy_converts_csc_and_canonicalizes_duplicates_and_zeros():
    noncanonical = sp.csr_matrix(
        (
            np.asarray([1, 1, 0, 1, 1, 1], dtype=np.float32),
            np.asarray([2, 1, 3, 2, 2, 3], dtype=np.int32),
            np.asarray([0, 3, 5, 6, 6], dtype=np.int32),
        ),
        shape=(4, 4),
    )
    expected = estimate_independent_cascade(
        noncanonical, [0], p=0.43, n_simulations=31,
        random_seed=71, backend="cpu",
    )
    device_noncanonical = _device_csr(noncanonical)
    actual_csr = estimate_independent_cascade(
        device_noncanonical, [0], p=0.43, n_simulations=31,
        random_seed=71, backend="cupy", batch_size=5,
    )
    device_csc = cpx_sparse.csc_matrix(device_noncanonical)
    actual_csc = estimate_independent_cascade(
        device_csc, [0], p=0.43, n_simulations=31,
        random_seed=71, backend="cupy", batch_size=5,
    )
    assert actual_csr == expected
    assert actual_csc == expected


@requires_cupy_gpu
@pytest.mark.gpu
def test_ic_cupy_rejects_rectangular_sparse_graph_before_kernel_launch():
    graph = cpx_sparse.csr_matrix((3, 4), dtype=cp.float32)
    with pytest.raises(ValueError, match="square"):
        estimate_independent_cascade(
            graph, [0], p=0.0, n_simulations=2, backend="cupy"
        )


@requires_cupy_gpu
@pytest.mark.gpu
def test_ic_cupy_host_int64_indices_and_degree_discount_match_cpu():
    graph = _undirected_csr(10, [(0, node) for node in range(1, 10)])
    graph.indices = graph.indices.astype(np.int64)
    graph.indptr = graph.indptr.astype(np.int64)
    device_graph = _device_csr(graph)
    assert degree_discount_seed_selection(device_graph, 4, p=0.17) == (
        degree_discount_seed_selection(graph, 4, p=0.17)
    )

    cpu_result = estimate_independent_cascade(
        graph, [0], p=0.17, n_simulations=29, random_seed=3, backend="cpu"
    )
    gpu_result = estimate_independent_cascade(
        graph, [0], p=0.17, n_simulations=29,
        random_seed=3, backend="cupy", batch_size=4,
    )
    assert gpu_result == cpu_result


@requires_cupy_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("bad_target", [-1, 4])
def test_ic_cupy_rejects_invalid_device_csr_indices(bad_target):
    graph = _device_csr(_directed_csr(4, [(0, 1), (1, 2)]))
    graph.indices[0] = bad_target
    with pytest.raises(ValueError, match="invalid CSR"):
        estimate_independent_cascade(
            graph, [0], p=0.2, n_simulations=2, backend="cupy"
        )


@requires_cupy_gpu
@pytest.mark.gpu
def test_ic_cupy_kernel_warmup_is_reusable():
    warm_independent_cascade_kernels()
    warm_independent_cascade_kernels()


@requires_cupy_gpu
@pytest.mark.gpu
def test_ic_cupy_forced_q64_queue_matches_cpu(monkeypatch):
    graph = _directed_csr(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    expected = estimate_independent_cascade(
        graph, [0], p=0.41, n_simulations=7, random_seed=29, backend="cpu"
    )
    monkeypatch.setattr(influence_module, "_UINT32_STATE_CAPACITY", 1)
    actual = estimate_independent_cascade(
        graph,
        [0],
        p=0.41,
        n_simulations=7,
        random_seed=29,
        backend="cupy",
        batch_size=3,
    )
    assert actual == expected
