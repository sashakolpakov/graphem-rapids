"""CPU contract checks for the single GraphEm implementation."""

# pylint: disable=missing-function-docstring

import inspect
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

try:
    import torch
except ImportError:  # pragma: no cover - depends on the test environment
    torch = None

import graphem_rapids
import graphem_rapids.embedder as implementation
from graphem_rapids.embedder import GraphEmbedder


pytestmark = pytest.mark.fast


def test_public_api_exposes_one_embedder():
    assert graphem_rapids.GraphEmbedder is GraphEmbedder
    assert "GraphEmbedder" in graphem_rapids.__all__
    assert not hasattr(graphem_rapids, "create_graphem")
    assert all("PyTorch" not in name and "CuVS" not in name for name in graphem_rapids.__all__)


def test_constructor_has_no_discarded_control_parameters():
    parameters = inspect.signature(GraphEmbedder).parameters
    discarded = {
        "backend",
        "force_mode",
        "learning_rate",
        "max_displacement",
        "initialization",
        "index_type",
        "midpoint_reference_size",
        "intersection_interval",
    }
    assert discarded.isdisjoint(parameters)


def test_missing_gpu_stack_is_an_explicit_error(monkeypatch):
    marker = ImportError("missing test dependency")
    monkeypatch.setattr(implementation, "_GPU_IMPORT_ERROR", marker)
    with pytest.raises(
        ImportError, match="requires CuPy, cupyx, and cuVS"
    ):
        GraphEmbedder(np.eye(4, dtype=np.float32))


def test_midpoint_query_batch_size_has_a_hard_canonical_bound():
    parameter = inspect.signature(GraphEmbedder).parameters[
        "midpoint_query_batch_size"
    ]
    assert parameter.default == implementation.MIDPOINT_QUERY_BATCH_SIZE_BOUND
    assert implementation._bounded_midpoint_query_batch_size(np.int64(1)) == 1
    assert (
        implementation._bounded_midpoint_query_batch_size(
            implementation.MIDPOINT_QUERY_BATCH_SIZE_BOUND
        )
        == implementation.MIDPOINT_QUERY_BATCH_SIZE_BOUND
    )
    for invalid in (True, 0, -1, 1.5, "4"):
        with pytest.raises((TypeError, ValueError)):
            implementation._bounded_midpoint_query_batch_size(invalid)
    with pytest.raises(ValueError, match="canonical bound"):
        implementation._bounded_midpoint_query_batch_size(
            implementation.MIDPOINT_QUERY_BATCH_SIZE_BOUND + 1
        )


class _ExactCpuBruteForce:
    """GPU-free exact-search oracle with adversarial cutoff-tie ordering."""

    calls = []

    @staticmethod
    def build(midpoints, metric):
        assert metric == "sqeuclidean"
        return np.asarray(midpoints, dtype=np.float32)

    @classmethod
    def search(cls, index, queries, width):
        queries = np.asarray(queries, dtype=np.float32)
        references = np.asarray(index, dtype=np.float32)
        deltas = queries[:, None, :] - references[None, :, :]
        squared_distances = np.sum(deltas * deltas, axis=2, dtype=np.float32)
        raw_distances = squared_distances.copy()
        raw_distances[squared_distances == 0] = np.float32(-1.0e-7)
        global_edge_ids = np.arange(references.shape[0], dtype=np.int64)
        neighbor_rows = []
        distance_rows = []
        for row in raw_distances:
            # cuVS promises distance ordering, not an edge-ID order at ties.
            # Deliberately reverse tied IDs so GraphEm must canonicalize them.
            order = np.lexsort((-global_edge_ids, row))[:width]
            neighbor_rows.append(global_edge_ids[order])
            distance_rows.append(row[order])
        cls.calls.append((int(queries.shape[0]), int(width)))
        return (
            np.asarray(distance_rows, dtype=np.float32),
            np.asarray(neighbor_rows, dtype=np.int64),
        )


def _run_cpu_midpoint_case(midpoints, sampled_edge_ids, n_neighbors, batch_size):
    midpoints = np.asarray(midpoints, dtype=np.float32)
    n_edges = int(midpoints.shape[0])
    embedder = object.__new__(GraphEmbedder)
    embedder.positions = np.repeat(midpoints, 2, axis=0)
    embedder.edges = np.arange(2 * n_edges, dtype=np.int64).reshape(
        n_edges, 2
    )
    embedder.sampled_edge_ids = np.asarray(sampled_edge_ids, dtype=np.int64)
    embedder.sample_size = int(embedder.sampled_edge_ids.size)
    embedder.n_edges = n_edges
    embedder.n_neighbors = n_neighbors
    embedder._midpoint_query_batch_size = batch_size
    embedder._midpoint_width_histogram = {}
    embedder._midpoint_negative_distance_repairs = 0
    embedder._midpoint_search_call_count = 0
    embedder._midpoint_search_call_width_histogram = {}
    embedder._midpoint_query_batch_histogram = {}
    embedder._midpoint_search_peak_device_bytes = None
    _ExactCpuBruteForce.calls = []
    result = embedder._midpoint_neighbors()
    return result, embedder, list(_ExactCpuBruteForce.calls)


def _run_cpu_midpoint_batch_fixture(batch_size):
    midpoints = np.array(
        [
            [1000.0, 1000.0],
            [1000.0, 1000.0],
            [1000.0, 1000.0],
            [1000.0, 1000.0],
            [1000.0, 1000.0],
            [1001.0, 1000.0],
            [1001.0, 1000.0],
            [1002.0, 1000.0],
            [1003.0, 1000.0],
            [1004.0, 1000.0],
        ],
        dtype=np.float32,
    )
    return _run_cpu_midpoint_case(
        midpoints,
        np.array([4, 5, 8, 9, 1], dtype=np.int64),
        2,
        batch_size,
    )


def test_query_batching_is_exact_across_ties_expansion_and_negative_repair(
    monkeypatch,
):
    monkeypatch.setattr(implementation, "cp", np)
    monkeypatch.setattr(implementation, "brute_force", _ExactCpuBruteForce)

    unbatched, unbatched_embedder, unbatched_calls = (
        _run_cpu_midpoint_batch_fixture(5)
    )
    batched_two, batched_two_embedder, batched_two_calls = (
        _run_cpu_midpoint_batch_fixture(2)
    )
    batched_one, batched_one_embedder, batched_one_calls = (
        _run_cpu_midpoint_batch_fixture(1)
    )

    expected = np.array(
        [[0, 1], [6, 0], [7, 9], [8, 7], [0, 2]], dtype=np.int64
    )
    np.testing.assert_array_equal(unbatched, expected)
    np.testing.assert_array_equal(batched_two, unbatched)
    np.testing.assert_array_equal(batched_one, unbatched)
    for embedder in (
        unbatched_embedder,
        batched_two_embedder,
        batched_one_embedder,
    ):
        assert embedder._midpoint_width_histogram == {4: 2, 8: 2, 10: 1}
        assert embedder._midpoint_negative_distance_repairs == 26

    assert unbatched_calls == [(5, 4), (3, 8), (1, 10)]
    assert batched_two_calls == [
        (2, 4),
        (2, 4),
        (1, 4),
        (2, 8),
        (1, 8),
        (1, 10),
    ]
    assert batched_one_calls == [(1, 4)] * 5 + [(1, 8)] * 3 + [(1, 10)]
    assert max(size for size, _width in batched_two_calls) == 2
    assert batched_two_embedder._midpoint_search_call_count == 6
    assert batched_two_embedder._midpoint_search_call_width_histogram == {
        4: 3,
        8: 2,
        10: 1,
    }
    assert batched_two_embedder._midpoint_query_batch_histogram == {1: 3, 2: 3}


@pytest.mark.parametrize("seed", [0, 1, 7, 19])
def test_randomized_query_batches_match_an_independent_exact_oracle(
    monkeypatch, seed
):
    monkeypatch.setattr(implementation, "cp", np)
    monkeypatch.setattr(implementation, "brute_force", _ExactCpuBruteForce)
    generator = np.random.default_rng(seed)
    midpoints = np.float32(1000.0) + generator.integers(
        -4, 5, size=(37, 3)
    ).astype(np.float32)
    sampled_edge_ids = generator.choice(37, size=17, replace=False)
    global_edge_ids = np.arange(37, dtype=np.int64)
    expected_rows = []
    for query_edge_id in sampled_edge_ids:
        deltas = midpoints - midpoints[query_edge_id]
        distances = np.sum(deltas * deltas, axis=1, dtype=np.float32)
        order = np.lexsort((global_edge_ids, distances))
        expected_rows.append(order[order != query_edge_id][:5])
    expected = np.asarray(expected_rows, dtype=np.int64)

    reference, reference_embedder, _reference_calls = _run_cpu_midpoint_case(
        midpoints, sampled_edge_ids, 5, 17
    )
    np.testing.assert_array_equal(reference, expected)
    for batch_size in (1, 3, 7):
        observed, embedder, calls = _run_cpu_midpoint_case(
            midpoints, sampled_edge_ids, 5, batch_size
        )
        np.testing.assert_array_equal(observed, reference)
        assert max(size for size, _width in calls) <= batch_size
        assert (
            embedder._midpoint_width_histogram
            == reference_embedder._midpoint_width_histogram
        )
        assert (
            embedder._midpoint_negative_distance_repairs
            == reference_embedder._midpoint_negative_distance_repairs
        )


def test_midpoint_search_revalidates_the_batch_bound_before_submission(
    monkeypatch,
):
    monkeypatch.setattr(implementation, "cp", np)
    monkeypatch.setattr(implementation, "brute_force", _ExactCpuBruteForce)
    with pytest.raises(ValueError, match="canonical bound"):
        _run_cpu_midpoint_batch_fixture(
            implementation.MIDPOINT_QUERY_BATCH_SIZE_BOUND + 1
        )
    assert not _ExactCpuBruteForce.calls


def test_query_batch_diagnostics_receipt_the_executed_policy(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    monkeypatch.setattr(implementation, "brute_force", _ExactCpuBruteForce)
    monkeypatch.setattr(np, "asnumpy", np.asarray, raising=False)
    _result, embedder, _calls = _run_cpu_midpoint_batch_fixture(2)
    embedder.n = 20
    embedder.degrees = np.ones(embedder.n, dtype=np.float32)
    embedder.n_components = 2
    embedder.L_min = 1.0
    embedder.k_attr = 0.2
    embedder.k_inter = 0.5
    embedder.seed = 0
    embedder._spectral_device_requested = "cuda"
    embedder._iteration = 1
    embedder.timings = {}
    embedder._spectral_diagnostics = {"normalized_laplacian": "fixture"}
    embedder._spectral_eigenvalues = []
    embedder._spectral_max_residual_norm_ratio = 0.0

    diagnostics = embedder.get_diagnostics()

    assert diagnostics["configuration"]["midpoint_query_batch_size"] == 2
    assert diagnostics["midpoint_query_batch_policy"].endswith("at-most-64-v1")
    assert diagnostics["midpoint_query_batch_size_bound"] == 64
    assert diagnostics["midpoint_query_batch_size_effective"] == 2
    assert diagnostics["midpoint_neighbor_id_validation"] == (
        "rowwise-unique-global-edge-id-before-negative-repair-v1"
    )
    assert diagnostics["midpoint_search_call_count"] == 6
    assert diagnostics["midpoint_search_call_width_histogram"] == {
        "4": 3,
        "8": 2,
        "10": 1,
    }
    assert diagnostics["midpoint_search_query_batch_histogram"] == {
        "1": 3,
        "2": 3,
    }
    assert diagnostics["midpoint_search_width_histogram"] == {
        "4": 2,
        "8": 2,
        "10": 1,
    }
    assert diagnostics["midpoint_search_peak_device_bytes"] is None
    assert diagnostics["midpoint_search_peak_device_bytes_scope"].startswith(
        "cuda-memgetinfo"
    )


def test_midpoint_memory_receipt_retains_the_highest_checkpoint(monkeypatch):
    observations = iter((100, 90, 120, 110))
    monkeypatch.setattr(
        GraphEmbedder,
        "_current_device_memory_used_bytes",
        staticmethod(lambda: next(observations)),
    )
    embedder = object.__new__(GraphEmbedder)
    embedder._midpoint_search_peak_device_bytes = None
    for _ in range(4):
        embedder._observe_midpoint_search_device_memory()
    assert embedder._midpoint_search_peak_device_bytes == 120


def test_identity_based_self_removal_handles_ties(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    raw = np.array([[4, 0, 2, 9], [8, 3, 5, 1]], dtype=np.int64)
    distances = np.array(
        [[0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3]], dtype=np.float32
    )
    queries = np.array([4, 8], dtype=np.int64)
    actual = GraphEmbedder._compact_nonself_neighbors(raw, distances, queries, 3)
    np.testing.assert_array_equal(actual, np.array([[0, 2, 9], [3, 5, 1]]))


def test_identity_based_self_removal_keeps_column_zero_when_self_absent(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    raw = np.array([[7, 2, 5, 3]], dtype=np.int64)
    distances = np.array([[0.0, 0.1, 0.2, 0.3]], dtype=np.float32)
    actual = GraphEmbedder._compact_nonself_neighbors(
        raw, distances, np.array([11], dtype=np.int64), 3
    )
    np.testing.assert_array_equal(actual, np.array([[7, 2, 5]]))


def test_identity_based_self_removal_fails_when_row_is_short(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    raw = np.array([[4, 4, 2]], dtype=np.int64)
    distances = np.array([[0.0, 0.0, 0.1]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="fewer than the required"):
        GraphEmbedder._compact_nonself_neighbors(
            raw, distances, np.array([4], dtype=np.int64), 2
        )


def test_midpoint_cutoff_tie_uses_global_edge_id(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    raw = np.array([[4, 1, 2, 3, 5]], dtype=np.int64)
    distances = np.array([[0.0, 0.1, 0.2, 0.2, 0.4]], dtype=np.float32)
    actual = GraphEmbedder._compact_nonself_neighbors(
        raw, distances, np.array([4], dtype=np.int64), 2
    )
    np.testing.assert_array_equal(actual, np.array([[1, 2]], dtype=np.int64))


def test_batched_midpoint_order_is_distance_then_global_id(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    raw = np.array([[9, 4, 3, 1, 2], [8, 7, 6, 2, 5]], dtype=np.int64)
    distances = np.array(
        [[0.2, 0.0, 0.1, 0.1, 0.1], [0.0, 0.3, 0.2, 0.2, 0.1]],
        dtype=np.float32,
    )
    actual = GraphEmbedder._compact_nonself_neighbors(
        raw, distances, np.array([4, 8], dtype=np.int64), 3
    )
    np.testing.assert_array_equal(
        actual,
        np.array([[1, 2, 3], [5, 2, 6]], dtype=np.int64),
    )


def test_bounded_negative_squared_distance_is_recomputed_directly(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    queries = np.array([[1.0, 1.0]], dtype=np.float32)
    references = np.array(
        [[np.nextafter(np.float32(1.0), np.float32(2.0)), 1.0]],
        dtype=np.float32,
    )
    raw = np.array([[-1.0e-7]], dtype=np.float32)
    repaired, count = GraphEmbedder._repair_negative_squared_distances(
        raw,
        np.array([[0]], dtype=np.int64),
        queries,
        references,
    )
    expected = np.sum(
        (queries[0] - references[0]) ** 2,
        dtype=np.float32,
    )
    assert count == 1
    assert repaired.dtype == np.float32
    assert repaired[0, 0] == expected
    assert raw[0, 0] < 0


def test_material_negative_squared_distance_exceeding_bound_fails(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    queries = np.array([[1.0, 1.0]], dtype=np.float32)
    references = queries.copy()
    with pytest.raises(FloatingPointError, match="exceeds the float32 error bound"):
        GraphEmbedder._repair_negative_squared_distances(
            np.array([[-1.0e-2]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
            queries,
            references,
        )


def test_midpoint_search_result_rejects_nonfinite_distance(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    result = (
        np.array([[0.0, np.nan]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.int64),
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        GraphEmbedder._search_result_arrays(result)


@pytest.mark.parametrize(
    "neighbor_ids",
    [
        [[0, 1, 1, 2]],
        [[0, 1, 2, 1]],
        [[0, 1, 0, 2]],
        [[0, 1, 2, 3], [4, 5, 4, 6]],
    ],
)
def test_rowwise_neighbor_id_validation_rejects_any_duplicate_position(
    monkeypatch, neighbor_ids
):
    monkeypatch.setattr(implementation, "cp", np)
    with pytest.raises(ValueError, match="duplicate global edge IDs"):
        GraphEmbedder._validate_unique_global_neighbor_ids(
            np.asarray(neighbor_ids, dtype=np.int64)
        )


def test_rowwise_neighbor_id_validation_accepts_unique_rows(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    GraphEmbedder._validate_unique_global_neighbor_ids(
        np.array([[3, 0, 2, 1], [7, 4, 6, 5]], dtype=np.int64)
    )


def test_duplicate_neighbor_ids_fail_before_negative_repair_and_keep_receipts(
    monkeypatch,
):
    monkeypatch.setattr(implementation, "cp", np)

    class DuplicateIdBruteForce:
        """Return one malformed exact-search row with repeated edge ID 1."""

        calls = []

        @staticmethod
        def build(midpoints, metric):
            assert metric == "sqeuclidean"
            return midpoints

        @classmethod
        def search(cls, _index, queries, width):
            cls.calls.append((int(queries.shape[0]), int(width)))
            assert width == 4
            return (
                np.array(
                    [[-4.0e-7, -3.0e-7, -2.0e-7, 1.0]],
                    dtype=np.float32,
                ),
                np.array([[0, 1, 1, 2]], dtype=np.int64),
            )

    monkeypatch.setattr(implementation, "brute_force", DuplicateIdBruteForce)
    embedder = object.__new__(GraphEmbedder)
    midpoints = np.array(
        [
            [1000.0, 1000.0],
            [1001.0, 1000.0],
            [1002.0, 1000.0],
            [1003.0, 1000.0],
        ],
        dtype=np.float32,
    )
    embedder.positions = np.repeat(midpoints, 2, axis=0)
    embedder.edges = np.arange(8, dtype=np.int64).reshape(4, 2)
    embedder.sampled_edge_ids = np.array([0], dtype=np.int64)
    embedder.sample_size = 1
    embedder.n_edges = 4
    embedder.n_neighbors = 2
    embedder._midpoint_query_batch_size = 64
    embedder._midpoint_width_histogram = {}
    embedder._midpoint_negative_distance_repairs = 0
    embedder._midpoint_search_call_count = 0
    embedder._midpoint_search_call_width_histogram = {}
    embedder._midpoint_query_batch_histogram = {}
    embedder._midpoint_search_peak_device_bytes = None

    with pytest.raises(ValueError, match="duplicate global edge IDs"):
        embedder._midpoint_neighbors()

    assert DuplicateIdBruteForce.calls == [(1, 4)]
    assert embedder._midpoint_negative_distance_repairs == 0
    assert embedder._midpoint_search_call_count == 1
    assert embedder._midpoint_search_call_width_histogram == {4: 1}
    assert embedder._midpoint_query_batch_histogram == {1: 1}
    assert not embedder._midpoint_width_histogram


def test_negative_repair_preserves_raw_boundary_completeness_proof(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)

    class FakeBruteForce:
        """Expose a repaired ordering that cannot prove width-four completeness."""

        widths = []

        @staticmethod
        def build(midpoints, metric):
            assert metric == "sqeuclidean"
            return midpoints

        @classmethod
        def search(cls, _index, _queries, width):
            cls.widths.append(width)
            if width == 4:
                return (
                    np.array(
                        [[-4.0e-7, -3.0e-7, -2.0e-7, -1.0e-7]],
                        dtype=np.float32,
                    ),
                    np.array([[4, 0, 1, 5]], dtype=np.int64),
                )
            assert width == 6
            return (
                np.array(
                    [[0.0, 1.0e-8, 2.0e-8, 3.0e-8, 4.0e-8, 5.0e-8]],
                    dtype=np.float32,
                ),
                np.array([[4, 0, 1, 2, 3, 5]], dtype=np.int64),
            )

    monkeypatch.setattr(implementation, "brute_force", FakeBruteForce)
    embedder = object.__new__(GraphEmbedder)
    midpoint_x = np.array(
        [
            np.nextafter(np.float32(1000.0), np.float32(2000.0)),
            np.float32(1000.0001220703125),
            np.float32(1000.0001831054688),
            np.float32(1000.000244140625),
            np.float32(1000.0),
            np.float32(1000.001),
        ],
        dtype=np.float32,
    )
    midpoints = np.column_stack(
        (midpoint_x, np.full(6, np.float32(1000.0), dtype=np.float32))
    )
    embedder.positions = np.repeat(midpoints, 2, axis=0)
    embedder.edges = np.arange(12, dtype=np.int64).reshape(6, 2)
    embedder.sampled_edge_ids = np.array([4], dtype=np.int64)
    embedder.sample_size = 1
    embedder.n_edges = 6
    embedder.n_neighbors = 2
    embedder._midpoint_width_histogram = {}
    embedder._midpoint_negative_distance_repairs = 0

    actual = embedder._midpoint_neighbors()

    np.testing.assert_array_equal(actual, np.array([[0, 1]], dtype=np.int64))
    assert FakeBruteForce.widths == [4, 6]
    assert embedder._midpoint_width_histogram == {6: 1}
    assert embedder._midpoint_negative_distance_repairs == 4


def test_midpoint_search_doubles_until_full_cutoff_tie_is_observed(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)

    class FakeBruteForce:
        """Deterministic cuVS stand-in that exposes the queried widths."""

        widths = []

        @staticmethod
        def build(midpoints, metric):
            assert metric == "sqeuclidean"
            return midpoints

        @classmethod
        def search(cls, _index, _queries, width):
            cls.widths.append(width)
            if width == 4:
                return (
                    np.array([[0.0, 0.2, 0.2, 0.2]], dtype=np.float32),
                    np.array([[4, 5, 3, 2]], dtype=np.int64),
                )
            assert width == 6
            return (
                np.array([[0.0, 0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32),
                np.array([[4, 5, 3, 2, 1, 0]], dtype=np.int64),
            )

    monkeypatch.setattr(implementation, "brute_force", FakeBruteForce)
    embedder = object.__new__(GraphEmbedder)
    embedder.positions = np.arange(14, dtype=np.float32).reshape(7, 2)
    embedder.edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=np.int64
    )
    embedder.sampled_edge_ids = np.array([4], dtype=np.int64)
    embedder.sample_size = 1
    embedder.n_edges = 6
    embedder.n_neighbors = 2
    embedder._midpoint_width_histogram = {}
    embedder._midpoint_negative_distance_repairs = 0

    actual = embedder._midpoint_neighbors()

    np.testing.assert_array_equal(actual, np.array([[0, 1]], dtype=np.int64))
    assert FakeBruteForce.widths == [4, 6]
    assert embedder._midpoint_width_histogram == {6: 1}


def test_query_edge_sampling_is_uniform_without_replacement():
    expected = np.array([5, 7, 4, 2, 3], dtype=np.int64)
    np.testing.assert_array_equal(
        GraphEmbedder._uniform_query_edge_ids(11, 5, 0), expected
    )
    observed_subsets = {
        tuple(sorted(GraphEmbedder._uniform_query_edge_ids(8, 4, seed)))
        for seed in range(4096)
    }
    assert len(observed_subsets) == 70


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable")
def test_torch_spectral_embedding_is_repeatable_on_degenerate_graph():
    left_size = 50
    right_size = 100
    sources = np.repeat(np.arange(left_size, dtype=np.int64), right_size)
    targets = np.tile(
        np.arange(left_size, left_size + right_size, dtype=np.int64), left_size
    )
    edges = np.column_stack((sources, targets))

    with pytest.warns(RuntimeWarning, match="using CPU"):
        first, first_diagnostics = GraphEmbedder._torch_spectral_embedding(
            edges, left_size + right_size, 4, 20250608, device="cpu"
        )
    with pytest.warns(RuntimeWarning, match="using CPU"):
        second, second_diagnostics = GraphEmbedder._torch_spectral_embedding(
            edges, left_size + right_size, 4, 20250608, device="cpu"
        )

    np.testing.assert_allclose(first.numpy(), second.numpy(), atol=1.0e-7, rtol=1.0e-7)
    np.testing.assert_allclose(
        first @ first.mT, second @ second.mT, atol=1.0e-6, rtol=1.0e-6
    )
    np.testing.assert_allclose(
        first_diagnostics["eigenvalues"],
        [0.0, 1.0, 1.0, 1.0, 1.0],
        atol=1.0e-8,
    )
    assert (
        first_diagnostics["maximum_eigenpair_residual_norm_ratio"] <= 1.0e-8
    )
    assert first_diagnostics["orthogonality_error"] <= 1.0e-8
    assert first_diagnostics["output_eigenpair_count"] == 5
    assert first_diagnostics["solver_block_width"] == 16
    assert first_diagnostics["maximum_iterations"] == 5000
    assert first_diagnostics["start_sha256"] == second_diagnostics["start_sha256"]
    repeat_metrics = GraphEmbedder._torch_subspace_repeat_metrics(first, second)
    assert repeat_metrics["projector_frobenius_distance"] <= 1.0e-5
    assert repeat_metrics["largest_principal_angle_radians"] <= 1.0e-5


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable")
def test_torch_spectral_embedding_preserves_components_and_isolate():
    first_component = np.column_stack(
        (np.arange(19, dtype=np.int64), np.arange(1, 20, dtype=np.int64))
    )
    second_component = np.array(
        [[20, 21], [21, 22], [22, 23], [23, 24], [24, 20]], dtype=np.int64
    )
    edges = np.vstack((first_component, second_component))

    with pytest.warns(RuntimeWarning, match="using CPU"):
        positions, diagnostics = GraphEmbedder._torch_spectral_embedding(
            edges, 26, 3, 7, device="cpu"
        )

    assert tuple(positions.shape) == (26, 3)
    assert positions.dtype == torch.float32
    assert bool(torch.isfinite(positions).all())
    assert sum(abs(value) < 1.0e-8 for value in diagnostics["eigenvalues"]) >= 3
    assert diagnostics["maximum_eigenpair_residual_norm_ratio"] <= 1.0e-8
    assert diagnostics["solver_block_width"] == 8


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable")
def test_torch_spectral_cpu_and_cuda_use_the_same_function():
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA is unavailable")
    edges = np.column_stack(
        (np.arange(11, dtype=np.int64), np.arange(1, 12, dtype=np.int64))
    )
    with pytest.warns(RuntimeWarning, match="using CPU"):
        cpu_positions, cpu_diagnostics = GraphEmbedder._torch_spectral_embedding(
            edges, 12, 2, 19, device="cpu"
        )
    cuda_positions, cuda_diagnostics = GraphEmbedder._torch_spectral_embedding(
        edges, 12, 2, 19, device="cuda"
    )
    assert cpu_diagnostics["solver_block_width"] == 4
    assert cuda_diagnostics["solver_block_width"] == 4
    np.testing.assert_allclose(
        cpu_diagnostics["eigenvalues"],
        cuda_diagnostics["eigenvalues"],
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    np.testing.assert_allclose(
        (cpu_positions @ cpu_positions.mT).numpy(),
        (cuda_positions @ cuda_positions.mT).cpu().numpy(),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable")
def test_torch_spectral_device_selection_is_loud_and_fail_closed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        implementation._resolve_spectral_device("cuda")
    with pytest.warns(RuntimeWarning, match="CUDA is unavailable"):
        selected, requested, reason = implementation._resolve_spectral_device("auto")
    assert selected.type == "cpu"
    assert requested == "auto"
    assert "False" in reason
    with pytest.warns(RuntimeWarning, match="explicitly selected"):
        selected, requested, reason = implementation._resolve_spectral_device("cpu")
    assert selected.type == "cpu"
    assert requested == "cpu"
    assert reason == "CPU was explicitly selected"


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable")
def test_torch_spectral_rejects_graphs_outside_lobpcg_domain():
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    with pytest.warns(RuntimeWarning, match="using CPU"):
        with pytest.raises(ValueError, match="n_vertices >= 3"):
            GraphEmbedder._torch_spectral_embedding(
                edges, 4, 2, 0, device="cpu"
            )


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable")
def test_torch_spectral_rejects_bad_solver_output(monkeypatch):
    edges = np.column_stack(
        (np.arange(63, dtype=np.int64), np.arange(1, 64, dtype=np.int64))
    )

    def bad_lobpcg(matrix, *, k, X, tracker, **_kwargs):
        full_width = X.shape[1]
        full_values = torch.zeros(full_width, dtype=torch.float64)
        full_vectors = torch.ones(
            (matrix.shape[0], full_width), dtype=torch.float64
        )

        class Worker:
            """Synthetic nonconverged Torch LOBPCG tracker state."""

            ivars = {"istep": 1, "converged_count": 0}
            E = full_values
            X = full_vectors

        tracker(Worker())
        return (
            full_values[:k],
            full_vectors[:, :k],
        )

    monkeypatch.setattr(torch, "lobpcg", bad_lobpcg)
    with pytest.warns(RuntimeWarning, match="using CPU"):
        with pytest.raises(RuntimeError, match="residual exceeds"):
            GraphEmbedder._torch_spectral_embedding(
                edges, 64, 2, 19, device="cpu"
            )


@pytest.mark.parametrize("sparse_format", ["csr", "coo"])
def test_host_adjacency_rejects_duplicate_entries_even_when_weights_sum_to_one(
    sparse_format,
):
    adjacency = sp.csr_matrix(
        (
            np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            np.array([1, 1, 0, 0], dtype=np.int32),
            np.array([0, 2, 4], dtype=np.int32),
        ),
        shape=(2, 2),
    )
    if sparse_format == "coo":
        adjacency = sp.coo_matrix(
            (
                np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
                (
                    np.array([0, 0, 1, 1], dtype=np.int32),
                    np.array([1, 1, 0, 0], dtype=np.int32),
                ),
            ),
            shape=(2, 2),
        )
    with pytest.raises(ValueError, match="duplicate entries"):
        GraphEmbedder._validate_host_adjacency(adjacency)


def test_xy_crossing_predicate_accepts_higher_dimensional_geometry():
    p1 = np.array([[-1.0, -1.0, 4.0, -2.0]], dtype=np.float32)
    p2 = np.array([[1.0, 1.0, 5.0, 3.0]], dtype=np.float32)
    q1 = np.array([[-1.0, 1.0, -7.0, 8.0]], dtype=np.float32)
    q2 = np.array([[1.0, -1.0, 6.0, -4.0]], dtype=np.float32)
    assert GraphEmbedder._strict_xy_crossing(p1, p2, q1, q2).tolist() == [True]


@pytest.mark.parametrize(
    "q1,q2",
    [
        ((1.0, 1.0), (2.0, 2.0)),
        ((-0.5, -0.5), (0.5, 0.5)),
        ((1.0, -1.0), (2.0, -2.0)),
    ],
)
def test_crossing_predicate_rejects_touching_collinear_and_separate(q1, q2):
    p1 = np.array([[-1.0, -1.0]], dtype=np.float32)
    p2 = np.array([[1.0, 1.0]], dtype=np.float32)
    first = np.array([q1], dtype=np.float32)
    second = np.array([q2], dtype=np.float32)
    assert GraphEmbedder._strict_xy_crossing(p1, p2, first, second).tolist() == [False]


def test_all_coordinate_centroid_repulsion_matches_cpu_oracle(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    embedder = object.__new__(GraphEmbedder)
    embedder.positions = np.array(
        [
            [-1.0, -1.0, 4.0],
            [1.0, 1.0, 5.0],
            [-1.0, 1.0, -7.0],
            [1.0, -1.0, 6.0],
        ],
        dtype=np.float32,
    )
    embedder.edges = np.array([[0, 1], [2, 3]], dtype=np.int64)
    embedder.sampled_edge_ids = np.array([0], dtype=np.int64)
    embedder.n_neighbors = 1
    embedder.k_inter = 0.5

    def sequential_cpu_sum(endpoint_ids, contributions, forces):
        ordered, starts, ends, vertices = GraphEmbedder._ordered_endpoint_segments(
            endpoint_ids, contributions
        )
        for start, end, vertex in zip(starts, ends, vertices):
            for row in range(int(start), int(end)):
                forces[int(vertex)] += ordered[row]
        return forces

    monkeypatch.setattr(
        embedder, "_reduce_endpoint_contributions", sequential_cpu_sum
    )
    forces = embedder._intersection_forces(np.array([[1]], dtype=np.int64))
    centroid = embedder.positions.mean(axis=0)
    expected = np.empty_like(embedder.positions)
    for index, point in enumerate(embedder.positions):
        displacement = point - centroid
        denominator = np.linalg.norm(displacement) + 1.0e-6
        expected[index] = 0.5 * displacement / (denominator * denominator)
    np.testing.assert_allclose(forces, expected, rtol=2.0e-6, atol=2.0e-7)
    assert np.any(np.abs(forces[:, 2]) > 0)


def test_population_axis_normalization(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    positions = np.array([[1.0, 10.0], [2.0, 12.0], [7.0, 19.0]], dtype=np.float32)
    actual = GraphEmbedder._normalize_positions(positions)
    expected = (positions - positions.mean(axis=0, keepdims=True)) / (
        positions.std(axis=0, ddof=0, keepdims=True) + 1.0e-6
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_population_axis_normalization_rejects_collapse(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    positions = np.ones((4, 3), dtype=np.float32)
    with pytest.raises(FloatingPointError, match="collapsed along a coordinate axis"):
        GraphEmbedder._normalize_positions(positions)


def test_ordered_endpoint_segments_are_vertex_then_contribution_stable(monkeypatch):
    monkeypatch.setattr(implementation, "cp", np)
    endpoint_ids = np.array([2, 1, 2, 0, 1], dtype=np.int32)
    contributions = np.arange(10, dtype=np.float32).reshape(5, 2)
    ordered, starts, ends, vertices = GraphEmbedder._ordered_endpoint_segments(
        endpoint_ids, contributions
    )
    np.testing.assert_array_equal(vertices, np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(starts, np.array([0, 1, 3], dtype=np.int64))
    np.testing.assert_array_equal(ends, np.array([1, 3, 5], dtype=np.int64))
    np.testing.assert_array_equal(ordered, contributions[[3, 1, 4, 0, 2]])


def test_force_kernels_are_sequential_and_preserve_exact_force_equations():
    source = implementation._DETERMINISTIC_FORCE_KERNELS
    assert "sqrtf(squared_norm) + 1.0e-6f" in source
    assert "attraction * (distance - preferred_length)" in source
    assert "row_offsets[vertex_id]" in source
    assert "neighbors[offset]" in source
    assert "starts[segment]" in source
    assert "ends[segment]" in source
    assert "atomicAdd" not in source
    assert "cp.add.at" not in inspect.getsource(GraphEmbedder)


def test_source_tree_contains_no_discarded_runtime_implementations():
    root = Path(__file__).parents[1] / "graphem_rapids"
    tracked_python = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(root.rglob("*.py"))
    ).lower()
    assert "graphembedderpytorch" not in tracked_python
    assert "graphembeddercuvs" not in tracked_python
    assert "force_mode" not in tracked_python


def test_source_uses_one_disconnected_graph_and_adaptive_tie_contract():
    source = inspect.getsource(GraphEmbedder)
    assert "connected_components" not in source
    assert "must not contain isolated vertices" not in source
    assert "if eigen_count >= self.n" in source
    assert "torch.lobpcg" in source
    assert "torch.sparse_coo_tensor" in source
    assert "torch.sparse.mm" in source
    assert "sp_" + "linalg" not in source
    assert "sp_" + "csgraph" not in source
    assert "eig" + "sh" not in source
    assert 'dtype=torch.float64' in source
    assert "stable=True" in source
    assert "tol=float(SPECTRAL_TOLERANCE)" in source
    assert "niter=SPECTRAL_MAX_ITERATIONS" in source
    assert 'method="ortho"' in source
    assert "largest=True" in source
    assert "cp.from_dlpack" in source
    assert "raw_search_boundary = distances[:, -1].copy()" in source
    assert "raw_search_boundary > cutoff" in source
    assert "search_width = min(self.n_edges, search_width * 2)" in source
    midpoint_source = inspect.getsource(GraphEmbedder._midpoint_neighbors)
    assert midpoint_source.index(
        "outside the global edge namespace"
    ) < midpoint_source.index(
        "_validate_unique_global_neighbor_ids"
    )
    assert midpoint_source.index(
        "_validate_unique_global_neighbor_ids"
    ) < midpoint_source.index(
        "_repair_negative_squared_distances"
    )


def test_source_tree_has_no_cpu_eigensolver_path():
    root = Path(__file__).parents[1]
    tracked = "\n".join(
        path.read_text(encoding="utf-8")
        for pattern in ("*.py", "*.md", "*.rst")
        for path in sorted(root.rglob(pattern))
        if ".git" not in path.parts
    ).lower()
    assert "scipy.sparse." + "linalg" not in tracked
    assert "scipy " + "eig" + "sh" not in tracked
    assert "_scipy_" + "spectral_embedding" not in tracked


def test_host_adjacency_preserves_an_isolated_vertex():
    adjacency = sp.csr_matrix(
        np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    observed = GraphEmbedder._validate_host_adjacency(adjacency)
    np.testing.assert_array_equal(observed.toarray(), adjacency.toarray())


def test_source_uses_cupy14_lexsort_contract_and_imports_cuvs_first():
    source = Path(implementation.__file__).read_text(encoding="utf-8")
    assert source.index("from cuvs.neighbors import brute_force") < source.index(
        "import cupy as cp"
    )
    assert "cp.lexsort(cp.stack((canonical[:, 1], canonical[:, 0]), axis=0))" in source
    assert "cp.lexsort(cp.stack((vertex_ids, -scores), axis=0))" in source
