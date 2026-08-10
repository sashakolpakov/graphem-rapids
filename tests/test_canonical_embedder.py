"""CPU contract checks for the single GraphEm implementation."""

# pylint: disable=missing-function-docstring

import inspect
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

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


def test_scipy_spectral_embedding_is_bitwise_repeatable_on_degenerate_graph():
    left_size = 10
    right_size = 20
    sources = np.repeat(np.arange(left_size, dtype=np.int64), right_size)
    targets = np.tile(
        np.arange(left_size, left_size + right_size, dtype=np.int64), left_size
    )
    edges = np.column_stack((sources, targets))

    first = GraphEmbedder._scipy_spectral_embedding(
        edges, left_size + right_size, 4, 20250608
    )
    second = GraphEmbedder._scipy_spectral_embedding(
        edges, left_size + right_size, 4, 20250608
    )

    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]
    np.testing.assert_allclose(first[1], [0.0, 1.0, 1.0, 1.0, 1.0], atol=1.0e-12)
    pivots = np.argmax(np.abs(first[0]), axis=0)
    assert np.all(first[0][pivots, np.arange(4)] >= 0)
    assert first[2] <= 1.0e-12


def test_scipy_spectral_embedding_preserves_components_and_isolate():
    first_component = np.column_stack(
        (np.arange(19, dtype=np.int64), np.arange(1, 20, dtype=np.int64))
    )
    second_component = np.array(
        [[20, 21], [21, 22], [22, 23], [23, 24], [24, 20]], dtype=np.int64
    )
    edges = np.vstack((first_component, second_component))

    positions, eigenvalues, max_residual = GraphEmbedder._scipy_spectral_embedding(
        edges, 26, 3, 7
    )

    assert positions.shape == (26, 3)
    assert positions.dtype == np.float32
    assert np.all(np.isfinite(positions))
    assert sum(abs(value) < 1.0e-10 for value in eigenvalues) >= 2
    assert max_residual <= 1.0e-8


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
    assert "sp_linalg.eigsh" in source
    assert "sp_csgraph.laplacian" in source
    assert 'which="SM"' in source
    assert "lobpcg" not in source.lower()
    assert "dtype=np.float64" in source
    assert 'kind="stable"' in source
    assert "tol=SPECTRAL_TOLERANCE" in source
    assert "maxiter=SPECTRAL_MAX_ITERATIONS" in source
    assert "raw_search_boundary = distances[:, -1].copy()" in source
    assert "raw_search_boundary > cutoff" in source
    assert "search_width = min(self.n_edges, search_width * 2)" in source


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
