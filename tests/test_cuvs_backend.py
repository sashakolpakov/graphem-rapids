"""Unit tests for cuVS backend."""

# pylint: disable=missing-function-docstring,too-many-public-methods

from types import SimpleNamespace

import pytest
import numpy as np
from graphem_rapids.backends.embedder_cuvs import GraphEmbedderCuVS

try:
    import cupy as cp
    import cuvs  # pylint: disable=unused-import
    from graphem_rapids.backends import embedder_cuvs as cuvs_backend
    CUVS_AVAILABLE = True
except ImportError:
    CUVS_AVAILABLE = False

from graphem_rapids.generators import generate_er, generate_random_regular


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("full_midpoint_index", "false"),
        ("full_midpoint_index", 0),
        ("full_midpoint_index", None),
        ("assume_canonical_edges", "false"),
        ("assume_canonical_edges", 1),
        ("assume_canonical_edges", None),
    ],
)
def test_cuvs_boolean_options_reject_coercion(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        GraphEmbedderCuVS(**{parameter: value})


@pytest.mark.skipif(not CUVS_AVAILABLE, reason="cuVS not available")
class TestCuVSBackend:
    """Test cuVS backend functionality."""

    @pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True])
    def test_seed_requires_numpy_cupy_uint32_domain(self, seed):
        edges = cp.asarray([[0, 1]], dtype=cp.int32)
        with pytest.raises(ValueError, match=r"between zero and 2\*\*32 - 1"):
            GraphEmbedderCuVS(
                edges=edges,
                n_vertices=2,
                initialization="randomized",
                k_inter=0,
                seed=seed,
                verbose=False,
            )

    def test_search_result_uses_integer_neighbor_output(self):
        distances = cp.asarray([[0.0, 1.0]], dtype=cp.float32)
        neighbors = cp.asarray([[3, 7]], dtype=cp.int64)
        parsed = GraphEmbedderCuVS._neighbors_from_search_result((distances, neighbors))
        assert cp.array_equal(parsed, neighbors)

    def test_midpoint_knn_returns_edge_ids(self):
        edges = cp.asarray([[0, 1], [2, 3], [4, 5]], dtype=cp.int32)
        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=6,
            assume_canonical_edges=True,
            n_components=2,
            initialization="randomized",
            spectral_iterations=2,
            index_type="brute_force",
            n_neighbors=1,
            sample_size=3,
            verbose=False,
            seed=4,
        )
        embedder.positions = cp.asarray(
            [[-0.1, 0], [0.1, 0], [0.9, 0], [1.1, 0], [9.9, 0], [10.1, 0]],
            dtype=cp.float32,
        )
        midpoints = 0.5 * (
            embedder.positions[embedder.edges[:, 0]]
            + embedder.positions[embedder.edges[:, 1]]
        )
        neighbors, sampled = embedder._locate_knn_midpoints_cuvs(midpoints, 1)
        assert cp.array_equal(sampled, cp.arange(3))
        assert cp.all(neighbors >= 0)
        assert cp.all(neighbors < embedder.n_edges)
        assert cp.array_equal(neighbors[:, 0], cp.asarray([1, 0, 1]))

    def test_gpu_edge_list_constructor_avoids_host_adjacency(self):
        edges = cp.asarray([[0, 1], [1, 2], [2, 3]], dtype=cp.int32)
        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=4,
            assume_canonical_edges=True,
            initialization="randomized",
            spectral_iterations=2,
            verbose=False,
        )
        assert embedder.adjacency is None
        assert embedder.n_edges == 3
        assert embedder.positions.shape == (4, 2)

    def test_isolates_are_zero_and_never_selected(self):
        edges = cp.asarray([[0, 1], [1, 2], [2, 3]], dtype=cp.int32)
        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=6,
            assume_canonical_edges=True,
            initialization="randomized",
            spectral_iterations=2,
            k_inter=0,
            verbose=False,
            seed=7,
        )

        assert embedder.n_active_vertices == 4
        assert cp.all(embedder.positions[4:] == 0)
        assert set(embedder.topk_nodes(4)) == {0, 1, 2, 3}
        assert set(embedder.diverse_topk_nodes(2)).issubset({0, 1, 2, 3})
        with pytest.raises(ValueError, match="non-isolated"):
            embedder.topk_nodes(5)

    def test_initial_active_edge_scale_matches_preferred_length(self):
        edges = cp.asarray([[0, 1], [1, 2], [2, 3]], dtype=cp.int32)
        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=6,
            assume_canonical_edges=True,
            initialization="randomized",
            spectral_iterations=3,
            L_min=2.5,
            k_inter=0,
            verbose=False,
            seed=11,
        )

        lengths = cp.linalg.norm(
            embedder.positions[edges[:, 1]] - embedder.positions[edges[:, 0]],
            axis=1,
        )
        assert float(cp.median(lengths).item()) == pytest.approx(2.5, rel=1e-4)
        assert cp.all(embedder.positions[4:] == 0)

    def test_randomized_positions_are_c_contiguous_for_fused_kernel(self):
        edges = cp.asarray([[0, 1], [1, 2], [2, 3]], dtype=cp.int32)
        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=4,
            assume_canonical_edges=True,
            initialization="randomized",
            spectral_iterations=2,
            k_inter=0,
            verbose=False,
            seed=19,
        )

        assert embedder.positions.flags.c_contiguous
        embedder.update_positions()
        assert embedder.positions.flags.c_contiguous

    def test_components_cannot_exceed_vertices(self):
        edges = cp.asarray([[0, 1]], dtype=cp.int32)
        with pytest.raises(ValueError, match="cannot exceed"):
            GraphEmbedderCuVS(
                edges=edges,
                n_vertices=2,
                n_components=3,
                k_inter=0,
                initialization="randomized",
                verbose=False,
            )

    def test_normalization_is_isotropic(self):
        edges = cp.asarray([[0, 1], [1, 2], [2, 3]], dtype=cp.int32)
        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=4,
            assume_canonical_edges=True,
            initialization="randomized",
            spectral_iterations=2,
            k_attr=0,
            k_inter=0,
            verbose=False,
        )
        embedder.positions = cp.asarray(
            [[-3.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [3.0, 1.0]],
            dtype=cp.float32,
        )
        embedder._normalization_radius = embedder._active_rms_radius(
            embedder.positions
        )
        before = embedder.positions.copy()
        embedder.update_positions()
        assert cp.allclose(embedder.positions, before, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("edge_dtype_name", ["int32", "int64"])
    def test_fused_spring_kernel_uses_typed_endpoint_atomics(self, edge_dtype_name):
        embedder = object.__new__(GraphEmbedderCuVS)
        embedder.n = 2
        embedder.n_components = 2
        embedder.k_attr = 1.0
        embedder.L_min = 1.0
        embedder.force_mode = "attractive"
        positions = cp.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=cp.float32)
        edges = cp.asarray([[0, 1]], dtype=getattr(cp, edge_dtype_name))

        forces = embedder._compute_spring_forces_cuvs(positions, edges)
        assert float(forces[0, 0].item()) == pytest.approx(1.0, rel=1e-5)
        assert float(forces[1, 0].item()) == pytest.approx(-1.0, rel=1e-5)
        assert cp.allclose(cp.sum(forces, axis=0), 0.0, atol=1e-6)

    @pytest.mark.parametrize("edge_dtype_name", ["int32", "int64"])
    def test_segment_separation_reduces_symmetric_x_crossing(self, edge_dtype_name):
        embedder = object.__new__(GraphEmbedderCuVS)
        embedder.n = 4
        embedder.n_components = 2
        embedder.k_inter = 1.0
        embedder.intersection_inclusion_scale = 1.0
        positions = cp.asarray(
            [[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]],
            dtype=cp.float32,
        )
        edges = cp.asarray(
            [[0, 1], [2, 3]], dtype=getattr(cp, edge_dtype_name)
        )
        neighbors = cp.asarray([[1]], dtype=cp.int64)
        sampled = cp.asarray([0], dtype=cp.int64)

        forces = embedder._compute_intersection_forces_cuvs(
            positions, edges, neighbors, sampled
        )
        assert cp.allclose(cp.sum(forces, axis=0), 0.0, atol=1e-6)
        assert cp.allclose(forces[0], forces[1], atol=1e-6)
        assert cp.allclose(forces[2], forces[3], atol=1e-6)
        assert cp.allclose(forces[0], -forces[2], atol=1e-6)

        def crossing_depth(points):
            p1, p2, q1, q2 = points
            first = p2 - p1
            second = q2 - q1
            relative = q1 - p1
            denominator = first[0] * second[1] - first[1] * second[0]
            first_parameter = (
                relative[0] * second[1] - relative[1] * second[0]
            ) / denominator
            second_parameter = (
                relative[0] * first[1] - relative[1] * first[0]
            ) / denominator
            return float(cp.min(cp.stack((
                first_parameter,
                1.0 - first_parameter,
                second_parameter,
                1.0 - second_parameter,
            ))).item())

        moved = positions + np.float32(0.1) * forces
        assert crossing_depth(moved) < crossing_depth(positions)

        embedder.intersection_inclusion_scale = 2.0
        scaled_sample_forces = embedder._compute_intersection_forces_cuvs(
            positions, edges, neighbors, sampled
        )
        embedder.intersection_inclusion_scale = 1.0
        mutual_forces = embedder._compute_intersection_forces_cuvs(
            positions,
            edges,
            cp.asarray([[1], [0]], dtype=cp.int64),
            cp.asarray([0, 1], dtype=cp.int64),
        )
        assert cp.allclose(mutual_forces, 2.0 * forces, atol=1e-6)
        assert cp.allclose(mutual_forces, scaled_sample_forces, atol=1e-6)

    def test_non_2d_intersections_fail_fast_and_float64_is_rejected(self):
        edges = cp.asarray([[0, 1], [1, 2]], dtype=cp.int32)
        with pytest.raises(ValueError, match="only for 2D"):
            GraphEmbedderCuVS(
                edges=edges,
                n_vertices=3,
                n_components=3,
                k_inter=0.1,
                initialization="randomized",
                verbose=False,
            )
        with pytest.raises(ValueError, match="float32"):
            GraphEmbedderCuVS(
                edges=edges,
                n_vertices=3,
                dtype=np.float64,
                initialization="randomized",
                verbose=False,
            )
        with pytest.raises(TypeError, match="integer dtype"):
            GraphEmbedderCuVS(
                edges=cp.asarray([[0.0, 1.5]], dtype=cp.float32),
                n_vertices=2,
                k_inter=0,
                initialization="randomized",
                verbose=False,
            )
        with pytest.raises(ValueError, match="n_components must be an integer"):
            GraphEmbedderCuVS(
                edges=edges,
                n_vertices=3,
                n_components=2.5,
                k_inter=0,
                initialization="randomized",
                verbose=False,
            )
        with pytest.raises(ValueError, match="must be finite"):
            GraphEmbedderCuVS(
                edges=edges,
                n_vertices=3,
                L_min=np.nan,
                k_inter=0,
                initialization="randomized",
                verbose=False,
            )

    def test_fused_launch_contracts_reject_malformed_shapes(self):
        embedder = object.__new__(GraphEmbedderCuVS)
        embedder.n = 3
        embedder.n_components = 2
        embedder.k_attr = 1.0
        embedder.L_min = 1.0
        embedder.force_mode = "attractive"
        positions = cp.zeros((3, 2), dtype=cp.float32)
        with pytest.raises(ValueError, match="edges must have shape"):
            embedder._compute_spring_forces_cuvs(
                positions, cp.asarray([0, 1], dtype=cp.int32)
            )
        with pytest.raises(ValueError, match="matching"):
            embedder._scatter_intersection_forces(
                cp.asarray([[0, 1]], dtype=cp.int32),
                cp.asarray([[1, 2]], dtype=cp.int32),
                cp.zeros((2, 2), dtype=cp.float32),
                positions,
            )

    def test_forced_ivf_uses_bounded_reference_and_global_edge_ids(self):
        edges = cp.arange(400, dtype=cp.int32).reshape(200, 2)
        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=400,
            assume_canonical_edges=True,
            initialization="randomized",
            spectral_iterations=2,
            index_type="ivf_flat",
            sample_size=16,
            midpoint_reference_size=64,
            n_neighbors=2,
            ivf_n_probes=99,
            verbose=False,
            seed=13,
        )
        midpoints = 0.5 * (
            embedder.positions[edges[:, 0]] + embedder.positions[edges[:, 1]]
        )
        neighbors, sampled = embedder._locate_knn_midpoints_cuvs(midpoints, 2)

        assert sampled.shape == (16,)
        assert neighbors.shape == (16, 2)
        assert cp.all(neighbors >= 0)
        assert cp.all(neighbors < embedder.n_edges)
        assert cp.all(neighbors != sampled[:, None])
        assert embedder._last_midpoint_reference_size == 64
        assert embedder._knn_fallbacks == 0
        diagnostics = embedder.get_diagnostics()
        assert diagnostics["query_inclusion_scale"] == pytest.approx(12.5)
        assert diagnostics["midpoint_reference_mode"] == "bounded_user"
        assert diagnostics["last_midpoint_reference_size"] == 64
        assert diagnostics["configured_ivf_n_probes"] == 99
        assert diagnostics["actual_ivf_n_probes"] == 8
        assert diagnostics["last_n_probes"] == 8

    @pytest.mark.parametrize("ivf_n_probes", [0, -1, 1.5, True])
    def test_ivf_probe_count_must_be_a_positive_integer(self, ivf_n_probes):
        """Reject values that cuVS cannot use as a probe count."""
        edges = cp.asarray([[0, 1]], dtype=cp.int32)
        with pytest.raises(ValueError, match="ivf_n_probes"):
            GraphEmbedderCuVS(
                edges=edges,
                n_vertices=2,
                assume_canonical_edges=True,
                initialization="randomized",
                k_inter=0,
                ivf_n_probes=ivf_n_probes,
                verbose=False,
            )

    @pytest.mark.parametrize(
        ("index_type", "module_name"),
        [("ivf_flat", "ivf_flat"), ("ivf_pq", "ivf_pq")],
    )
    def test_each_ivf_index_caps_configured_probes_to_n_lists(
        self, monkeypatch, index_type, module_name
    ):
        """Apply the configured probe cap to both supported IVF variants."""
        search_arguments = {}

        class Params:
            """Capture index construction options accepted by the fake module."""

            def __init__(self, **kwargs):
                self.options = kwargs

        class SearchParams:
            """Capture search options accepted by the fake module."""

            def __init__(self, **kwargs):
                search_arguments.update(kwargs)

        fake_module = SimpleNamespace(
            IndexParams=Params,
            SearchParams=SearchParams,
            build=lambda _params, _midpoints: object(),
        )
        monkeypatch.setattr(cuvs_backend, module_name, fake_module)
        embedder = object.__new__(GraphEmbedderCuVS)
        embedder.index_type = index_type
        embedder.n_edges = 100
        embedder.n_components = 8
        embedder.ivf_n_probes = 99
        embedder.logger = SimpleNamespace(warning=lambda *_args: None)

        embedder._build_knn_index(cp.zeros((100, 8), dtype=cp.float32))

        assert search_arguments["n_probes"] == 10
        assert embedder._last_n_probes == 10

    def test_candidate_pair_budget_fails_before_layout_allocation(self):
        edges = cp.arange(400, dtype=cp.int32).reshape(200, 2)
        with pytest.raises(MemoryError, match="query-neighbor entries"):
            GraphEmbedderCuVS(
                edges=edges,
                n_vertices=400,
                assume_canonical_edges=True,
                initialization="randomized",
                sample_size=16,
                midpoint_reference_size=64,
                n_neighbors=4,
                max_candidate_pairs=32,
                verbose=False,
            )

    def test_large_cuvs_failure_never_enters_exact_fallback(self, monkeypatch):
        embedder = object.__new__(GraphEmbedderCuVS)
        embedder.n_edges = 100_001
        embedder.sample_size = 1
        embedder.midpoint_reference_size = 100_001
        embedder._last_query_sample_size = 0
        embedder._last_midpoint_reference_size = 0
        embedder._knn_fallbacks = 0
        embedder._knn_last_error = None
        embedder.knn_index = None
        embedder._knn_search_params = None
        embedder.batch_size = 16
        midpoints = cp.zeros((100_001, 2), dtype=cp.float32)

        def fail_build(_reference):
            raise RuntimeError("synthetic cuVS failure")

        monkeypatch.setattr(embedder, "_build_knn_index", fail_build)
        with pytest.raises(RuntimeError, match="exact fallback is disabled"):
            embedder._locate_knn_midpoints_cuvs(midpoints, 1)
        assert embedder._knn_fallbacks == 1

    def test_cuvs_backend_initialization(self):
        """Test cuVS backend initialization."""
        adjacency = generate_random_regular(n=50, d=4, seed=42)

        embedder = GraphEmbedderCuVS(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
            verbose=False
        )

        assert embedder.n == 50
        assert embedder.n_components == 2
        assert embedder.positions.shape == (50, 2)

    def test_cuvs_backend_dimensions(self):
        """Test cuVS backend with different dimensions."""
        adjacency = generate_random_regular(n=40, d=4, seed=42)

        for dim in [2, 3, 4]:
            embedder = GraphEmbedderCuVS(
                adjacency=adjacency,
                n_components=dim,
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1 if dim == 2 else 0.0,
                n_neighbors=15,
                sample_size=200,
                verbose=False
            )

            assert embedder.n_components == dim
            assert embedder.positions.shape == (40, dim)

    def test_cuvs_layout_execution(self):
        """Test cuVS backend layout algorithm execution."""
        adjacency = generate_random_regular(n=40, d=4, seed=42)

        embedder = GraphEmbedderCuVS(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=10,
            sample_size=128,
            verbose=False
        )

        initial_positions = embedder.get_positions().copy()
        embedder.run_layout(num_iterations=3)
        final_positions = embedder.get_positions()

        # Check that positions changed
        assert not np.array_equal(initial_positions, final_positions)
        assert final_positions.shape == (40, 2)
        assert np.all(np.isfinite(final_positions))

    def test_cuvs_memory_efficiency(self):
        """Test cuVS backend memory efficiency with larger graphs."""
        adjacency = generate_er(n=200, p=0.02, seed=42)

        embedder = GraphEmbedderCuVS(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=512,
            verbose=False
        )

        # Test that the embedder was created successfully
        assert embedder.positions.shape == (200, 2)
        assert np.all(np.isfinite(embedder.get_positions()))

    def test_cuvs_disconnected_graph(self):
        """Test cuVS backend with disconnected graph."""
        # Create two disconnected hexagons (12 vertices to meet n_neighbors requirement)
        import scipy.sparse as sp  # pylint: disable=import-outside-toplevel
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],  # Hexagon 1
            [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 6]  # Hexagon 2
        ])

        # Convert to adjacency matrix
        n_vertices = 12
        adjacency = sp.csr_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(n_vertices, n_vertices)
        )
        adjacency = adjacency + adjacency.T

        embedder = GraphEmbedderCuVS(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=5,
            sample_size=12,
            verbose=False
        )

        embedder.run_layout(num_iterations=2)
        assert embedder.positions.shape == (12, 2)
        assert np.all(np.isfinite(embedder.get_positions()))

    def test_cuvs_layout_stability(self):
        """Test that cuVS backend layout runs are numerically stable."""
        adjacency = generate_random_regular(n=30, d=4, seed=42)

        embedder = GraphEmbedderCuVS(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=64,
            verbose=False
        )

        for _ in range(3):
            embedder.run_layout(num_iterations=2)

            positions = embedder.get_positions()
            assert np.all(np.isfinite(positions))

            max_coord = np.max(np.abs(positions))
            assert max_coord < 1000  # Reasonable bound

    def test_cuvs_large_graphs(self):
        """Test cuVS backend with large graphs."""
        adjacency = generate_er(n=500, p=0.008, seed=42)

        embedder = GraphEmbedderCuVS(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=512,
            verbose=False
        )

        assert embedder.positions.shape == (500, 2)
        assert np.all(np.isfinite(embedder.get_positions()))

        # Run a few iterations to ensure it works
        embedder.run_layout(num_iterations=2)
        assert np.all(np.isfinite(embedder.get_positions()))

    def test_cuvs_parameter_validation(self):
        """Test cuVS backend parameter validation."""
        adjacency = generate_random_regular(n=50, d=4, seed=42)

        # Test invalid dimension
        with pytest.raises((ValueError, AssertionError)):
            GraphEmbedderCuVS(
                adjacency=adjacency,
                n_components=0,  # Invalid
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                n_neighbors=15,
                sample_size=256,
                verbose=False
            )

        # Test negative k_attr
        with pytest.raises((ValueError, AssertionError)):
            GraphEmbedderCuVS(
                adjacency=adjacency,
                n_components=2,
                k_attr=-1.0,  # Invalid
                L_min=10.0,
                k_inter=0.1,
                n_neighbors=15,
                sample_size=256,
                verbose=False
            )

    def test_cuvs_knn_performance(self):
        """Test cuVS backend KNN performance optimization."""
        adjacency = generate_er(n=100, p=0.05, seed=42)

        # Test with different n_neighbors values
        for n_neighbors_val in [5, 10, 20]:
            embedder = GraphEmbedderCuVS(
                adjacency=adjacency,
                n_components=2,
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                n_neighbors=n_neighbors_val,
                sample_size=256,
                verbose=False
            )

            embedder.run_layout(num_iterations=2)
            assert np.all(np.isfinite(embedder.get_positions()))

    def test_cuvs_batch_processing(self):
        """Test cuVS backend with different batch sizes."""
        adjacency = generate_er(n=100, p=0.03, seed=42)

        # Test with different batch sizes (note: batch_size removed from API, test sample_size instead)
        for sample_size_val in [64, 256, 512]:
            embedder = GraphEmbedderCuVS(
                adjacency=adjacency,
                n_components=2,
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                n_neighbors=15,
                sample_size=sample_size_val,
                verbose=False
            )

            embedder.run_layout(num_iterations=2)
            assert np.all(np.isfinite(embedder.get_positions()))

    def test_cuvs_sample_size_effects(self):
        """Test cuVS backend with different sample sizes."""
        adjacency = generate_er(n=100, p=0.04, seed=42)

        # Test with different sample sizes
        for sample_size in [128, 256, 512]:
            embedder = GraphEmbedderCuVS(
                adjacency=adjacency,
                n_components=2,
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                n_neighbors=15,
                sample_size=sample_size,
                verbose=False
            )

            embedder.run_layout(num_iterations=2)
            assert np.all(np.isfinite(embedder.get_positions()))

    def test_cuvs_force_parameters(self):
        """Test cuVS backend with different force parameters."""
        adjacency = generate_random_regular(n=50, d=4, seed=42)

        # Test with different force parameters
        force_configs = [
            {'k_attr': 0.1, 'k_inter': 0.05},
            {'k_attr': 0.5, 'k_inter': 0.1},
            {'k_attr': 1.0, 'k_inter': 0.2}
        ]

        for config in force_configs:
            embedder = GraphEmbedderCuVS(
                adjacency=adjacency,
                n_components=2,
                L_min=10.0,
                n_neighbors=15,
                sample_size=256,
                **config,
                verbose=False
            )

            embedder.run_layout(num_iterations=2)
            assert np.all(np.isfinite(embedder.get_positions()))

    def test_cuvs_gpu_memory_management(self):
        """Test cuVS backend GPU memory management."""
        adjacency = generate_er(n=200, p=0.02, seed=42)

        embedder = GraphEmbedderCuVS(
            adjacency=adjacency,
            n_components=3,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.0,
            n_neighbors=15,
            sample_size=512,
            verbose=False
        )

        # Test that multiple layout runs work without memory issues
        for _ in range(3):
            embedder.run_layout(num_iterations=2)
            assert np.all(np.isfinite(embedder.get_positions()))

    def test_cuvs_data_transfer_integrity(self):
        """Test data integrity in cuVS backend CPU-GPU transfers."""
        adjacency = generate_random_regular(n=50, d=4, seed=42)

        embedder = GraphEmbedderCuVS(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
            verbose=False
        )

        # Test that positions are properly transferred back from GPU
        initial_positions = embedder.get_positions().copy()
        embedder.run_layout(num_iterations=1)
        final_positions = embedder.get_positions()

        # Verify data types and shapes are preserved
        assert isinstance(final_positions, np.ndarray)
        assert final_positions.shape == initial_positions.shape
        assert final_positions.dtype in [np.float32, np.float64]

    def test_cuvs_numerical_precision(self):
        """Test cuVS backend numerical precision."""
        adjacency = generate_random_regular(n=40, d=4, seed=42)

        embedder = GraphEmbedderCuVS(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
            verbose=False
        )

        # Run layout and check for reasonable numerical values
        embedder.run_layout(num_iterations=5)

        # Check that positions are not NaN or infinity
        positions = embedder.get_positions()
        assert not np.any(np.isnan(positions))
        assert not np.any(np.isinf(positions))

        # Check that positions are within reasonable bounds
        max_coord = np.max(np.abs(positions))
        assert max_coord < 1e6  # Should not explode numerically
