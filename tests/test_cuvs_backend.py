"""Unit tests for cuVS backend."""

import pytest
import numpy as np

try:
    import cupy as cp
    import cuvs  # pylint: disable=unused-import
    from graphem_rapids.backends.embedder_cuvs import GraphEmbedderCuVS
    CUVS_AVAILABLE = True
except ImportError:
    CUVS_AVAILABLE = False

from graphem_rapids.generators import generate_er, generate_random_regular


@pytest.mark.skipif(not CUVS_AVAILABLE, reason="cuVS not available")
class TestCuVSBackend:
    """Test cuVS backend functionality."""

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
                k_inter=0.1,
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
            k_inter=0.1,
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
