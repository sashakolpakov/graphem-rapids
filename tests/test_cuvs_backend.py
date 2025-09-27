"""Unit tests for cuVS backend."""

import pytest
import numpy as np

try:
    import cudf  # pylint: disable=unused-import
    import cuml  # pylint: disable=unused-import
    import cuvs  # pylint: disable=unused-import
    from graphem_rapids.backends.embedder_cuvs import GraphEmbedderCuVS
    CUVS_AVAILABLE = True
except ImportError:
    CUVS_AVAILABLE = False

from graphem_rapids.generators import erdos_renyi_graph, generate_random_regular


@pytest.mark.skipif(not CUVS_AVAILABLE, reason="cuVS not available")
class TestCuVSBackend:
    """Test cuVS backend functionality."""

    def test_cuvs_backend_initialization(self):
        """Test cuVS backend initialization."""
        edges = generate_random_regular(n=50, d=4, seed=42)

        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=50,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(256, len(edges)),
            batch_size=1024,
            verbose=False
        )

        assert embedder.n == 50
        assert embedder.dimension == 2
        assert embedder.positions.shape == (50, 2)

    def test_cuvs_backend_dimensions(self):
        """Test cuVS backend with different dimensions."""
        edges = generate_random_regular(n=40, d=4, seed=42)

        for dim in [2, 3, 4]:
            embedder = GraphEmbedderCuVS(
                edges=edges,
                n_vertices=40,
                dimension=dim,
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                knn_k=15,
                sample_size=min(200, len(edges)),
                batch_size=1024,
                verbose=False
            )

            assert embedder.dimension == dim
            assert embedder.positions.shape == (40, dim)

    def test_cuvs_layout_execution(self):
        """Test cuVS backend layout algorithm execution."""
        edges = generate_random_regular(n=40, d=4, seed=42)

        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=40,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=10,
            sample_size=min(128, len(edges)),
            batch_size=1024,
            verbose=False
        )

        initial_positions = embedder.positions.copy()
        embedder.run_layout(num_iterations=3)

        # Check that positions changed
        assert not np.array_equal(initial_positions, embedder.positions)
        assert embedder.positions.shape == (40, 2)
        assert np.all(np.isfinite(embedder.positions))

    def test_cuvs_memory_efficiency(self):
        """Test cuVS backend memory efficiency with larger graphs."""
        edges = erdos_renyi_graph(n=200, p=0.02, seed=42)

        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=200,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(512, len(edges)),
            batch_size=1024,
            verbose=False
        )

        # Test that the embedder was created successfully
        assert embedder.positions.shape == (200, 2)
        assert np.all(np.isfinite(embedder.positions))

    def test_cuvs_disconnected_graph(self):
        """Test cuVS backend with disconnected graph."""
        # Create two disconnected triangles
        edges = np.array([
            [0, 1], [1, 2], [2, 0],  # Triangle 1
            [3, 4], [4, 5], [5, 3]   # Triangle 2
        ])

        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=6,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=5,
            sample_size=min(6, len(edges)),
            batch_size=64,
            verbose=False
        )

        embedder.run_layout(num_iterations=2)
        assert embedder.positions.shape == (6, 2)
        assert np.all(np.isfinite(embedder.positions))

    def test_cuvs_layout_stability(self):
        """Test that cuVS backend layout runs are numerically stable."""
        edges = generate_random_regular(n=30, d=4, seed=42)

        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=30,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(64, len(edges)),
            batch_size=1024,
            verbose=False
        )

        for _ in range(3):
            embedder.run_layout(num_iterations=2)

            assert np.all(np.isfinite(embedder.positions))

            max_coord = np.max(np.abs(embedder.positions))
            assert max_coord < 1000  # Reasonable bound

    def test_cuvs_large_graphs(self):
        """Test cuVS backend with large graphs."""
        edges = erdos_renyi_graph(n=500, p=0.008, seed=42)

        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=500,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(512, len(edges)),
            batch_size=2048,
            verbose=False
        )

        assert embedder.positions.shape == (500, 2)
        assert np.all(np.isfinite(embedder.positions))

        # Run a few iterations to ensure it works
        embedder.run_layout(num_iterations=2)
        assert np.all(np.isfinite(embedder.positions))

    def test_cuvs_parameter_validation(self):
        """Test cuVS backend parameter validation."""
        edges = generate_random_regular(n=50, d=4, seed=42)

        # Test invalid dimension
        with pytest.raises((ValueError, AssertionError)):
            GraphEmbedderCuVS(
                edges=edges,
                n_vertices=50,
                dimension=0,  # Invalid
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                knn_k=15,
                sample_size=min(256, len(edges)),
                batch_size=1024,
                verbose=False
            )

        # Test negative k_attr
        with pytest.raises((ValueError, AssertionError)):
            GraphEmbedderCuVS(
                edges=edges,
                n_vertices=50,
                dimension=2,
                k_attr=-1.0,  # Invalid
                L_min=10.0,
                k_inter=0.1,
                knn_k=15,
                sample_size=min(256, len(edges)),
                batch_size=1024,
                verbose=False
            )

    def test_cuvs_knn_performance(self):
        """Test cuVS backend KNN performance optimization."""
        edges = erdos_renyi_graph(n=100, p=0.05, seed=42)

        # Test with different KNN k values
        for knn_k in [5, 10, 20]:
            embedder = GraphEmbedderCuVS(
                edges=edges,
                n_vertices=100,
                dimension=2,
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                knn_k=knn_k,
                sample_size=min(256, len(edges)),
                batch_size=1024,
                verbose=False
            )

            embedder.run_layout(num_iterations=2)
            assert np.all(np.isfinite(embedder.positions))

    def test_cuvs_batch_processing(self):
        """Test cuVS backend with different batch sizes."""
        edges = erdos_renyi_graph(n=100, p=0.03, seed=42)

        # Test with different batch sizes
        for batch_size in [64, 256, 1024]:
            embedder = GraphEmbedderCuVS(
                edges=edges,
                n_vertices=100,
                dimension=2,
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                knn_k=15,
                sample_size=min(256, len(edges)),
                batch_size=batch_size,
                verbose=False
            )

            embedder.run_layout(num_iterations=2)
            assert np.all(np.isfinite(embedder.positions))

    def test_cuvs_sample_size_effects(self):
        """Test cuVS backend with different sample sizes."""
        edges = erdos_renyi_graph(n=100, p=0.04, seed=42)

        # Test with different sample sizes
        for sample_size in [128, 256, 512]:
            embedder = GraphEmbedderCuVS(
                edges=edges,
                n_vertices=100,
                dimension=2,
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                knn_k=15,
                sample_size=min(sample_size, len(edges)),
                batch_size=1024,
                verbose=False
            )

            embedder.run_layout(num_iterations=2)
            assert np.all(np.isfinite(embedder.positions))

    def test_cuvs_force_parameters(self):
        """Test cuVS backend with different force parameters."""
        edges = generate_random_regular(n=50, d=4, seed=42)

        # Test with different force parameters
        force_configs = [
            {'k_attr': 0.1, 'k_inter': 0.05},
            {'k_attr': 0.5, 'k_inter': 0.1},
            {'k_attr': 1.0, 'k_inter': 0.2}
        ]

        for config in force_configs:
            embedder = GraphEmbedderCuVS(
                edges=edges,
                n_vertices=50,
                dimension=2,
                L_min=10.0,
                knn_k=15,
                sample_size=min(256, len(edges)),
                batch_size=1024,
                **config,
                verbose=False
            )

            embedder.run_layout(num_iterations=2)
            assert np.all(np.isfinite(embedder.positions))

    def test_cuvs_gpu_memory_management(self):
        """Test cuVS backend GPU memory management."""
        edges = erdos_renyi_graph(n=200, p=0.02, seed=42)

        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=200,
            dimension=3,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(512, len(edges)),
            batch_size=1024,
            verbose=False
        )

        # Test that multiple layout runs work without memory issues
        for _ in range(3):
            embedder.run_layout(num_iterations=2)
            assert np.all(np.isfinite(embedder.positions))

    def test_cuvs_data_transfer_integrity(self):
        """Test data integrity in cuVS backend CPU-GPU transfers."""
        edges = generate_random_regular(n=50, d=4, seed=42)

        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=50,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(256, len(edges)),
            batch_size=1024,
            verbose=False
        )

        # Test that positions are properly transferred back from GPU
        initial_positions = embedder.positions.copy()
        embedder.run_layout(num_iterations=1)

        # Verify data types and shapes are preserved
        assert isinstance(embedder.positions, np.ndarray)
        assert embedder.positions.shape == initial_positions.shape
        assert embedder.positions.dtype in [np.float32, np.float64]

    def test_cuvs_numerical_precision(self):
        """Test cuVS backend numerical precision."""
        edges = generate_random_regular(n=40, d=4, seed=42)

        embedder = GraphEmbedderCuVS(
            edges=edges,
            n_vertices=40,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(256, len(edges)),
            batch_size=1024,
            verbose=False
        )

        # Run layout and check for reasonable numerical values
        embedder.run_layout(num_iterations=5)

        # Check that positions are not NaN or infinity
        assert not np.any(np.isnan(embedder.positions))
        assert not np.any(np.isinf(embedder.positions))

        # Check that positions are within reasonable bounds
        max_coord = np.max(np.abs(embedder.positions))
        assert max_coord < 1e6  # Should not explode numerically


