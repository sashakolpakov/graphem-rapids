"""Unit tests for PyTorch backend."""

import pytest
import numpy as np
import torch
from graphem_rapids.backends.embedder_pytorch import GraphEmbedderPyTorch
from graphem_rapids.generators import erdos_renyi_graph, generate_random_regular


class TestPyTorchBackend:
    """Test PyTorch backend functionality."""

    @pytest.mark.fast
    def test_pytorch_backend_initialization(self):
        """Test PyTorch backend initialization."""
        edges = generate_random_regular(n=50, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
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
        assert isinstance(embedder.positions, np.ndarray)  # Public API returns numpy arrays
        assert isinstance(embedder._positions, torch.Tensor)  # Internal storage is tensor
        assert embedder.device.type in ['cpu', 'cuda']

    @pytest.mark.fast
    def test_pytorch_cpu_device(self):
        """Test PyTorch backend with CPU device."""
        edges = generate_random_regular(n=30, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
            edges=edges,
            n_vertices=30,
            dimension=3,
            device='cpu',
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(256, len(edges)),
            batch_size=1024,
            verbose=False
        )

        assert embedder.device.type == 'cpu'
        assert embedder._positions.device.type == 'cpu'  # Check internal tensor device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.gpu
    @pytest.mark.fast
    def test_pytorch_cuda_device(self):
        """Test PyTorch backend with CUDA device."""
        edges = generate_random_regular(n=30, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
            edges=edges,
            n_vertices=30,
            dimension=3,
            device='cuda',
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(256, len(edges)),
            batch_size=1024,
            verbose=False
        )

        assert embedder.device.type == 'cuda'
        assert embedder._positions.device.type == 'cuda'  # Check internal tensor device

    @pytest.mark.fast
    def test_pytorch_backend_dimensions(self):
        """Test PyTorch backend with different dimensions."""
        edges = generate_random_regular(n=40, d=4, seed=42)

        for dim in [2, 3, 4]:
            embedder = GraphEmbedderPyTorch(
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

    @pytest.mark.fast
    def test_pytorch_layout_execution(self):
        """Test PyTorch backend layout algorithm execution."""
        edges = generate_random_regular(n=40, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
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

        initial_positions = embedder.positions.copy()  # numpy copy for public API
        embedder.run_layout(num_iterations=3)

        # Check that positions changed
        assert not np.allclose(initial_positions, embedder.positions)
        assert embedder.positions.shape == (40, 2)
        assert np.all(np.isfinite(embedder.positions))

    @pytest.mark.fast
    def test_pytorch_memory_management(self):
        """Test memory management features of PyTorch backend."""
        edges = erdos_renyi_graph(n=100, p=0.05, seed=42)

        embedder = GraphEmbedderPyTorch(
            edges=edges,
            n_vertices=100,
            dimension=2,
            memory_efficient=True,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(256, len(edges)),
            batch_size=1024,
            verbose=False
        )

        # Test that the embedder was created successfully with memory management
        assert embedder.memory_efficient is True
        assert embedder.positions.shape == (100, 2)

    @pytest.mark.fast
    def test_pytorch_different_dtypes(self):
        """Test PyTorch backend with different data types."""
        edges = generate_random_regular(n=50, d=4, seed=42)

        # Test float32
        embedder_f32 = GraphEmbedderPyTorch(
            edges=edges,
            n_vertices=50,
            dimension=2,
            dtype=torch.float32,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(256, len(edges)),
            batch_size=1024,
            verbose=False
        )
        assert embedder_f32._positions.dtype == torch.float32  # Check internal tensor dtype

        # Test float64
        embedder_f64 = GraphEmbedderPyTorch(
            edges=edges,
            n_vertices=50,
            dimension=2,
            dtype=torch.float64,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(256, len(edges)),
            batch_size=1024,
            verbose=False
        )
        assert embedder_f64._positions.dtype == torch.float64  # Check internal tensor dtype

    @pytest.mark.fast
    def test_pytorch_disconnected_graph(self):
        """Test PyTorch backend with disconnected graph."""
        # Create two disconnected triangles
        edges = np.array([
            [0, 1], [1, 2], [2, 0],  # Triangle 1
            [3, 4], [4, 5], [5, 3]   # Triangle 2
        ])

        embedder = GraphEmbedderPyTorch(
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

    @pytest.mark.fast
    def test_pytorch_layout_stability(self):
        """Test that PyTorch backend layout runs are numerically stable."""
        edges = generate_random_regular(n=30, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
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

    @pytest.mark.fast
    def test_pytorch_large_graphs(self):
        """Test PyTorch backend with large graphs."""
        edges = erdos_renyi_graph(n=200, p=0.02, seed=42)

        embedder = GraphEmbedderPyTorch(
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

        assert embedder.positions.shape == (200, 2)
        assert np.all(np.isfinite(embedder.positions))

        # Run a few iterations to ensure it works
        embedder.run_layout(num_iterations=2)
        assert np.all(np.isfinite(embedder.positions))

    @pytest.mark.fast
    def test_pytorch_parameter_validation(self):
        """Test PyTorch backend parameter validation."""
        edges = generate_random_regular(n=50, d=4, seed=42)

        # Test invalid dimension
        with pytest.raises((ValueError, AssertionError)):
            GraphEmbedderPyTorch(
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

        # Test invalid device
        with pytest.raises((ValueError, RuntimeError)):
            GraphEmbedderPyTorch(
                edges=edges,
                n_vertices=50,
                dimension=2,
                device='invalid_device',  # Invalid
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                knn_k=15,
                sample_size=min(256, len(edges)),
                batch_size=1024,
                verbose=False
            )

    @pytest.mark.fast
    def test_pytorch_batch_processing(self):
        """Test PyTorch backend with different batch sizes."""
        edges = erdos_renyi_graph(n=100, p=0.03, seed=42)

        # Test with small batch size
        embedder_small = GraphEmbedderPyTorch(
            edges=edges,
            n_vertices=100,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(256, len(edges)),
            batch_size=32,
            verbose=False
        )

        # Test with large batch size
        embedder_large = GraphEmbedderPyTorch(
            edges=edges,
            n_vertices=100,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(256, len(edges)),
            batch_size=1024,
            verbose=False
        )

        # Both should work
        embedder_small.run_layout(num_iterations=2)
        embedder_large.run_layout(num_iterations=2)

        assert np.all(np.isfinite(embedder_small.positions))
        assert np.all(np.isfinite(embedder_large.positions))

    @pytest.mark.fast
    def test_pytorch_reproducibility(self):
        """Test that PyTorch backend produces reproducible results."""
        edges = generate_random_regular(n=50, d=4, seed=42)

        # Create two identical embedders with same random seed
        # IMPORTANT: Set seed before creating each embedder instance
        torch.manual_seed(123)
        embedder1 = GraphEmbedderPyTorch(
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
        embedder1.run_layout(num_iterations=3)

        torch.manual_seed(123)
        embedder2 = GraphEmbedderPyTorch(
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
        embedder2.run_layout(num_iterations=3)

        # Get final positions
        pos1, pos2 = embedder1.positions, embedder2.positions

        # Graph layout algorithms often produce equivalent results under transformations
        # (rotation, reflection). Test for these equivalent layouts.

        # Check all possible axis reflections that preserve graph structure
        transformations = [
            ("no reflection", pos2 * np.array([1, 1])),     # identity
            ("x-axis reflection", pos2 * np.array([-1, 1])), # x-axis reflection
            ("y-axis reflection", pos2 * np.array([1, -1])), # y-axis reflection
            ("both axes reflection", pos2 * np.array([-1, -1])), # both axes reflection
        ]

        for transform_name, transformed_pos2 in transformations:
            if np.allclose(pos1, transformed_pos2, rtol=1e-6, atol=1e-6):
                return  # Found a valid transformation

        # If no transformation matches, the algorithm is not reproducible
        pytest.fail("Positions differ significantly between identical runs, "
                   "even accounting for valid transformations")

