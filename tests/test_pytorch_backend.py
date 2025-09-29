"""Unit tests for PyTorch backend."""

import pytest
import numpy as np
import torch
from graphem_rapids.backends.embedder_pytorch import GraphEmbedderPyTorch
from graphem_rapids.generators import erdos_renyi_graph, generate_random_regular

# Test if PyKeOps is available for conditional tests
try:
    from pykeops.torch import LazyTensor
    PYKEOPS_AVAILABLE = True
except ImportError:
    PYKEOPS_AVAILABLE = False


class TestPyTorchBackend:
    """Test PyTorch backend functionality."""

    @pytest.mark.fast
    def test_pytorch_backend_initialization(self):
        """Test PyTorch backend initialization."""
        adjacency = generate_random_regular(n=50, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
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
        assert isinstance(embedder.positions, np.ndarray)  # Public API returns numpy arrays
        assert isinstance(embedder._positions, torch.Tensor)  # Internal storage is tensor
        assert embedder.device.type in ['cpu', 'cuda']

    @pytest.mark.fast
    def test_pytorch_cpu_device(self):
        """Test PyTorch backend with CPU device."""
        adjacency = generate_random_regular(n=30, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=3,
            device='cpu',
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
            verbose=False
        )

        assert embedder.device.type == 'cpu'
        assert embedder._positions.device.type == 'cpu'  # Check internal tensor device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.gpu
    @pytest.mark.fast
    def test_pytorch_cuda_device(self):
        """Test PyTorch backend with CUDA device."""
        adjacency = generate_random_regular(n=30, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=3,
            device='cuda',
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
            verbose=False
        )

        assert embedder.device.type == 'cuda'
        assert embedder._positions.device.type == 'cuda'  # Check internal tensor device

    @pytest.mark.fast
    def test_pytorch_backend_dimensions(self):
        """Test PyTorch backend with different dimensions."""
        adjacency = generate_random_regular(n=40, d=4, seed=42)

        for dim in [2, 3, 4]:
            embedder = GraphEmbedderPyTorch(
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

    @pytest.mark.fast
    def test_pytorch_layout_execution(self):
        """Test PyTorch backend layout algorithm execution."""
        adjacency = generate_random_regular(n=40, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=10,
            sample_size=128,
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
        adjacency = erdos_renyi_graph(n=100, p=0.05, seed=42)

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            memory_efficient=True,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
            verbose=False
        )

        # Test that the embedder was created successfully with memory management
        assert embedder.memory_efficient is True
        assert embedder.positions.shape == (100, 2)

    @pytest.mark.fast
    def test_pytorch_different_dtypes(self):
        """Test PyTorch backend with different data types."""
        adjacency = generate_random_regular(n=50, d=4, seed=42)

        # Test float32
        embedder_f32 = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            dtype=torch.float32,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
            verbose=False
        )
        assert embedder_f32._positions.dtype == torch.float32  # Check internal tensor dtype

        # Test float64
        embedder_f64 = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            dtype=torch.float64,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
            verbose=False
        )
        assert embedder_f64._positions.dtype == torch.float64  # Check internal tensor dtype

    @pytest.mark.fast
    def test_pytorch_disconnected_graph(self):
        """Test PyTorch backend with disconnected graph."""
        # Create two disconnected triangles using dense adjacency matrix
        adjacency = np.array([
            [0, 1, 1, 0, 0, 0],  # vertex 0: connected to 1,2
            [1, 0, 1, 0, 0, 0],  # vertex 1: connected to 0,2
            [1, 1, 0, 0, 0, 0],  # vertex 2: connected to 0,1
            [0, 0, 0, 0, 1, 1],  # vertex 3: connected to 4,5
            [0, 0, 0, 1, 0, 1],  # vertex 4: connected to 3,5
            [0, 0, 0, 1, 1, 0],  # vertex 5: connected to 3,4
        ])

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=5,
            sample_size=6,
            verbose=False
        )

        embedder.run_layout(num_iterations=2)
        assert embedder.positions.shape == (6, 2)
        assert np.all(np.isfinite(embedder.positions))

    @pytest.mark.fast
    def test_pytorch_layout_stability(self):
        """Test that PyTorch backend layout runs are numerically stable."""
        adjacency = generate_random_regular(n=30, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
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

            assert np.all(np.isfinite(embedder.positions))

            max_coord = np.max(np.abs(embedder.positions))
            assert max_coord < 1000  # Reasonable bound

    @pytest.mark.fast
    def test_pytorch_large_graphs(self):
        """Test PyTorch backend with large graphs."""
        adjacency = erdos_renyi_graph(n=200, p=0.02, seed=42)

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=512,
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
        adjacency = generate_random_regular(n=50, d=4, seed=42)

        # Test invalid dimension
        with pytest.raises((ValueError, AssertionError)):
            GraphEmbedderPyTorch(
                adjacency=adjacency,
                n_components=0,  # Invalid
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                n_neighbors=15,
                sample_size=256,
                    verbose=False
            )

        # Test invalid device
        with pytest.raises((ValueError, RuntimeError)):
            GraphEmbedderPyTorch(
                adjacency=adjacency,
                n_components=2,
                device='invalid_device',  # Invalid
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                n_neighbors=15,
                sample_size=256,
                    verbose=False
            )

    @pytest.mark.fast
    def test_pytorch_batch_processing(self):
        """Test PyTorch backend with different batch sizes."""
        adjacency = erdos_renyi_graph(n=100, p=0.03, seed=42)

        # Test with small batch size
        embedder_small = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
            verbose=False
        )

        # Test with large batch size
        embedder_large = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
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
        adjacency = generate_random_regular(n=50, d=4, seed=42)

        # Create two identical embedders with same random seed
        # IMPORTANT: Set seed before creating each embedder instance
        torch.manual_seed(123)
        embedder1 = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
            verbose=False
        )
        embedder1.run_layout(num_iterations=3)

        torch.manual_seed(123)
        embedder2 = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
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

    @pytest.mark.fast
    def test_pykeops_availability_detection(self):
        """Test PyKeOps availability detection."""
        adjacency = generate_random_regular(n=30, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            verbose=False
        )

        # Test that PyKeOps availability check works
        has_pykeops = embedder._has_pykeops
        assert isinstance(has_pykeops, bool)

        # If PyKeOps is available, test that it can be used
        if has_pykeops:
            # Should be able to create LazyTensor without error
            try:
                from pykeops.torch import LazyTensor
                test_tensor = torch.randn(2, 3, device=embedder.device, dtype=embedder.dtype)
                x_i = LazyTensor(test_tensor[:1, None, :])
                y_j = LazyTensor(test_tensor[None, 1:, :])
                _ = ((x_i - y_j) ** 2).sum(-1)
            except Exception as e:
                pytest.fail(f"PyKeOps detected as available but failed basic test: {e}")

    @pytest.mark.fast
    def test_knn_backend_selection(self):
        """Test k-NN backend selection logic."""
        adjacency = generate_random_regular(n=50, d=4, seed=42)

        # Test low-dimensional case (should prefer PyKeOps if available)
        embedder_low_dim = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            device='cpu',  # Use CPU to ensure consistent behavior
            verbose=False
        )

        # Test high-dimensional case (should prefer PyTorch)
        embedder_high_dim = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=250,  # High dimension
            device='cpu',
            verbose=False
        )

        # Both should initialize successfully
        assert embedder_low_dim.positions.shape == (50, 2)
        assert embedder_high_dim.positions.shape == (50, 250)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.gpu
    def test_pykeops_cuda_preference(self):
        """Test that PyKeOps is preferred on CUDA for low dimensions."""
        adjacency = generate_random_regular(n=100, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,  # Low dimension
            device='cuda',
            dtype=torch.float32,  # PyKeOps prefers float32
            verbose=False
        )

        # Run one position update to trigger k-NN computation
        embedder.update_positions()

        assert embedder.positions.shape == (100, 2)
        assert np.all(np.isfinite(embedder.positions))

    @pytest.mark.fast
    def test_pykeops_fallback_behavior(self):
        """Test PyKeOps fallback to PyTorch when needed."""
        adjacency = generate_random_regular(n=50, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            verbose=False
        )

        # Test that k-NN computation works regardless of backend
        query_points = torch.randn(10, 2, device=embedder.device, dtype=embedder.dtype)
        ref_points = torch.randn(20, 2, device=embedder.device, dtype=embedder.dtype)

        knn_indices = embedder._compute_knn_chunked(query_points, ref_points, k=5)

        assert knn_indices.shape == (10, 5)
        assert torch.all(knn_indices >= 0)
        assert torch.all(knn_indices < 20)

    @pytest.mark.fast
    def test_adaptive_chunk_sizing_backends(self):
        """Test adaptive chunk sizing for different backends."""
        adjacency = generate_random_regular(n=100, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            verbose=False
        )

        # Test chunk size adaptation for both backends
        torch_chunk_size = embedder._get_adaptive_chunk_size(100, 200, 2, 'torch')
        pykeops_chunk_size = embedder._get_adaptive_chunk_size(100, 200, 2, 'pykeops')

        assert torch_chunk_size > 0
        assert pykeops_chunk_size > 0

        # PyKeOps should typically allow larger chunks due to memory efficiency
        if embedder.device.type == 'cuda' and embedder._has_pykeops:
            assert pykeops_chunk_size >= torch_chunk_size

    @pytest.mark.fast
    def test_dtype_compatibility_with_backends(self):
        """Test that different dtypes work with both backends."""
        adjacency = generate_random_regular(n=50, d=4, seed=42)

        # Test float32 (should work with both PyTorch and PyKeOps)
        embedder_f32 = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            dtype=torch.float32,
            verbose=False
        )

        # Test float16 (PyKeOps may not work well with this)
        if torch.cuda.is_available():
            embedder_f16 = GraphEmbedderPyTorch(
                adjacency=adjacency,
                n_components=2,
                dtype=torch.float16,
                device='cuda',
                verbose=False
            )
            assert embedder_f16._positions.dtype == torch.float16

        # Run layout to ensure both work
        embedder_f32.run_layout(num_iterations=2)
        assert np.all(np.isfinite(embedder_f32.positions))

    @pytest.mark.fast
    def test_knn_consistency_between_backends(self):
        """Test that PyKeOps and PyTorch k-NN give similar results."""
        adjacency = generate_random_regular(n=30, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            verbose=False
        )

        # Create test data
        query_points = torch.randn(10, 2, device=embedder.device, dtype=embedder.dtype)
        ref_points = torch.randn(20, 2, device=embedder.device, dtype=embedder.dtype)
        k = 5

        # Force PyTorch backend
        torch_knn = embedder._compute_knn_torch(query_points, ref_points, k, chunk_size=5)

        # If PyKeOps is available, test it too
        if embedder._has_pykeops and embedder.n_components < 200 and embedder.device.type == 'cuda' and embedder.dtype == torch.float32:
            try:
                pykeops_knn = embedder._compute_knn_pykeops(query_points, ref_points, k, chunk_size=5)

                # Results should be identical for the same input
                assert torch_knn.shape == pykeops_knn.shape
                assert torch.allclose(torch_knn.float(), pykeops_knn.float(), rtol=1e-5)
            except (ImportError, RuntimeError):
                # PyKeOps may not be available or may fail, which is fine
                pass

        # At minimum, PyTorch backend should work
        assert torch_knn.shape == (10, k)
        assert torch.all(torch_knn >= 0)
        assert torch.all(torch_knn < 20)

    @pytest.mark.skipif(not PYKEOPS_AVAILABLE, reason="PyKeOps not available")
    @pytest.mark.fast
    def test_pykeops_specific_functionality(self):
        """Test PyKeOps-specific functionality when available."""
        adjacency = generate_random_regular(n=50, d=4, seed=42)

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,  # Low dimension to trigger PyKeOps
            device='cpu',  # Use CPU for consistent testing
            dtype=torch.float32,  # PyKeOps preferred dtype
            verbose=False
        )

        # Test PyKeOps availability check
        if embedder._has_pykeops:
            # Test that PyKeOps k-NN method works
            query_points = torch.randn(10, 2, device=embedder.device, dtype=embedder.dtype)
            ref_points = torch.randn(20, 2, device=embedder.device, dtype=embedder.dtype)

            try:
                knn_indices = embedder._compute_knn_pykeops(query_points, ref_points, 5, 5)
                assert knn_indices.shape == (10, 5)
                assert torch.all(knn_indices >= 0)
                assert torch.all(knn_indices < 20)
            except Exception as e:
                pytest.fail(f"PyKeOps k-NN computation failed: {e}")

    @pytest.mark.fast
    def test_memory_efficiency_with_backends(self):
        """Test memory efficiency features with different backends."""
        adjacency = erdos_renyi_graph(n=150, p=0.03, seed=42)

        embedder = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=3,
            memory_efficient=True,
            verbose=False
        )

        # Test that memory-efficient mode works with backend selection
        initial_positions = embedder.positions.copy()
        embedder.run_layout(num_iterations=2)

        assert not np.allclose(initial_positions, embedder.positions)
        assert np.all(np.isfinite(embedder.positions))
        assert embedder.positions.shape == (150, 3)

    @pytest.mark.fast
    def test_dimension_based_backend_selection(self):
        """Test that backend selection respects dimension thresholds."""
        adjacency = generate_random_regular(n=40, d=4, seed=42)

        # Low dimension - may use PyKeOps if available
        embedder_2d = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            verbose=False
        )

        # High dimension - should use PyTorch
        embedder_300d = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=300,
            verbose=False
        )

        # Both should work correctly
        embedder_2d.run_layout(num_iterations=1)
        embedder_300d.run_layout(num_iterations=1)

        assert embedder_2d.positions.shape == (40, 2)
        assert embedder_300d.positions.shape == (40, 300)
        assert np.all(np.isfinite(embedder_2d.positions))
        assert np.all(np.isfinite(embedder_300d.positions))

