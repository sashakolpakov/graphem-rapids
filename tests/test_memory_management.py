"""Tests for memory management utilities."""

from unittest.mock import patch
import pytest
from graphem_rapids.utils.memory_management import (
    MemoryManager,
    get_optimal_chunk_size,
    get_gpu_memory_info,
    cleanup_gpu_memory,
    monitor_memory_usage,
    check_memory_requirements
)


class TestMemoryManager:
    """Test MemoryManager context manager."""

    @pytest.mark.fast
    def test_memory_manager_initialization(self):
        """Test MemoryManager initialization."""
        manager = MemoryManager(cleanup_on_exit=True)
        assert manager.cleanup_on_exit is True

        manager = MemoryManager(cleanup_on_exit=False)
        assert manager.cleanup_on_exit is False

    @pytest.mark.fast
    def test_memory_manager_context(self):
        """Test MemoryManager as context manager."""
        with MemoryManager(cleanup_on_exit=False) as manager:
            assert isinstance(manager, MemoryManager)

    @pytest.mark.fast
    @patch('graphem_rapids.utils.memory_management.cleanup_gpu_memory')
    def test_memory_manager_cleanup_on_exit(self, mock_cleanup):
        """Test MemoryManager cleanup on exit."""
        with MemoryManager(cleanup_on_exit=True):
            pass
        mock_cleanup.assert_called_once()

    @pytest.mark.fast
    @patch('graphem_rapids.utils.memory_management.cleanup_gpu_memory')
    def test_memory_manager_no_cleanup(self, mock_cleanup):
        """Test MemoryManager without cleanup."""
        with MemoryManager(cleanup_on_exit=False):
            pass
        mock_cleanup.assert_not_called()


class TestChunkSizeOptimization:
    """Test chunk size optimization functions."""

    @pytest.mark.fast
    def test_chunk_size_positive(self):
        """Test chunk size is always positive."""
        for n_vertices in [100, 1000, 10000, 100000]:
            for dimension in [2, 3, 10]:
                chunk_size = get_optimal_chunk_size(n_vertices, dimension)
                assert chunk_size > 0

    @pytest.mark.fast
    def test_chunk_size_reasonable(self):
        """Test chunk size is reasonable."""
        # Small graphs should have small chunks
        small_chunk = get_optimal_chunk_size(100, 2)
        assert small_chunk <= 100

        # Large graphs should have larger chunks
        large_chunk = get_optimal_chunk_size(100000, 2)
        assert large_chunk > small_chunk

    @pytest.mark.fast
    def test_chunk_size_dimension_scaling(self):
        """Test chunk size scales appropriately with dimension."""
        chunk_2d = get_optimal_chunk_size(10000, 2)
        chunk_10d = get_optimal_chunk_size(10000, 10)

        # Higher dimensions should have smaller chunks to manage memory
        assert chunk_10d <= chunk_2d

    @pytest.mark.fast
    def test_chunk_size_bounds(self):
        """Test chunk size respects bounds."""
        # Should never exceed total vertices
        for n_vertices in [50, 500, 5000]:
            chunk_size = get_optimal_chunk_size(n_vertices, 2)
            assert chunk_size <= n_vertices

        # Should have reasonable minimum
        chunk_size = get_optimal_chunk_size(1000000, 2)
        assert chunk_size >= 100  # At least 100


class TestGPUMemoryInfo:
    """Test GPU memory information functions."""

    @pytest.mark.fast
    def test_gpu_memory_info_structure(self):
        """Test GPU memory info returns correct structure."""
        info = get_gpu_memory_info()

        required_keys = ['total', 'allocated', 'cached', 'free', 'available']
        for key in required_keys:
            assert key in info

        # All values should be non-negative
        for key in required_keys:
            assert info[key] >= 0

        assert isinstance(info['available'], bool)

    @pytest.mark.fast
    def test_gpu_memory_info_no_torch(self):
        """Test GPU memory info when torch not available."""
        # This test relies on ImportError handling in get_gpu_memory_info
        # We can't easily mock the import, so we'll just test the current behavior
        info = get_gpu_memory_info()

        # Should work regardless of torch availability
        required_keys = ['total', 'allocated', 'cached', 'free', 'available']
        for key in required_keys:
            assert key in info
            assert info[key] >= 0

    @pytest.mark.fast
    def test_gpu_memory_info_structure_validation(self):
        """Test GPU memory info structure validation."""
        info = get_gpu_memory_info()

        # Basic structure validation
        required_keys = ['total', 'allocated', 'cached', 'free', 'available']
        for key in required_keys:
            assert key in info
            if key != 'available':
                assert isinstance(info[key], (int, float))
                assert info[key] >= 0
        assert isinstance(info['available'], bool)

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_gpu_memory_info_real(self):
        """Test GPU memory info with real GPU (if available)."""
        info = get_gpu_memory_info()

        if info['available']:
            # If GPU is available, check reasonable values
            assert info['total'] > 0
            assert info['free'] <= info['total']
            assert info['allocated'] + info['free'] <= info['total'] + info['cached']


class TestMemoryCleanup:
    """Test memory cleanup functions."""

    @pytest.mark.fast
    def test_cleanup_gpu_memory_basic(self):
        """Test basic GPU memory cleanup functionality."""
        # Should not raise exceptions regardless of torch/CUDA availability
        try:
            cleanup_gpu_memory()
            # If no exception, test passes
            assert True
        except Exception as e:  # pylint: disable=broad-exception-caught
            pytest.fail(f"cleanup_gpu_memory() raised {type(e).__name__}: {e}")


class TestMemoryMonitoring:
    """Test memory monitoring decorator."""

    @pytest.mark.fast
    def test_memory_monitor_decorator(self):
        """Test memory monitoring decorator."""

        @monitor_memory_usage
        def dummy_function(x, y):
            return x + y

        # Should work normally
        result = dummy_function(2, 3)
        assert result == 5

    @pytest.mark.fast
    @patch('graphem_rapids.utils.memory_management.get_gpu_memory_info')
    def test_memory_monitor_with_logging(self, mock_memory_info):
        """Test memory monitoring with logging."""
        mock_memory_info.return_value = {
            'total': 8.0, 'allocated': 2.0, 'cached': 1.0,
            'free': 5.0, 'available': True
        }

        @monitor_memory_usage
        def dummy_function():
            return "test"

        result = dummy_function()
        assert result == "test"
        assert mock_memory_info.call_count >= 1


class TestMemoryRequirements:
    """Test memory requirements checking."""

    @pytest.mark.fast
    @patch('graphem_rapids.utils.memory_management.get_gpu_memory_info')
    def test_memory_requirements_sufficient(self, mock_memory_info):
        """Test memory requirements when sufficient memory."""
        mock_memory_info.return_value = {
            'total': 8.0, 'allocated': 2.0, 'cached': 1.0,
            'free': 5.0, 'available': True
        }

        result = check_memory_requirements(
            n_vertices=1000,
            dimension=2,
            backend='pytorch'
        )

        assert result['sufficient'] is True
        assert result['recommendation'] == 'pytorch'
        assert 'required_gb' in result

    @pytest.mark.fast
    def test_memory_requirements_large_graph(self):
        """Test memory requirements for large graphs."""
        result = check_memory_requirements(
            n_vertices=100000,  # Large graph
            dimension=3,
            backend='pytorch'
        )

        # Should return valid structure regardless of hardware
        required_keys = ['required_gb', 'available_gb', 'sufficient', 'recommendation', 'estimated_chunk_size']
        for key in required_keys:
            assert key in result

        assert isinstance(result['sufficient'], bool)
        assert result['recommendation'] in ['cpu', 'pytorch', 'pytorch_chunked']
        assert result['required_gb'] > 0

    @pytest.mark.fast
    def test_memory_requirements_cpu_backend(self):
        """Test memory requirements for CPU backend."""
        result = check_memory_requirements(
            n_vertices=1000,
            dimension=2,
            backend='cpu'
        )

        # CPU backend should always be available
        assert result['sufficient'] is True
        assert result['recommendation'] == 'cpu'

    @pytest.mark.fast
    @patch('graphem_rapids.utils.memory_management.get_gpu_memory_info')
    def test_memory_requirements_no_gpu(self, mock_memory_info):
        """Test memory requirements when no GPU available."""
        mock_memory_info.return_value = {
            'total': 0.0, 'allocated': 0.0, 'cached': 0.0,
            'free': 0.0, 'available': False
        }

        result = check_memory_requirements(
            n_vertices=1000,
            dimension=2,
            backend='pytorch'
        )

        assert result['recommendation'] == 'cpu'

    @pytest.mark.slow
    def test_memory_requirements_scaling(self):
        """Test memory requirements scale with problem size."""
        small_result = check_memory_requirements(
            n_vertices=1000,
            dimension=2,
            backend='cpu'
        )

        large_result = check_memory_requirements(
            n_vertices=10000,
            dimension=3,
            backend='cpu'
        )

        assert large_result['required_gb'] > small_result['required_gb']
        assert large_result['estimated_chunk_size'] >= small_result['estimated_chunk_size']