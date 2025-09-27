"""Tests for backend selection utilities."""

import os
from unittest.mock import patch
import pytest
from graphem_rapids.utils.backend_selection import (
    BackendConfig,
    get_optimal_backend,
    get_data_complexity_score,
    estimate_memory_usage,
    get_default_config,
    check_torch_availability,
    check_rapids_availability
)


class TestBackendConfig:
    """Test BackendConfig class."""

    @pytest.mark.fast
    def test_backend_config_initialization(self):
        """Test BackendConfig initialization."""
        config = BackendConfig(n_vertices=1000, dimension=2)
        assert config.n_vertices == 1000
        assert config.dimension == 2
        assert config.force_backend is None
        assert config.prefer_gpu is True
        assert config.memory_limit is None
        assert config.verbose is True

    @pytest.mark.fast
    def test_backend_config_validation(self):
        """Test BackendConfig parameter validation."""
        # Valid backends
        for backend in ['pytorch', 'cuvs', 'cpu', 'auto']:
            config = BackendConfig(n_vertices=100, force_backend=backend)
            assert config.force_backend == backend

        # Invalid backend
        with pytest.raises(ValueError, match="Invalid backend"):
            BackendConfig(n_vertices=100, force_backend="invalid")

    @pytest.mark.fast
    def test_backend_config_custom_params(self):
        """Test BackendConfig with custom parameters."""
        config = BackendConfig(
            n_vertices=5000,
            dimension=3,
            force_backend='pytorch',
            prefer_gpu=False,
            memory_limit=8.0,
            verbose=False
        )
        assert config.n_vertices == 5000
        assert config.dimension == 3
        assert config.force_backend == 'pytorch'
        assert config.prefer_gpu is False
        assert config.memory_limit == 8.0
        assert config.verbose is False


class TestComplexityScoring:
    """Test complexity scoring functions."""

    @pytest.mark.fast
    def test_complexity_score_range(self):
        """Test complexity score is in valid range."""
        for n_vertices in [100, 1000, 10000, 1000000]:
            for dimension in [2, 3, 10]:
                config = BackendConfig(n_vertices=n_vertices, dimension=dimension)
                score = get_data_complexity_score(config)
                assert 0 <= score <= 1, f"Score {score} out of range for n={n_vertices}, d={dimension}"

    @pytest.mark.fast
    def test_complexity_increases_with_size(self):
        """Test complexity increases with graph size."""
        small_config = BackendConfig(n_vertices=1000, dimension=2)
        large_config = BackendConfig(n_vertices=100000, dimension=2)

        small_score = get_data_complexity_score(small_config)
        large_score = get_data_complexity_score(large_config)

        assert large_score > small_score

    @pytest.mark.fast
    def test_complexity_increases_with_dimension(self):
        """Test complexity increases with dimension."""
        config_2d = BackendConfig(n_vertices=10000, dimension=2)
        config_10d = BackendConfig(n_vertices=10000, dimension=10)

        score_2d = get_data_complexity_score(config_2d)
        score_10d = get_data_complexity_score(config_10d)

        assert score_10d > score_2d


class TestMemoryEstimation:
    """Test memory estimation functions."""

    @pytest.mark.fast
    def test_memory_estimation_positive(self):
        """Test memory estimation returns positive values."""
        config = BackendConfig(n_vertices=1000, dimension=2)
        memory_gb = estimate_memory_usage(config)
        assert memory_gb > 0

    @pytest.mark.fast
    def test_memory_scales_with_vertices(self):
        """Test memory estimation scales with number of vertices."""
        small_config = BackendConfig(n_vertices=1000, dimension=2)
        large_config = BackendConfig(n_vertices=10000, dimension=2)

        small_memory = estimate_memory_usage(small_config)
        large_memory = estimate_memory_usage(large_config)

        assert large_memory > small_memory

    @pytest.mark.fast
    def test_memory_scales_with_dimension(self):
        """Test memory estimation scales with dimension."""
        config_2d = BackendConfig(n_vertices=5000, dimension=2)
        config_3d = BackendConfig(n_vertices=5000, dimension=3)

        memory_2d = estimate_memory_usage(config_2d)
        memory_3d = estimate_memory_usage(config_3d)

        assert memory_3d > memory_2d

    @pytest.mark.slow
    def test_memory_estimation_realistic(self):
        """Test memory estimates are realistic."""
        # Small graph should use reasonable memory
        small_config = BackendConfig(n_vertices=1000, dimension=2)
        small_memory = estimate_memory_usage(small_config)
        assert small_memory < 1.0  # Less than 1GB

        # Large graph should use more memory but not excessive
        large_config = BackendConfig(n_vertices=100000, dimension=3)
        large_memory = estimate_memory_usage(large_config)
        assert 1.0 < large_memory < 100.0  # Between 1-100GB


@patch.dict(os.environ, {}, clear=True)
class TestEnvironmentConfiguration:
    """Test environment-based configuration."""

    @pytest.mark.fast
    def test_default_environment_config(self):
        """Test default environment configuration."""
        config = get_default_config()
        assert config['prefer_gpu'] is True
        assert config['force_backend'] is None
        assert config['memory_limit'] is None
        assert config['verbose'] is False

    @pytest.mark.fast
    @patch.dict(os.environ, {'GRAPHEM_PREFER_GPU': 'false'})
    def test_environment_prefer_gpu(self):
        """Test GRAPHEM_PREFER_GPU environment variable."""
        config = get_default_config()
        assert config['prefer_gpu'] is False

    @pytest.mark.fast
    @patch.dict(os.environ, {'GRAPHEM_BACKEND': 'pytorch'})
    def test_environment_force_backend(self):
        """Test GRAPHEM_BACKEND environment variable."""
        config = get_default_config()
        assert config['force_backend'] == 'pytorch'

    @pytest.mark.fast
    @patch.dict(os.environ, {'GRAPHEM_MEMORY_LIMIT': '8.0'})
    def test_environment_memory_limit(self):
        """Test GRAPHEM_MEMORY_LIMIT environment variable."""
        config = get_default_config()
        assert config['memory_limit'] == 8.0

    @pytest.mark.fast
    @patch.dict(os.environ, {'GRAPHEM_VERBOSE': 'true'})
    def test_environment_verbose(self):
        """Test GRAPHEM_VERBOSE environment variable."""
        config = get_default_config()
        assert config['verbose'] is True


class TestHardwareChecks:
    """Test hardware availability checks."""

    @pytest.mark.fast
    def test_torch_availability_check(self):
        """Test torch availability check structure."""
        info = check_torch_availability()

        required_keys = ['available', 'cuda_available', 'cuda_device_count',
                        'cuda_device_name', 'memory_gb', 'compute_capability']
        for key in required_keys:
            assert key in info

        assert isinstance(info['available'], bool)
        assert isinstance(info['cuda_available'], bool)
        assert isinstance(info['cuda_device_count'], int)

    @pytest.mark.fast
    def test_rapids_availability_check(self):
        """Test RAPIDS availability check structure."""
        info = check_rapids_availability()

        required_keys = ['available', 'cuvs_available', 'cuml_available',
                        'cudf_available', 'version']
        for key in required_keys:
            assert key in info

        assert isinstance(info['available'], bool)
        assert isinstance(info['cuvs_available'], bool)
        assert isinstance(info['cuml_available'], bool)
        assert isinstance(info['cudf_available'], bool)


class TestBackendSelection:
    """Test backend selection logic."""

    @pytest.mark.fast
    @patch('graphem_rapids.utils.backend_selection.check_torch_availability')
    @patch('graphem_rapids.utils.backend_selection.check_rapids_availability')
    def test_forced_backend_selection(self, mock_rapids, mock_torch):
        """Test forced backend selection."""
        mock_torch.return_value = {'available': False, 'cuda_available': False}
        mock_rapids.return_value = {'available': False, 'cuvs_available': False}

        for backend in ['pytorch', 'cuvs', 'cpu']:
            config = BackendConfig(n_vertices=1000, force_backend=backend, verbose=False)
            selected = get_optimal_backend(config)
            assert selected == backend

    @pytest.mark.fast
    @patch('graphem_rapids.utils.backend_selection.check_torch_availability')
    @patch('graphem_rapids.utils.backend_selection.check_rapids_availability')
    def test_auto_selection_small_graph(self, mock_rapids, mock_torch):
        """Test automatic selection for small graphs."""
        mock_torch.return_value = {'available': True, 'cuda_available': True}
        mock_rapids.return_value = {'available': True, 'cuvs_available': True}

        config = BackendConfig(n_vertices=5000, verbose=False)
        backend = get_optimal_backend(config)
        assert backend == 'pytorch'  # Small graphs prefer PyTorch

    @pytest.mark.fast
    @patch('graphem_rapids.utils.backend_selection.check_torch_availability')
    @patch('graphem_rapids.utils.backend_selection.check_rapids_availability')
    def test_auto_selection_large_graph(self, mock_rapids, mock_torch):
        """Test automatic selection for large graphs."""
        mock_torch.return_value = {'available': True, 'cuda_available': True}
        mock_rapids.return_value = {'available': True, 'cuvs_available': True}

        config = BackendConfig(n_vertices=200000, verbose=False)
        backend = get_optimal_backend(config)
        assert backend == 'cuvs'  # Large graphs prefer cuVS

    @pytest.mark.fast
    @patch('graphem_rapids.utils.backend_selection.check_torch_availability')
    @patch('graphem_rapids.utils.backend_selection.check_rapids_availability')
    def test_fallback_no_hardware(self, mock_rapids, mock_torch):
        """Test fallback when no suitable hardware."""
        mock_torch.return_value = {'available': False, 'cuda_available': False}
        mock_rapids.return_value = {'available': False, 'cuvs_available': False}

        config = BackendConfig(n_vertices=10000, verbose=False)
        backend = get_optimal_backend(config)
        assert backend == 'cpu'

    @pytest.mark.fast
    @patch('graphem_rapids.utils.backend_selection.check_torch_availability')
    @patch('graphem_rapids.utils.backend_selection.check_rapids_availability')
    @patch('graphem_rapids.utils.backend_selection.estimate_memory_usage')
    def test_memory_limit_fallback(self, mock_memory, mock_rapids, mock_torch):
        """Test fallback to CPU when memory limit exceeded."""
        mock_torch.return_value = {'available': True, 'cuda_available': True}
        mock_rapids.return_value = {'available': True, 'cuvs_available': True}
        mock_memory.return_value = 10.0  # Require 10GB

        config = BackendConfig(n_vertices=50000, memory_limit=5.0, verbose=False)
        backend = get_optimal_backend(config)
        assert backend == 'cpu'

    @pytest.mark.slow
    def test_real_backend_selection(self):
        """Test backend selection with real hardware checks."""
        # This test uses actual hardware availability
        config = BackendConfig(n_vertices=1000, verbose=False)
        backend = get_optimal_backend(config)
        assert backend in ['pytorch', 'cuvs', 'cpu']