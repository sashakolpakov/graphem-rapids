"""
Backend selection utilities for GraphEm Rapids.

This module provides automatic backend selection based on data characteristics,
hardware availability, and performance considerations.
"""

import os
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BackendConfig:
    """Configuration for backend selection."""
    n_vertices: int
    dimension: int = 2
    force_backend: str = None
    prefer_gpu: bool = True
    memory_limit: float = None  # GB
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.force_backend and self.force_backend not in ['pytorch', 'cuvs', 'cpu', 'auto']:
            raise ValueError(f"Invalid backend: {self.force_backend}")


def check_torch_availability():
    """Check PyTorch availability and CUDA support."""
    info = {
        'available': False,
        'cuda_available': False,
        'cuda_device_count': 0,
        'cuda_device_name': None,
        'memory_gb': 0.0,
        'compute_capability': None
    }

    try:
        import torch  # pylint: disable=import-outside-toplevel
        info['available'] = True
        info['cuda_available'] = torch.cuda.is_available()

        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['compute_capability'] = torch.cuda.get_device_capability(0)

    except ImportError:
        logger.debug("PyTorch not available")

    return info


def check_rapids_availability():
    """Check RAPIDS availability and cuVS support."""
    info = {
        'available': False,
        'cuvs_available': False,
        'cuml_available': False,
        'cudf_available': False,
        'version': None
    }

    try:
        import cudf  # pylint: disable=import-outside-toplevel
        info['available'] = True
        info['cudf_available'] = True
        info['version'] = cudf.__version__

        try:
            import cuml  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
            info['cuml_available'] = True
        except ImportError:
            pass

        try:
            import cuvs  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
            info['cuvs_available'] = True
        except ImportError:
            pass

    except ImportError:
        logger.debug("RAPIDS not available")

    return info


def get_data_complexity_score(config):
    """
    Calculate complexity score based on data characteristics.

    Parameters
    ----------
    config : BackendConfig
        Backend configuration.

    Returns
    -------
    float
        Complexity score (0-1, higher means more complex).
    """
    # Base complexity from vertex count
    vertex_score = min(config.n_vertices / 1_000_000, 1.0)

    # Dimension penalty (higher dimensions are more expensive)
    dimension_score = min(config.dimension / 10.0, 1.0)

    # Combined score
    complexity = (vertex_score * 0.8) + (dimension_score * 0.2)

    return complexity


def get_optimal_backend(config):
    """
    Select optimal backend based on configuration and hardware.

    Parameters
    ----------
    config : BackendConfig
        Backend configuration.

    Returns
    -------
    str
        Optimal backend name ('pytorch', 'cuvs', 'cpu').
    """
    # Check forced backend
    if config.force_backend:
        if config.force_backend == 'auto':
            pass  # Continue with automatic selection
        else:
            return config.force_backend

    # Check hardware availability
    torch_info = check_torch_availability()
    rapids_info = check_rapids_availability()

    # Calculate complexity
    complexity = get_data_complexity_score(config)

    if config.verbose:
        logger.info("Data complexity score: %.3f", complexity)
        logger.info("PyTorch available: %s", torch_info['available'])
        logger.info("CUDA available: %s", torch_info['cuda_available'])
        logger.info("RAPIDS available: %s", rapids_info['available'])
        logger.info("cuVS available: %s", rapids_info['cuvs_available'])

    # Selection logic based on DiRe Rapids patterns
    if config.n_vertices > 100_000 and rapids_info['cuvs_available'] and config.prefer_gpu:
        # Large datasets: prefer cuVS if available
        if config.verbose:
            logger.info("Selected backend: cuVS (large dataset)")
        return 'cuvs'

    if config.n_vertices > 10_000 and torch_info['cuda_available'] and config.prefer_gpu:
        # Medium datasets: prefer PyTorch with CUDA
        if config.memory_limit:
            estimated_memory = estimate_memory_usage(config)
            if estimated_memory > config.memory_limit:
                if config.verbose:
                    logger.warning("Estimated memory usage (%.1fGB) exceeds limit", estimated_memory)
                return 'cpu'

        if config.verbose:
            logger.info("Selected backend: PyTorch CUDA (medium dataset)")
        return 'pytorch'

    if torch_info['available']:
        # Small datasets or no GPU: PyTorch CPU
        if config.verbose:
            logger.info("Selected backend: PyTorch CPU (small dataset or no GPU)")
        return 'pytorch'

    # Fallback
    if config.verbose:
        logger.warning("No suitable backend found, falling back to CPU")
    return 'cpu'


def estimate_memory_usage(config):
    """
    Estimate memory usage in GB for given configuration.

    Parameters
    ----------
    config : BackendConfig
        Backend configuration.

    Returns
    -------
    float
        Estimated memory usage in GB.
    """
    # Rough estimates based on graph embedding requirements
    n = config.n_vertices
    d = config.dimension

    # Position matrix: n × d × 4 bytes (float32)
    position_memory = n * d * 4

    # Edge-related operations (assume sparse graph with O(n) edges)
    edge_memory = n * 8  # Typical sparse graph

    # Force computation buffers
    force_memory = n * d * 4 * 3  # Multiple force arrays

    # KNN operations (temporary arrays)
    knn_memory = min(n * 100, 10_000_000) * 4  # Capped at reasonable limit

    # Total in bytes, convert to GB with safety margin
    total_bytes = position_memory + edge_memory + force_memory + knn_memory
    total_gb = total_bytes / (1024**3) * 1.5  # 50% safety margin

    return total_gb


def log_backend_selection(config, selected_backend):
    """
    Log backend selection details.

    Parameters
    ----------
    config : BackendConfig
        Backend configuration.
    selected_backend : str
        Selected backend name.
    """
    torch_info = check_torch_availability()
    rapids_info = check_rapids_availability()

    print("\n=== GraphEm Rapids Backend Selection ===")
    print(f"Dataset: {config.n_vertices:,} vertices, {config.dimension}D")
    print(f"Complexity score: {get_data_complexity_score(config):.3f}")
    print(f"Estimated memory: {estimate_memory_usage(config):.1f} GB")
    print("\nHardware:")
    print(f"  PyTorch: {'✓' if torch_info['available'] else '✗'}")
    if torch_info['cuda_available']:
        print(f"  CUDA: ✓ ({torch_info['cuda_device_name']}, {torch_info['memory_gb']:.1f} GB)")
    else:
        print("  CUDA: ✗")
    print(f"  RAPIDS: {'✓' if rapids_info['available'] else '✗'}")
    print(f"  cuVS: {'✓' if rapids_info['cuvs_available'] else '✗'}")
    print(f"\nSelected backend: {selected_backend.upper()}")
    print("========================================\n")


# Environment-based configuration
def get_default_config():
    """Get default configuration from environment variables."""
    return {
        'prefer_gpu': os.environ.get('GRAPHEM_PREFER_GPU', 'true').lower() == 'true',
        'force_backend': os.environ.get('GRAPHEM_BACKEND'),
        'memory_limit': float(os.environ.get('GRAPHEM_MEMORY_LIMIT', '0')) or None,
        'verbose': os.environ.get('GRAPHEM_VERBOSE', 'false').lower() == 'true'
    }