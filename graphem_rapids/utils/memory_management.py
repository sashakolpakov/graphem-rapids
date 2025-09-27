"""
Memory management utilities for GraphEm Rapids.

This module provides memory optimization and monitoring utilities
for efficient graph embedding computation.
"""

import logging
import gc

logger = logging.getLogger(__name__)


def get_gpu_memory_info():
    """
    Get GPU memory information.

    Returns
    -------
    dict
        GPU memory info with keys 'total', 'allocated', 'cached', 'free' in GB.
    """
    info = {
        'total': 0.0,
        'allocated': 0.0,
        'cached': 0.0,
        'free': 0.0,
        'available': False
    }

    try:
        import torch  # pylint: disable=import-outside-toplevel
        if torch.cuda.is_available():
            info['available'] = True
            info['total'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['allocated'] = torch.cuda.memory_allocated() / (1024**3)
            info['cached'] = torch.cuda.memory_reserved() / (1024**3)
            info['free'] = info['total'] - info['allocated']
    except ImportError:
        pass

    return info


def get_optimal_chunk_size(
    n_vertices,
    dimension,
    available_memory_gb=None,
    safety_factor=0.7
):
    """
    Calculate optimal chunk size for memory-efficient processing.

    Parameters
    ----------
    n_vertices : int
        Number of vertices in the graph.
    dimension : int
        Embedding dimension.
    available_memory_gb : float, optional
        Available GPU memory in GB. If None, automatically detected.
    safety_factor : float, default=0.7
        Safety factor to avoid OOM (0-1).

    Returns
    -------
    int
        Optimal chunk size.
    """
    if available_memory_gb is None:
        gpu_info = get_gpu_memory_info()
        if gpu_info['available']:
            available_memory_gb = gpu_info['free'] * safety_factor
        else:
            # Assume 8GB for CPU systems
            available_memory_gb = 8.0 * safety_factor

    # Estimate memory per vertex (positions + forces + temporary arrays)
    bytes_per_vertex = dimension * 4 * 5  # float32, multiple arrays
    vertices_per_gb = (1024**3) / bytes_per_vertex

    # Calculate chunk size
    chunk_size = int(available_memory_gb * vertices_per_gb)

    # Ensure reasonable bounds
    min_chunk = min(1000, n_vertices)
    max_chunk = n_vertices
    chunk_size = max(min_chunk, min(chunk_size, max_chunk))

    logger.debug("Calculated chunk size: %d (available memory: %.1fGB)", chunk_size, available_memory_gb)

    return chunk_size


def cleanup_gpu_memory():
    """Clean up GPU memory by clearing cache and running garbage collection."""
    try:
        import torch  # pylint: disable=import-outside-toplevel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    # Run garbage collection
    gc.collect()


def monitor_memory_usage(func):
    """
    Decorator to monitor memory usage of a function.

    Parameters
    ----------
    func : callable
        Function to monitor.

    Returns
    -------
    callable
        Wrapped function with memory monitoring.
    """
    def wrapper(*args, **kwargs):
        # Get initial memory
        initial_info = get_gpu_memory_info()

        try:
            result = func(*args, **kwargs)

            # Get final memory
            final_info = get_gpu_memory_info()

            # Log memory usage
            if initial_info['available'] and final_info['available']:
                memory_used = final_info['allocated'] - initial_info['allocated']
                logger.info("Memory usage for %s: %.2f GB", func.__name__, memory_used)

            return result

        except Exception as e:
            # Clean up on error
            cleanup_gpu_memory()
            raise e

    return wrapper


class MemoryManager:
    """Context manager for memory management."""

    def __init__(self, cleanup_on_exit=True):
        """
        Initialize memory manager.

        Parameters
        ----------
        cleanup_on_exit : bool, default=True
            Whether to clean up memory on exit.
        """
        self.cleanup_on_exit = cleanup_on_exit
        self.initial_info = None

    def __enter__(self):
        """Enter context and record initial memory state."""
        self.initial_info = get_gpu_memory_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and optionally clean up memory."""
        if self.cleanup_on_exit:
            cleanup_gpu_memory()

        # Log memory usage
        if self.initial_info and self.initial_info['available']:
            final_info = get_gpu_memory_info()
            memory_used = final_info['allocated'] - self.initial_info['allocated']
            if abs(memory_used) > 0.01:  # Only log significant changes
                logger.info("Net memory change: %+.2f GB", memory_used)

    def get_memory_info(self):
        """Get current memory information."""
        return get_gpu_memory_info()

    def cleanup(self):
        """Manually trigger memory cleanup."""
        cleanup_gpu_memory()


def adaptive_batch_size(
    total_items,
    base_batch_size=1024,
    max_memory_gb=None
):
    """
    Calculate adaptive batch size based on available memory.

    Parameters
    ----------
    total_items : int
        Total number of items to process.
    base_batch_size : int, default=1024
        Base batch size.
    max_memory_gb : float, optional
        Maximum memory to use in GB.

    Returns
    -------
    int
        Adaptive batch size.
    """
    if max_memory_gb is None:
        gpu_info = get_gpu_memory_info()
        if gpu_info['available']:
            max_memory_gb = gpu_info['free'] * 0.8  # 80% of free memory
        else:
            max_memory_gb = 4.0  # Conservative default

    # Simple heuristic: adjust batch size based on memory
    memory_factor = max(0.1, min(2.0, max_memory_gb / 4.0))  # Scale around 4GB
    adaptive_size = int(base_batch_size * memory_factor)

    # Ensure reasonable bounds
    adaptive_size = max(64, min(adaptive_size, total_items))

    logger.debug("Adaptive batch size: %d (memory factor: %.2f)", adaptive_size, memory_factor)

    return adaptive_size


def check_memory_requirements(
    n_vertices,
    dimension,
    backend='pytorch'
):
    """
    Check if current system can handle the memory requirements.

    Parameters
    ----------
    n_vertices : int
        Number of vertices.
    dimension : int
        Embedding dimension.
    backend : str, default='pytorch'
        Backend to use.

    Returns
    -------
    dict
        Memory requirement analysis.
    """
    # Estimate memory requirements
    position_memory = n_vertices * dimension * 4  # float32 positions
    force_memory = position_memory * 2  # Force arrays
    knn_memory = min(n_vertices * 100 * 4, 1024**3)  # KNN operations, capped at 1GB
    overhead = (position_memory + force_memory) * 0.3  # 30% overhead

    total_required_bytes = position_memory + force_memory + knn_memory + overhead
    total_required_gb = total_required_bytes / (1024**3)

    # Get available memory
    gpu_info = get_gpu_memory_info()

    result = {
        'required_gb': total_required_gb,
        'available_gb': gpu_info['free'] if gpu_info['available'] else 8.0,
        'sufficient': False,
        'recommendation': 'cpu',
        'estimated_chunk_size': get_optimal_chunk_size(n_vertices, dimension)
    }

    if backend in ('cuvs', 'pytorch'):
        if gpu_info['available'] and gpu_info['free'] > total_required_gb * 1.2:
            result['sufficient'] = True
            result['recommendation'] = backend
        elif gpu_info['available'] and gpu_info['free'] > total_required_gb * 0.5:
            result['sufficient'] = True
            result['recommendation'] = f"{backend}_chunked"
        else:
            result['recommendation'] = 'cpu'
    else:
        # CPU backend - assume sufficient
        result['sufficient'] = True
        result['recommendation'] = 'cpu'

    return result