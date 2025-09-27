"""
GraphEm Rapids: A graph embedding library with PyTorch and RAPIDS acceleration.

This package provides high-performance graph embedding with multiple computational backends:
- PyTorch backend for CUDA acceleration
- RAPIDS cuVS backend for large-scale datasets
- CPU fallback for compatibility
"""

import os
import warnings

# Import backend classes and utilities
from .backends.embedder_pytorch import GraphEmbedderPyTorch
from .utils.backend_selection import get_optimal_backend, BackendConfig
from .benchmark import run_benchmark, benchmark_correlations, run_influence_benchmark
from .generators import (
    erdos_renyi_graph,
    generate_sbm,
    generate_ba,
    generate_ws,
    generate_caveman,
    generate_geometric,
    generate_scale_free,
    generate_road_network,
    generate_balanced_tree,
    generate_power_cluster,
    generate_random_regular,
    generate_bipartite_graph,
    generate_relaxed_caveman
)
from .influence import (
    graphem_seed_selection,
    ndlib_estimated_influence,
    greedy_seed_selection
)
from .visualization import (
    report_corr,
    report_full_correlation_matrix,
    plot_radial_vs_centrality,
    display_benchmark_results
)
from .datasets import load_dataset

# Version info
__version__ = '0.1.0'

# Backend availability flags
_TORCH_AVAILABLE = False
_RAPIDS_AVAILABLE = False
_CUVS_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import cudf
    import cuml
    _RAPIDS_AVAILABLE = True
    try:
        import cuvs
        _CUVS_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass

# Conditional import for cuVS backend (design choice)
if _RAPIDS_AVAILABLE and _CUVS_AVAILABLE:
    from .backends.embedder_cuvs import GraphEmbedderCuVS  # pylint: disable=wrong-import-position
else:
    GraphEmbedderCuVS = None


def create_graphem(
    edges,
    n_vertices,
    dimension=2,
    backend=None,
    **kwargs
):
    """
    Create a GraphEmbedder with automatic backend selection.

    Parameters
    ----------
    edges : array-like
        Array of edge pairs (i, j).
    n_vertices : int
        Number of vertices in the graph.
    dimension : int, default=2
        Dimension of the embedding.
    backend : str, optional
        Force specific backend ('pytorch', 'cuvs', 'auto').
        If None, automatically selects optimal backend.
    **kwargs
        Additional arguments passed to the embedder constructor.

    Returns
    -------
    embedder : GraphEmbedder
        Graph embedder instance with optimal backend.

    Examples
    --------
    >>> import graphem_rapids as gr
    >>> edges = gr.erdos_renyi_graph(n=500, p=0.01)
    >>> embedder = gr.create_graphem(edges, n_vertices=500, dimension=3)
    >>> embedder.run_layout(num_iterations=50)
    >>> embedder.display_layout()
    """
    # Configure backend
    config = BackendConfig(
        n_vertices=n_vertices,
        dimension=dimension
    )
    config.force_backend = backend

    # Get optimal backend
    optimal_backend = get_optimal_backend(config)

    # Create embedder with selected backend
    if optimal_backend == 'cuvs' and _RAPIDS_AVAILABLE and _CUVS_AVAILABLE and GraphEmbedderCuVS is not None:
        return GraphEmbedderCuVS(edges, n_vertices, dimension, **kwargs)

    if optimal_backend in ['pytorch', 'cuda'] and _TORCH_AVAILABLE:
        return GraphEmbedderPyTorch(edges, n_vertices, dimension, **kwargs)

    # Fallback to PyTorch CPU
    kwargs['device'] = 'cpu'
    return GraphEmbedderPyTorch(edges, n_vertices, dimension, **kwargs)


def get_backend_info():
    """
    Get information about available backends.

    Returns
    -------
    dict
        Dictionary with backend availability and hardware info.
    """
    info = {
        'torch_available': _TORCH_AVAILABLE,
        'rapids_available': _RAPIDS_AVAILABLE,
        'cuvs_available': _CUVS_AVAILABLE,
        'cuda_available': False,
        'cuda_device_count': 0,
        'cuda_device_name': None,
        'recommended_backend': 'cpu'
    }

    if _TORCH_AVAILABLE:
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)

            if _RAPIDS_AVAILABLE and _CUVS_AVAILABLE:
                info['recommended_backend'] = 'cuvs'
            else:
                info['recommended_backend'] = 'pytorch'

    return info


# Export main interface
__all__ = [
    # Main factory function
    'create_graphem',

    # Backend classes
    'GraphEmbedderPyTorch',

    # Graph generators
    'erdos_renyi_graph',
    'generate_sbm',
    'generate_ba',
    'generate_ws',
    'generate_caveman',
    'generate_geometric',
    'generate_scale_free',
    'generate_road_network',
    'generate_balanced_tree',
    'generate_power_cluster',
    'generate_random_regular',
    'generate_bipartite_graph',
    'generate_relaxed_caveman',

    # Influence maximization
    'graphem_seed_selection',
    'ndlib_estimated_influence',
    'greedy_seed_selection',

    # Visualization
    'report_corr',
    'report_full_correlation_matrix',
    'plot_radial_vs_centrality',
    'display_benchmark_results',

    # Datasets
    'load_dataset',

    # Utilities
    'get_backend_info',

    # Benchmark functionality
    'run_benchmark',
    'benchmark_correlations',
    'run_influence_benchmark',
]

# Add RAPIDS classes to exports if available
if _RAPIDS_AVAILABLE and _CUVS_AVAILABLE:
    __all__.append('GraphEmbedderCuVS')


# Show backend info on import
def _show_backend_info():
    info = get_backend_info()
    backend_status = []

    if info['torch_available']:
        backend_status.append("PyTorch ✓")
        if info['cuda_available']:
            backend_status.append(f"CUDA ✓ ({info['cuda_device_count']} device(s))")
        else:
            backend_status.append("CUDA ✗")
    else:
        backend_status.append("PyTorch ✗")

    if info['rapids_available']:
        backend_status.append("RAPIDS ✓")
        if info['cuvs_available']:
            backend_status.append("cuVS ✓")
        else:
            backend_status.append("cuVS ✗")
    else:
        backend_status.append("RAPIDS ✗")

    print(f"GraphEm Rapids v{__version__} - Backends: {' | '.join(backend_status)}")
    print(f"Recommended backend: {info['recommended_backend'].upper()}")


# Show info on import unless in testing environment
if not os.environ.get('GRAPHEM_RAPIDS_QUIET'):
    try:
        _show_backend_info()
    except Exception:  # pylint: disable=broad-exception-caught
        pass  # Silently fail if there are import issues