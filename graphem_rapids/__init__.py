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
    generate_er,
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
    generate_complete_bipartite_graph,
    generate_relaxed_caveman,
    generate_delaunay_triangulation
)
from .influence import (
    InfluenceEstimate,
    degree_discount_seed_selection,
    estimate_independent_cascade,
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
__version__ = '0.3.0.dev0'

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
    import cupy
    import cuvs
    _RAPIDS_AVAILABLE = True
    _CUVS_AVAILABLE = True
except ImportError:
    pass

# Conditional import for cuVS backend (design choice)
if _RAPIDS_AVAILABLE and _CUVS_AVAILABLE:
    from .backends.embedder_cuvs import GraphEmbedderCuVS  # pylint: disable=wrong-import-position
else:
    GraphEmbedderCuVS = None


def create_graphem(
    adjacency=None,
    n_components=2,
    backend=None,
    edges=None,
    n_vertices=None,
    **kwargs
):
    """
    Create a GraphEmbedder with automatic backend selection.

    Parameters
    ----------
    adjacency : array-like or scipy.sparse matrix
        Adjacency matrix (n_vertices × n_vertices). Can be sparse or dense.
        For unweighted graphs, should contain 1s for edges, 0s otherwise.
    n_components : int, default=2
        Number of components (dimensions) in the embedding.
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
    >>> # Generate sparse adjacency matrix
    >>> adjacency = gr.generate_er(n=500, p=0.01)
    >>> embedder = gr.create_graphem(adjacency, n_components=3)
    >>> embedder.run_layout(num_iterations=50)
    >>> embedder.display_layout()
    """
    if (adjacency is None) == (edges is None):
        raise ValueError("provide exactly one of adjacency or edges")
    if adjacency is not None:
        graph_n_vertices = adjacency.shape[0]
    elif n_vertices is not None:
        graph_n_vertices = int(n_vertices)
    else:
        raise ValueError("n_vertices is required with a GPU edge list")

    # Configure backend
    config = BackendConfig(
        n_vertices=graph_n_vertices,
        n_components=n_components
    )
    config.force_backend = backend

    # Get optimal backend
    optimal_backend = get_optimal_backend(config)

    # Create embedder with selected backend
    if optimal_backend == 'cuvs' and _CUVS_AVAILABLE and GraphEmbedderCuVS is not None:
        return GraphEmbedderCuVS(
            adjacency=adjacency,
            edges=edges,
            n_vertices=graph_n_vertices,
            n_components=n_components,
            **kwargs,
        )

    if edges is not None:
        raise ImportError("GPU edge-list input requires the cuVS backend")

    if optimal_backend in ['pytorch', 'cuda'] and _TORCH_AVAILABLE:
        return GraphEmbedderPyTorch(adjacency, n_components, **kwargs)

    # Fallback to PyTorch CPU
    kwargs['device'] = 'cpu'
    return GraphEmbedderPyTorch(adjacency, n_components, **kwargs)


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
    'generate_er',
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
    'generate_complete_bipartite_graph',
    'generate_relaxed_caveman',
    'generate_delaunay_triangulation',

    # Influence maximization
    'InfluenceEstimate',
    'degree_discount_seed_selection',
    'estimate_independent_cascade',
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


# Library imports stay silent so benchmark stdout can be machine-readable. The
# console entry point remains available, and interactive users may opt in.
if os.environ.get('GRAPHEM_RAPIDS_SHOW_INFO', 'false').lower() == 'true':
    try:
        _show_backend_info()
    except Exception:  # pylint: disable=broad-exception-caught
        pass  # Silently fail if there are import issues
