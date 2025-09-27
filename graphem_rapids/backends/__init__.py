"""Backend implementations for GraphEm Rapids."""

from .embedder_pytorch import GraphEmbedderPyTorch

__all__ = ['GraphEmbedderPyTorch']

# Conditionally import RAPIDS backend if available
try:
    import cuvs
    from .embedder_cuvs import GraphEmbedderCuVS
    __all__.append('GraphEmbedderCuVS')
except ImportError:
    pass