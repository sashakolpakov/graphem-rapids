"""Backend implementations for GraphEm Rapids."""

__all__ = []

# Load cuVS before PyTorch when both are installed.  PyTorch wheels bundle CUDA
# runtime libraries, and loading those first can shadow the newer libnvJitLink
# required by a pinned RAPIDS build in the same process.
try:
    import cuvs
    from .embedder_cuvs import GraphEmbedderCuVS
    __all__.append('GraphEmbedderCuVS')
except ImportError:
    pass

from .embedder_pytorch import GraphEmbedderPyTorch

__all__.append('GraphEmbedderPyTorch')
