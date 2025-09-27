<p align="center">
  <img src="images/logo.png" alt="graphem rapids logo" height="240"/>
</p>

<h1 align="center">GraphEm Rapids: High-Performance Graph Embedding</h1>

<p align="center">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"/>
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"/>
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch 2.0+"/>
  </a>
  <a href="https://rapids.ai/">
    <img src="https://img.shields.io/badge/RAPIDS-cuVS-76B900.svg" alt="RAPIDS cuVS"/>
  </a>
</p>

GraphEm Rapids is a high-performance implementation of the GraphEm graph embedding library, with PyTorch and RAPIDS for enhanced scalability and GPU acceleration.

## Key Features

- **Multiple Backends**: PyTorch, RAPIDS cuVS, and CPU fallback
- **Automatic Backend Selection**: Optimal backend chosen based on data size and hardware
- **Large-Scale Support**: Handles graphs with millions of vertices using RAPIDS
- **Memory Efficient**: Adaptive chunking and memory management
- **GPU Accelerated**: Full CUDA support with PyTorch and RAPIDS

## Installation

### Basic Installation (PyTorch backend)
```bash
pip install graphem-rapids
```

### With CUDA Support
```bash
pip install graphem-rapids[cuda]
```

### With Full RAPIDS Support
```bash
pip install graphem-rapids[rapids]
# or for everything
pip install graphem-rapids[all]
```

### Development Installation
```bash
git clone https://github.com/sashakolpakov/graphem-rapids.git
cd graphem-rapids
pip install -e .
```

## Quick Start

### Automatic Backend Selection
```python
import graphem_rapids as gr

# Generate a graph
edges = gr.erdos_renyi_graph(n=10000, p=0.001)

# Create embedder with automatic backend selection
embedder = gr.create_graphem(edges, n_vertices=10000, dimension=3)

# Run layout
embedder.run_layout(num_iterations=50)

# Display
embedder.display_layout()
```

### Explicit Backend Selection
```python
# Force PyTorch backend
embedder = gr.GraphEmbedderPyTorch(
    edges, n_vertices=10000, dimension=3,
    device='cuda'  # or 'cpu'
)

# Force RAPIDS cuVS backend (for large graphs)
embedder = gr.GraphEmbedderCuVS(
    edges, n_vertices=100000, dimension=3,
    index_type='ivf_flat'
)
```

### Backend Information
```python
# Check available backends
info = gr.get_backend_info()
print(f"CUDA available: {info['cuda_available']}")
print(f"Recommended: {info['recommended_backend']}")
```

## Architecture

GraphEm Rapids provides multiple computational backends:

### PyTorch Backend
- **Best for**: Medium-scale graphs (1K-100K vertices)
- **Features**: CUDA acceleration, memory-efficient chunking
- **Fallback**: Automatic CPU mode when GPU unavailable

### RAPIDS cuVS Backend
- **Best for**: Large-scale graphs (100K+ vertices)
- **Features**: Optimized KNN with cuVS indices, CuPy operations
- **Index Types**: Brute force, IVF-Flat, IVF-PQ (automatic selection)

### Automatic Selection
The `create_graphem()` function automatically selects the optimal backend based on:
- Dataset size (number of vertices)
- Available hardware (CUDA, RAPIDS)
- Memory constraints
- User preferences

## Configuration

### Environment Variables
```bash
export GRAPHEM_BACKEND=pytorch     # Force backend
export GRAPHEM_PREFER_GPU=true     # Prefer GPU backends
export GRAPHEM_MEMORY_LIMIT=8      # Memory limit in GB
export GRAPHEM_VERBOSE=true        # Verbose logging
export GRAPHEM_RAPIDS_QUIET=true   # Suppress startup messages
```

### Programmatic Configuration
```python
from graphem_rapids.utils import BackendConfig

config = BackendConfig(
    n_vertices=50000,
    dimension=3,
    force_backend='cuvs',
    memory_limit=16.0,  # GB
    prefer_gpu=True
)

embedder = gr.create_graphem(edges, n_vertices=50000, **config.__dict__)
```

## Influence Maximization

GraphEm Rapids maintains full compatibility with influence maximization algorithms:

```python
# Select influential nodes using embedding-based method
seeds = gr.graphem_seed_selection(embedder, k=10)

# Compare with traditional methods
import networkx as nx
G = nx.from_edgelist(edges)
influence, _ = gr.ndlib_estimated_influence(G, seeds, p=0.1)
print(f"Estimated influence: {influence} nodes")
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Test specific backends:
```bash
pytest tests/test_pytorch_backend.py
pytest tests/test_cuvs_backend.py
```

## Benchmarking

Run performance benchmarks:
```bash
python benchmarks/run_benchmarks.py
```

Compare backends:
```bash
python benchmarks/compare_backends.py --sizes 1000,10000,100000
```

## Advanced Usage

### Custom Memory Management
```python
from graphem_rapids.utils import MemoryManager

with MemoryManager(cleanup_on_exit=True):
    embedder = gr.create_graphem(edges, n_vertices=50000)
    embedder.run_layout(50)
    # Automatic cleanup on exit
```

### Chunked Processing for Large Graphs
```python
from graphem_rapids.utils import get_optimal_chunk_size

chunk_size = get_optimal_chunk_size(n_vertices=1000000, dimension=3)
embedder = gr.GraphEmbedderPyTorch(
    edges, n_vertices=1000000,
    batch_size=chunk_size,
    memory_efficient=True
)
```

### cuVS Index Configuration
```python
embedder = gr.GraphEmbedderCuVS(
    edges, n_vertices=500000,
    index_type='ivf_pq',  # Options: 'brute_force', 'ivf_flat', 'ivf_pq'
    sample_size=2048,     # Larger samples for better accuracy
    batch_size=8192       # Larger batches for better throughput
)
```

## Documentation

- [API Reference](https://sashakolpakov.github.io/graphem-rapids/)
- [User Guide](docs/user_guide.md)
- [Backend Selection Guide](docs/backend_selection.md)
- [Performance Tuning](docs/performance.md)

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

If you use GraphEm Rapids in your research, please cite:

```bibtex
@misc{kolpakov-rivin-2025fast,
  title={Fast Geometric Embedding for Node Influence Maximization},
  author={Kolpakov, Alexander and Rivin, Igor},
  year={2025},
  eprint={2506.07435},
  archivePrefix={arXiv},
  primaryClass={cs.SI},
  url={https://arxiv.org/abs/2506.07435}
}
```