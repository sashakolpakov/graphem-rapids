<p align="center">
  <img src="images/logo.png" alt="graphem rapids logo" height="120"/>
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

GraphEm Rapids is a high-performance implementation of the [GraphEm](https://github.com/sashakolpakov/graphem) graph embedding library, leveraging PyTorch and RAPIDS cuVS for enhanced scalability and GPU acceleration. It uses a force-directed layout algorithm with geometric intersection detection to produce high-quality graph embeddings that correlate strongly with network centrality measures.

## Key Features

- **Unified Adjacency Matrix Interface**: Simple, consistent API accepting scipy sparse matrices
- **Multiple Backends**: PyTorch (CUDA/CPU), RAPIDS cuVS, with automatic selection
- **Automatic Backend Selection**: Intelligently chooses optimal backend based on graph size and hardware
- **Large-Scale Support**: Handles graphs with millions of vertices via RAPIDS cuVS
- **Memory Efficient**: Adaptive chunking, memory monitoring, and GPU memory management
- **GPU Accelerated**: Full CUDA support with PyTorch and RAPIDS cuVS indices
- **Flexible Parameters**: Renamed to sklearn-style `n_components` and `n_neighbors` for consistency

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

### Basic Usage with Automatic Backend Selection
```python
import graphem_rapids as gr

# Generate a graph (returns sparse adjacency matrix)
adjacency = gr.erdos_renyi_graph(n=1000, p=0.01)

# Create embedder with automatic backend selection
embedder = gr.create_graphem(adjacency, n_components=3)

# Run force-directed layout
embedder.run_layout(num_iterations=50)

# Get final positions
positions = embedder.get_positions()  # numpy array (n_vertices, n_components)

# Display visualization (2D or 3D)
embedder.display_layout()
```

### Explicit Backend Selection
```python
# Force PyTorch backend with custom parameters
embedder = gr.GraphEmbedderPyTorch(
    adjacency,
    n_components=3,
    device='cuda',  # or 'cpu'
    L_min=1.0,      # Minimum spring length
    k_attr=0.2,     # Attraction constant
    k_inter=0.5,    # Intersection repulsion constant
    n_neighbors=10  # Number of nearest neighbors for intersection detection
)

# Force RAPIDS cuVS backend (for large graphs > 100K vertices)
embedder = gr.GraphEmbedderCuVS(
    adjacency,
    n_components=3,
    index_type='ivf_flat',  # 'auto', 'brute_force', 'ivf_flat', 'ivf_pq'
    sample_size=1024        # Larger sample size for better accuracy
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
from graphem_rapids.utils.backend_selection import BackendConfig, get_optimal_backend

# Create configuration for backend selection
config = BackendConfig(
    n_vertices=50000,
    n_components=3,
    force_backend='cuvs',  # or 'pytorch', 'auto', 'cpu'
    memory_limit=16.0,     # GB
    prefer_gpu=True,
    verbose=True
)

# Get recommended backend
backend = get_optimal_backend(config)
print(f"Recommended backend: {backend}")

# Create embedder with specific backend
embedder = gr.create_graphem(adjacency, n_components=3, backend=backend)
```

## Influence Maximization

GraphEm Rapids includes fast influence maximization via radial distance in the embedding space:

```python
# Generate graph and compute embedding
adjacency = gr.erdos_renyi_graph(n=1000, p=0.01)
embedder = gr.create_graphem(adjacency, n_components=3)
embedder.run_layout(num_iterations=50)

# Select influential nodes using embedding-based method (fast)
seeds = gr.graphem_seed_selection(embedder, k=10)

# Evaluate influence spread using Independent Cascade model
import networkx as nx
G = nx.from_scipy_sparse_array(adjacency)
influence, _ = gr.ndlib_estimated_influence(G, seeds, p=0.1, iterations=100)
print(f"Estimated influence: {influence:.1f} nodes ({influence/len(G)*100:.1f}%)")

# Compare with greedy seed selection (slow but optimal)
greedy_seeds, _ = gr.greedy_seed_selection(G, k=10, p=0.1)
```

## Testing

Run the test suite:
```bash
pytest
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
from graphem_rapids.utils.memory_management import MemoryManager, get_gpu_memory_info

# Check available GPU memory
mem_info = get_gpu_memory_info()
print(f"GPU memory: {mem_info['free']:.1f}GB free / {mem_info['total']:.1f}GB total")

# Use context manager for automatic cleanup
with MemoryManager(cleanup_on_exit=True):
    embedder = gr.create_graphem(adjacency, n_components=3)
    embedder.run_layout(num_iterations=50)
    # GPU memory automatically cleaned up on exit
```

### Chunked Processing for Large Graphs
```python
from graphem_rapids.utils.memory_management import get_optimal_chunk_size

# Calculate optimal chunk size based on available memory
chunk_size = get_optimal_chunk_size(
    n_vertices=1000000,
    n_components=3,
    backend='pytorch'  # or 'pykeops', 'cuvs'
)
print(f"Optimal chunk size: {chunk_size}")

# PyTorch backend handles chunking automatically
embedder = gr.GraphEmbedderPyTorch(
    adjacency,
    n_components=3,
    memory_efficient=True
)
```

### cuVS Index Configuration
```python
# Fine-tune RAPIDS cuVS backend for large-scale graphs
embedder = gr.GraphEmbedderCuVS(
    adjacency,
    n_components=3,
    index_type='ivf_pq',  # 'auto', 'brute_force', 'ivf_flat', 'ivf_pq'
    sample_size=2048,     # Larger samples for accuracy (vs 1024 default)
    n_neighbors=20,       # Number of nearest neighbors for intersection detection
    L_min=1.0,            # Spring parameters
    k_attr=0.2,
    k_inter=0.5
)

# Index type selection guide:
# - 'auto': Automatic selection based on graph size (recommended)
# - 'brute_force': Exact KNN, best for < 100K vertices
# - 'ivf_flat': Good balance for 100K-1M vertices
# - 'ivf_pq': Memory-efficient for > 1M vertices
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

If you use GraphEm in research, please cite our work [![arXiv](https://img.shields.io/badge/arXiv-2506.07435-b31b1b.svg)](https://arxiv.org/abs/2506.07435)

**BibTeX:**

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

**APA Style:**

Kolpakov, A., & Rivin, I. (2025). Fast Geometric Embedding for Node Influence Maximization. arXiv preprint arXiv:2506.07435.
