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
  <a href="https://pepy.tech/projects/graphem-rapids">
    <img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/graphem-rapids">
  </a>
</p>

High-performance [GraphEm](https://github.com/sashakolpakov/graphem) implementation using PyTorch and RAPIDS cuVS. Force-directed layout with geometric intersection detection produces embeddings that correlate strongly with centrality measures.

## Features

- **Unified API**: Scipy sparse adjacency matrices, sklearn-style parameters (`n_components`, `n_neighbors`)
- **Multiple Backends**: PyTorch (1K-100K vertices), RAPIDS cuVS (100K+ vertices), automatic selection
- **GPU Acceleration**: CUDA support, memory-efficient chunking, automatic CPU fallback
- **Graph Generators**: Erdős-Rényi, scale-free, SBM, bipartite, Delaunay, and more
- **Influence Maximization**: Fast embedding-based seed selection

## Installation

```bash
pip install graphem-rapids              # PyTorch backend
pip install graphem-rapids[cuda]        # + CUDA support
pip install graphem-rapids[rapids]      # + RAPIDS cuVS
pip install graphem-rapids[all]         # Everything
```

## Quick Start

```python
import graphem_rapids as gr

# Generate graph (returns sparse adjacency matrix)
adjacency = gr.generate_er(n=1000, p=0.01)

# Create embedder (automatic backend selection)
embedder = gr.create_graphem(adjacency, n_components=3)

# Run layout
embedder.run_layout(num_iterations=50)

# Get positions and visualize
positions = embedder.get_positions()  # numpy array (n, d)
embedder.display_layout()             # 2D or 3D plot
```

## Backend Selection

### Automatic (Recommended)
```python
embedder = gr.create_graphem(adjacency, n_components=3)
```

### Explicit PyTorch
```python
embedder = gr.GraphEmbedderPyTorch(
    adjacency, n_components=3, device='cuda',
    L_min=1.0, k_attr=0.2, k_inter=0.5, n_neighbors=10,
    batch_size=None  # Automatic (or manual: 1024)
)
```

### Explicit RAPIDS cuVS
```python
embedder = gr.GraphEmbedderCuVS(
    adjacency, n_components=3,
    index_type='auto',  # 'brute_force', 'ivf_flat', 'ivf_pq'
    sample_size=1024, batch_size=None
)
```

**Index Types**: `brute_force` (<100K), `ivf_flat` (100K-1M), `ivf_pq` (>1M vertices)

### Check Backends
```python
info = gr.get_backend_info()
print(f"CUDA: {info['cuda_available']}, Recommended: {info['recommended_backend']}")
```

## Configuration

**Environment Variables:**
```bash
export GRAPHEM_BACKEND=pytorch        # Force backend
export GRAPHEM_PREFER_GPU=true        # Prefer GPU
export GRAPHEM_MEMORY_LIMIT=8         # GB
export GRAPHEM_VERBOSE=true
```

**Programmatic:**
```python
from graphem_rapids.utils.backend_selection import BackendConfig, get_optimal_backend

config = BackendConfig(n_vertices=50000, force_backend='cuvs', memory_limit=16.0)
backend = get_optimal_backend(config)
embedder = gr.create_graphem(adjacency, backend=backend)
```

## Graph Generators

All generators return scipy sparse adjacency matrices:

```python
# Random
gr.generate_er(n=1000, p=0.01, seed=42)
gr.generate_random_regular(n=100, d=3, seed=42)

# Scale-free & small-world
gr.generate_ba(n=300, m=3, seed=42)             # Barabási-Albert
gr.generate_ws(n=1000, k=6, p=0.3, seed=42)     # Watts-Strogatz
gr.generate_scale_free(n=100, seed=42)

# Community structures
gr.generate_sbm(n_per_block=75, num_blocks=4, p_in=0.15, p_out=0.01, seed=42)
gr.generate_caveman(l=10, k=10)
gr.generate_relaxed_caveman(l=10, k=10, p=0.1, seed=42)

# Bipartite
gr.generate_bipartite_graph(n_top=50, n_bottom=100, p=0.2, seed=42)
gr.generate_complete_bipartite_graph(n_top=50, n_bottom=100)

# Geometric
gr.generate_geometric(n=100, radius=0.2, dim=2, seed=42)
gr.generate_delaunay_triangulation(n=100, seed=42)
gr.generate_road_network(width=30, height=30)   # 2D grid

# Trees
gr.generate_balanced_tree(r=2, h=10)
```

## Influence Maximization

```python
adjacency = gr.generate_er(n=1000, p=0.01)
embedder = gr.create_graphem(adjacency, n_components=3)
embedder.run_layout(num_iterations=50)

# Fast: embedding-based selection
seeds = gr.graphem_seed_selection(embedder, k=10)

# Evaluate with Independent Cascade model
import networkx as nx
G = nx.from_scipy_sparse_array(adjacency)
influence, _ = gr.ndlib_estimated_influence(G, seeds, p=0.1, iterations_count=100)

# Compare with greedy (slow, optimal)
greedy_seeds, _ = gr.greedy_seed_selection(G, k=10, p=0.1)
```

## Advanced

### Memory Management
```python
from graphem_rapids.utils.memory_management import MemoryManager, get_gpu_memory_info

mem_info = get_gpu_memory_info()
print(f"GPU: {mem_info['free']:.1f}GB free / {mem_info['total']:.1f}GB total")

adjacency = gr.generate_er(n=1000, p=0.01)
with MemoryManager(cleanup_on_exit=True):
    embedder = gr.create_graphem(adjacency)
    embedder.run_layout(50)
```

### Batch Size Tuning
```python
from graphem_rapids.utils.memory_management import get_optimal_chunk_size

adjacency = gr.generate_er(n=1000, p=0.01)

# Automatic (recommended)
embedder = gr.GraphEmbedderPyTorch(adjacency, batch_size=None)

# Manual
embedder = gr.GraphEmbedderPyTorch(adjacency, batch_size=1024)

# Programmatic
optimal = get_optimal_chunk_size(n_vertices=1000000, n_components=3, backend='pytorch')
embedder = gr.GraphEmbedderPyTorch(adjacency, batch_size=optimal)
```

## Testing & Benchmarking

```bash
pytest                                          # Run all tests
pytest tests/test_pytorch_backend.py            # Specific backend
python benchmarks/run_benchmarks.py             # Performance tests
python benchmarks/compare_backends.py --sizes 1000,10000,100000
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and contribution guidelines.

## Citation

[![arXiv](https://img.shields.io/badge/arXiv-2506.07435-b31b1b.svg)](https://arxiv.org/abs/2506.07435)

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

## License

MIT License - see [LICENSE](LICENSE) file.
