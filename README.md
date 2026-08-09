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
  </p>

  <p align="center">
  <a href="https://pepy.tech/projects/graphem-rapids">
    <img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/graphem-rapids">
  </a>
  <a href="https://sashakolpakov.github.io/graphem-rapids/">
    <img src="https://img.shields.io/website-up-down-green-red/https/sashakolpakov.github.io/graphem-rapids?label=API%20Documentation" alt="Docs Status"/>
  </a>
</p>

High-performance [GraphEm](https://github.com/sashakolpakov/graphem)
implementation using PyTorch and RAPIDS cuVS. Force-directed layouts with
geometric intersection detection produce radial node-ranking scores for
influence experiments; a separate GPU-generic evidence workflow compares those
scores against cheaper centrality and influence-maximization baselines.

## Features

- **Unified API**: SciPy sparse adjacency matrices or GPU-resident canonical edge lists
- **Multiple Backends**: PyTorch for small/medium graphs and a CuPy/cuVS large-graph path
- **GPU-Native Scaling**: sparse randomized initialization, fused edge kernels,
  bounded midpoint indexing, and on-device top-k selection
- **Graph Generators**: Erdős-Rényi, scale-free, SBM, bipartite, Delaunay, and more
- **Influence Maximization**: embedding and degree-discount selection with reproducible batched IC evaluation

## Installation

```bash
pip install graphem-rapids              # PyTorch backend
pip install graphem-rapids[cuda]        # + CUDA support
pip install graphem-rapids[rapids]      # + RAPIDS cuVS
pip install graphem-rapids[all]         # Everything
```

The version-pinned GPU capacity, stress, ANN-recall, and influence evidence
workflow lives in the separate
[`fast-geometric-repro`](https://github.com/sashakolpakov/fast-geometric-repro)
repository. It supports diagnostic runs on compatible CUDA GPUs and has a
separate H100 production-evidence profile. This repository remains the
hardware-generic installable library and its unit tests.

> **Migration note for 0.3:** `graphem_seed_selection(embedder, k)` no longer
> performs 20 hidden layout iterations. Run the layout first, as in the example
> below, or pass `num_iterations=20` explicitly to retain the earlier behavior.

## Quick Start [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sashakolpakov/graphem-rapids/blob/main/examples/graphem_rapids_notebook.ipynb)

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

Crossing forces are 2D-only. When automatic selection chooses cuVS for a
higher-dimensional layout, `create_graphem` supplies `k_inter=0` if the caller
did not set it. An explicit nonzero `k_inter` remains an error outside 2D.

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
    adjacency, n_components=2,
    index_type='auto',
    sample_size=None,              # scale samples with graph size
    midpoint_reference_size=None,  # bounded stochastic reference set
    force_mode='legacy',           # or paper-consistent 'attractive'
    intersection_interval=5,       # refresh midpoint ANN every five steps
    max_candidate_pairs=8_388_608, # fail-fast candidate-memory bound
)
```

Crossing forces are geometric segment operations and are therefore defined only
in 2D. For a higher-dimensional spring-only layout, pass (for example)
`n_components=4, k_inter=0` explicitly.

The index is rebuilt over current **edge midpoints**, and returned row ids are edge
ids. Automatic selection uses brute force below 100K midpoints and IVF-Flat above
that threshold. IVF-PQ is not selected automatically because product quantization
is a poor fit for the usual 2–6 dimensional layout.

For graphs that should never pass through a host adjacency matrix:

```python
import cupy as cp

# One canonical row (u < v) per undirected edge, already on the GPU.
edges = cp.asarray([[0, 1], [1, 2], [2, 3]], dtype=cp.int32)
embedder = gr.create_graphem(
    edges=edges,
    n_vertices=4,
    backend='cuvs',
    n_components=2,
    assume_canonical_edges=True,
    initialization='randomized',
)
embedder.run_layout(20)
top_nodes = embedder.topk_nodes(2)  # transfers two ids, not all positions
```

The paper's Hooke equation and the v0.2.x implementation use opposite signs.
`force_mode='legacy'` preserves existing results; `force_mode='attractive'`
implements the written equation. Keep this choice explicit until a quality sweep
selects a default.

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

# Evaluate 256 independent cascades (CPU or CuPy CUDA backend)
estimate = gr.estimate_independent_cascade(
    adjacency, seeds, p=0.1, n_simulations=256, random_seed=42
)
print(estimate.mean, estimate.stderr)

# Scalable cascade-aware baseline
discount_seeds = gr.degree_discount_seed_selection(adjacency, k=10, p=0.1)
```

The NDlib and exhaustive greedy functions remain compatibility helpers. Their
`iterations_count` is a diffusion time-step limit, not a Monte Carlo trial count,
and the old greedy routine is not a scalable or exact baseline.

The CUDA evaluator uses a packed visited bitset and compact activation queue.
When an external allocator such as RMM owns reusable memory, callers can pass an
allocator-aware ``available_memory_bytes`` value; generated immutable CSR inputs
may opt into ``assume_validated_csr=True`` to avoid rescanning every edge for each
seed method.

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
python benchmarks/diagnose_force_modes.py        # Paper vs legacy spring sign
```

Use
[`fast-geometric-repro`](https://github.com/sashakolpakov/fast-geometric-repro)
for the fail-closed GPU runbook, reviewer/engineering triage, external IMM
adapter, environment lock, and evidence schemas. H100 is one explicitly scoped
production benchmark profile, not a package requirement.

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
