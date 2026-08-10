# GraphEm RAPIDS

GraphEm embeds an undirected graph in a low-dimensional Euclidean space and
scores each vertex by its distance from the origin. This repository exposes one
canonical implementation: a Torch normalized-Laplacian initializer followed by
CuPy/cuVS force refinement on CUDA.

## Execution contract

`GraphEmbedder` always requires CuPy, cupyx, cuVS, and a working CUDA device for
graph storage, midpoint search, forces, and layout updates. Its `device`
argument selects the device used by the shared Torch spectral function:

- `device="cuda"` is the default and the required production/benchmark mode;
- an unavailable explicit CUDA request is an error and never downgrades;
- explicit `device="cpu"` uses the same Torch tensor implementation and emits a
  `RuntimeWarning`; and
- `device="auto"` selects CUDA when available and otherwise selects CPU with a
  `RuntimeWarning` and a recorded reason.

CPU spectral selection is a small-fixture diagnostic, not a CPU layout backend.
There is no second eigensolver or alternate layout implementation.

## Canonical algorithm

For every run, GraphEm:

1. validates one unweighted, undirected, loop-free, duplicate-free graph;
2. builds the float64 symmetric normalized Laplacian with the declared isolate
   convention and solves its shifted sparse eigenspace with
   `torch.lobpcg(method="ortho")`;
3. checks finite eigenpairs, residual norms, and orthogonality before dropping
   the first eigenvector;
4. draws one deterministic uniform query-edge subset without replacement using
   the recorded PCG64/Floyd recipe;
5. recomputes every edge midpoint on every iteration and performs exact cuVS
   brute-force search against the complete midpoint array, submitting at most
   64 queries per call;
6. rejects duplicate global edge IDs within each returned query row before
   negative-distance repair, removes self-neighbours by global edge identity,
   and completes cutoff ties in increasing global edge-ID order;
7. applies restoring spring forces and strict crossing forces in the first two
   coordinates, with centroid repulsion in every embedding component;
8. takes the full force step and normalizes each coordinate by its population
   standard deviation; and
9. ranks vertices by decreasing Euclidean radius, breaking score ties by vertex
   ID.

Both force systems are mandatory. There is no learning-rate multiplier,
displacement clipping, initial edge-length rescaling, approximate midpoint
index, cached neighbour set, alternate score orientation, or alternate execution
path.

## Installation

The canonical qualified environment uses Python 3.11, CUDA 12.9, Torch 2.11,
CuPy 14.1.1, and cuVS 26.06. Install the CUDA-matched Torch wheel before the
package so dependency resolution does not select a different CUDA runtime:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -e .
```

Any proposed dependency update must pass the same graph, finiteness, spectral
residual, and orthogonality gates before it is treated as supported. Production
claims must record the exact source, image, dependency, CUDA, and GPU identities.

## Usage

```python
import graphem_rapids as gr

adjacency = gr.generate_er(n=1_000, p=0.1, seed=0)
embedder = gr.GraphEmbedder(
    adjacency=adjacency,
    n_components=3,
    L_min=40.0,
    k_attr=1.0,
    k_inter=1.0,
    n_neighbors=15,
    sample_size=2_048,
    midpoint_query_batch_size=64,
    seed=0,
    device="cuda",
)
embedder.run_layout(num_iterations=30)

positions = embedder.get_positions()
scores = embedder.get_scores()
farthest_vertices = embedder.get_top_k(50)
diagnostics = embedder.get_diagnostics()
```

The edge-list interface is explicit:

```python
embedder = gr.GraphEmbedder(
    edges=edge_array,
    n_vertices=vertex_count,
    n_components=3,
    n_neighbors=15,
    sample_size=2_048,
    midpoint_query_batch_size=64,
    device="cuda",
)
```

Exactly one of `adjacency` and `edges` is accepted. An edge list must use
integer vertex IDs in `[0, n_vertices)`. For either input form,
`sample_size <= n_edges`, `n_neighbors < n_edges`, and
`n_vertices >= 3 * (n_components + 1)` must hold. The exact midpoint search
accepts `midpoint_query_batch_size` in `[1, 64]`; 64 is both the default and a
hard ceiling, so an edge-heavy run cannot restore the all-queries-at-once
allocation by configuration.

Midpoint diagnostics record the configured and effective batch sizes, the hard
policy bound, every submitted batch-size count, the number of cuVS search
calls, call-width and resolved-query-width histograms, and the highest
device-wide allocation footprint observed at declared `cudaMemGetInfo`
checkpoints. The checkpoint receipt is an in-process diagnostic; qualification
artifacts still record an external GPU-memory high-water mark. The receipt also
binds the rowwise unique-global-edge-ID validation policy that runs before any
negative-distance repair.

## Reproduction policy

The primary golden and scaling matrices are untuned. Their graph preprocessing,
dimensions, force parameters, query sizes, iteration counts, and seeds are
frozen before result inspection. Each cell emits an immutable success or typed
failure record with raw artifacts and checksums; any cell can later be replaced
by a new linked attempt without changing unrelated cells.

Tuning and graph discovery begin only after the untuned baseline is sealed.
They use disjoint cell identities, retain the paired baseline, and are reported
as post-hoc until a separate confirmation set supports a default-performance
claim. Normal floating-point drift is measured with residual, orthogonality,
eigenvalue, and subspace diagnostics; byte equality is provenance, not a
numerical acceptance gate.

Comparator, influence, and scaling failures remain typed outcomes. A failed
method is not replaced, omitted, or reported with another method's values.

## Issue lineage

- [Bounded cuVS query-batching issue #8](https://github.com/sashakolpakov/graphem/issues/8)
- [Torch/CUDA spectral initializer issue #9](https://github.com/sashakolpakov/graphem/issues/9)
- [Global midpoint-ID issue #6](https://github.com/sashakolpakov/graphem-rapids/issues/6)
- [Original GraphEm midpoint-ID issue #7](https://github.com/sashakolpakov/graphem/issues/7)

## Development checks

```bash
python -m pytest -q
python -m pylint graphem_rapids tests
python -m py_compile graphem_rapids/embedder.py
sphinx-build -W --keep-going -n -b html docs /tmp/graphem-docs
npx --yes markdownlint-cli2@0.23.2 "*.md"
```

GPU claims require a separately sealed GPU qualification artifact. Local import,
documentation, and small-fixture checks are not performance or scientific
results.

## Citation

The historical paper is [Fast Geometric Embedding for Node Influence
Maximization](https://arxiv.org/abs/2506.07435). Revised citation metadata will
be published with the rewritten manuscript and verified result bundle.
