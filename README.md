# GraphEm RAPIDS

GraphEm embeds an undirected graph in a low-dimensional Euclidean space and
scores each vertex by its distance from the origin.  This repository contains
one CUDA/RAPIDS implementation of the corrected paper algorithm.

The implementation has no runtime algorithm selector.  Construction fails
when the required GPU stack, graph invariants, eigensolve, midpoint search, or
numerical checks do not pass.

## Canonical algorithm

For every run, GraphEm:

1. validates one unweighted, undirected, loop-free, duplicate-free graph;
2. computes one deterministic block-LOBPCG normalized-Laplacian eigenspace and
   drops the first eigenvector, using diagonal zero for isolated vertices and
   diagonal one for vertices of positive degree;
3. draws one deterministic uniform query-edge subset without replacement using
   the recorded PCG64/Floyd recipe;
4. on every iteration, recomputes all edge midpoints and performs exact cuVS
   brute-force search against the full midpoint array;
5. removes self-neighbours by global edge-ID equality and resolves equal
   squared distances by increasing global edge ID after adaptive exact
   overquery proves that the complete cutoff tie has been observed;
6. applies restoring spring forces;
7. detects strict crossings in the xy projection and applies the four-endpoint
   centroid force in every embedding component;
8. takes the full force step and normalizes every coordinate by its population
   standard deviation; and
9. ranks vertices by decreasing Euclidean radius.

Both force systems are mandatory.  There is no learning-rate multiplier,
displacement clipping, initial edge-length rescaling, approximate midpoint
index, cached neighbour set, alternate score orientation, or second embedder.

## Installation

The package requires a CUDA 12 RAPIDS environment with CuPy and cuVS.
The pinned 26.06 runtime also requires Python 3.11 or newer.  Disconnected
graphs and isolated vertices use the same normalized-Laplacian and force
rules; vertices without incident edges simply receive no edge force.

```bash
python -m pip install -e .
```

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
    seed=0,
)
embedder.run_layout(num_iterations=30)

positions = embedder.get_positions()
scores = embedder.get_scores()
farthest_vertices = embedder.get_top_k(50)
```

The edge-list interface is explicit:

```python
embedder = gr.GraphEmbedder(
    edges=edge_array,
    n_vertices=vertex_count,
    n_components=3,
)
```

Exactly one of `adjacency` and `edges` is accepted.

## Reproduction gates

The corrected implementation is accepted in this order:

1. CPU mathematical oracles and global-ID fixtures;
2. literal reproduction of the archived TPU/JOSS measurements;
3. corrected-ID and restoring-spring quality cells on Delaunay and ER graphs;
4. GPU parity on small fixed graphs;
5. the fixed golden paper matrix;
6. a predeclared scale ladder; and
7. independent-cascade and Ripples/IMM comparisons.

Scaling evidence is not accepted when the golden matrix fails.  Every method
emits its own success or typed failure record.  A failed comparator is never
replaced, omitted, or reported using another method's values.

The new manuscript will live with a separate checksum-verifying reproduction
suite.  Its tables, figures, result macros, and recomputation ledger will be
generated only from downloaded and verified result bundles.

## Confirmed midpoint-ID defects

- [graphem issue #7](https://github.com/sashakolpakov/graphem/issues/7)
- [graphem-rapids issue #6](https://github.com/sashakolpakov/graphem-rapids/issues/6)

The controlled repair retains the strong paper signal: at five iterations the
corrected ER-1000 cell has Spearman rho 0.9017 against degree and 0.9063 against
PageRank.  The corrected Delaunay cell improves to 0.8105 and 0.7995.

## Development checks

```bash
python -m pytest -q
python -m pylint graphem_rapids tests
python -m py_compile graphem_rapids/embedder.py
```

GPU claims require the separately sealed reproduction bundle; passing CPU
tests alone is not a performance or scientific result.

## Citation

The historical paper is [Fast Geometric Embedding for Node Influence
Maximization](https://arxiv.org/abs/2506.07435).  Revised citation metadata will
be published with the rewritten manuscript and verified result bundle.
