# H100 validation runbook

The H100 run is a correctness and Pareto test, not a demonstration with a
preselected favorable graph. Run it from a clean checkout of the benchmark branch
and retain the JSONL output plus the exact commit hash.

## Environment

Use a RAPIDS 26.06 environment with CUDA 12 and Python 3.12 or 3.13. The scalable
path requires CuPy, cuVS, SciPy, PyTorch, and pytest; cuDF is no longer required
merely to load the backend. Install cuDF and cuGraph as well for the k-core and
ForceAtlas2 baselines. Install this repository editable, then record the environment
before running:

```bash
git rev-parse HEAD
nvidia-smi
python -c "import cupy, cuvs; print(cupy.__version__, cuvs.__version__)"
python -m pytest -q tests/test_cuvs_backend.py tests/test_influence_scalable.py
```

The cuVS tests are important. They verify that the indexed rows and returned
identifiers are edge midpoints/edge ids and that the current cuVS tuple ordering is
handled correctly.

## Smoke test

```bash
python benchmarks/benchmark_h100.py \
  --sizes 100_000:1_000_000 \
  --layout-iterations 3 \
  --spectral-iterations 4 \
  --cascade-trials 16 \
  --output h100-smoke.jsonl
```

This should complete both spring conventions without a host eigensolve. Inspect
the JSON record for non-finite or missing timings before starting the scale sweep.

## Scale sweep

```bash
python benchmarks/benchmark_h100.py \
  --sizes 1_000_000:10_000_000,10_000_000:100_000_000 \
  --layout-iterations 20 \
  --spectral-iterations 16 \
  --cascade-trials 128 \
  --with-cugraph \
  --output h100-scale.jsonl
```

The synthetic R-MAT generator is intentionally included in the measured workflow
and canonicalizes duplicate undirected edges on the GPU. For real data, also run
the same engine from a canonical two-column CuPy edge array with
`assume_canonical_edges=True`; this avoids host adjacency construction and the
synthetic generator's sort.

## Decision rules

Do not select the spring convention from correlation alone. Choose a default only
after checking all of the following:

- cascade spread confidence intervals against degree, degree-discount, and
  PageRank;
- end-to-end selection time, including initialization and layout;
- peak device memory and whether the 10M/100M case completes without fallback;
- sampled rank correlation, to identify whether GraphEm has merely reproduced
  degree at much greater cost;
- stage diagnostics. A large `midpoint_knn` share means the low-dimensional ANN
  strategy still needs replacement or multilevel amortization.

GraphEm is materially useful only if it adds cascade quality or a distinct useful
ranking at acceptable cost. If PageRank or degree-discount dominates both GraphEm
variants, the right next step is a cascade-aware geometric reranker or a multilevel
method—not a larger correlation table.

## Second-stage external baseline

After the internal harness is sound, compare the same graph and IC probabilities
against cuRipples or another reverse-reachable/sketch implementation. Record graph
loading and preprocessing separately but include them in the end-to-end total.
The old exhaustive NDlib greedy routine is retained only for API compatibility and
must not appear as the sole scalable baseline.
