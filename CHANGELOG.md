# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Build cuVS indexes over edge midpoints and consume integer edge-neighbor ids;
  v0.2.x indexed vertices and interpreted distances as edge indices.
- Use the RAPIDS 26.06 parameter-object API with an explicit, diagnosed tiled
  fallback instead of silently entering a quadratic broadcast path.
- Make the paper-consistent attractive spring sign available explicitly while
  retaining the v0.2.x sign as `force_mode="legacy"` pending quality benchmarks.
- Make NDlib optional so importing graph generation and embedding does not require
  an influence-simulation package.
- Exclude isolated vertices from radial seed selection in both built-in
  backends, scale randomized initialization to the preferred edge length, and
  use isotropic normalization.
- Define crossing forces only in 2D and replace radial four-endpoint repulsion
  with a segment-separation force that can resolve a symmetric crossing.
- Canonicalize and validate CPU/GPU CSR inputs before cascade evaluation; unsafe
  CSC/BSR interpretation, explicit zeros, duplicates, and malformed indices no
  longer reach the CUDA kernel.
- Load cuVS before CuPy/PyTorch CUDA libraries, use CuPy 14's stacked lexsort
  key contract, and explicitly transfer host CSR buffers before making them
  contiguous. These fixes were exercised on the pinned RAPIDS 26.06 H100 stack.

### Added

- GPU-resident canonical edge-list input and randomized sparse spectral
  initialization for graphs that cannot pass through SciPy or a CPU eigensolver.
- Fused FP32 spring and crossing accumulation, bounded midpoint references and
  candidate arrays, configurable crossing intervals, on-device active-only
  radial top-k, and stage diagnostics.
- Reproducible CPU/CuPy Independent Cascade evaluation with independent Monte
  Carlo trials, a packed visited bitset and compact activation queue, plus
  degree-discount and geometric-diversity seed selection.
- Stable diagnostics, explicit ANN controls, allocator-aware cascade planning,
  and GPU-resident input contracts consumed by the separately versioned
  hardware-generic ``fast-geometric-repro`` evidence suite. H100 remains an
  explicitly scoped production benchmark profile rather than a package target.

### Changed

- Move unused profiling tools out of the package's mandatory runtime
  dependencies and into the explicit ``profiling`` extra.
- Keep higher-dimensional ``create_graphem`` calls backend-independent by
  supplying ``k_inter=0`` when the factory selects cuVS and the caller omitted
  the 2D-only crossing-force option.
- Reject boolean, fractional, and string seed/count inputs instead of silently
  coercing them to node ids or returning the wrong number of seeds.
- Independent Cascade live-edge worlds are now keyed by
  ``(trial, source, target)`` rather than CSR storage offset. Results remain
  deterministic for a seed but intentionally differ from earlier releases and
  are invariant to canonical CSR storage order.
- ``graphem_seed_selection`` no longer runs 20 hidden layout iterations by
  default. Pass ``num_iterations`` explicitly when selection should also mutate
  the layout; pass ``num_iterations=20`` to retain the v0.2 behavior.

## [0.2.1] - 2025-11-08

### Fixed
- **Sphinx documentation**: Fixed duplicate object description warning for `GraphEmbedderPyTorch.positions`
- **NDlib integration**: Fixed critical bug in `ndlib_estimated_influence()` causing infinite hangs
  - Changed from `add_node_configuration()` to `add_model_initial_configuration("Infected", seeds)` for proper seed initialization
  - Fixed influence calculation to read from `node_count` dict instead of empty `status` delta dict
  - Eliminates "Initial infection missing" warnings
  - Seeds now correctly propagate influence through the network
- **Benchmark functions**: Fixed TypeError in `benchmark_correlations()` and `run_influence_benchmark()`
  - Replaced `len(edges)` with `adjacency.shape[0]` and `adjacency.nnz`
  - Use `nx.from_scipy_sparse_array()` instead of manual graph construction
  - Fixes "sparse array length is ambiguous" error when using benchmarks

### Added
- **Graph generators**: New generators for specialized graph types
  - `generate_bipartite_graph(n_top, n_bottom, p, seed)` - Random bipartite with edge probability control
  - `generate_complete_bipartite_graph(n_top, n_bottom)` - Complete bipartite K_{n,m}
  - `generate_delaunay_triangulation(n, seed)` - Planar graphs from Delaunay triangulation
- **Tests**: Comprehensive test coverage for all new generators
- **Documentation**: Added CONTRIBUTING.md with contribution guidelines
- **Examples**: Added Jupyter notebook with comprehensive examples and benchmarks

## [0.2.0] - 2025-10-30

### Added
- **batch_size parameter**: Added configurable `batch_size` parameter to both `GraphEmbedderPyTorch` and `GraphEmbedderCuVS` constructors
  - Default value: `None` (automatic selection based on available memory)
  - Manual override: Users can specify custom batch sizes (e.g., `batch_size=1024`) for fine-tuned memory management
  - Backward compatible: Existing code without `batch_size` continues to work with automatic selection
  - Resolves Issue #1: Parameter [batch_size] cannot be assigned

### Fixed
- **cuVS metric parameter**: Corrected distance metric parameter in cuVS index builds
  - Changed from invalid `metric='l2'` to official `metric='sqeuclidean'`
  - Affects: `brute_force.build()`, `ivf_flat.build()`, and `ivf_pq.build()`
  - Ensures compatibility with cuVS API specification
  - No user-facing changes (internal fix)

### Changed
- Unified internal batch size handling: Eliminated redundant `chunk_size` attribute in favor of consistent `batch_size` usage throughout codebase
- Improved logging: Added informative log messages for automatic vs. manual batch size selection

### Documentation
- Updated README.md with comprehensive `batch_size` parameter documentation
- Added "Batch Size Configuration for Large Graphs" section with examples
- Updated all code examples to include `batch_size` parameter
- Added batch size usage guide explaining when to use automatic vs. manual values

### Testing
- All 98 tests pass successfully
- Verified backward compatibility with existing test suite
- Tested automatic batch size selection
- Tested manual batch size specification

## [0.1.0] - 2025-09-30

### Added
- Initial release of GraphEm Rapids
- PyTorch backend with CUDA acceleration
- RAPIDS cuVS backend for large-scale graphs
- Automatic backend selection based on graph size and hardware
- Force-directed layout algorithm with geometric intersection detection
- Graph generators (Erdős-Rényi, Scale-free, SBM, Caveman, etc.)
- Influence maximization via radial distance
- Memory management utilities and adaptive chunking
- Comprehensive test suite with 98+ tests
- Documentation and examples

[0.2.1]: https://github.com/sashakolpakov/graphem-rapids/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/sashakolpakov/graphem-rapids/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sashakolpakov/graphem-rapids/releases/tag/v0.1.0
