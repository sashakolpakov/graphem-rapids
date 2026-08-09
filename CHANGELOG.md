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

### Added

- GPU-resident canonical edge-list input and randomized sparse spectral
  initialization for graphs that cannot pass through SciPy or a CPU eigensolver.
- Bounded edge-force batching, less-contended bincount reductions, configurable
  crossing intervals, on-device radial top-k, and stage diagnostics.
- Reproducible CPU/CuPy Independent Cascade evaluation with independent Monte
  Carlo trials, plus degree-discount and geometric-diversity seed selection.
- End-to-end H100 and spring-sign diagnostic benchmarks, including degree,
  PageRank, degree-discount, cascade quality, and peak device memory.

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
