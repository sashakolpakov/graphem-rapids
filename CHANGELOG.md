# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/sashakolpakov/graphem-rapids/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sashakolpakov/graphem-rapids/releases/tag/v0.1.0
