# Changelog

## Unreleased

- Exposed one canonical `GraphEmbedder` API for the corrected GraphEm dynamics.
- Added a single float64 Torch tensor spectral function parameterized by
  `torch.device`; production defaults to CUDA, explicit CUDA requests fail when
  unavailable, and diagnostic CPU selection warns and records its reason.
- Implemented the symmetric normalized-Laplacian isolate convention, shifted
  sparse Torch LOBPCG solve, deterministic analytic start block, and external
  finiteness, residual, and orthogonality gates.
- Increased the fixed LOBPCG budget and starting-block policy after H100 tests
  on small, disconnected, road-network, LiveJournal, and capacity fixtures.
- Corrected midpoint neighbours to use global edge IDs, identity-based self
  removal, and distance/ID ordering with adaptive exact overquery at tied
  cutoffs.
- Bounded every exact cuVS midpoint-search call to at most 64 queries, exposed
  smaller batch selection, and receipted search calls, submitted batch sizes,
  adaptive widths, and checkpointed device-memory high water without changing
  neighbor semantics.
- Corrected the spring to restoring Hooke dynamics and retained strict xy
  crossings, all-coordinate centroid force, full updates, and per-coordinate
  population normalization.
- Removed approximate midpoint indexes, cached neighbour sets, random
  initialization, clipping, learning-rate scaling, initial edge rescaling, and
  score-orientation alternatives.
- Made graph, dependency, spectral, search, and numerical failures explicit.
- Added warnings-as-errors Sphinx builds, Markdown lint, and documentation link
  checks.
