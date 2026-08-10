# Changelog

## Unreleased

- Replaced the previous multi-engine package with one canonical RAPIDS
  `GraphEmbedder`.
- Corrected midpoint neighbours to use global edge IDs, identity-based self
  removal, and deterministic distance/ID ordering with adaptive exact
  overquery at tied cutoffs.
- Replaced single-vector ARPACK initialization with one deterministic block
  LOBPCG solve and an explicit residual gate.
- Defined the normalized Laplacian for disconnected graphs and isolated
  vertices without changing or dropping vertices.
- Corrected the spring to restoring Hooke dynamics.
- Restored the paper's xy crossing predicate, all-coordinate centroid force,
  full update, and per-coordinate population normalization.
- Removed approximate midpoint indexes, cached neighbour sets, random
  initialization, clipping, learning-rate scaling, initial edge rescaling, and
  score-orientation alternatives.
- Made graph, dependency, eigensolve, search, and numerical failures explicit.
- Added CPU contract checks and retained comparator errors as typed outcomes.
