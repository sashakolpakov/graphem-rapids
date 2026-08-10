# Changelog

## Unreleased

- Replaced the previous multi-engine package with one canonical RAPIDS
  `GraphEmbedder`.
- Corrected midpoint neighbours to use global edge IDs and identity-based self
  removal.
- Corrected the spring to restoring Hooke dynamics.
- Restored the paper's xy crossing predicate, all-coordinate centroid force,
  full update, and per-coordinate population normalization.
- Removed approximate midpoint indexes, cached neighbour sets, random
  initialization, clipping, learning-rate scaling, initial edge rescaling, and
  score-orientation alternatives.
- Made graph, dependency, eigensolve, search, and numerical failures explicit.
- Added CPU contract checks and retained comparator errors as typed outcomes.
