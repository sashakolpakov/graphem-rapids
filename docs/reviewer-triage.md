# Reviewer triage and engineering response

This document separates publication work from changes that can materially improve
GraphEm as a large-graph system. The engineering claims below refer to the v0.2.1
implementation and should be re-checked after each backend rewrite.

## Comment-by-comment verdict

| Comment | Verdict | Engineering action |
|---|---|---|
| R1 W1: limited novelty | Publication/positioning | Do not spend engineering time manufacturing novelty. The one useful part is to benchmark standard ForceAtlas2 radius as another cheap layout baseline. |
| R1 W2: heuristic theory | Hybrid | Ignore demands for a grand centrality theorem; do add adversarial graph families and document low-degree bridges, directed graphs, and stability failures. |
| R1 W3: missing degree/PageRank/k-core/layout baselines | Material | Add all of them. A slower proxy for degree is not a useful system. |
| R1 W4: tiny/narrow influence tests | Material | Use multiple graph families, large real graphs, independent repeated cascades, and confidence intervals. |
| R1 W5: greedy runtime is an inadequate comparison | Material | Measure every pipeline stage and compare against degree-discount plus a sketch/RR, cuRipples, PaC-IM, or pruned-Monte-Carlo implementation. |
| R1 W6: unsupported scalability | Material | Run million- and ten-million-node H100 cases with end-to-end time and peak memory. |
| R1 W7: centrality inter-correlation | Material | Measure incremental value over degree/PageRank and include graphs where their rankings diverge. |
| R1 W8: no ablation | Material | Ablate initialization, spring sign, crossing repulsion, normalization, dimension, sampling, and ANN choice. This also catches implementation defects. |
| R1 W9: generic embedding comparison is a straw man | Correct but mostly evaluative | Drop it as a headline systems comparison. UMAP/TriMAP/PaCMAP radius is not a relevant performance target. |
| R2: formal theorem/proof requests | Publication-only | Revisit only if resubmission venue demands it; it does not make the engine better. |
| R2: parameter table and sensitivity | Hybrid | Keep a machine-readable benchmark configuration and a compact robustness sweep; skip ceremonial tables detached from runtime/quality decisions. |
| R2: larger graphs, runtime, memory, scalable centrality approximation | Material | This is a core acceptance gate. |
| R2: recent representation-learning related work | Publication-only | No implementation work unless a method is a credible ranking/cascade baseline. |
| R3: motivation and cheap correlated measures | Material | Require a Pareto win or a genuinely different useful ranking relative to degree/PageRank. |
| R3: Ohsaka et al. AAAI 2014 and scalable IM | Material | Treat Ohsaka as the minimum historical bar and add a newer parallel/sketch baseline for the H100 study. |
| R3: node2vec positioning | Publication-only | A conceptual paragraph may help a paper, but node2vec radius is not the missing systems baseline. |
| R3: undefined notation, section structure, incomplete sentence | Publication-only | Fix in a future manuscript pass, not in the scaling workstream. |

## Material comments

### Compare against inexpensive node-ranking baselines

This is essential. A geometric score is only useful if it occupies a useful point
on the quality/runtime/memory Pareto frontier. Correlation with expensive
betweenness and closeness does not establish that when degree, PageRank, Katz, and
approximate betweenness are cheap and available in cuGraph. The benchmark must
therefore report ranking agreement, cascade spread, runtime, and peak memory for
all of these methods rather than treating exact centrality as the only alternative.

### Use scalable influence-maximization baselines

This is essential. The existing comparison is against an exhaustive, noisy greedy
routine on tiny graphs. That is not a credible systems baseline. At minimum, use a
degree-discount IC heuristic and a properly repeated cascade evaluator. For the
large H100 evaluation, compare with a reverse-reachable/sketch implementation or
cuRipples/PaC-IM when it can be built in the test environment. Ohsaka et al.'s
pruned Monte Carlo method is historically relevant, but it is not the current
scaling ceiling.

### Test large graphs and report scaling

This is the main engineering requirement. Report end-to-end and stage-level time,
host and device peak memory, edges processed per second, and output quality. Use
graphs large enough that setup, transfers, and index construction cannot be hidden
by millisecond kernels. Include both sparse and hub-dominated families, since
atomic scatter behavior and cascade quality differ substantially between them.

### State the method's success and failure regimes

This matters to product design. Radial distance is expected to work best when the
desired importance signal tracks degree/PageRank-like hub structure. It can miss
low-degree bridges, structurally important peripheral vertices, directed
asymmetry, and cases where several high-score seeds cover the same community.
These are not prose caveats only: the benchmark matrix needs graph families that
expose each failure mode.

### Parameter sensitivity

Only the engineering part is material: defaults must remain stable across graph
scale and degree distribution, and quality must not depend on a narrowly tuned
combination of force constants. A compact sensitivity sweep belongs in the H100
benchmark. A large parameter table written only to satisfy a referee does not.

## Publication-only comments

The following may improve a paper, but they do not make the implementation faster,
more accurate, or more scalable by themselves:

- packaging the fixed-point discussion as a formal theorem;
- expanding generic related work on graph representation learning;
- adding a conceptual node2vec comparison when node2vec is not a competitive
  centrality or influence-maximization baseline;
- merging a short section, defining already inferable notation, fixing an
  incomplete sentence, and other exposition edits.

Undefined parameters are an exception when they also exist in the API: those must
be documented and validated because they affect reproducibility.

## v0.2.1 implementation findings

These issues can explain the disappointing GPU results without invoking a general
failure of CUDA or RAPIDS:

1. The cuVS index is built over **vertex positions**, queried with **edge
   midpoints**, and its returned indices are then used as **edge indices**. The
   indexed objects and the consumed identifiers do not match.
2. Stable cuVS returns `(distances, neighbors)` from brute-force and IVF searches.
   The backend reads the first result as neighbor indices.
3. The index is initialized before any midpoint data exists and is only rebuilt
   periodically, even though positions and midpoints change on every iteration.
4. The implementation uses the pre-26.06 IVF construction/search calling
   convention. Errors are caught broadly and silently turn into a quadratic
   broadcast-distance fallback.
5. Spectral initialization always uses CPU SciPy `eigsh`, so the nominal RAPIDS
   backend begins with a non-scaling CPU stage and a host-to-device transfer.
6. Spring accumulation uses `cupy.add.at`, whose contended atomics are especially
   poor on hub-dominated graphs, and allocates full edge temporaries each iteration.
7. GPU cache cleanup is invoked around inner operations. This defeats allocator
   reuse and makes timings include repeated allocation/synchronization overhead.
8. The documented Hooke force is attractive, but all code backends implement its
   negative. Because the published ranking behavior may depend on this discrepancy,
   the two signs must be benchmarked explicitly before choosing a default.
9. A fixed number of sampled edges makes the crossing signal vanish as the graph
   grows, while an all-edge spring pass remains O(E). Sampling must scale with the
   graph or be replaced by a multilevel/spatial approximation.
10. Importing the package requires NDlib even when influence evaluation is unused.
11. `iterations_count` in the NDlib helper counts diffusion time steps in one
    stochastic cascade, not independent Monte Carlo trials. The greedy routine
    compares candidates using unrelated single draws, so its quality and timing
    are not a sound baseline.

## Acceptance gates for the rebuild

The new engine should not be called scalable merely because it completes one large
run. It must satisfy all of the following:

- correctness tests for force direction, midpoint-neighbor identifiers, cuVS
  result ordering, seed selection, and deterministic cascade evaluation;
- no mandatory CPU eigensolve or NetworkX/NDlib conversion on the large-graph
  path;
- bounded edge batching with no O(V^2), O(E^2), or dense adjacency allocation;
- end-to-end H100 runs at 1M/10M and then 10M/100M vertices/edges when the graph
  generator and host memory permit;
- stage timings and peak host/device memory, including graph construction and
  transfers;
- cascade quality against random, degree, degree-discount, PageRank, and a
  sketch/RR or pruned-Monte-Carlo baseline;
- explicit Pareto reporting. If degree/PageRank dominates GraphEm, that result is
  a design constraint, not something to hide with a correlation table.
