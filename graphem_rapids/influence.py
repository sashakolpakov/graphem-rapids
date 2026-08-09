"""Influence selection and reproducible Independent Cascade evaluation.

The large-graph path in this module does not depend on NetworkX or NDlib.  It
accepts SciPy CSR matrices directly and can evaluate many cascades on a CUDA GPU
with a compact CuPy kernel.  The NDlib helpers remain as compatibility wrappers.
"""

from dataclasses import dataclass
import heapq
import math

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class InfluenceEstimate:
    """Summary of independent Monte Carlo cascade trials."""

    mean: float
    std: float
    stderr: float
    trials: int
    minimum: int
    maximum: int
    samples: tuple


def graphem_seed_selection(
    embedder,
    k,
    num_iterations=20,
    diversity=0.0,
    candidate_pool_size=None,
):
    """Select the ``k`` largest radial scores without forcing a host copy.

    New GPU backends expose ``topk_nodes`` so a million-row position matrix need
    not be copied to NumPy just to retain a handful of node identifiers.  The
    fallback preserves the original embedder protocol.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if not 0.0 <= diversity <= 1.0:
        raise ValueError("diversity must be between zero and one")
    if num_iterations:
        embedder.run_layout(num_iterations=num_iterations)
    if diversity and hasattr(embedder, "diverse_topk_nodes"):
        return [
            int(node)
            for node in embedder.diverse_topk_nodes(
                k,
                diversity=diversity,
                candidate_pool_size=candidate_pool_size,
            )
        ]
    if hasattr(embedder, "topk_nodes"):
        return [int(node) for node in embedder.topk_nodes(k)]

    positions = embedder.get_positions()
    if k > len(positions):
        raise ValueError("k cannot exceed the number of vertices")
    radial_distances = np.linalg.norm(positions, axis=1)
    if k == 0:
        return []
    candidates = np.argpartition(radial_distances, -k)[-k:]
    order = np.argsort(radial_distances[candidates])[::-1]
    return candidates[order].astype(np.int64).tolist()


def _graph_to_csr(graph):
    """Return a sorted CSR adjacency matrix without densifying ``graph``."""
    graph_module = type(graph).__module__
    if (
        graph_module.startswith("cupyx.scipy.sparse")
        and hasattr(graph, "indptr")
        and hasattr(graph, "indices")
    ):
        return graph
    if sp.issparse(graph):
        adjacency = sp.csr_matrix(graph, dtype=np.float32)
    else:
        try:
            import networkx as nx  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise TypeError("graph must be a SciPy sparse matrix when NetworkX is unavailable") from exc
        if not isinstance(graph, nx.Graph):
            raise TypeError("graph must be a SciPy sparse matrix or NetworkX graph")
        nodes = list(graph.nodes())
        if nodes != list(range(len(nodes))):
            raise ValueError("NetworkX graph nodes must be consecutive integers starting at zero")
        adjacency = sp.csr_matrix(
            nx.to_scipy_sparse_array(graph, nodelist=nodes, dtype=np.float32, format="csr")
        )

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency matrix must be square")
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def _is_device_csr(adjacency):
    return type(adjacency).__module__.startswith("cupyx.scipy.sparse")


def _validate_cascade_inputs(adjacency, seeds, probability, n_simulations):
    seeds = np.asarray(seeds, dtype=np.int64)
    if seeds.ndim != 1:
        raise ValueError("seeds must be one-dimensional")
    if len(seeds) != len(np.unique(seeds)):
        raise ValueError("seeds must not contain duplicates")
    if len(seeds) and (seeds.min() < 0 or seeds.max() >= adjacency.shape[0]):
        raise ValueError("seed node is outside the graph")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("p must be between zero and one")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    return seeds


def _probability_threshold(probability):
    maximum = int(np.iinfo(np.uint64).max)
    value = min(maximum, int(round(probability * maximum)))
    return np.frombuffer(value.to_bytes(8, byteorder="little"), dtype=np.uint64)[0]


def _estimate_ic_cpu(adjacency, seeds, probability, n_simulations, random_seed):
    indptr = adjacency.indptr
    indices = adjacency.indices
    n_vertices = adjacency.shape[0]
    spreads = np.empty(n_simulations, dtype=np.int64)

    threshold = _probability_threshold(probability)
    random_seed_u64 = np.uint64(random_seed)
    for trial in range(n_simulations):
        visited = np.zeros(n_vertices, dtype=np.bool_)
        visited[seeds] = True
        frontier = seeds.copy()

        while frontier.size:
            next_parts = []
            for source in frontier:
                neighbors = indices[indptr[source]:indptr[source + 1]]
                if probability == 1.0:
                    activated = neighbors
                elif probability == 0.0 or neighbors.size == 0:
                    continue
                else:
                    edge_ids = np.arange(
                        indptr[source], indptr[source + 1], dtype=np.uint64
                    )
                    trial_key = np.multiply(
                        np.uint64(trial),
                        np.uint64(0xD2B74407B1CE6E93),
                        dtype=np.uint64,
                    )
                    edge_key = np.multiply(
                        edge_ids,
                        np.uint64(0xCA5A826395121157),
                        dtype=np.uint64,
                    )
                    values = random_seed_u64 ^ trial_key ^ edge_key
                    values += np.uint64(0x9E3779B97F4A7C15)
                    values = (values ^ (values >> np.uint64(30))) * np.uint64(
                        0xBF58476D1CE4E5B9
                    )
                    values = (values ^ (values >> np.uint64(27))) * np.uint64(
                        0x94D049BB133111EB
                    )
                    values ^= values >> np.uint64(31)
                    activated = neighbors[values <= threshold]
                if activated.size:
                    next_parts.append(activated)

            if not next_parts:
                break
            candidates = np.unique(np.concatenate(next_parts))
            frontier = candidates[~visited[candidates]]
            visited[frontier] = True

        spreads[trial] = np.count_nonzero(visited)
    return spreads


_IC_KERNEL_SOURCE = r"""
extern "C" {
__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__global__ void ic_expand(
    const long long* indptr,
    const long long* indices,
    const unsigned char* frontier,
    unsigned char* next_frontier,
    const long long n_vertices,
    const long long trial_offset,
    const unsigned long long random_seed,
    const unsigned long long threshold,
    const int probability_mode,
    const long long total_states
) {
    long long state = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (; state < total_states; state += stride) {
        if (!frontier[state]) continue;
        const long long trial = state / n_vertices;
        const long long source = state - trial * n_vertices;
        for (long long edge = indptr[source]; edge < indptr[source + 1]; ++edge) {
            bool live = probability_mode == 1;
            if (probability_mode == 0) live = false;
            if (probability_mode == 2) {
                const unsigned long long simulation = (unsigned long long)(trial_offset + trial);
                const unsigned long long key = random_seed
                    ^ (simulation * 0xd2b74407b1ce6e93ULL)
                    ^ ((unsigned long long)edge * 0xca5a826395121157ULL);
                live = mix64(key) <= threshold;
            }
            if (live) {
                const long long target = indices[edge];
                // Concurrent writers store the same idempotent value, so no
                // contended atomic is required.
                next_frontier[trial * n_vertices + target] = (unsigned char)1;
            }
        }
    }
}
}
"""


def _estimate_ic_cupy(
    adjacency,
    seeds,
    probability,
    n_simulations,
    random_seed,
    batch_size,
):
    try:
        import cupy as cp  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - requires CUDA environment
        raise ImportError("backend='cupy' requires CuPy") from exc

    n_vertices = adjacency.shape[0]
    indptr = cp.asarray(adjacency.indptr, dtype=cp.int64)
    indices = cp.asarray(adjacency.indices, dtype=cp.int64)
    seeds_gpu = cp.asarray(seeds, dtype=cp.int64)
    kernel = cp.RawKernel(_IC_KERNEL_SOURCE, "ic_expand")

    if batch_size is None:
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        # visited, frontier, and next_frontier each use one byte per state.
        memory_limited = max(1, int((free_bytes * 0.25) // max(3 * n_vertices, 1)))
        batch_size = min(256, n_simulations, memory_limited)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    probability_mode = 0 if probability == 0.0 else 1 if probability == 1.0 else 2
    threshold = _probability_threshold(probability)
    spreads = np.empty(n_simulations, dtype=np.int64)

    for offset in range(0, n_simulations, batch_size):
        current_batch = min(batch_size, n_simulations - offset)
        shape = (current_batch, n_vertices)
        visited = cp.zeros(shape, dtype=cp.uint8)
        frontier = cp.zeros(shape, dtype=cp.uint8)
        frontier[:, seeds_gpu] = 1
        visited[:, seeds_gpu] = 1
        total_states = current_batch * n_vertices
        blocks = min(65535, max(1, math.ceil(total_states / 256)))

        while bool(cp.any(frontier)):
            next_frontier = cp.zeros_like(frontier)
            kernel(
                (blocks,),
                (256,),
                (
                    indptr,
                    indices,
                    frontier,
                    next_frontier,
                    np.int64(n_vertices),
                    np.int64(offset),
                    np.uint64(random_seed),
                    threshold,
                    np.int32(probability_mode),
                    np.int64(total_states),
                ),
            )
            cp.logical_and(next_frontier, cp.logical_not(visited), out=frontier)
            cp.bitwise_or(visited, frontier, out=visited)

        spreads[offset:offset + current_batch] = cp.asnumpy(cp.count_nonzero(visited, axis=1))
    return spreads


def estimate_independent_cascade(
    graph,
    seeds,
    p=0.1,
    n_simulations=256,
    random_seed=0,
    backend="auto",
    batch_size=None,
):
    """Estimate Independent Cascade spread with independent repeated trials.

    Parameters
    ----------
    graph : scipy.sparse matrix or networkx.Graph
        Directed CSR rows are interpreted as outgoing edges. NetworkX undirected
        graphs naturally produce both directions.
    seeds : sequence of int
        Initially active vertices.
    p : float
        Per-directed-edge activation probability.
    n_simulations : int
        Number of independent cascades, not the number of diffusion time steps.
    backend : {'auto', 'cpu', 'cupy'}
        ``auto`` uses CuPy only when a CUDA device is available.

    Returns
    -------
    InfluenceEstimate
        Mean, sample standard deviation, standard error, and range.
    """
    adjacency = _graph_to_csr(graph)
    seeds = _validate_cascade_inputs(adjacency, seeds, p, n_simulations)
    if backend not in ("auto", "cpu", "cupy"):
        raise ValueError("backend must be 'auto', 'cpu', or 'cupy'")

    use_cupy = backend == "cupy"
    if backend == "auto":
        try:
            import cupy as cp  # pylint: disable=import-outside-toplevel
            use_cupy = cp.cuda.runtime.getDeviceCount() > 0
        except (ImportError, RuntimeError):
            use_cupy = False

    if use_cupy:
        spreads = _estimate_ic_cupy(
            adjacency, seeds, p, n_simulations, random_seed, batch_size
        )
    else:
        if _is_device_csr(adjacency):
            raise ValueError("a CuPy CSR graph requires backend='cupy'")
        spreads = _estimate_ic_cpu(adjacency, seeds, p, n_simulations, random_seed)

    std = float(np.std(spreads, ddof=1)) if n_simulations > 1 else 0.0
    return InfluenceEstimate(
        mean=float(np.mean(spreads)),
        std=std,
        stderr=std / math.sqrt(n_simulations),
        trials=int(n_simulations),
        minimum=int(np.min(spreads)),
        maximum=int(np.max(spreads)),
        samples=tuple(int(value) for value in spreads),
    )


def degree_discount_seed_selection(graph, k, p=0.1, candidate_pool_size=None):
    """Select IC seeds with the scalable degree-discount heuristic.

    The implementation uses a lazy heap and updates only neighbors of selected
    vertices. ``candidate_pool_size`` can bound heap memory on extremely large
    graphs; omitted means the exact full candidate set.
    """
    adjacency = _graph_to_csr(graph)
    n_vertices = adjacency.shape[0]
    if not 0 <= k <= n_vertices:
        raise ValueError("k must be between zero and the number of vertices")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between zero and one")
    if k == 0:
        return []

    if _is_device_csr(adjacency):
        if candidate_pool_size is not None and candidate_pool_size < n_vertices:
            raise ValueError("candidate_pool_size is not supported for GPU CSR input")
        import cupy as cp  # pylint: disable=import-outside-toplevel
        degrees = cp.diff(adjacency.indptr).astype(cp.float32)
        selected = cp.zeros(n_vertices, dtype=cp.bool_)
        selected_neighbor_count = cp.zeros(n_vertices, dtype=cp.float32)
        scores = degrees.copy()
        result = []
        for _ in range(k):
            node = int(cp.argmax(scores).item())
            result.append(node)
            selected[node] = True
            scores[node] = -cp.inf
            start = int(adjacency.indptr[node].item())
            stop = int(adjacency.indptr[node + 1].item())
            neighbors = adjacency.indices[start:stop]
            if neighbors.size == 0:
                continue
            neighbors = neighbors[~selected[neighbors]]
            selected_neighbor_count[neighbors] += 1.0
            t_value = selected_neighbor_count[neighbors]
            degree = degrees[neighbors]
            scores[neighbors] = degree - 2.0 * t_value - (degree - t_value) * t_value * p
        return result

    degrees = np.diff(adjacency.indptr).astype(np.float64)
    if candidate_pool_size is None or candidate_pool_size >= n_vertices:
        candidates = np.arange(n_vertices, dtype=np.int64)
    else:
        if candidate_pool_size < k:
            raise ValueError("candidate_pool_size must be at least k")
        candidates = np.argpartition(degrees, -candidate_pool_size)[-candidate_pool_size:]

    allowed = np.zeros(n_vertices, dtype=np.bool_)
    allowed[candidates] = True
    selected = np.zeros(n_vertices, dtype=np.bool_)
    selected_neighbor_count = np.zeros(n_vertices, dtype=np.int32)
    scores = degrees.copy()
    versions = np.zeros(n_vertices, dtype=np.int32)
    heap = [(-float(scores[node]), int(node), 0) for node in candidates]
    heapq.heapify(heap)
    result = []

    while len(result) < k:
        if not heap:
            raise RuntimeError("candidate pool was exhausted")
        negative_score, node, version = heapq.heappop(heap)
        if selected[node] or version != versions[node]:
            continue
        # The score is stored in the heap so stale floating-point entries cannot
        # win after multiple neighbor updates.
        if not np.isclose(-negative_score, scores[node]):
            continue
        selected[node] = True
        result.append(node)

        start, stop = adjacency.indptr[node], adjacency.indptr[node + 1]
        for neighbor in adjacency.indices[start:stop]:
            if selected[neighbor] or not allowed[neighbor]:
                continue
            selected_neighbor_count[neighbor] += 1
            t_value = float(selected_neighbor_count[neighbor])
            degree = degrees[neighbor]
            scores[neighbor] = degree - 2.0 * t_value - (degree - t_value) * t_value * p
            versions[neighbor] += 1
            heapq.heappush(
                heap,
                (-float(scores[neighbor]), int(neighbor), int(versions[neighbor])),
            )
    return result


def ndlib_estimated_influence(G, seeds, p=0.1, iterations_count=200):
    """Compatibility wrapper for one NDlib cascade.

    ``iterations_count`` is a maximum number of diffusion *time steps*. It is not
    a Monte Carlo sample count. New benchmarks should use
    :func:`estimate_independent_cascade`.
    """
    try:
        import ndlib.models.ModelConfig as mc  # pylint: disable=import-outside-toplevel
        import ndlib.models.epidemics as ep  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError("ndlib_estimated_influence requires NDlib") from exc

    model = ep.IndependentCascadesModel(G)
    config = mc.Configuration()
    for edge in G.edges():
        config.add_edge_configuration("threshold", edge, p)
    config.add_model_initial_configuration("Infected", seeds)
    model.set_initial_status(config)
    iterations = model.iteration_bunch(iterations_count)
    if not iterations:
        return len(seeds), 0
    counts = iterations[-1].get("node_count", {})
    influenced_count = counts.get(1, 0) + counts.get(2, 0)
    return influenced_count, len(iterations)


def greedy_seed_selection(G, k, p=0.1, iterations_count=200):
    """Legacy exhaustive greedy selection using one NDlib draw per candidate."""
    seeds = []
    total_iters = 0
    nodes = list(G.nodes())
    for _ in range(k):
        best_node = None
        best_influence = -1
        for node in nodes:
            if node in seeds:
                continue
            influence, used = ndlib_estimated_influence(
                G, seeds + [node], p=p, iterations_count=iterations_count
            )
            total_iters += used
            if influence > best_influence:
                best_influence = influence
                best_node = node
        if best_node is None:
            break
        seeds.append(best_node)
    return seeds, total_iters
