"""Influence selection and reproducible Independent Cascade evaluation.

The large-graph path in this module does not depend on NetworkX or NDlib.  It
accepts SciPy CSR matrices directly and can evaluate many cascades on a CUDA GPU
with a compact CuPy kernel.  The NDlib helpers remain as compatibility wrappers.
"""

from dataclasses import dataclass
import heapq
import math
import numbers

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


def _nonnegative_integer(value, name):
    """Return ``value`` as an int after rejecting booleans and coercions."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Integral)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validated_candidate_pool_size(candidate_pool_size, k):
    """Validate an optional candidate-pool size against ``k``."""
    if candidate_pool_size is None:
        return None
    candidate_pool_size = _nonnegative_integer(
        candidate_pool_size, "candidate_pool_size"
    )
    if candidate_pool_size < k:
        raise ValueError("candidate_pool_size must be at least k")
    return candidate_pool_size


def graphem_seed_selection(
    embedder,
    k,
    num_iterations=0,
):
    """Select the ``k`` largest canonical GraphEm radii."""
    k = _nonnegative_integer(k, "k")
    embedder_size = getattr(embedder, "n", None)
    if embedder_size is not None and k > embedder_size:
        raise ValueError("k cannot exceed the number of vertices")
    if num_iterations is None:
        num_iterations = 0
    num_iterations = _nonnegative_integer(num_iterations, "num_iterations")
    if num_iterations:
        embedder.run_layout(num_iterations=num_iterations)
    if not hasattr(embedder, "get_top_k"):
        raise TypeError("embedder must implement the canonical get_top_k method")
    return [int(node) for node in embedder.get_top_k(k)]


def _is_device_sparse(graph):
    """Return whether ``graph`` is a CuPy sparse object without importing CuPy."""
    return type(graph).__module__.startswith("cupyx.scipy.sparse")


def _graph_to_csr(graph, assume_validated_csr=False):  # pylint: disable=too-many-branches
    """Return a sorted CSR adjacency matrix without densifying ``graph``."""
    if not isinstance(assume_validated_csr, (bool, np.bool_)):
        raise ValueError("assume_validated_csr must be boolean")
    if _is_device_sparse(graph):
        try:
            import cupy as cp  # pylint: disable=import-outside-toplevel
            import cupyx.scipy.sparse as cpx_sparse  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - inconsistent installation
            raise TypeError("CuPy sparse input requires CuPy") from exc
        if not cpx_sparse.issparse(graph):
            raise TypeError("unrecognized CuPy sparse graph")
        if graph.shape[0] != graph.shape[1]:
            raise ValueError("adjacency matrix must be square")

        is_csr = getattr(graph, "format", None) == "csr"
        if assume_validated_csr and not is_csr:
            raise ValueError("assume_validated_csr requires CSR input")
        adjacency = graph if is_csr else graph.tocsr(copy=True)
        if assume_validated_csr:
            return adjacency
        n_vertices = int(adjacency.shape[0])
        if any(
            array.ndim != 1
            for array in (adjacency.indptr, adjacency.indices, adjacency.data)
        ):
            raise ValueError("invalid CSR index or indptr structure")
        invalid_sizes = (
            int(adjacency.indptr.size) != n_vertices + 1
            or int(adjacency.indices.size) != int(adjacency.data.size)
        )
        invalid_endpoints = (
            int(adjacency.indptr[0].item()) != 0
            or int(adjacency.indptr[-1].item()) != int(adjacency.indices.size)
        )
        if invalid_sizes or invalid_endpoints:
            raise ValueError("invalid CSR index or indptr structure")
        _validate_device_csr_arrays(cp, adjacency)
        is_canonical = bool(adjacency.has_canonical_format)
        has_explicit_zeros = (
            adjacency.nnz > 0
            and int(cp.count_nonzero(adjacency.data).item()) != int(adjacency.nnz)
        )
        if not is_csr or not is_canonical or has_explicit_zeros:
            if adjacency is graph:
                adjacency = adjacency.copy()
            adjacency.sum_duplicates()
            adjacency.eliminate_zeros()
            adjacency.sort_indices()
        return adjacency

    if sp.issparse(graph):
        if assume_validated_csr and getattr(graph, "format", None) != "csr":
            raise ValueError("assume_validated_csr requires CSR input")
        graph_indices = getattr(graph, "indices", None)
        graph_indptr = getattr(graph, "indptr", None)
        preserve_int64_indices = (
            graph_indices is not None
            and graph_indptr is not None
            and (
                graph_indices.dtype.itemsize > np.dtype(np.int32).itemsize
                or graph_indptr.dtype.itemsize > np.dtype(np.int32).itemsize
            )
        )
        adjacency = sp.csr_matrix(graph, dtype=np.float32, copy=True)
        if preserve_int64_indices:
            # SciPy downcasts small copied CSR structures even when callers
            # deliberately supply int64 buffers. Restore them so the typed CUDA
            # path can be tested without allocating >2**31 entries.
            adjacency.indices = adjacency.indices.astype(np.int64, copy=False)
            adjacency.indptr = adjacency.indptr.astype(np.int64, copy=False)
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
    if assume_validated_csr:
        return adjacency
    try:
        adjacency.check_format(full_check=True)
    except ValueError as exc:
        raise ValueError("invalid CSR index or indptr structure") from exc
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def _is_device_csr(adjacency):
    return _is_device_sparse(adjacency) and getattr(adjacency, "format", None) == "csr"


def _validate_cascade_inputs(adjacency, seeds, probability, n_simulations, random_seed):
    if type(seeds).__module__.startswith("cupy"):
        import cupy as cp  # pylint: disable=import-outside-toplevel
        seeds = cp.asnumpy(seeds)
    try:
        seeds = np.asarray(seeds)
    except (TypeError, ValueError) as exc:
        raise ValueError("seeds must be a one-dimensional integer sequence") from exc
    if seeds.ndim != 1:
        raise ValueError("seeds must be one-dimensional")
    if seeds.size and seeds.dtype.kind not in "iu":
        raise ValueError("seeds must contain only integer node ids")
    if len(seeds) != len(np.unique(seeds)):
        raise ValueError("seeds must not contain duplicates")
    if len(seeds) and (seeds.min() < 0 or seeds.max() >= adjacency.shape[0]):
        raise ValueError("seed node is outside the graph")
    seeds = seeds.astype(np.int64, copy=False)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("p must be between zero and one")
    if (
        isinstance(n_simulations, (bool, np.bool_))
        or not isinstance(n_simulations, numbers.Integral)
        or n_simulations <= 0
    ):
        raise ValueError("n_simulations must be a positive integer")
    if isinstance(random_seed, (bool, np.bool_)) or not isinstance(
        random_seed, numbers.Integral
    ):
        raise ValueError("random_seed must be an integer")
    maximum_seed = int(np.iinfo(np.uint64).max)
    if not 0 <= int(random_seed) <= maximum_seed:
        raise ValueError("random_seed must be between zero and 2**64 - 1")
    return seeds, int(n_simulations), int(random_seed)


def _probability_threshold(probability):
    maximum = int(np.iinfo(np.uint64).max)
    value = min(maximum, int(round(probability * maximum)))
    return np.frombuffer(value.to_bytes(8, byteorder="little"), dtype=np.uint64)[0]


_TRIAL_HASH_MULTIPLIER = np.uint64(0xD2B74407B1CE6E93)
_SOURCE_HASH_MULTIPLIER = np.uint64(0xCA5A826395121157)
_TARGET_HASH_MULTIPLIER = np.uint64(0x9E3779B185EBCA87)


def _live_edge_hashes_cpu(trial, source, targets, random_seed):
    """Return canonical counter-based hashes for directed ``source -> targets``."""
    values = np.multiply(
        np.asarray(targets, dtype=np.uint64),
        _TARGET_HASH_MULTIPLIER,
        dtype=np.uint64,
    )
    values ^= np.multiply(
        np.uint64(source), _SOURCE_HASH_MULTIPLIER, dtype=np.uint64
    )
    values ^= np.multiply(
        np.uint64(trial), _TRIAL_HASH_MULTIPLIER, dtype=np.uint64
    )
    values ^= np.uint64(random_seed)
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values = (values ^ (values >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return values ^ (values >> np.uint64(31))


def _estimate_ic_cpu(adjacency, seeds, probability, n_simulations, random_seed):
    indptr = adjacency.indptr
    indices = adjacency.indices
    n_vertices = adjacency.shape[0]
    if seeds.size == 0:
        return np.zeros(n_simulations, dtype=np.int64)
    if probability == 0.0:
        return np.full(n_simulations, seeds.size, dtype=np.int64)

    simulated_trials = 1 if probability == 1.0 else n_simulations
    spreads = np.empty(simulated_trials, dtype=np.int64)

    threshold = _probability_threshold(probability)
    for trial in range(simulated_trials):
        visited = np.zeros(n_vertices, dtype=np.bool_)
        visited[seeds] = True
        queue = seeds.tolist()
        head = 0

        while head < len(queue):
            source = queue[head]
            head += 1
            neighbors = indices[indptr[source]:indptr[source + 1]]
            if neighbors.size == 0:
                continue
            if probability == 1.0:
                activated = neighbors
            else:
                values = _live_edge_hashes_cpu(
                    trial, source, neighbors, random_seed
                )
                activated = neighbors[values <= threshold]
            for target in activated:
                target = int(target)
                if not visited[target]:
                    visited[target] = True
                    queue.append(target)

        spreads[trial] = len(queue)
    if probability == 1.0:
        return np.full(n_simulations, spreads[0], dtype=np.int64)
    return spreads


_IC_KERNEL_SOURCE = r"""
__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ bool live_edge(
    const unsigned long long random_seed,
    const unsigned long long simulation,
    const unsigned long long source,
    const unsigned long long target,
    const unsigned long long threshold,
    const int probability_mode
) {
    if (probability_mode == 1) return true;
    const unsigned long long key = random_seed
        ^ (simulation * 0xd2b74407b1ce6e93ULL)
        ^ (source * 0xca5a826395121157ULL)
        ^ (target * 0x9e3779b185ebca87ULL);
    return mix64(key) <= threshold;
}

template <typename IndexT>
__device__ __forceinline__ void validate_csr_impl(
    const IndexT* indptr,
    const IndexT* indices,
    const unsigned long long n_vertices,
    const unsigned long long nnz,
    unsigned int* invalid)
{
    unsigned long long item =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)blockDim.x * gridDim.x;
    const unsigned long long count = n_vertices > nnz ? n_vertices : nnz;
    for (; item < count; item += stride) {
        if (item < n_vertices && indptr[item] > indptr[item + 1ULL]) {
            atomicExch(invalid, 1U);
        }
        if (item < nnz) {
            const IndexT target = indices[item];
            if (target < 0 || (unsigned long long)target >= n_vertices) {
                atomicExch(invalid, 1U);
            }
        }
    }
}

extern "C" __global__ void validate_csr_i32(
    const int* indptr, const int* indices,
    unsigned long long n_vertices, unsigned long long nnz,
    unsigned int* invalid)
{
    validate_csr_impl<int>(indptr, indices, n_vertices, nnz, invalid);
}

extern "C" __global__ void validate_csr_i64(
    const long long* indptr, const long long* indices,
    unsigned long long n_vertices, unsigned long long nnz,
    unsigned int* invalid)
{
    validate_csr_impl<long long>(indptr, indices, n_vertices, nnz, invalid);
}

template <typename StateT>
__device__ __forceinline__ void initialize_impl(
    StateT* queue,
    unsigned int* visited,
    const long long* seeds,
    const unsigned long long n_seeds,
    const unsigned long long n_vertices,
    const unsigned long long total_seed_states
) {
    unsigned long long item =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride =
        (unsigned long long)blockDim.x * gridDim.x;
    for (; item < total_seed_states; item += stride) {
        const unsigned long long trial = item / n_seeds;
        const unsigned long long seed_index = item - trial * n_seeds;
        const unsigned long long node = (unsigned long long)seeds[seed_index];
        const unsigned long long state = trial * n_vertices + node;
        queue[item] = (StateT)state;
        atomicOr(
            visited + (state >> 5),
            1U << (unsigned int)(state & 31ULL)
        );
    }
}

extern "C" __global__ void ic_initialize_q32(
    unsigned int* queue,
    unsigned int* visited,
    const long long* seeds,
    const unsigned long long n_seeds,
    const unsigned long long n_vertices,
    const unsigned long long total_seed_states
) {
    initialize_impl<unsigned int>(
        queue, visited, seeds, n_seeds, n_vertices, total_seed_states
    );
}

extern "C" __global__ void ic_initialize_q64(
    unsigned long long* queue,
    unsigned int* visited,
    const long long* seeds,
    const unsigned long long n_seeds,
    const unsigned long long n_vertices,
    const unsigned long long total_seed_states
) {
    initialize_impl<unsigned long long>(
        queue, visited, seeds, n_seeds, n_vertices, total_seed_states
    );
}

template <typename IndexT, typename StateT>
__device__ __forceinline__ void expand_impl(
    const IndexT* indptr,
    const IndexT* indices,
    StateT* queue,
    unsigned int* visited,
    unsigned long long* counts,
    unsigned long long* queue_tail,
    const unsigned long long n_vertices,
    const unsigned long long trial_offset,
    const unsigned long long random_seed,
    const unsigned long long threshold,
    const int probability_mode,
    const unsigned long long queue_head,
    const unsigned long long frontier_tail,
    const unsigned long long queue_capacity
) {
    const unsigned int lane = threadIdx.x & 31U;
    unsigned long long work = (
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x
    ) >> 5;
    const unsigned long long warp_stride = (
        (unsigned long long)gridDim.x * blockDim.x
    ) >> 5;
    const unsigned long long active_count = frontier_tail - queue_head;

    for (; work < active_count; work += warp_stride) {
        const unsigned long long state =
            (unsigned long long)queue[queue_head + work];
        const unsigned long long trial = state / n_vertices;
        const unsigned long long source = state - trial * n_vertices;
        const unsigned long long start = (unsigned long long)indptr[source];
        const unsigned long long stop = (unsigned long long)indptr[source + 1ULL];
        const unsigned long long simulation = trial_offset + trial;

        for (unsigned long long edge_base = start;
             edge_base < stop;
             edge_base += 32ULL) {
            const unsigned long long edge = edge_base + lane;
            unsigned long long target_state = 0ULL;
            bool claimed = false;
            if (edge < stop) {
                const unsigned long long target =
                    (unsigned long long)indices[edge];
                if (target < n_vertices && live_edge(
                    random_seed,
                    simulation,
                    source,
                    target,
                    threshold,
                    probability_mode
                )) {
                    target_state = trial * n_vertices + target;
                    const unsigned int bit =
                        1U << (unsigned int)(target_state & 31ULL);
                    unsigned int* const word = visited + (target_state >> 5);
                    const unsigned int old = atomicOr(word, bit);
                    claimed = !(old & bit);
                }
            }

            const unsigned int claimed_mask =
                __ballot_sync(0xffffffffU, claimed);
            const unsigned int claimed_count = __popc(claimed_mask);
            unsigned long long reservation = 0ULL;
            if (lane == 0U && claimed_count) {
                reservation = atomicAdd(
                    queue_tail, (unsigned long long)claimed_count
                );
                const unsigned long long remaining =
                    reservation < queue_capacity
                    ? queue_capacity - reservation
                    : 0ULL;
                const unsigned long long accepted =
                    remaining < (unsigned long long)claimed_count
                    ? remaining
                    : (unsigned long long)claimed_count;
                if (accepted) atomicAdd(counts + trial, accepted);
            }
            reservation = __shfl_sync(0xffffffffU, reservation, 0);
            if (claimed) {
                const unsigned int lower_lanes =
                    lane == 0U ? 0U : ((1U << lane) - 1U);
                const unsigned long long position = reservation +
                    (unsigned long long)__popc(claimed_mask & lower_lanes);
                if (position < queue_capacity) {
                    queue[position] = (StateT)target_state;
                }
            }
        }
    }
}

extern "C" __global__ void ic_expand_i32_q32(
    const int* indptr, const int* indices, unsigned int* queue,
    unsigned int* visited, unsigned long long* counts,
    unsigned long long* queue_tail, unsigned long long n_vertices,
    unsigned long long trial_offset, unsigned long long random_seed,
    unsigned long long threshold, int probability_mode,
    unsigned long long queue_head, unsigned long long frontier_tail,
    unsigned long long queue_capacity
) {
    expand_impl<int, unsigned int>(
        indptr, indices, queue, visited, counts, queue_tail, n_vertices,
        trial_offset, random_seed, threshold, probability_mode, queue_head,
        frontier_tail, queue_capacity
    );
}

extern "C" __global__ void ic_expand_i64_q32(
    const long long* indptr, const long long* indices, unsigned int* queue,
    unsigned int* visited, unsigned long long* counts,
    unsigned long long* queue_tail, unsigned long long n_vertices,
    unsigned long long trial_offset, unsigned long long random_seed,
    unsigned long long threshold, int probability_mode,
    unsigned long long queue_head, unsigned long long frontier_tail,
    unsigned long long queue_capacity
) {
    expand_impl<long long, unsigned int>(
        indptr, indices, queue, visited, counts, queue_tail, n_vertices,
        trial_offset, random_seed, threshold, probability_mode, queue_head,
        frontier_tail, queue_capacity
    );
}

extern "C" __global__ void ic_expand_i32_q64(
    const int* indptr, const int* indices, unsigned long long* queue,
    unsigned int* visited, unsigned long long* counts,
    unsigned long long* queue_tail, unsigned long long n_vertices,
    unsigned long long trial_offset, unsigned long long random_seed,
    unsigned long long threshold, int probability_mode,
    unsigned long long queue_head, unsigned long long frontier_tail,
    unsigned long long queue_capacity
) {
    expand_impl<int, unsigned long long>(
        indptr, indices, queue, visited, counts, queue_tail, n_vertices,
        trial_offset, random_seed, threshold, probability_mode, queue_head,
        frontier_tail, queue_capacity
    );
}

extern "C" __global__ void ic_expand_i64_q64(
    const long long* indptr, const long long* indices,
    unsigned long long* queue, unsigned int* visited,
    unsigned long long* counts, unsigned long long* queue_tail,
    unsigned long long n_vertices, unsigned long long trial_offset,
    unsigned long long random_seed, unsigned long long threshold,
    int probability_mode, unsigned long long queue_head,
    unsigned long long frontier_tail, unsigned long long queue_capacity
) {
    expand_impl<long long, unsigned long long>(
        indptr, indices, queue, visited, counts, queue_tail, n_vertices,
        trial_offset, random_seed, threshold, probability_mode, queue_head,
        frontier_tail, queue_capacity
    );
}
"""


_IC_MODULE_CACHE = {}
_UINT32_STATE_CAPACITY = int(np.iinfo(np.uint32).max) + 1


def _get_ic_module(cp):
    """Compile the IC CUDA module once per device and retain it in-process."""
    device_id = int(cp.cuda.Device().id)
    module = _IC_MODULE_CACHE.get(device_id)
    if module is None:
        module = cp.RawModule(code=_IC_KERNEL_SOURCE, options=("-std=c++11",))
        module.compile()
        _IC_MODULE_CACHE[device_id] = module
    return module


def _validate_device_csr_arrays(cp, adjacency):
    """Validate CSR bounds/monotonicity with one no-temporary device pass."""
    if (
        adjacency.indptr.dtype == cp.dtype(cp.int32)
        and adjacency.indices.dtype == cp.dtype(cp.int32)
    ):
        suffix = "i32"
    elif (
        adjacency.indptr.dtype == cp.dtype(cp.int64)
        and adjacency.indices.dtype == cp.dtype(cp.int64)
    ):
        suffix = "i64"
    else:
        raise ValueError("CSR indices and indptr must share int32 or int64 dtype")
    if not adjacency.indptr.flags.c_contiguous or not adjacency.indices.flags.c_contiguous:
        raise ValueError("CSR indices and indptr must be contiguous")

    count = max(int(adjacency.shape[0]), int(adjacency.indices.size))
    if count == 0:
        return
    invalid = cp.zeros(1, dtype=cp.uint32)
    kernel = _get_ic_module(cp).get_function(f"validate_csr_{suffix}")
    threads = 256
    blocks = min(65535, max(1, (count + threads - 1) // threads))
    kernel(
        (blocks,),
        (threads,),
        (
            adjacency.indptr,
            adjacency.indices,
            np.uint64(adjacency.shape[0]),
            np.uint64(adjacency.indices.size),
            invalid,
        ),
    )
    if int(invalid.item()):
        raise ValueError("invalid CSR index or indptr structure")


def warm_independent_cascade_kernels():
    """Eagerly compile and cache CUDA cascade kernels on the current device."""
    try:
        import cupy as cp  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - requires CUDA environment
        raise ImportError("warming cascade kernels requires CuPy") from exc
    module = _get_ic_module(cp)
    for name in (
        "validate_csr_i32",
        "validate_csr_i64",
        "ic_initialize_q32",
        "ic_initialize_q64",
        "ic_expand_i32_q32",
        "ic_expand_i64_q32",
        "ic_expand_i32_q64",
        "ic_expand_i64_q64",
    ):
        module.get_function(name)


def _queue_workspace_bytes(n_vertices, batch_size):
    total_states = int(n_vertices) * int(batch_size)
    queue_item_size = 4 if total_states <= _UINT32_STATE_CAPACITY else 8
    visited_words = (total_states + 31) // 32
    return total_states * queue_item_size + visited_words * 4 + batch_size * 8 + 8


def _available_cupy_memory(cp):
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    try:
        allocator = cp.cuda.get_allocator()
        pool = cp.get_default_memory_pool()
        if getattr(allocator, "__self__", None) is pool:
            free_bytes += pool.free_bytes()
    except AttributeError:  # pragma: no cover - alternate allocator
        pass
    return int(free_bytes)


def _select_cascade_batch_size(
    cp,
    n_vertices,
    n_simulations,
    batch_size,
    available_memory_bytes=None,
):
    available = (
        _available_cupy_memory(cp)
        if available_memory_bytes is None
        else int(available_memory_bytes)
    )
    if batch_size is not None:
        if (
            isinstance(batch_size, (bool, np.bool_))
            or not isinstance(batch_size, numbers.Integral)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        selected = min(int(batch_size), n_simulations)
        required = _queue_workspace_bytes(n_vertices, selected)
        if required > int(available * 0.8):
            raise MemoryError(
                "cascade batch workspace requires "
                f"{required / 2**30:.2f} GiB but only "
                f"{available / 2**30:.2f} GiB is available"
            )
        return selected

    maximum = min(256, n_simulations)
    budget = int(available * 0.25)
    low, high, selected = 1, maximum, 0
    while low <= high:
        candidate = (low + high) // 2
        if _queue_workspace_bytes(n_vertices, candidate) <= budget:
            selected = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    if selected == 0:
        required = _queue_workspace_bytes(n_vertices, 1)
        raise MemoryError(
            "one cascade trial requires "
            f"{required / 2**30:.2f} GiB of queue workspace; "
            f"the 25% device-memory budget is {budget / 2**30:.2f} GiB"
        )
    return selected


def _prepare_csr_indices(cp, adjacency):
    use_int32 = (
        adjacency.indptr.dtype == cp.dtype(cp.int32)
        and adjacency.indices.dtype == cp.dtype(cp.int32)
    )
    dtype = cp.int32 if use_int32 else cp.int64
    return (
        cp.ascontiguousarray(cp.asarray(adjacency.indptr, dtype=dtype)),
        cp.ascontiguousarray(cp.asarray(adjacency.indices, dtype=dtype)),
        "i32" if use_int32 else "i64",
    )


def _estimate_ic_cupy(
    adjacency,
    seeds,
    probability,
    n_simulations,
    random_seed,
    batch_size,
    available_memory_bytes,
):
    try:
        import cupy as cp  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - requires CUDA environment
        raise ImportError("backend='cupy' requires CuPy") from exc

    if seeds.size == 0:
        return np.zeros(n_simulations, dtype=np.int64)
    if probability == 0.0:
        return np.full(n_simulations, seeds.size, dtype=np.int64)

    n_vertices = int(adjacency.shape[0])
    simulated_trials = 1 if probability == 1.0 else n_simulations
    indptr, indices, index_suffix = _prepare_csr_indices(cp, adjacency)
    seeds_gpu = cp.ascontiguousarray(cp.asarray(seeds, dtype=cp.int64))
    module = _get_ic_module(cp)
    batch_size = _select_cascade_batch_size(
        cp,
        n_vertices,
        simulated_trials,
        batch_size,
        available_memory_bytes,
    )

    probability_mode = 1 if probability == 1.0 else 2
    threshold = _probability_threshold(probability)
    spreads = np.empty(simulated_trials, dtype=np.int64)

    for offset in range(0, simulated_trials, batch_size):
        current_batch = min(batch_size, simulated_trials - offset)
        total_states = current_batch * n_vertices
        queue_suffix = "q32" if total_states <= _UINT32_STATE_CAPACITY else "q64"
        queue_dtype = cp.uint32 if queue_suffix == "q32" else cp.uint64
        queue = cp.empty(total_states, dtype=queue_dtype)
        visited = cp.zeros((total_states + 31) // 32, dtype=cp.uint32)
        counts = cp.full(current_batch, seeds.size, dtype=cp.uint64)

        initial_tail = current_batch * seeds.size
        initialize = module.get_function(f"ic_initialize_{queue_suffix}")
        initialize_blocks = min(65535, max(1, (initial_tail + 255) // 256))
        initialize(
            (initialize_blocks,),
            (256,),
            (
                queue,
                visited,
                seeds_gpu,
                np.uint64(seeds.size),
                np.uint64(n_vertices),
                np.uint64(initial_tail),
            ),
        )

        queue_tail = cp.asarray(np.uint64(initial_tail))
        expand = module.get_function(f"ic_expand_{index_suffix}_{queue_suffix}")
        head = 0
        tail = initial_tail
        while head < tail:
            active_count = tail - head
            blocks = min(65535, max(1, (active_count + 7) // 8))
            expand(
                (blocks,),
                (256,),
                (
                    indptr,
                    indices,
                    queue,
                    visited,
                    counts,
                    queue_tail,
                    np.uint64(n_vertices),
                    np.uint64(offset),
                    np.uint64(random_seed),
                    threshold,
                    np.int32(probability_mode),
                    np.uint64(head),
                    np.uint64(tail),
                    np.uint64(total_states),
                ),
            )
            new_tail = int(queue_tail.item())
            if new_tail > total_states:
                raise RuntimeError("cascade activation queue exceeded its state capacity")
            head, tail = tail, new_tail

        spreads[offset:offset + current_batch] = cp.asnumpy(counts).astype(
            np.int64, copy=False
        )
    if probability == 1.0:
        return np.full(n_simulations, spreads[0], dtype=np.int64)
    return spreads


def estimate_independent_cascade(
    graph,
    seeds,
    p=0.1,
    n_simulations=256,
    random_seed=0,
    *,
    backend,
    batch_size=None,
    available_memory_bytes=None,
    assume_validated_csr=False,
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
    backend : {'cpu', 'cupy'}
        Required execution engine. It is never inferred or substituted.
    available_memory_bytes : int, optional
        Allocator-aware capacity used for GPU workspace planning. This is useful
        with external pools such as RMM whose reusable blocks are not included in
        CUDA's raw free-memory counter.
    assume_validated_csr : bool, default=False
        Skip repeated CSR structural/canonical checks. Use only for an immutable
        CSR already validated by the caller; benchmark-generated graphs use this
        to avoid rescanning all edges for every seed method.

    Returns
    -------
    InfluenceEstimate
        Mean, sample standard deviation, standard error, and range.
    """
    if backend not in ("cpu", "cupy"):
        raise ValueError("backend must be explicitly 'cpu' or 'cupy'")
    if batch_size is not None and (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, numbers.Integral)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer or None")
    if available_memory_bytes is not None and (
        isinstance(available_memory_bytes, (bool, np.bool_))
        or not isinstance(available_memory_bytes, numbers.Integral)
        or available_memory_bytes <= 0
    ):
        raise ValueError("available_memory_bytes must be a positive integer or None")
    adjacency = _graph_to_csr(
        graph, assume_validated_csr=assume_validated_csr
    )
    seeds, n_simulations, random_seed = _validate_cascade_inputs(
        adjacency, seeds, p, n_simulations, random_seed
    )

    if backend == "cupy":
        spreads = _estimate_ic_cupy(
            adjacency,
            seeds,
            p,
            n_simulations,
            random_seed,
            batch_size,
            available_memory_bytes,
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


def degree_discount_seed_selection(
    graph,
    k,
    p=0.1,
    candidate_pool_size=None,
    assume_validated_csr=False,
):
    """Select IC seeds with the scalable degree-discount heuristic.

    The implementation uses a lazy heap and updates only neighbors of selected
    vertices. ``candidate_pool_size`` can bound heap memory on extremely large
    graphs; omitted means the exact full candidate set.
    """
    k = _nonnegative_integer(k, "k")
    candidate_pool_size = _validated_candidate_pool_size(candidate_pool_size, k)
    adjacency = _graph_to_csr(
        graph, assume_validated_csr=assume_validated_csr
    )
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
        degrees = cp.diff(adjacency.indptr)
        selected = cp.zeros(n_vertices, dtype=cp.bool_)
        selected_neighbor_count = cp.zeros(n_vertices, dtype=cp.int32)
        scores = degrees.astype(cp.float64)
        result = []
        for _ in range(k):
            node = int(cp.argmax(scores).item())
            result.append(node)
            selected[node] = True
            scores[node] = -cp.inf
            start, stop = cp.asnumpy(adjacency.indptr[node:node + 2]).tolist()
            neighbors = adjacency.indices[start:stop]
            if neighbors.size == 0:
                continue
            neighbors = neighbors[~selected[neighbors]]
            selected_neighbor_count[neighbors] += 1
            t_value = selected_neighbor_count[neighbors].astype(cp.float64)
            degree = degrees[neighbors].astype(cp.float64)
            scores[neighbors] = degree - 2.0 * t_value - (degree - t_value) * t_value * p
        return result

    degrees = np.diff(adjacency.indptr).astype(np.float64)
    if candidate_pool_size is None or candidate_pool_size >= n_vertices:
        candidates = np.arange(n_vertices, dtype=np.int64)
    else:
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
    """Exhaustive single-draw NDlib selection retained as an explicit comparator."""
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
