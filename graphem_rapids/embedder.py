"""Canonical GPU implementation of GraphEm.

The module intentionally exposes one algorithm.  It follows the executed
paper protocol where that protocol is well-defined, while applying confirmed
correctness repairs to spring dynamics, spectral initialization, and global
edge identities for midpoint neighbours.
"""

# PyTorch exposes several compiled linalg callables without signatures that
# pylint can recognize.
# pylint: disable=not-callable

from __future__ import annotations

import hashlib
import logging
import numbers
import time
from typing import Optional
import warnings

import numpy as np
import scipy.sparse as sp

try:  # Imports remain lazy so documentation and CPU contract tests can import.
    from cuvs.neighbors import brute_force
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except ImportError as gpu_import_error:  # pragma: no cover - host dependent
    cp = None
    cpx_sparse = None
    brute_force = None
    _GPU_IMPORT_ERROR = gpu_import_error
else:  # pragma: no cover - host dependent
    _GPU_IMPORT_ERROR = None

try:  # Torch is independently optional for import-time contract inspection.
    import torch
except ImportError as torch_import_error:  # pragma: no cover - host dependent
    torch = None
    _TORCH_IMPORT_ERROR = torch_import_error
else:  # pragma: no cover - host dependent
    _TORCH_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)
EPSILON = np.float32(1.0e-6)
FLOAT32_UNIT_ROUNDOFF = np.float64(2.0**-24)
SPECTRAL_TOLERANCE = np.float64(1.0e-10)
SPECTRAL_MAX_ITERATIONS = 5000
SPECTRAL_RESIDUAL_BOUND = np.float64(1.0e-8)
SPECTRAL_ORTHOGONALITY_BOUND = np.float64(1.0e-8)
SPECTRAL_SHIFT = np.float64(3.0)
SPECTRAL_CLUSTER_BOUND = np.float64(1.0e-8)
SPECTRAL_MINIMUM_BLOCK_WIDTH = 16
SPECTRAL_START_ALGORITHM = "analytic-sine-cosine-qr-float64-v1"
TORCH_SPECTRAL_BACKEND = "torch-lobpcg-shifted-normalized-laplacian-v2"
MIDPOINT_QUERY_BATCH_SIZE_BOUND = 64
MIDPOINT_QUERY_BATCH_POLICY = "fixed-explicit-at-most-64-v1"
MIDPOINT_MEMORY_OBSERVATION = "cuda-memgetinfo-search-checkpoints-v1"
MIDPOINT_NEIGHBOR_ID_VALIDATION = (
    "rowwise-unique-global-edge-id-before-negative-repair-v1"
)


_DETERMINISTIC_FORCE_KERNELS = r"""
#define DEFINE_SPRING_KERNEL(NAME, INDEX_TYPE)                                  \
extern "C" __global__ void NAME(                                                \
    const float* positions, const long long* row_offsets,                       \
    const INDEX_TYPE* neighbors, const long long n_vertices,                    \
    const int n_components,                                                     \
    const float preferred_length, const float attraction, float* forces)        \
{                                                                                \
    const long long vertex_id =                                                  \
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;           \
    if (vertex_id >= n_vertices) return;                                          \
    const long long vertex_offset = vertex_id * n_components;                    \
    for (int component = 0; component < n_components; ++component) {             \
        forces[vertex_offset + component] = 0.0f;                                \
    }                                                                             \
    const long long begin = static_cast<long long>(row_offsets[vertex_id]);       \
    const long long end = static_cast<long long>(row_offsets[vertex_id + 1]);     \
    for (long long offset = begin; offset < end; ++offset) {                      \
        const long long neighbor = static_cast<long long>(neighbors[offset]);     \
        const long long neighbor_offset = neighbor * n_components;               \
        float squared_norm = 0.0f;                                                \
        for (int component = 0; component < n_components; ++component) {         \
            const float delta = positions[neighbor_offset + component]           \
                - positions[vertex_offset + component];                           \
            squared_norm += delta * delta;                                        \
        }                                                                         \
        const float distance = sqrtf(squared_norm) + 1.0e-6f;                    \
        const float multiplier = attraction * (distance - preferred_length)      \
            / distance;                                                           \
        for (int component = 0; component < n_components; ++component) {         \
            const float delta = positions[neighbor_offset + component]           \
                - positions[vertex_offset + component];                           \
            forces[vertex_offset + component] += multiplier * delta;             \
        }                                                                         \
    }                                                                             \
}

DEFINE_SPRING_KERNEL(graphem_spring_i32, int)
DEFINE_SPRING_KERNEL(graphem_spring_i64, long long)

#define DEFINE_SEGMENT_KERNEL(NAME, INDEX_TYPE)                                 \
extern "C" __global__ void NAME(                                                \
    const float* contributions, const long long* starts,                        \
    const long long* ends, const INDEX_TYPE* vertices,                          \
    const long long n_segments, const int n_components, float* forces)          \
{                                                                               \
    const long long output_id =                                                 \
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;          \
    const long long output_count = n_segments * n_components;                   \
    if (output_id >= output_count) return;                                      \
    const long long segment = output_id / n_components;                         \
    const int component = static_cast<int>(output_id % n_components);           \
    float accumulated = 0.0f;                                                   \
    for (long long row = starts[segment]; row < ends[segment]; ++row) {          \
        accumulated += contributions[row * n_components + component];           \
    }                                                                            \
    const long long vertex = static_cast<long long>(vertices[segment]);          \
    forces[vertex * n_components + component] = accumulated;                    \
}

DEFINE_SEGMENT_KERNEL(graphem_segment_i32, int)
DEFINE_SEGMENT_KERNEL(graphem_segment_i64, long long)
"""


def _require_gpu() -> None:
    if _GPU_IMPORT_ERROR is not None:
        raise ImportError(
            "GraphEm requires CuPy, cupyx, and cuVS; the "
            "canonical GPU implementation cannot start without them"
        ) from _GPU_IMPORT_ERROR


def _require_torch() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise ImportError(
            "GraphEm spectral initialization requires PyTorch; install the "
            "pinned CUDA-enabled build used by the canonical executor"
        ) from _TORCH_IMPORT_ERROR


def _resolve_spectral_device(requested):
    """Resolve one Torch spectral device without an implicit CUDA downgrade."""
    _require_torch()
    requested_text = str(requested).lower()
    if requested_text == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), requested_text, None
        reason = "torch.cuda.is_available() returned False"
        warnings.warn(
            "GraphEm spectral initialization is using CPU because CUDA is "
            f"unavailable ({reason}); golden and scaling runs must pin "
            "device='cuda'",
            RuntimeWarning,
            stacklevel=3,
        )
        return torch.device("cpu"), requested_text, reason

    try:
        resolved = torch.device(requested)
    except (RuntimeError, TypeError) as error:
        raise ValueError("device must be 'cuda', 'cpu', 'auto', or a torch.device") \
            from error
    if resolved.type not in {"cuda", "cpu"}:
        raise ValueError("GraphEm spectral initialization supports only CUDA or CPU")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA spectral initialization was requested but Torch reports "
                "that CUDA is unavailable"
            )
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA spectral device index {resolved.index} is unavailable"
            )
        return resolved, requested_text, None

    reason = "CPU was explicitly selected"
    warnings.warn(
        "GraphEm spectral initialization is using CPU because device='cpu' "
        "was explicitly selected; golden and scaling runs must pin device='cuda'",
        RuntimeWarning,
        stacklevel=3,
    )
    return resolved, requested_text, reason


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _bounded_midpoint_query_batch_size(value: object) -> int:
    parsed = _positive_integer("midpoint_query_batch_size", value)
    if parsed > MIDPOINT_QUERY_BATCH_SIZE_BOUND:
        raise ValueError(
            "midpoint_query_batch_size cannot exceed the canonical "
            f"bound of {MIDPOINT_QUERY_BATCH_SIZE_BOUND}"
        )
    return parsed


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return parsed


class GraphEmbedder:  # pylint: disable=too-many-instance-attributes
    """Embed one undirected simple graph with the canonical GraphEm dynamics.

    Exactly one of ``adjacency`` or ``edges`` must be supplied.  The graph must
    be loop-free, duplicate-free, and large enough for the requested block
    eigenspace and midpoint neighbourhood.  Disconnected graphs and isolated
    vertices use the same normalized-Laplacian convention as connected graphs.
    """

    _force_modules = {}

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        adjacency=None,
        n_components: int = 2,
        L_min: float = 1.0,
        k_attr: float = 0.2,
        k_inter: float = 0.5,
        n_neighbors: int = 10,
        sample_size: int = 256,
        seed: int = 0,
        verbose: bool = True,
        logger_instance: Optional[logging.Logger] = None,
        *,
        device="cuda",
        midpoint_query_batch_size: int = MIDPOINT_QUERY_BATCH_SIZE_BOUND,
        edges=None,
        n_vertices: Optional[int] = None,
    ):
        _require_gpu()
        if (adjacency is None) == (edges is None):
            raise ValueError("provide exactly one of adjacency or edges")

        self.n_components = _positive_integer("n_components", n_components)
        if self.n_components < 2:
            raise ValueError("n_components must be at least two")
        self.L_min = _positive_finite("L_min", L_min)
        self.k_attr = _positive_finite("k_attr", k_attr)
        self.k_inter = _positive_finite("k_inter", k_inter)
        self.n_neighbors = _positive_integer("n_neighbors", n_neighbors)
        requested_sample_size = _positive_integer("sample_size", sample_size)
        requested_query_batch_size = _bounded_midpoint_query_batch_size(
            midpoint_query_batch_size
        )
        self._midpoint_query_batch_size = requested_query_batch_size
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
            raise TypeError("seed must be an integer")
        if not 0 <= int(seed) <= int(np.iinfo(np.uint32).max):
            raise ValueError("seed must be between zero and 2**32 - 1")
        self.seed = int(seed)
        if not isinstance(verbose, (bool, np.bool_)):
            raise TypeError("verbose must be boolean")
        self.verbose = bool(verbose)
        self.logger = logger_instance if logger_instance is not None else LOGGER
        self._spectral_device_request = device
        self._spectral_device_requested = str(device).lower()
        self._spectral_device = None
        self._spectral_device_reason = None

        device_edges, vertex_count = self._canonical_graph(
            adjacency=adjacency,
            edges=edges,
            n_vertices=n_vertices,
        )
        self.edges = device_edges
        self.n = vertex_count
        self.n_edges = int(device_edges.shape[0])
        eigen_count = self.n_components + 1
        if eigen_count >= self.n:
            raise ValueError(
                "n_components + 1 must be smaller than the vertex count"
            )
        if self.n_edges <= self.n_neighbors:
            raise ValueError("n_neighbors must be smaller than the edge count")
        if requested_sample_size > self.n_edges:
            raise ValueError("sample_size cannot exceed the edge count")
        self.sample_size = requested_sample_size
        self.sampled_edge_ids = self._fixed_query_edge_ids()

        self._adjacency = self._device_adjacency()
        self.degrees = cp.asarray(self._adjacency.sum(axis=1)).reshape(-1)
        self._neighbor_offsets = cp.ascontiguousarray(
            self._adjacency.indptr, dtype=cp.int64
        )
        self._neighbor_ids = cp.ascontiguousarray(self._adjacency.indices)
        self._midpoint_width_histogram = {}
        self._midpoint_negative_distance_repairs = 0
        self._midpoint_search_call_count = 0
        self._midpoint_search_call_width_histogram = {}
        self._midpoint_query_batch_histogram = {}
        self._midpoint_search_peak_device_bytes = None

        self.timings = {
            "initialization_seconds": 0.0,
            "spring_seconds": 0.0,
            "midpoint_search_seconds": 0.0,
            "intersection_seconds": 0.0,
            "normalization_seconds": 0.0,
        }
        started = time.perf_counter()
        self.positions = self._spectral_initialization()
        cp.cuda.get_current_stream().synchronize()
        self.timings["initialization_seconds"] = time.perf_counter() - started
        self._iteration = 0

    @staticmethod
    def _validate_host_adjacency(adjacency) -> sp.csr_matrix:
        if sp.issparse(adjacency):
            raw = adjacency.tocoo(copy=True)
            if raw.nnz > 1:
                order = np.lexsort((raw.col, raw.row))
                rows = raw.row[order]
                columns = raw.col[order]
                if np.any((rows[1:] == rows[:-1]) & (columns[1:] == columns[:-1])):
                    raise ValueError("adjacency must not contain duplicate entries")
        matrix = sp.csr_matrix(adjacency, dtype=np.float32, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("adjacency must be square")
        if matrix.shape[0] == 0:
            raise ValueError("adjacency must not be empty")
        matrix.sort_indices()
        if not matrix.has_canonical_format:
            raise ValueError("adjacency must not contain duplicate entries")
        matrix.eliminate_zeros()
        if matrix.nnz and not np.all(np.isfinite(matrix.data)):
            raise ValueError("adjacency values must be finite")
        if matrix.diagonal().any():
            raise ValueError("self loops are not permitted")
        if matrix.nnz and not np.all(matrix.data == 1.0):
            raise ValueError("the canonical algorithm requires an unweighted graph")
        difference = matrix - matrix.transpose()
        difference.eliminate_zeros()
        if difference.nnz:
            raise ValueError("adjacency must be symmetric")
        return matrix

    def _canonical_graph(self, *, adjacency, edges, n_vertices):
        if adjacency is not None:
            matrix = self._validate_host_adjacency(adjacency)
            upper = sp.triu(matrix, k=1, format="coo")
            host_edges = np.column_stack((upper.row, upper.col))
            vertex_count = int(matrix.shape[0])
            device_edges = cp.asarray(host_edges)
        else:
            device_edges = cp.asarray(edges)
            if device_edges.ndim != 2 or device_edges.shape[1] != 2:
                raise ValueError("edges must have shape (n_edges, 2)")
            if device_edges.dtype.kind not in "iu":
                raise TypeError("edge endpoints must be integers")
            if n_vertices is None:
                raise ValueError("n_vertices is required with an edge list")
            vertex_count = _positive_integer("n_vertices", n_vertices)

        if vertex_count > int(np.iinfo(np.int32).max):
            raise ValueError("vertex count exceeds the canonical global edge-ID range")

        if int(device_edges.shape[0]) == 0:
            raise ValueError("the canonical graph must contain edges")
        edge_dtype = cp.int32 if vertex_count < np.iinfo(np.int32).max else cp.int64
        device_edges = cp.asarray(device_edges, dtype=edge_dtype)
        if bool(cp.any(device_edges < 0).item()) or bool(
            cp.any(device_edges >= vertex_count).item()
        ):
            raise ValueError("edge endpoint is outside the graph")
        if bool(cp.any(device_edges[:, 0] == device_edges[:, 1]).item()):
            raise ValueError("self loops are not permitted")

        source = cp.minimum(device_edges[:, 0], device_edges[:, 1])
        target = cp.maximum(device_edges[:, 0], device_edges[:, 1])
        canonical = cp.column_stack((source, target))
        order = cp.lexsort(cp.stack((canonical[:, 1], canonical[:, 0]), axis=0))
        canonical = canonical[order]
        if canonical.shape[0] > 1 and bool(
            cp.any(cp.all(canonical[1:] == canonical[:-1], axis=1)).item()
        ):
            raise ValueError("duplicate undirected edges are not permitted")
        return cp.ascontiguousarray(canonical), vertex_count

    def _device_adjacency(self):
        source = self.edges[:, 0]
        target = self.edges[:, 1]
        rows = cp.concatenate((source, target))
        columns = cp.concatenate((target, source))
        values = cp.ones(rows.shape[0], dtype=cp.float32)
        adjacency = cpx_sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(self.n, self.n),
            dtype=cp.float32,
        )
        adjacency.sort_indices()
        if not adjacency.has_sorted_indices:
            raise RuntimeError("canonical CSR neighbor rows are not sorted")
        return adjacency

    @staticmethod
    def _uniform_query_edge_ids(n_edges, sample_size, seed):
        """Return a uniform sample without replacement using O(sample_size) memory."""
        if not 0 < sample_size <= n_edges:
            raise ValueError("sample_size must be between one and n_edges")
        if sample_size == n_edges:
            return np.arange(n_edges, dtype=np.int64)
        generator = np.random.default_rng(seed)
        selected = set()
        result = []
        for upper in range(n_edges - sample_size, n_edges):
            candidate = int(generator.integers(0, upper + 1))
            if candidate in selected:
                candidate = upper
            selected.add(candidate)
            result.append(candidate)
        if len(result) != sample_size or len(selected) != sample_size:
            raise RuntimeError("uniform query-edge sampling produced duplicate IDs")
        return np.asarray(result, dtype=np.int64)

    def _fixed_query_edge_ids(self):
        if self.sample_size == self.n_edges:
            return cp.arange(self.n_edges, dtype=cp.int64)
        return cp.asarray(
            self._uniform_query_edge_ids(
                self.n_edges,
                self.sample_size,
                self.seed,
            )
        )

    @staticmethod
    def _torch_edge_tensor(edges, device):
        """Move canonical edges to one Torch device without a CUDA host copy."""
        if isinstance(edges, torch.Tensor):
            edge_tensor = edges.detach().to(device=device, dtype=torch.int64)
        elif cp is not None and isinstance(edges, cp.ndarray):
            if device.type == "cuda":
                edge_tensor = torch.utils.dlpack.from_dlpack(edges)
                edge_tensor = edge_tensor.to(device=device, dtype=torch.int64)
            else:
                edge_tensor = torch.as_tensor(
                    cp.asnumpy(edges), dtype=torch.int64, device=device
                )
        else:
            edge_tensor = torch.as_tensor(
                np.asarray(edges), dtype=torch.int64, device=device
            )
        return edge_tensor.contiguous()

    @staticmethod
    def _orient_tensor_columns(vectors):
        column_ids = torch.arange(vectors.shape[1], device=vectors.device)
        pivots = torch.argmax(torch.abs(vectors), dim=0)
        pivot_values = vectors[pivots, column_ids]
        signs = torch.where(
            pivot_values < 0,
            vectors.new_tensor(-1.0),
            vectors.new_tensor(1.0),
        )
        return vectors * signs.unsqueeze(0)

    @staticmethod
    def _tensor_sha256(tensor, numpy_dtype, chunk_rows=1 << 20):
        """Hash a tensor's canonical little-endian evidence stream."""
        digest = hashlib.sha256()
        for begin in range(0, tensor.shape[0], chunk_rows):
            host = (
                tensor[begin : begin + chunk_rows]
                .detach()
                .contiguous()
                .cpu()
                .numpy()
                .astype(numpy_dtype, copy=False)
            )
            digest.update(host.tobytes(order="C"))
        return digest.hexdigest()

    @staticmethod
    def _torch_spectral_start(n_vertices, eigen_count, seed, device):
        """Build a deterministic full-rank orthonormal Torch block."""
        vertex_ids = torch.arange(
            1, n_vertices + 1, dtype=torch.float64, device=device
        ).unsqueeze(1)
        column_ids = torch.arange(
            1, eigen_count + 1, dtype=torch.float64, device=device
        ).unsqueeze(0)
        phase = (
            float(seed + 1) * 0.6180339887498949
            + column_ids * 1.4142135623730951
        )
        raw = torch.sin(vertex_ids * (column_ids + 0.5) + phase)
        raw = raw + torch.cos(vertex_ids * (column_ids + 1.5) + phase)
        start, factor = torch.linalg.qr(raw, mode="reduced")
        diagonal = torch.abs(torch.diagonal(factor))
        maximum = torch.max(diagonal)
        rank_ratio = torch.min(diagonal) / torch.clamp(
            maximum, min=torch.finfo(torch.float64).tiny
        )
        if not bool(torch.isfinite(rank_ratio).item()) or float(rank_ratio.item()) <= 1e-12:
            raise FloatingPointError("spectral start block is not numerically full rank")
        start = GraphEmbedder._orient_tensor_columns(start.contiguous())
        identity = torch.eye(eigen_count, dtype=torch.float64, device=device)
        orthogonality_error = torch.max(
            torch.abs(start.mT @ start - identity)
        )
        if not bool(torch.isfinite(orthogonality_error).item()) or float(
            orthogonality_error.item()
        ) > float(SPECTRAL_ORTHOGONALITY_BOUND):
            raise FloatingPointError(
                "spectral start block failed the orthogonality gate"
            )
        return start, float(rank_ratio.item()), float(orthogonality_error.item())

    @staticmethod
    def _torch_subspace_repeat_metrics(reference, candidate):
        """Measure numerical subspace drift without making it a runtime gate."""
        if reference.ndim != 2 or candidate.ndim != 2:
            raise ValueError("subspace samples must be matrices")
        if tuple(reference.shape) != tuple(candidate.shape):
            raise ValueError("subspace samples must have identical shapes")
        device = reference.device
        first = reference.to(device=device, dtype=torch.float64)
        second = candidate.to(device=device, dtype=torch.float64)
        first, _ = torch.linalg.qr(first, mode="reduced")
        second, _ = torch.linalg.qr(second, mode="reduced")
        singular_values = torch.linalg.svdvals(first.mT @ second).clamp(0.0, 1.0)
        projector_frobenius = torch.sqrt(
            torch.clamp(
                2.0 * first.shape[1] - 2.0 * torch.sum(singular_values**2),
                min=0.0,
            )
        )
        largest_principal_angle = torch.acos(torch.min(singular_values))
        return {
            "projector_frobenius_distance": float(projector_frobenius.item()),
            "largest_principal_angle_radians": float(
                largest_principal_angle.item()
            ),
            "canonical_correlations": [
                float(value) for value in singular_values.detach().cpu().tolist()
            ],
        }

    @staticmethod
    def _torch_spectral_embedding(
        edges, n_vertices, n_components, seed, device="cuda"
    ):  # pylint: disable=too-many-locals
        """Compute one device-parameterized normalized-Laplacian embedding."""
        resolved, requested, selection_reason = _resolve_spectral_device(device)
        eigen_count = n_components + 1
        if n_vertices < 3 * eigen_count:
            raise ValueError(
                "Torch LOBPCG requires n_vertices >= 3 * (n_components + 1)"
            )
        solver_block_width = min(
            max(SPECTRAL_MINIMUM_BLOCK_WIDTH, eigen_count),
            n_vertices // 3,
        )

        if resolved.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved)

        def synchronize():
            if resolved.type == "cuda":
                torch.cuda.synchronize(resolved)

        timings = {}
        total_started = time.perf_counter()
        stage_started = time.perf_counter()
        edge_tensor = GraphEmbedder._torch_edge_tensor(edges, resolved)
        if edge_tensor.ndim != 2 or edge_tensor.shape[1] != 2:
            raise ValueError("edges must have shape (n_edges, 2)")
        source, target = edge_tensor[:, 0], edge_tensor[:, 1]
        endpoints = edge_tensor.reshape(-1)
        degrees = torch.bincount(endpoints, minlength=n_vertices).to(torch.float64)
        positive_degree = degrees > 0
        inverse_sqrt_degree = torch.zeros_like(degrees)
        inverse_sqrt_degree[positive_degree] = torch.rsqrt(degrees[positive_degree])
        weights = inverse_sqrt_degree[source] * inverse_sqrt_degree[target]
        diagonal_ids = torch.arange(n_vertices, dtype=torch.int64, device=resolved)
        rows = torch.cat((source, target, diagonal_ids))
        columns = torch.cat((target, source, diagonal_ids))
        diagonal = torch.where(
            positive_degree,
            degrees.new_tensor(2.0),
            degrees.new_tensor(float(SPECTRAL_SHIFT)),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse invariant checks are implicitly disabled.*",
                category=UserWarning,
            )
            shifted = torch.sparse_coo_tensor(
                torch.stack((rows, columns)),
                torch.cat((weights, weights, diagonal)),
                size=(n_vertices, n_vertices),
                dtype=torch.float64,
                device=resolved,
                check_invariants=False,
            ).coalesce()
        edge_tensor_sha256 = GraphEmbedder._tensor_sha256(edge_tensor, "<i8")
        operator_receipt = (
            "three-identity-minus-symmetric-normalized-laplacian|"
            f"float64|{n_vertices}|{edge_tensor.shape[0]}|{edge_tensor_sha256}"
        )
        operator_sha256 = hashlib.sha256(operator_receipt.encode("utf-8")).hexdigest()
        synchronize()
        timings["operator_seconds"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        start, start_rank_ratio, start_orthogonality_error = (
            GraphEmbedder._torch_spectral_start(
                n_vertices, solver_block_width, seed, resolved
            )
        )
        start_sha256 = GraphEmbedder._tensor_sha256(start, "<f8")
        synchronize()
        timings["start_seconds"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        tracker_state = {
            "iterations": 0,
            "converged_count": 0,
        }

        def tracker(worker):
            tracker_state["iterations"] = int(worker.ivars["istep"])
            tracker_state["converged_count"] = int(
                worker.ivars.get("converged_count", 0)
            )

        shifted_eigenvalues, eigenvectors = torch.lobpcg(
            shifted,
            k=eigen_count,
            X=start,
            niter=SPECTRAL_MAX_ITERATIONS,
            tol=float(SPECTRAL_TOLERANCE),
            largest=True,
            method="ortho",
            tracker=tracker,
        )
        solver = "torch.lobpcg"
        synchronize()
        timings["solver_seconds"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        eigenvalues = float(SPECTRAL_SHIFT) - shifted_eigenvalues
        order = torch.argsort(eigenvalues, stable=True)
        eigenvalues = eigenvalues[order]
        eigenvectors = GraphEmbedder._orient_tensor_columns(
            eigenvectors[:, order].contiguous()
        )
        if not bool(torch.isfinite(eigenvalues).all().item()) or not bool(
            torch.isfinite(eigenvectors).all().item()
        ):
            raise FloatingPointError("spectral solver returned non-finite values")
        laplacian_times_vectors = (
            float(SPECTRAL_SHIFT) * eigenvectors
            - torch.sparse.mm(shifted, eigenvectors)
        )
        residual = (
            laplacian_times_vectors
            - eigenvectors * eigenvalues.unsqueeze(0)
        )
        denominator = torch.clamp(
            torch.linalg.vector_norm(eigenvectors, dim=0),
            min=torch.finfo(torch.float64).tiny,
        )
        residual_norm_ratios = torch.linalg.vector_norm(residual, dim=0) / denominator
        identity = torch.eye(eigen_count, dtype=torch.float64, device=resolved)
        orthogonality_error = torch.max(
            torch.abs(eigenvectors.mT @ eigenvectors - identity)
        )
        positions = eigenvectors[:, 1 : n_components + 1].to(torch.float32)
        if tuple(positions.shape) != (n_vertices, n_components):
            raise RuntimeError("spectral solver returned an unexpected shape")
        if not bool(torch.isfinite(residual_norm_ratios).all().item()) or not bool(
            torch.isfinite(positions).all().item()
        ):
            raise FloatingPointError("spectral solver returned non-finite values")
        maximum_residual = float(torch.max(residual_norm_ratios).item())
        maximum_orthogonality_error = float(orthogonality_error.item())
        if maximum_residual > float(SPECTRAL_RESIDUAL_BOUND):
            raise RuntimeError(
                "spectral solver residual exceeds the accepted bound: "
                f"{maximum_residual:.6e} > {float(SPECTRAL_RESIDUAL_BOUND):.6e}"
            )
        if maximum_orthogonality_error > float(SPECTRAL_ORTHOGONALITY_BOUND):
            raise RuntimeError(
                "spectral solver orthogonality error exceeds the accepted bound: "
                f"{maximum_orthogonality_error:.6e} > "
                f"{float(SPECTRAL_ORTHOGONALITY_BOUND):.6e}"
            )
        eigenvalues_sha256 = GraphEmbedder._tensor_sha256(eigenvalues, "<f8")
        eigenvectors_sha256 = GraphEmbedder._tensor_sha256(eigenvectors, "<f8")
        output_sha256 = GraphEmbedder._tensor_sha256(positions, "<f4")
        synchronize()
        timings["audit_seconds"] = time.perf_counter() - stage_started
        timings["total_seconds"] = time.perf_counter() - total_started

        eigenvalue_list = eigenvalues.detach().cpu().tolist()
        gaps = [
            float(eigenvalue_list[index + 1] - eigenvalue_list[index])
            for index in range(len(eigenvalue_list) - 1)
        ]
        clusters = []
        cluster_start = 0
        for index, gap in enumerate(gaps):
            if abs(gap) > float(SPECTRAL_CLUSTER_BOUND):
                clusters.append([cluster_start, index + 1])
                cluster_start = index + 1
        clusters.append([cluster_start, len(eigenvalue_list)])
        diagnostics = {
            "backend": TORCH_SPECTRAL_BACKEND,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "device_requested": requested,
            "device_selected": str(resolved),
            "device_selection_reason": selection_reason,
            "operator": "three-identity-minus-symmetric-normalized-laplacian",
            "operator_shift": float(SPECTRAL_SHIFT),
            "operator_eigenvalue_mapping": (
                "normalized-laplacian-eigenvalue=three-minus-operator-eigenvalue"
            ),
            "operator_sha256": operator_sha256,
            "operator_edge_tensor_sha256": edge_tensor_sha256,
            "operator_nnz": int(shifted._nnz()),
            "isolated_vertices": int(torch.sum(~positive_degree).item()),
            "normalized_laplacian": (
                "symmetric-normalized-isolate-diagonal-zero-float64-v1"
            ),
            "operator_dtype": "float64",
            "solver": solver,
            "method": "ortho",
            "largest": True,
            "output_eigenpair_count": eigen_count,
            "solver_block_width": solver_block_width,
            "solver_oversampling_count": solver_block_width - eigen_count,
            "solver_block_width_policy": (
                "min(max(16,n_components+1),floor(n_vertices/3))"
            ),
            "tolerance": float(SPECTRAL_TOLERANCE),
            "maximum_iterations": SPECTRAL_MAX_ITERATIONS,
            "domain_requirement": (
                "n_vertices>=3*output_eigenpair_count;"
                "solver_block_width<=floor(n_vertices/3);otherwise-fail-closed"
            ),
            "observed_iterations": tracker_state["iterations"],
            "reported_converged_count": tracker_state["converged_count"],
            "start_algorithm": SPECTRAL_START_ALGORITHM,
            "start_formula": (
                "sin((i+1)*(j+1+0.5)+phase_j)+"
                "cos((i+1)*(j+1+1.5)+phase_j);"
                "phase_j=(seed+1)*golden_ratio+(j+1)*sqrt(2);thin-qr"
            ),
            "start_sha256": start_sha256,
            "start_rank_ratio": start_rank_ratio,
            "start_orthogonality_error": start_orthogonality_error,
            "eigenvalue_order": "normalized-laplacian-ascending-stable",
            "sign_orientation": "lowest-argmax-absolute-pivot-nonnegative",
            "eigenvalues": [float(value) for value in eigenvalue_list],
            "eigenvalues_sha256": eigenvalues_sha256,
            "eigenvectors_sha256": eigenvectors_sha256,
            "output_float32_sha256": output_sha256,
            "eigenpair_residual_norm_ratios": [
                float(value) for value in residual_norm_ratios.detach().cpu().tolist()
            ],
            "residual_numerator": "l2-norm-of-Lv-minus-lambda-v",
            "residual_denominator": "l2-norm-of-v",
            "maximum_eigenpair_residual_norm_ratio": maximum_residual,
            "orthogonality_error": maximum_orthogonality_error,
            "residual_bound": float(SPECTRAL_RESIDUAL_BOUND),
            "orthogonality_bound": float(SPECTRAL_ORTHOGONALITY_BOUND),
            "cluster_gap_bound": float(SPECTRAL_CLUSTER_BOUND),
            "eigenvalue_gaps": gaps,
            "eigenvalue_clusters_half_open": clusters,
            "torch_peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(resolved))
                if resolved.type == "cuda"
                else None
            ),
            "torch_peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(resolved))
                if resolved.type == "cuda"
                else None
            ),
            "timings": timings,
        }
        return positions.contiguous(), diagnostics

    def _spectral_initialization(self):
        positions, diagnostics = self._torch_spectral_embedding(
            self.edges,
            self.n,
            self.n_components,
            self.seed,
            self._spectral_device_request,
        )
        self._spectral_diagnostics = diagnostics
        self._spectral_device = torch.device(diagnostics["device_selected"])
        self._spectral_device_reason = diagnostics["device_selection_reason"]
        self._spectral_eigenvalues = diagnostics["eigenvalues"]
        self._spectral_max_residual_norm_ratio = diagnostics[
            "maximum_eigenpair_residual_norm_ratio"
        ]
        if positions.device.type == "cuda":
            device_positions = cp.from_dlpack(positions.detach())
        else:
            device_positions = cp.asarray(positions.detach().cpu().numpy())
        return cp.ascontiguousarray(device_positions, dtype=cp.float32)

    @staticmethod
    def _search_result_arrays(result):
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("cuVS brute-force search must return two arrays")
        distances, neighbors = result
        distances = cp.asarray(distances)
        neighbors = cp.asarray(neighbors)
        if distances.dtype != cp.float32 or neighbors.dtype.kind not in "iu":
            raise TypeError("unexpected cuVS brute-force result dtypes")
        if distances.shape != neighbors.shape:
            raise ValueError("cuVS distances and neighbor IDs must share a shape")
        if not bool(cp.all(cp.isfinite(distances)).item()):
            raise FloatingPointError("cuVS returned non-finite midpoint distances")
        if distances.shape[1] > 1 and bool(
            cp.any(distances[:, 1:] < distances[:, :-1]).item()
        ):
            raise ValueError("cuVS midpoint distances are not sorted")
        return distances, neighbors

    @staticmethod
    def _validate_unique_global_neighbor_ids(neighbors):
        """Fail if one cuVS query row repeats a global edge ID."""
        neighbors = cp.asarray(neighbors)
        if neighbors.ndim != 2:
            raise ValueError("cuVS neighbor IDs must be a two-dimensional array")
        if neighbors.dtype.kind not in "iu":
            raise TypeError("cuVS neighbor IDs must be integers")
        if neighbors.shape[1] < 2:
            return
        sorted_ids = cp.sort(neighbors, axis=1)
        duplicate = bool(cp.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]).item())
        if duplicate:
            raise ValueError(
                "cuVS returned duplicate global edge IDs within a query row"
            )

    @staticmethod
    def _repair_negative_squared_distances(
        distances, neighbors, queries, reference_midpoints
    ):
        """Recompute bounded negative cuVS squared distances directly."""
        negative = distances < 0
        repair_count = int(cp.sum(negative).item())
        if repair_count == 0:
            return distances, repair_count

        negative_rows, _negative_columns = cp.nonzero(negative)
        query_values = cp.asarray(queries[negative_rows], dtype=cp.float32)
        reference_values = cp.asarray(
            reference_midpoints[neighbors[negative]], dtype=cp.float32
        )
        deltas = query_values - reference_values
        direct = cp.sum(deltas * deltas, axis=1, dtype=cp.float32)

        query64 = query_values.astype(cp.float64)
        reference64 = reference_values.astype(cp.float64)
        absolute_query = cp.abs(query64)
        absolute_reference = cp.abs(reference64)
        scale = cp.sum(
            absolute_query * absolute_query
            + absolute_reference * absolute_reference
            + np.float64(2.0) * absolute_query * absolute_reference,
            axis=1,
            dtype=cp.float64,
        )
        operation_count = 2 * queries.shape[1] + 5
        gamma = (operation_count * FLOAT32_UNIT_ROUNDOFF) / (
            1.0 - operation_count * FLOAT32_UNIT_ROUNDOFF
        )
        error_bound = np.float64(gamma) * scale
        raw_negative = distances[negative].astype(cp.float64)
        discrepancy = cp.abs(raw_negative - direct.astype(cp.float64))
        valid = (
            cp.isfinite(direct)
            & (direct >= 0)
            & cp.isfinite(error_bound)
            & (discrepancy <= error_bound)
        )
        if not bool(cp.all(valid).item()):
            raise FloatingPointError(
                "negative cuVS squared distance exceeds the float32 error bound"
            )
        repaired = distances.copy()
        repaired[negative] = direct
        return repaired, repair_count

    @staticmethod
    def _lexicographic_nonself_candidates(neighbors, distances, query_edge_ids):
        neighbors = cp.asarray(neighbors)
        distances = cp.asarray(distances)
        query_edge_ids = cp.asarray(query_edge_ids)
        if neighbors.ndim != 2 or query_edge_ids.shape != (neighbors.shape[0],):
            raise ValueError("neighbor rows must align with query edge IDs")
        if distances.shape != neighbors.shape:
            raise ValueError("neighbor distances must align with neighbor IDs")
        if neighbors.dtype.kind not in "iu" or query_edge_ids.dtype.kind not in "iu":
            raise TypeError("neighbor and query IDs must be integers")
        usable = neighbors != query_edge_ids[:, None]
        usable_counts = cp.sum(usable, axis=1)
        distance_keys = cp.where(usable, distances, cp.inf)
        id_keys = cp.where(usable, neighbors, cp.iinfo(neighbors.dtype).max)
        row_keys = cp.repeat(
            cp.arange(neighbors.shape[0], dtype=cp.int64), neighbors.shape[1]
        )
        flat_order = cp.lexsort(
            cp.stack(
                (
                    id_keys.reshape(-1),
                    distance_keys.reshape(-1),
                    row_keys,
                ),
                axis=0,
            )
        )
        ordered_rows = row_keys[flat_order].reshape(neighbors.shape)
        expected_rows = cp.arange(neighbors.shape[0], dtype=cp.int64)[:, None]
        if bool(cp.any(ordered_rows != expected_rows).item()):
            raise RuntimeError("batched midpoint ordering mixed query rows")
        column_order = (flat_order % neighbors.shape[1]).reshape(neighbors.shape)
        ordered_neighbors = cp.take_along_axis(neighbors, column_order, axis=1)
        ordered_distances = cp.take_along_axis(distance_keys, column_order, axis=1)
        return ordered_neighbors, ordered_distances, usable_counts

    @staticmethod
    def _compact_nonself_neighbors(neighbors, distances, query_edge_ids, count):
        ordered_neighbors, _, usable_counts = (
            GraphEmbedder._lexicographic_nonself_candidates(
                neighbors, distances, query_edge_ids
            )
        )
        if bool(cp.any(usable_counts < count).item()):
            raise RuntimeError("cuVS returned fewer than the required non-self neighbors")
        compacted = ordered_neighbors[:, :count]
        query_edge_ids = cp.asarray(query_edge_ids)
        if bool(cp.any(compacted == query_edge_ids[:, None]).item()):
            raise RuntimeError("self neighbor survived identity-based removal")
        return compacted

    @staticmethod
    def _current_device_memory_used_bytes():
        """Return device-wide used bytes at one declared CUDA checkpoint."""
        cuda = getattr(cp, "cuda", None)
        runtime = getattr(cuda, "runtime", None)
        if runtime is None:  # GPU-free NumPy contract tests.
            return None
        free_bytes, total_bytes = runtime.memGetInfo()
        free_bytes = int(free_bytes)
        total_bytes = int(total_bytes)
        if free_bytes < 0 or total_bytes <= 0 or free_bytes > total_bytes:
            raise RuntimeError("CUDA returned an invalid device-memory snapshot")
        return total_bytes - free_bytes

    def _observe_midpoint_search_device_memory(self):
        used_bytes = self._current_device_memory_used_bytes()
        if used_bytes is None:
            return
        previous = getattr(self, "_midpoint_search_peak_device_bytes", None)
        if previous is None or used_bytes > previous:
            self._midpoint_search_peak_device_bytes = used_bytes

    def _midpoint_neighbors(self):
        midpoints = cp.ascontiguousarray(
            np.float32(0.5)
            * (self.positions[self.edges[:, 0]] + self.positions[self.edges[:, 1]]),
            dtype=cp.float32,
        )
        self._observe_midpoint_search_device_memory()
        queries = cp.ascontiguousarray(midpoints[self.sampled_edge_ids])
        self._observe_midpoint_search_device_memory()
        index = brute_force.build(midpoints, metric="sqeuclidean")
        self._observe_midpoint_search_device_memory()
        search_width = min(self.n_edges, self.n_neighbors + 2)
        unresolved = cp.arange(self.sample_size, dtype=cp.int64)
        resolved = cp.empty(
            (self.sample_size, self.n_neighbors), dtype=cp.int64
        )
        query_batch_size = min(
            self.sample_size,
            _bounded_midpoint_query_batch_size(
                getattr(
                    self,
                    "_midpoint_query_batch_size",
                    MIDPOINT_QUERY_BATCH_SIZE_BOUND,
                )
            ),
        )
        while int(unresolved.size):
            next_unresolved = []
            for batch_start in range(0, int(unresolved.size), query_batch_size):
                batch_rows = unresolved[
                    batch_start : batch_start + query_batch_size
                ]
                query_subset = queries[batch_rows]
                submitted_count = int(batch_rows.size)
                self._midpoint_search_call_count = (
                    getattr(self, "_midpoint_search_call_count", 0) + 1
                )
                call_width_histogram = getattr(
                    self, "_midpoint_search_call_width_histogram", {}
                )
                call_width_histogram[search_width] = (
                    call_width_histogram.get(search_width, 0) + 1
                )
                self._midpoint_search_call_width_histogram = (
                    call_width_histogram
                )
                batch_histogram = getattr(
                    self, "_midpoint_query_batch_histogram", {}
                )
                batch_histogram[submitted_count] = (
                    batch_histogram.get(submitted_count, 0) + 1
                )
                self._midpoint_query_batch_histogram = batch_histogram
                self._observe_midpoint_search_device_memory()
                result = brute_force.search(index, query_subset, search_width)
                self._observe_midpoint_search_device_memory()
                distances, neighbors = self._search_result_arrays(result)
                raw_search_boundary = distances[:, -1].copy()
                expected_shape = (submitted_count, search_width)
                if neighbors.shape != expected_shape:
                    raise ValueError(
                        f"cuVS returned neighbor shape {neighbors.shape}, "
                        f"expected {expected_shape}"
                    )
                if bool(cp.any(neighbors < 0).item()) or bool(
                    cp.any(neighbors >= self.n_edges).item()
                ):
                    raise ValueError(
                        "cuVS returned an ID outside the global edge namespace"
                    )
                self._validate_unique_global_neighbor_ids(neighbors)
                distances, repair_count = self._repair_negative_squared_distances(
                    distances, neighbors, query_subset, midpoints
                )
                self._midpoint_negative_distance_repairs += repair_count
                self._observe_midpoint_search_device_memory()

                query_edge_ids = self.sampled_edge_ids[batch_rows]
                ordered_neighbors, ordered_distances, usable_counts = (
                    self._lexicographic_nonself_candidates(
                        neighbors, distances, query_edge_ids
                    )
                )
                if bool(cp.any(usable_counts < self.n_neighbors).item()):
                    raise RuntimeError(
                        "cuVS returned fewer than the required non-self neighbors"
                    )
                selected = ordered_neighbors[:, : self.n_neighbors]
                cutoff = ordered_distances[:, self.n_neighbors - 1]
                if search_width == self.n_edges:
                    complete = cp.ones(batch_rows.shape, dtype=cp.bool_)
                else:
                    complete = raw_search_boundary > cutoff
                completed_count = int(cp.sum(complete).item())
                if completed_count:
                    completed_rows = batch_rows[complete]
                    resolved[completed_rows] = selected[complete].astype(
                        cp.int64, copy=False
                    )
                    self._midpoint_width_histogram[search_width] = (
                        self._midpoint_width_histogram.get(search_width, 0)
                        + completed_count
                    )
                remaining_rows = batch_rows[~complete]
                if int(remaining_rows.size):
                    next_unresolved.append(remaining_rows)
                del (
                    batch_rows,
                    query_subset,
                    result,
                    distances,
                    neighbors,
                    raw_search_boundary,
                    query_edge_ids,
                    ordered_neighbors,
                    ordered_distances,
                    usable_counts,
                    selected,
                    cutoff,
                    complete,
                    remaining_rows,
                )

            if next_unresolved:
                unresolved = cp.concatenate(tuple(next_unresolved))
            else:
                unresolved = cp.empty(0, dtype=cp.int64)
            if int(unresolved.size):
                if search_width == self.n_edges:
                    raise RuntimeError(
                        "full midpoint reference did not resolve every query"
                    )
                search_width = min(self.n_edges, search_width * 2)

        if bool(cp.any(resolved == self.sampled_edge_ids[:, None]).item()):
            raise RuntimeError("self neighbor survived identity-based removal")
        return resolved

    def _spring_forces(self):
        forces = cp.zeros_like(self.positions)
        device_id = cp.cuda.Device().id
        module = self._force_modules.get(device_id)
        if module is None:
            module = cp.RawModule(
                code=_DETERMINISTIC_FORCE_KERNELS,
                options=("--std=c++11",),
            )
            self._force_modules[device_id] = module
        suffix = "i32" if self._neighbor_ids.dtype == cp.int32 else "i64"
        kernel = module.get_function(f"graphem_spring_{suffix}")
        threads = 256
        blocks = (self.n + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                self.positions,
                self._neighbor_offsets,
                self._neighbor_ids,
                np.int64(self.n),
                np.int32(self.n_components),
                np.float32(self.L_min),
                np.float32(self.k_attr),
                forces,
            ),
        )
        return forces

    @staticmethod
    def _strict_xy_crossing(p1, p2, q1, q2):
        def orientation(first, second, third):
            return (second[:, 0] - first[:, 0]) * (
                third[:, 1] - first[:, 1]
            ) - (second[:, 1] - first[:, 1]) * (
                third[:, 0] - first[:, 0]
            )

        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)
        first_opposed = ((o1 > 0) & (o2 < 0)) | ((o1 < 0) & (o2 > 0))
        second_opposed = ((o3 > 0) & (o4 < 0)) | ((o3 < 0) & (o4 > 0))
        return first_opposed & second_opposed

    @staticmethod
    def _ordered_endpoint_segments(endpoint_ids, contributions):
        """Order endpoint contributions for deterministic sequential sums."""
        endpoint_ids = cp.asarray(endpoint_ids)
        contributions = cp.asarray(contributions, dtype=cp.float32)
        if endpoint_ids.ndim != 1 or contributions.ndim != 2:
            raise ValueError("endpoint contribution arrays have invalid rank")
        if contributions.shape[0] != endpoint_ids.shape[0]:
            raise ValueError("endpoint IDs and contributions must align")
        if endpoint_ids.dtype.kind not in "iu":
            raise TypeError("endpoint IDs must be integers")
        if int(endpoint_ids.size) == 0:
            raise ValueError("endpoint contribution arrays must not be empty")
        contribution_ids = cp.arange(endpoint_ids.shape[0], dtype=cp.int64)
        order = cp.lexsort(cp.stack((contribution_ids, endpoint_ids), axis=0))
        ordered_ids = cp.ascontiguousarray(endpoint_ids[order])
        ordered_contributions = cp.ascontiguousarray(contributions[order])
        boundaries = cp.empty(ordered_ids.shape[0], dtype=cp.bool_)
        boundaries[0] = True
        boundaries[1:] = ordered_ids[1:] != ordered_ids[:-1]
        starts = cp.flatnonzero(boundaries).astype(cp.int64, copy=False)
        ends = cp.concatenate(
            (starts[1:], cp.asarray([ordered_ids.shape[0]], dtype=cp.int64))
        )
        vertices = cp.ascontiguousarray(ordered_ids[starts])
        return ordered_contributions, starts, ends, vertices

    def _reduce_endpoint_contributions(self, endpoint_ids, contributions, forces):
        ordered, starts, ends, vertices = self._ordered_endpoint_segments(
            endpoint_ids, contributions
        )
        device_id = cp.cuda.Device().id
        module = self._force_modules.get(device_id)
        if module is None:
            module = cp.RawModule(
                code=_DETERMINISTIC_FORCE_KERNELS,
                options=("--std=c++11",),
            )
            self._force_modules[device_id] = module
        suffix = "i32" if vertices.dtype == cp.int32 else "i64"
        kernel = module.get_function(f"graphem_segment_{suffix}")
        segment_count = int(vertices.shape[0])
        output_count = segment_count * self.n_components
        threads = 256
        blocks = (output_count + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                ordered,
                starts,
                ends,
                vertices,
                np.int64(segment_count),
                np.int32(self.n_components),
                forces,
            ),
        )
        return forces

    def _intersection_forces(self, neighbor_edge_ids):
        forces = cp.zeros_like(self.positions)
        candidate_i = cp.repeat(self.sampled_edge_ids, self.n_neighbors)
        candidate_j = neighbor_edge_ids.reshape(-1)
        ordered = candidate_i < candidate_j
        if not bool(cp.any(ordered).item()):
            return forces
        first_edges = self.edges[candidate_i[ordered]]
        second_edges = self.edges[candidate_j[ordered]]
        shared = (
            (first_edges[:, 0] == second_edges[:, 0])
            | (first_edges[:, 0] == second_edges[:, 1])
            | (first_edges[:, 1] == second_edges[:, 0])
            | (first_edges[:, 1] == second_edges[:, 1])
        )
        first_edges = first_edges[~shared]
        second_edges = second_edges[~shared]
        if int(first_edges.shape[0]) == 0:
            return forces

        p1 = self.positions[first_edges[:, 0]]
        p2 = self.positions[first_edges[:, 1]]
        q1 = self.positions[second_edges[:, 0]]
        q2 = self.positions[second_edges[:, 1]]
        crossing = self._strict_xy_crossing(p1, p2, q1, q2)
        if not bool(cp.any(crossing).item()):
            return forces
        first_edges = first_edges[crossing]
        second_edges = second_edges[crossing]
        p1, p2 = p1[crossing], p2[crossing]
        q1, q2 = q1[crossing], q2[crossing]
        centroid = np.float32(0.25) * (p1 + p2 + q1 + q2)

        def endpoint_force(points):
            displacement = points - centroid
            denominator = cp.linalg.norm(displacement, axis=1, keepdims=True) + EPSILON
            return np.float32(self.k_inter) * displacement / (denominator * denominator)

        endpoint_ids = cp.concatenate(
            (
                first_edges[:, 0],
                first_edges[:, 1],
                second_edges[:, 0],
                second_edges[:, 1],
            )
        )
        contributions = cp.concatenate(
            (
                endpoint_force(p1),
                endpoint_force(p2),
                endpoint_force(q1),
                endpoint_force(q2),
            ),
            axis=0,
        )
        return self._reduce_endpoint_contributions(
            endpoint_ids, contributions, forces
        )

    def update_positions(self):
        """Apply one complete spring, crossing, update, and normalization step."""
        started = time.perf_counter()
        spring = self._spring_forces()
        cp.cuda.get_current_stream().synchronize()
        self.timings["spring_seconds"] += time.perf_counter() - started

        started = time.perf_counter()
        neighbors = self._midpoint_neighbors()
        cp.cuda.get_current_stream().synchronize()
        self.timings["midpoint_search_seconds"] += time.perf_counter() - started

        started = time.perf_counter()
        intersections = self._intersection_forces(neighbors)
        cp.cuda.get_current_stream().synchronize()
        self.timings["intersection_seconds"] += time.perf_counter() - started

        started = time.perf_counter()
        updated = self._normalize_positions(self.positions + spring + intersections)
        if not bool(cp.all(cp.isfinite(updated)).item()):
            raise FloatingPointError("GraphEm update produced non-finite positions")
        self.positions = cp.ascontiguousarray(updated, dtype=cp.float32)
        cp.cuda.get_current_stream().synchronize()
        self.timings["normalization_seconds"] += time.perf_counter() - started
        self._iteration += 1
        return self.positions

    @staticmethod
    def _normalize_positions(positions):
        """Center and scale each coordinate using population statistics."""
        if not bool(cp.all(cp.isfinite(positions)).item()):
            raise FloatingPointError("GraphEm positions are non-finite before normalization")
        means = cp.mean(positions, axis=0, keepdims=True)
        standard_deviations = cp.std(positions, axis=0, ddof=0, keepdims=True)
        if not bool(cp.all(cp.isfinite(means)).item()) or not bool(
            cp.all(cp.isfinite(standard_deviations)).item()
        ):
            raise FloatingPointError("GraphEm normalization statistics are non-finite")
        if bool(cp.any(standard_deviations <= EPSILON).item()):
            raise FloatingPointError("GraphEm embedding collapsed along a coordinate axis")
        return (positions - means) / (standard_deviations + EPSILON)

    def run_layout(self, num_iterations: int = 100):
        """Run exactly ``num_iterations`` complete GraphEm steps."""
        iterations = _nonnegative_integer("num_iterations", num_iterations)
        for iteration in range(iterations):
            self.update_positions()
            if self.verbose and (iteration + 1) % 10 == 0:
                self.logger.info("completed GraphEm iteration %d/%d", iteration + 1, iterations)
        return self.positions

    def get_positions(self, as_numpy: bool = True):
        """Return the complete embedding positions."""
        if not isinstance(as_numpy, (bool, np.bool_)):
            raise TypeError("as_numpy must be boolean")
        return cp.asnumpy(self.positions) if as_numpy else self.positions.copy()

    def get_scores(self, as_numpy: bool = True):
        """Return Euclidean radius for every vertex."""
        if not isinstance(as_numpy, (bool, np.bool_)):
            raise TypeError("as_numpy must be boolean")
        scores = cp.linalg.norm(self.positions, axis=1)
        return cp.asnumpy(scores) if as_numpy else scores

    def get_top_k(self, k: int, as_numpy: bool = True):
        """Return vertex IDs ordered by decreasing radius and then ID."""
        if not isinstance(as_numpy, (bool, np.bool_)):
            raise TypeError("as_numpy must be boolean")
        count = _nonnegative_integer("k", k)
        if count > self.n:
            raise ValueError("k cannot exceed the vertex count")
        if count == 0:
            empty = cp.empty(0, dtype=cp.int64)
            return cp.asnumpy(empty) if as_numpy else empty
        scores = self.get_scores(as_numpy=False)
        vertex_ids = cp.arange(self.n, dtype=cp.int64)
        order = cp.lexsort(cp.stack((vertex_ids, -scores), axis=0))
        selected = order[:count]
        return cp.asnumpy(selected) if as_numpy else selected

    def get_diagnostics(self):
        """Return configuration and primitive timing scopes for this instance."""
        query_ids = cp.asnumpy(self.sampled_edge_ids).astype("<i8", copy=False)
        query_edges = cp.asnumpy(self.edges[self.sampled_edge_ids]).astype(
            "<i8", copy=False
        )
        return {
            "algorithm": "graphem-canonical",
            "graph": {
                "vertices": self.n,
                "edges": self.n_edges,
                "isolated_vertices": int(cp.sum(self.degrees == 0).item()),
            },
            "configuration": {
                "n_components": self.n_components,
                "L_min": self.L_min,
                "k_attr": self.k_attr,
                "k_inter": self.k_inter,
                "n_neighbors": self.n_neighbors,
                "sample_size": self.sample_size,
                "seed": self.seed,
                "device": self._spectral_device_requested,
                "midpoint_query_batch_size": self._midpoint_query_batch_size,
            },
            "iterations": self._iteration,
            "timings": dict(self.timings),
            "query_selection": "uniform-without-replacement-pcg64-floyd-v1",
            "query_edge_count": self.sample_size,
            "query_edge_ids_sha256": hashlib.sha256(
                query_ids.tobytes(order="C")
            ).hexdigest(),
            "query_edge_endpoints_sha256": hashlib.sha256(
                query_edges.tobytes(order="C")
            ).hexdigest(),
            "midpoint_reference": "all-global-edges",
            "midpoint_selection": (
                "adaptive-exact-sqeuclidean-then-global-edge-id-v1"
            ),
            "midpoint_negative_distance_repair": (
                "direct-float32-with-gamma-2d-plus-5-bound-v1"
            ),
            "midpoint_negative_distance_repair_count": (
                self._midpoint_negative_distance_repairs
            ),
            "midpoint_neighbor_id_validation": (
                MIDPOINT_NEIGHBOR_ID_VALIDATION
            ),
            "midpoint_query_batch_policy": MIDPOINT_QUERY_BATCH_POLICY,
            "midpoint_query_batch_size_bound": MIDPOINT_QUERY_BATCH_SIZE_BOUND,
            "midpoint_query_batch_size_effective": min(
                self.sample_size, self._midpoint_query_batch_size
            ),
            "midpoint_search_call_count": self._midpoint_search_call_count,
            "midpoint_search_call_width_histogram": {
                str(width): count
                for width, count in sorted(
                    self._midpoint_search_call_width_histogram.items()
                )
            },
            "midpoint_search_query_batch_histogram": {
                str(size): count
                for size, count in sorted(
                    self._midpoint_query_batch_histogram.items()
                )
            },
            "midpoint_search_peak_device_bytes": (
                self._midpoint_search_peak_device_bytes
            ),
            "midpoint_search_peak_device_bytes_scope": (
                MIDPOINT_MEMORY_OBSERVATION
            ),
            "midpoint_search_width_histogram": {
                str(width): count
                for width, count in sorted(self._midpoint_width_histogram.items())
            },
            "spectral_initialization": TORCH_SPECTRAL_BACKEND,
            "spectral_solver": dict(self._spectral_diagnostics),
            "normalized_laplacian": self._spectral_diagnostics[
                "normalized_laplacian"
            ],
            "spectral_eigenvalues": list(self._spectral_eigenvalues),
            "spectral_max_eigenpair_residual_norm_ratio": (
                self._spectral_max_residual_norm_ratio
            ),
            "score_orientation": "farthest-radius-first",
        }
