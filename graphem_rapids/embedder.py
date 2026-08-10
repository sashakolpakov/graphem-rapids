"""Canonical GPU implementation of GraphEm.

The module intentionally exposes one algorithm.  It follows the executed
paper protocol where that protocol is well-defined, while applying the two
confirmed correctness repairs: restoring spring dynamics and global edge IDs
for midpoint neighbours.
"""

from __future__ import annotations

import hashlib
import logging
import numbers
import time
from typing import Optional

import numpy as np
import scipy.sparse as sp

try:  # Imports remain lazy so documentation and CPU contract tests can import.
    from cuvs.neighbors import brute_force
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    import cupyx.scipy.sparse.csgraph as cpx_csgraph
    import cupyx.scipy.sparse.linalg as cpx_linalg
except ImportError as gpu_import_error:  # pragma: no cover - host dependent
    cp = None
    cpx_sparse = None
    cpx_csgraph = None
    cpx_linalg = None
    brute_force = None
    _GPU_IMPORT_ERROR = gpu_import_error
else:  # pragma: no cover - host dependent
    _GPU_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)
EPSILON = np.float32(1.0e-6)


_SPRING_KERNEL = r"""
#define DEFINE_SPRING_KERNEL(NAME, INDEX_TYPE)                                  \
extern "C" __global__ void NAME(                                                \
    const float* positions, const INDEX_TYPE* edges,                            \
    const long long n_edges, const int n_components,                            \
    const float preferred_length, const float attraction, float* forces)        \
{                                                                                \
    const long long edge_id =                                                    \
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;           \
    if (edge_id >= n_edges) return;                                              \
    const long long source = static_cast<long long>(edges[2 * edge_id]);         \
    const long long target = static_cast<long long>(edges[2 * edge_id + 1]);     \
    const long long source_offset = source * n_components;                       \
    const long long target_offset = target * n_components;                       \
    float squared_norm = 0.0f;                                                   \
    for (int component = 0; component < n_components; ++component) {             \
        const float delta = positions[target_offset + component]                 \
            - positions[source_offset + component];                              \
        squared_norm += delta * delta;                                            \
    }                                                                             \
    const float distance = sqrtf(squared_norm) + 1.0e-6f;                        \
    const float multiplier = attraction * (distance - preferred_length)          \
        / distance;                                                               \
    for (int component = 0; component < n_components; ++component) {             \
        const float delta = positions[target_offset + component]                 \
            - positions[source_offset + component];                              \
        const float force = multiplier * delta;                                  \
        atomicAdd(&forces[source_offset + component], force);                     \
        atomicAdd(&forces[target_offset + component], -force);                    \
    }                                                                             \
}

DEFINE_SPRING_KERNEL(graphem_spring_i32, int)
DEFINE_SPRING_KERNEL(graphem_spring_i64, long long)
"""


def _require_gpu() -> None:
    if _GPU_IMPORT_ERROR is not None:
        raise ImportError(
            "GraphEm requires CuPy, cupyx, pylibcugraph, and cuVS; the "
            "canonical GPU implementation cannot start without them"
        ) from _GPU_IMPORT_ERROR


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
    be loop-free, duplicate-free, contain no isolated vertices, and be large
    enough for the requested eigenspace and midpoint neighbourhood.
    """

    _spring_modules = {}

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
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
            raise TypeError("seed must be an integer")
        if not 0 <= int(seed) <= int(np.iinfo(np.uint32).max):
            raise ValueError("seed must be between zero and 2**32 - 1")
        self.seed = int(seed)
        if not isinstance(verbose, (bool, np.bool_)):
            raise TypeError("verbose must be boolean")
        self.verbose = bool(verbose)
        self.logger = logger_instance if logger_instance is not None else LOGGER

        device_edges, vertex_count = self._canonical_graph(
            adjacency=adjacency,
            edges=edges,
            n_vertices=n_vertices,
        )
        self.edges = device_edges
        self.n = vertex_count
        self.n_edges = int(device_edges.shape[0])
        if self.n < self.n_components + 4:
            raise ValueError("graph is too small for the requested spectral embedding")
        if self.n_edges <= self.n_neighbors:
            raise ValueError("n_neighbors must be smaller than the edge count")
        if requested_sample_size > self.n_edges:
            raise ValueError("sample_size cannot exceed the edge count")
        self.sample_size = requested_sample_size
        self.sampled_edge_ids = self._fixed_query_edge_ids()

        self._adjacency = self._device_adjacency()
        self.degrees = cp.asarray(self._adjacency.sum(axis=1)).reshape(-1)
        if bool(cp.any(self.degrees <= 0).item()):
            raise ValueError("the canonical graph must not contain isolated vertices")
        component_count = int(
            cpx_csgraph.connected_components(
                self._adjacency,
                directed=False,
                return_labels=False,
            )
        )
        if component_count != 1:
            raise ValueError("the canonical graph must be connected")

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
            raise ValueError("vertex count exceeds the pinned connectivity implementation")

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
        return cpx_sparse.csr_matrix(
            (values, (rows, columns)),
            shape=(self.n, self.n),
            dtype=cp.float32,
        )

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

    def _spectral_initialization(self):
        degree_inverse_sqrt = cp.reciprocal(cp.sqrt(self.degrees)).astype(cp.float32)
        normalized = self._adjacency.multiply(degree_inverse_sqrt[:, None])
        normalized = normalized.multiply(degree_inverse_sqrt[None, :])
        laplacian = cpx_sparse.eye(self.n, dtype=cp.float32, format="csr") - normalized
        vertex_ids = cp.arange(self.n, dtype=cp.float32)
        phase = np.float32((self.seed + 1) * 0.6180339887498949)
        initial_vector = cp.sin(vertex_ids + phase) + cp.cos(
            vertex_ids * np.float32(0.5) + phase
        )
        initial_vector /= cp.linalg.norm(initial_vector)
        eigen_count = self.n_components + 1
        krylov_dimension = min(self.n - 1, max(2 * eigen_count + 1, 20))
        if not eigen_count + 1 < krylov_dimension < self.n:
            raise ValueError("graph dimensions do not admit a valid eigensolver subspace")
        eigenvalues, eigenvectors = cpx_linalg.eigsh(
            laplacian,
            k=eigen_count,
            which="SA",
            v0=initial_vector,
            ncv=krylov_dimension,
            tol=1.0e-6,
        )
        order = cp.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        positions = cp.ascontiguousarray(
            eigenvectors[:, 1 : self.n_components + 1], dtype=cp.float32
        )
        if positions.shape != (self.n, self.n_components):
            raise RuntimeError("spectral solver returned an unexpected shape")
        if not bool(cp.all(cp.isfinite(eigenvalues)).item()) or not bool(
            cp.all(cp.isfinite(positions)).item()
        ):
            raise FloatingPointError("spectral solver returned non-finite values")
        residual = laplacian @ eigenvectors - eigenvectors * eigenvalues[None, :]
        denominator = cp.maximum(cp.linalg.norm(eigenvectors, axis=0), EPSILON)
        relative_residual = cp.linalg.norm(residual, axis=0) / denominator
        if bool(cp.any(relative_residual > np.float32(5.0e-4)).item()):
            raise RuntimeError("spectral solver residual exceeds the accepted bound")
        return positions

    @staticmethod
    def _search_result_arrays(result):
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("cuVS brute-force search must return two arrays")
        distances, neighbors = result
        distances = cp.asarray(distances)
        neighbors = cp.asarray(neighbors)
        if distances.dtype.kind not in "fc" or neighbors.dtype.kind not in "iu":
            raise TypeError("unexpected cuVS brute-force result dtypes")
        if distances.shape != neighbors.shape:
            raise ValueError("cuVS distances and neighbor IDs must share a shape")
        if not bool(cp.all(cp.isfinite(distances)).item()) or bool(
            cp.any(distances < 0).item()
        ):
            raise FloatingPointError("cuVS returned invalid midpoint distances")
        if distances.shape[1] > 1 and bool(
            cp.any(distances[:, 1:] < distances[:, :-1]).item()
        ):
            raise ValueError("cuVS midpoint distances are not sorted")
        return distances, neighbors

    @staticmethod
    def _compact_nonself_neighbors(neighbors, distances, query_edge_ids, count):
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
        if bool(cp.any(usable_counts < count).item()):
            raise RuntimeError("cuVS returned fewer than the required non-self neighbors")
        source_columns = cp.broadcast_to(
            cp.arange(neighbors.shape[1], dtype=cp.int32), neighbors.shape
        )
        ranked_columns = cp.where(usable, source_columns, neighbors.shape[1])
        order = cp.argsort(ranked_columns, axis=1)
        compacted = cp.take_along_axis(neighbors, order, axis=1)[:, :count]
        ordered_distances = cp.take_along_axis(distances, order, axis=1)
        if bool(cp.any(compacted == query_edge_ids[:, None]).item()):
            raise RuntimeError("self neighbor survived identity-based removal")
        if distances.shape[1] > count:
            has_competitor = usable_counts > count
            tied_cutoff = ordered_distances[:, count] == ordered_distances[:, count - 1]
            if bool(cp.any(has_competitor & tied_cutoff).item()):
                raise RuntimeError(
                    "midpoint-neighbor membership is ambiguous at the distance cutoff"
                )
        return compacted

    def _midpoint_neighbors(self):
        midpoints = cp.ascontiguousarray(
            np.float32(0.5)
            * (self.positions[self.edges[:, 0]] + self.positions[self.edges[:, 1]]),
            dtype=cp.float32,
        )
        queries = cp.ascontiguousarray(midpoints[self.sampled_edge_ids])
        index = brute_force.build(midpoints, metric="sqeuclidean")
        search_width = min(self.n_edges, self.n_neighbors + 2)
        result = brute_force.search(index, queries, search_width)
        distances, neighbors = self._search_result_arrays(result)
        expected_shape = (self.sample_size, search_width)
        if neighbors.shape != expected_shape:
            raise ValueError(
                f"cuVS returned neighbor shape {neighbors.shape}, expected {expected_shape}"
            )
        if bool(cp.any(neighbors < 0).item()) or bool(
            cp.any(neighbors >= self.n_edges).item()
        ):
            raise ValueError("cuVS returned an ID outside the global edge namespace")
        return self._compact_nonself_neighbors(
            neighbors,
            distances,
            self.sampled_edge_ids,
            self.n_neighbors,
        )

    def _spring_forces(self):
        forces = cp.zeros_like(self.positions)
        device_id = cp.cuda.Device().id
        module = self._spring_modules.get(device_id)
        if module is None:
            module = cp.RawModule(code=_SPRING_KERNEL, options=("--std=c++11",))
            self._spring_modules[device_id] = module
        suffix = "i32" if self.edges.dtype == cp.int32 else "i64"
        kernel = module.get_function(f"graphem_spring_{suffix}")
        threads = 256
        blocks = (self.n_edges + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                self.positions,
                self.edges,
                np.int64(self.n_edges),
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

        cp.add.at(forces, first_edges[:, 0], endpoint_force(p1))
        cp.add.at(forces, first_edges[:, 1], endpoint_force(p2))
        cp.add.at(forces, second_edges[:, 0], endpoint_force(q1))
        cp.add.at(forces, second_edges[:, 1], endpoint_force(q2))
        return forces

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
            "graph": {"vertices": self.n, "edges": self.n_edges},
            "configuration": {
                "n_components": self.n_components,
                "L_min": self.L_min,
                "k_attr": self.k_attr,
                "k_inter": self.k_inter,
                "n_neighbors": self.n_neighbors,
                "sample_size": self.sample_size,
                "seed": self.seed,
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
            "score_orientation": "farthest-radius-first",
        }
