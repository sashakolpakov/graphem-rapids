"""
RAPIDS cuVS-based implementation of GraphEmbedder for large-scale datasets.

This module provides graph embedding using RAPIDS cuVS for efficient
large-scale nearest neighbor computations and GPU-accelerated processing.
"""

import logging
import math
import numbers
import time
import warnings

import numpy as np
import plotly.graph_objects as go
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import laplacian
from tqdm import tqdm

# RAPIDS imports
try:
    import cupy as cp
    from cuvs.neighbors import brute_force, ivf_flat, ivf_pq
    CUVS_AVAILABLE = True
except ImportError:
    CUVS_AVAILABLE = False
    warnings.warn(
        "RAPIDS cuVS not available. This backend requires RAPIDS cuVS installation.",
        ImportWarning
    )

logger = logging.getLogger(__name__)


_SPRING_KERNEL_SOURCE = r"""
extern "C" __global__ void spring_forces_i32(
    const float* positions,
    const int* edges,
    const long long n_edges,
    const int n_components,
    const float preferred_length,
    const float attraction,
    const float direction,
    float* forces)
{
    const long long edge_id =
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (edge_id >= n_edges) return;

    const long long source = static_cast<long long>(edges[2 * edge_id]);
    const long long target = static_cast<long long>(edges[2 * edge_id + 1]);
    const long long source_offset = source * n_components;
    const long long target_offset = target * n_components;

    float squared_distance = 0.0f;
    for (int component = 0; component < n_components; ++component) {
        const float delta =
            positions[target_offset + component] -
            positions[source_offset + component];
        squared_distance += delta * delta;
    }
    const float distance = sqrtf(squared_distance);
    if (distance <= 1.0e-12f) return;
    const float multiplier = direction * attraction *
        (distance - preferred_length) / (distance + 1.0e-6f);

    for (int component = 0; component < n_components; ++component) {
        const float delta =
            positions[target_offset + component] -
            positions[source_offset + component];
        const float force = multiplier * delta;
        atomicAdd(&forces[source_offset + component], force);
        atomicAdd(&forces[target_offset + component], -force);
    }
}

extern "C" __global__ void spring_forces_i64(
    const float* positions,
    const long long* edges,
    const long long n_edges,
    const int n_components,
    const float preferred_length,
    const float attraction,
    const float direction,
    float* forces)
{
    const long long edge_id =
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (edge_id >= n_edges) return;

    const long long source = edges[2 * edge_id];
    const long long target = edges[2 * edge_id + 1];
    const long long source_offset = source * n_components;
    const long long target_offset = target * n_components;

    float squared_distance = 0.0f;
    for (int component = 0; component < n_components; ++component) {
        const float delta =
            positions[target_offset + component] -
            positions[source_offset + component];
        squared_distance += delta * delta;
    }
    const float distance = sqrtf(squared_distance);
    if (distance <= 1.0e-12f) return;
    const float multiplier = direction * attraction *
        (distance - preferred_length) / (distance + 1.0e-6f);

    for (int component = 0; component < n_components; ++component) {
        const float delta =
            positions[target_offset + component] -
            positions[source_offset + component];
        const float force = multiplier * delta;
        atomicAdd(&forces[source_offset + component], force);
        atomicAdd(&forces[target_offset + component], -force);
    }
}

extern "C" __global__ void intersection_scatter_i32(
    const int* first_edges,
    const int* second_edges,
    const float* endpoint_forces,
    const long long n_interactions,
    float* forces)
{
    const long long interaction =
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (interaction >= n_interactions) return;
    const float force_x = endpoint_forces[2 * interaction];
    const float force_y = endpoint_forces[2 * interaction + 1];
    const long long first_source = first_edges[2 * interaction];
    const long long first_target = first_edges[2 * interaction + 1];
    const long long second_source = second_edges[2 * interaction];
    const long long second_target = second_edges[2 * interaction + 1];
    atomicAdd(&forces[2 * first_source], force_x);
    atomicAdd(&forces[2 * first_source + 1], force_y);
    atomicAdd(&forces[2 * first_target], force_x);
    atomicAdd(&forces[2 * first_target + 1], force_y);
    atomicAdd(&forces[2 * second_source], -force_x);
    atomicAdd(&forces[2 * second_source + 1], -force_y);
    atomicAdd(&forces[2 * second_target], -force_x);
    atomicAdd(&forces[2 * second_target + 1], -force_y);
}

extern "C" __global__ void intersection_scatter_i64(
    const long long* first_edges,
    const long long* second_edges,
    const float* endpoint_forces,
    const long long n_interactions,
    float* forces)
{
    const long long interaction =
        static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (interaction >= n_interactions) return;
    const float force_x = endpoint_forces[2 * interaction];
    const float force_y = endpoint_forces[2 * interaction + 1];
    const long long first_source = first_edges[2 * interaction];
    const long long first_target = first_edges[2 * interaction + 1];
    const long long second_source = second_edges[2 * interaction];
    const long long second_target = second_edges[2 * interaction + 1];
    atomicAdd(&forces[2 * first_source], force_x);
    atomicAdd(&forces[2 * first_source + 1], force_y);
    atomicAdd(&forces[2 * first_target], force_x);
    atomicAdd(&forces[2 * first_target + 1], force_y);
    atomicAdd(&forces[2 * second_source], -force_x);
    atomicAdd(&forces[2 * second_source + 1], -force_y);
    atomicAdd(&forces[2 * second_target], -force_x);
    atomicAdd(&forces[2 * second_target + 1], -force_y);
}
"""


class GraphEmbedderCuVS:  # pylint: disable=too-many-instance-attributes
    """
    RAPIDS cuVS-based graph embedder for large-scale datasets.

    This class provides graph embedding using RAPIDS cuVS for efficient
    large-scale KNN computations and GPU-accelerated force computations.
    Optimized for datasets with >100K vertices.

    Attributes
    ----------
    adjacency : scipy.sparse.csr_matrix
        Sparse adjacency matrix (n_vertices × n_vertices).
    edges : cupy.ndarray
        Edge list extracted from adjacency matrix as (n_edges, 2) array.
    n : int
        Number of vertices in the graph.
    n_components : int
        Number of components (dimensions) in the embedding space.
    positions : cupy.ndarray
        Current vertex positions as (n_vertices, n_components) array.
    """

    _spring_modules = {}
    _MAX_EXACT_FALLBACK_VECTORS = 100_000
    _MAX_EXACT_FALLBACK_PAIRS = 50_000_000
    _DEFAULT_MAX_CANDIDATE_PAIRS = 8_388_608
    _INITIAL_EDGE_SAMPLE = 65_536

    # pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-statements
    def __init__(
        self,
        adjacency=None,
        n_components=2,
        L_min=1.0,
        k_attr=0.2,
        k_inter=0.5,
        n_neighbors=10,
        sample_size=None,
        batch_size=None,
        index_type='auto',
        dtype=np.float32,
        verbose=True,
        logger_instance=None,
        seed=None,
        initialization='auto',
        spectral_max_vertices=50_000,
        spectral_iterations=24,
        force_mode='legacy',
        learning_rate=0.1,
        max_displacement=1.0,
        intersection_interval=None,
        edge_chunk_size=None,
        profile=False,
        edges=None,
        n_vertices=None,
        assume_canonical_edges=False,
        midpoint_reference_size=None,
        full_midpoint_index=False,
        max_candidate_pairs=_DEFAULT_MAX_CANDIDATE_PAIRS,
        ivf_n_probes=8,
    ):
        """
        Initialize the cuVS GraphEmbedder.

        Parameters
        ----------
        adjacency : array-like or scipy.sparse matrix
            Adjacency matrix (n_vertices × n_vertices). Can be sparse or dense.
            For unweighted graphs, should contain 1s for edges, 0s otherwise.
        n_components : int, default=2
            Number of components (dimensions) in the embedding.
        L_min : float, default=1.0
            Minimum spring length.
        k_attr : float, default=0.2
            Attraction force constant.
        k_inter : float, default=0.5
            Intersection repulsion force constant.
        n_neighbors : int, default=10
            Number of nearest neighbors for intersection detection.
        sample_size : int, optional
            Sampled midpoint queries. ``None`` scales the sample with edge count.
        midpoint_reference_size : int, optional
            Number of edge midpoints indexed as candidate references. ``None``
            uses a bounded automatic sample; it never indexes every edge solely
            because the graph is large.
        full_midpoint_index : bool, default=False
            Opt in to indexing all edge midpoints for maximum candidate quality.
            This has O(E) build memory and time on every index refresh.
        max_candidate_pairs : int, default=8,388,608
            Fail-fast bound for the sampled-query by neighbor result arrays.
        ivf_n_probes : int, default=8
            Requested number of inverted lists searched by IVF-Flat or IVF-PQ.
            The actual value is capped at the number of lists in the index.
        batch_size : int, optional
            Batch size for processing. If None, automatically selects based on available memory.
            Can be manually set (e.g., batch_size=1024) for custom memory management.
        index_type : str, default='auto'
            cuVS index type ('brute_force', 'ivf_flat', 'ivf_pq', 'auto').
        dtype : numpy.dtype, default=np.float32
            Data type for computations. The scalable backend requires float32.
        verbose : bool, default=True
            Enable verbose logging.
        logger_instance : logging.Logger, optional
            Custom logger instance.
        seed : int, optional
            Random seed for reproducibility. If provided, sets numpy and cupy seeds.
        """
        boolean_parameters = {
            "assume_canonical_edges": assume_canonical_edges,
            "full_midpoint_index": full_midpoint_index,
        }
        for parameter_name, value in boolean_parameters.items():
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{parameter_name} must be boolean")
        assume_canonical_edges = bool(assume_canonical_edges)
        full_midpoint_index = bool(full_midpoint_index)

        if seed is not None and (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, numbers.Integral)
            or not 0 <= int(seed) <= int(np.iinfo(np.uint32).max)
        ):
            raise ValueError("seed must be an integer between zero and 2**32 - 1")

        # Set random seeds for reproducibility if provided
        if seed is not None:
            np.random.seed(int(seed))
            if CUVS_AVAILABLE:
                cp.random.seed(int(seed))

        if not CUVS_AVAILABLE:
            raise ImportError(
                "RAPIDS cuVS is not available. Please install RAPIDS cuVS or use PyTorch backend."
            )

        # Setup logging
        if logger_instance is not None:
            self.logger = logger_instance
        else:
            self.logger = logger
            if verbose:
                logging.basicConfig(level=logging.INFO)

        integer_parameters = {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "spectral_iterations": spectral_iterations,
            "max_candidate_pairs": max_candidate_pairs,
            "ivf_n_probes": ivf_n_probes,
        }
        for parameter_name, value in integer_parameters.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, numbers.Integral
            ):
                raise ValueError(f"{parameter_name} must be an integer")
        for parameter_name, value in (
            ("sample_size", sample_size),
            ("batch_size", batch_size),
            ("intersection_interval", intersection_interval),
            ("edge_chunk_size", edge_chunk_size),
            ("midpoint_reference_size", midpoint_reference_size),
        ):
            if value is not None and (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, numbers.Integral)
            ):
                raise ValueError(f"{parameter_name} must be an integer or None")
        if not all(
            np.isfinite(float(value))
            for value in (L_min, k_attr, k_inter, learning_rate)
        ) or (max_displacement is not None and not np.isfinite(max_displacement)):
            raise ValueError("force and update parameters must be finite")
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if k_attr < 0 or k_inter < 0:
            raise ValueError("force constants must be non-negative")
        if L_min < 0:
            raise ValueError("L_min must be non-negative")
        if n_neighbors < 0:
            raise ValueError("n_neighbors must be non-negative")
        if force_mode not in ('legacy', 'attractive'):
            raise ValueError("force_mode must be 'legacy' or 'attractive'")
        if initialization not in ('auto', 'spectral', 'randomized'):
            raise ValueError("initialization must be 'auto', 'spectral', or 'randomized'")
        if index_type not in ('auto', 'brute_force', 'ivf_flat', 'ivf_pq'):
            raise ValueError("unknown cuVS index_type")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_displacement is not None and max_displacement <= 0:
            raise ValueError("max_displacement must be positive or None")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive or None")
        if max_candidate_pairs <= 0:
            raise ValueError("max_candidate_pairs must be positive")
        if ivf_n_probes <= 0:
            raise ValueError("ivf_n_probes must be positive")
        if spectral_iterations <= 0:
            raise ValueError("spectral_iterations must be positive")
        if np.dtype(dtype) != np.dtype(np.float32):
            raise ValueError(
                "the scalable cuVS backend requires dtype=float32; "
                "cuVS does not support float64 midpoint indexes"
            )
        if k_inter > 0 and n_components != 2:
            raise ValueError(
                "intersection forces are defined only for 2D layouts; "
                "set n_components=2 or k_inter=0"
            )

        if (adjacency is None) == (edges is None):
            raise ValueError("provide exactly one of adjacency or edges")
        if edges is None:
            adjacency = self._validate_adjacency(adjacency)
            self.adjacency = adjacency
            self.n = adjacency.shape[0]
            upper = sp.triu(adjacency, k=1, format='coo')
            host_edges = np.column_stack((upper.row, upper.col))
            edge_dtype = cp.int32 if self.n < np.iinfo(np.int32).max else cp.int64
            device_edges = cp.asarray(host_edges, dtype=edge_dtype)
        else:
            self.adjacency = None
            if isinstance(edges, tuple) and len(edges) == 2:
                device_edges = cp.column_stack((cp.asarray(edges[0]), cp.asarray(edges[1])))
            else:
                device_edges = cp.asarray(edges)
            if device_edges.ndim != 2 or device_edges.shape[1] != 2:
                raise ValueError("edges must have shape (n_edges, 2)")
            if n_vertices is None:
                if device_edges.shape[0] == 0:
                    raise ValueError("n_vertices is required for an empty edge list")
                n_vertices = int(cp.max(device_edges).item()) + 1
            if isinstance(n_vertices, (bool, np.bool_)) or not isinstance(
                n_vertices, numbers.Integral
            ):
                raise ValueError("n_vertices must be an integer")
            if n_vertices <= 0:
                raise ValueError("n_vertices must be positive")
            self.n = int(n_vertices)
            if device_edges.dtype.kind not in "iu":
                raise TypeError("edge endpoints must have an integer dtype")
            edge_dtype = cp.int32 if self.n < np.iinfo(np.int32).max else cp.int64
            device_edges = device_edges.astype(edge_dtype, copy=False)
            if bool(cp.any(device_edges < 0)) or bool(cp.any(device_edges >= self.n)):
                raise ValueError("edge endpoint is outside the graph")
            device_edges = device_edges[device_edges[:, 0] != device_edges[:, 1]]
            if not assume_canonical_edges and device_edges.shape[0]:
                source = cp.minimum(device_edges[:, 0], device_edges[:, 1])
                target = cp.maximum(device_edges[:, 0], device_edges[:, 1])
                device_edges = cp.column_stack((source, target))
                order = cp.lexsort((device_edges[:, 1], device_edges[:, 0]))
                device_edges = device_edges[order]
                unique = cp.ones(device_edges.shape[0], dtype=cp.bool_)
                unique[1:] = cp.any(device_edges[1:] != device_edges[:-1], axis=1)
                device_edges = device_edges[unique]

        if n_components > self.n:
            raise ValueError("n_components cannot exceed the number of vertices")

        # Store parameters
        self.n_components = n_components
        self.dtype = np.float32
        self.L_min = L_min
        self.k_attr = k_attr
        self.k_inter = k_inter
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size  # None for automatic, or user-defined value
        self.initialization = initialization
        self.spectral_max_vertices = int(spectral_max_vertices)
        self.spectral_iterations = int(spectral_iterations)
        self.force_mode = force_mode
        self.learning_rate = float(learning_rate)
        self.max_displacement = max_displacement
        self.profile = bool(profile)
        self.timings = {
            'initialization': 0.0,
            'spring': 0.0,
            'midpoint_knn': 0.0,
            'intersections': 0.0,
            'normalization': 0.0,
        }
        self._iteration = 0
        self._knn_fallbacks = 0
        self._knn_last_error = None
        self._cached_knn_indices = None
        self._cached_sampled_indices = None
        self._initial_edge_length_before_scale = 0.0
        self._initial_edge_length_after_scale = 0.0
        self._normalization_radius = 1.0

        # Calculate number of edges for sample size validation
        self.edges = cp.ascontiguousarray(device_edges)
        self.n_edges = int(self.edges.shape[0])
        if self.n_edges:
            self.degrees = cp.bincount(
                self.edges.reshape(-1), minlength=self.n
            ).astype(cp.float32, copy=False)
        else:
            self.degrees = cp.zeros(self.n, dtype=cp.float32)
        self._active_mask = self.degrees > 0
        self.n_active_vertices = int(cp.count_nonzero(self._active_mask).item())
        if sample_size is None:
            # Retain a linear-in-active-graph component through the target
            # 10M/100M regime, while bounding candidate-pair memory.
            square_root_target = int(8 * math.sqrt(max(self.n_edges, 1)))
            linear_target = max(
                (self.n_active_vertices + 63) // 64,
                (self.n_edges + 511) // 512,
            )
            sample_size = min(
                262_144,
                max(4_096, square_root_target, linear_target),
            )
        if sample_size < 0:
            raise ValueError("sample_size must be non-negative or None")
        self.sample_size = min(int(sample_size), self.n_edges)
        self.intersection_inclusion_scale = (
            float(self.n_edges) / float(self.sample_size)
            if self.sample_size > 0 else 0.0
        )

        self.full_midpoint_index = full_midpoint_index
        if self.full_midpoint_index and midpoint_reference_size is not None:
            raise ValueError(
                "midpoint_reference_size and full_midpoint_index are mutually exclusive"
            )
        if self.full_midpoint_index:
            reference_size = self.n_edges
            self._midpoint_reference_mode = 'full'
        elif midpoint_reference_size is None:
            reference_size = min(
                self.n_edges,
                2_000_000,
                max(262_144, 4 * self.sample_size, self.n_neighbors + 1),
            )
            self._midpoint_reference_mode = 'bounded_auto'
        else:
            reference_size = int(midpoint_reference_size)
            if reference_size < 0:
                raise ValueError("midpoint_reference_size cannot be negative")
            if reference_size < self.sample_size:
                raise ValueError(
                    "midpoint_reference_size must be at least sample_size"
                )
            if reference_size <= self.n_neighbors and self.n_edges > self.n_neighbors:
                raise ValueError(
                    "midpoint_reference_size must exceed n_neighbors"
                )
            reference_size = min(self.n_edges, max(reference_size, 0))
            self._midpoint_reference_mode = 'bounded_user'
        self.midpoint_reference_size = int(reference_size)
        effective_neighbors = min(
            self.n_neighbors,
            max(self.midpoint_reference_size - 1, 0),
            max(self.n_edges - 1, 0),
        )
        search_pairs = self.sample_size * (effective_neighbors + 1)
        self.max_candidate_pairs = int(max_candidate_pairs)
        if search_pairs > self.max_candidate_pairs:
            raise MemoryError(
                "midpoint search would materialize "
                f"{search_pairs:,} query-neighbor entries; reduce sample_size or "
                f"n_neighbors (limit {self.max_candidate_pairs:,})"
            )
        self._last_midpoint_reference_size = 0
        self._last_query_sample_size = 0
        self._last_n_probes = None

        self.index_type = index_type
        self.ivf_n_probes = int(ivf_n_probes)
        self.verbose = verbose

        # Memory management for large datasets
        # Use user-defined batch_size if provided, otherwise calculate optimal one automatically
        if self.batch_size is None:
            self.batch_size = min(max(self.sample_size, 1), 16_384)
            if self.verbose:
                self.logger.info("Using automatic batch size: %d", self.batch_size)
        else:
            if self.verbose:
                self.logger.info("Using user-defined batch size: %d", self.batch_size)

        if self.verbose:
            self.logger.info("Initialized GraphEmbedderCuVS")
            self.logger.info("Graph: %d vertices, %d edges, %dD", self.n, len(self.edges), self.n_components)
            self.logger.info("Index type: %s", self.index_type)

        if edge_chunk_size is None:
            edge_chunk_size = min(max(self.n_edges, 1), 2_000_000)
        if edge_chunk_size <= 0:
            raise ValueError("edge_chunk_size must be positive")
        self.edge_chunk_size = int(edge_chunk_size)
        if intersection_interval is None:
            intersection_interval = 5 if self.n_edges >= 1_000_000 else 1
        if intersection_interval <= 0:
            raise ValueError("intersection_interval must be positive")
        self.intersection_interval = int(intersection_interval)

        # Compute initial embedding
        self.positions = self._compute_laplacian_embedding()

        # The index must be built over current edge midpoints, not vertices.
        self.knn_index = None
        self._knn_kind = None
        self._knn_search_params = None

    def _validate_adjacency(self, adjacency):
        """
        Validate and convert adjacency matrix to scipy sparse format.

        Parameters
        ----------
        adjacency : array-like or scipy.sparse matrix
            Input adjacency matrix

        Returns
        -------
        scipy.sparse.csr_matrix
            Validated adjacency matrix in CSR format
        """
        # Handle scipy sparse matrices first
        if sp.issparse(adjacency):
            adjacency = adjacency.tocsr()  # Ensure CSR format
        elif isinstance(adjacency, np.ndarray):
            # Already a numpy array
            pass
        else:
            # Try to convert to numpy array
            adjacency = np.asarray(adjacency)

        # Check if square
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got shape {adjacency.shape}")
        if adjacency.shape[0] == 0:
            raise ValueError("adjacency matrix cannot be empty")

        # Convert to sparse format if needed
        if not sp.issparse(adjacency):
            adjacency = sp.csr_matrix(adjacency)

        adjacency.sum_duplicates()
        adjacency.eliminate_zeros()
        return adjacency

    def _compute_laplacian_embedding(self):
        """Compute an exact small-graph or randomized GPU spectral layout."""
        start = self._stage_start()
        method = self.initialization
        if method == 'auto':
            method = (
                'spectral'
                if self.adjacency is not None and self.n <= self.spectral_max_vertices
                else 'randomized'
            )
        if method == 'spectral' and self.adjacency is None:
            raise ValueError("spectral initialization requires an adjacency matrix")
        self.logger.info("Computing %s initialization", method)

        if self.n == 1 or self.n_active_vertices == 0:
            positions = cp.zeros((1, self.n_components), dtype=self.dtype)
            if self.n != 1:
                positions = cp.zeros((self.n, self.n_components), dtype=self.dtype)
        elif method == 'spectral':
            positions = self._compute_scipy_spectral_embedding()
        else:
            positions = self._compute_randomized_gpu_embedding()

        positions = self._prepare_initial_positions(positions)
        self._stage_end('initialization', start)
        return positions

    def _center_active_positions(self, positions):
        """Center active vertices and keep degree-zero vertices at the origin."""
        positions = positions.astype(cp.float32, copy=False)
        if self.n_active_vertices == 0:
            return cp.zeros_like(positions)
        positions[~self._active_mask] = 0.0
        active_mean = cp.sum(positions, axis=0, keepdims=True) / np.float32(
            self.n_active_vertices
        )
        positions -= active_mean
        positions[~self._active_mask] = 0.0
        return positions

    def _active_rms_radius(self, positions):
        """Return a rotationally invariant RMS radius over active vertices."""
        if self.n_active_vertices == 0:
            return 0.0
        squared_radius = cp.sum(positions * positions) / np.float32(
            self.n_active_vertices
        )
        return float(cp.sqrt(cp.maximum(squared_radius, 0.0)).item())

    def _sample_edge_lengths(self, positions):
        """Sample edge lengths without allocating a permutation of all edges."""
        if self.n_edges == 0:
            return cp.empty(0, dtype=cp.float32)
        sample_count = min(self.n_edges, self._INITIAL_EDGE_SAMPLE)
        if sample_count == self.n_edges:
            sampled_edges = self.edges
        else:
            sampled_ids = cp.random.randint(
                0,
                self.n_edges,
                size=sample_count,
                dtype=cp.int64,
            )
            sampled_edges = self.edges[sampled_ids]
        differences = (
            positions[sampled_edges[:, 1]] - positions[sampled_edges[:, 0]]
        )
        return cp.linalg.norm(differences, axis=1)

    def _prepare_initial_positions(self, positions):
        """Mask isolates and put the initial active-edge scale near ``L_min``."""
        positions = self._center_active_positions(positions)
        self._initial_edge_length_before_scale = 0.0
        self._initial_edge_length_after_scale = 0.0

        lengths = self._sample_edge_lengths(positions)
        if lengths.size:
            positive = lengths[lengths > 1.0e-12]
            if positive.size:
                median_length = float(cp.median(positive).item())
                self._initial_edge_length_before_scale = median_length
                if self.L_min > 0 and np.isfinite(median_length):
                    positions *= np.float32(self.L_min / median_length)

        radius = self._active_rms_radius(positions)
        if radius <= 1.0e-12 or not np.isfinite(radius):
            # This is reachable only for a degenerate initialization. Keep a
            # finite invariant instead of allowing the first update to divide
            # by a graph-size-dependent zero scale.
            radius = 1.0
        self._normalization_radius = radius

        scaled_lengths = self._sample_edge_lengths(positions)
        if scaled_lengths.size:
            positive = scaled_lengths[scaled_lengths > 1.0e-12]
            if positive.size:
                self._initial_edge_length_after_scale = float(
                    cp.median(positive).item()
                )
        positions[~self._active_mask] = 0.0
        return cp.ascontiguousarray(positions, dtype=cp.float32)

    def _compute_scipy_spectral_embedding(self):
        """Small-graph reference eigensolve on the host."""
        upper = sp.triu(self.adjacency, k=1, format='csr')
        adjacency = upper + upper.transpose()
        adjacency.data = np.ones_like(adjacency.data, dtype=np.float32)
        k = min(self.n_components + 1, self.n - 1)
        if k <= self.n_components:
            return cp.asarray(
                np.random.standard_normal((self.n, self.n_components)),
                dtype=self.dtype,
            )
        try:
            normalized_laplacian = laplacian(adjacency, normed=True)
            _, eigenvectors = spla.eigsh(
                normalized_laplacian,
                k=k,
                which='SM',
                tol=1e-3,
            )
            embedding = eigenvectors[:, 1:self.n_components + 1]
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.warning("Host eigendecomposition failed: %s", exc)
            embedding = np.random.standard_normal((self.n, self.n_components))
        return cp.asarray(embedding, dtype=self.dtype)

    def _compute_randomized_gpu_embedding(self):
        """Approximate low-frequency Laplacian modes with sparse subspace iteration.

        The shifted normalized adjacency ``(I + D^-1/2 A D^-1/2) / 2`` has
        spectrum in ``[0, 1]``. Repeated sparse multiplies therefore retain the
        desired low-frequency Laplacian subspace without a CPU eigensolve. The
        trivial degree vector is projected out after every multiply.
        """
        try:
            import cupyx.scipy.sparse as cpx_sparse  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - requires CUDA environment
            raise ImportError("randomized initialization requires cupyx.scipy.sparse") from exc

        source = self.edges[:, 0]
        target = self.edges[:, 1]
        rows = cp.concatenate((source, target))
        cols = cp.concatenate((target, source))
        values = cp.ones(rows.shape[0], dtype=self.dtype)
        adjacency = cpx_sparse.csr_matrix(
            (values, (rows, cols)), shape=(self.n, self.n), dtype=self.dtype
        )
        del rows, cols, values
        degrees = self.degrees
        inverse_sqrt_degree = cp.where(degrees > 0, 1.0 / cp.sqrt(degrees), 0.0)
        trivial = cp.sqrt(degrees).reshape(-1, 1)
        trivial_norm = cp.sum(trivial * trivial)

        basis = cp.random.standard_normal(
            (self.n, self.n_components), dtype=self.dtype
        )
        basis[~self._active_mask] = 0.0
        for _ in range(max(self.spectral_iterations, 1)):
            normalized = inverse_sqrt_degree[:, None] * basis
            normalized = adjacency @ normalized
            normalized = inverse_sqrt_degree[:, None] * normalized
            basis = 0.5 * (basis + normalized)
            basis[~self._active_mask] = 0.0
            if self.n_edges:
                basis -= trivial @ ((trivial.T @ basis) / trivial_norm)
            basis, _ = cp.linalg.qr(basis, mode='reduced')
            basis[~self._active_mask] = 0.0
        return basis.astype(self.dtype, copy=False)

    def _stage_start(self):
        if not self.profile:
            return None
        cp.cuda.get_current_stream().synchronize()
        return time.perf_counter()

    def _stage_end(self, name, started):
        if started is None:
            return
        cp.cuda.get_current_stream().synchronize()
        self.timings[name] += time.perf_counter() - started

    def _select_index_type(self, n_vectors=None):
        """Select an index for the edge-midpoint dataset."""
        if self.index_type != 'auto':
            return self.index_type
        n_vectors = self.n_edges if n_vectors is None else n_vectors
        # Product quantization has no useful compression regime in GraphEm's
        # usual 2--6 dimensions. IVF-Flat avoids an invalid/slow low-d PQ setup.
        return 'ivf_flat' if n_vectors >= 100_000 else 'brute_force'

    def _build_knn_index(self, midpoints):
        """Build a cuVS index whose row ids are edge ids."""
        index_type = self._select_index_type(midpoints.shape[0])
        if index_type == 'ivf_pq' and self.n_components < 8:
            self.logger.warning("IVF-PQ is unsuitable below 8D; using IVF-Flat")
            index_type = 'ivf_flat'
        self._knn_kind = index_type
        self._knn_search_params = None
        self._last_n_probes = None

        if index_type == 'brute_force':
            self.knn_index = brute_force.build(midpoints, metric='sqeuclidean')
            return

        n_lists = max(2, min(int(math.sqrt(midpoints.shape[0])), 16_384))
        actual_n_probes = min(self.ivf_n_probes, n_lists)
        if index_type == 'ivf_flat':
            params = ivf_flat.IndexParams(
                n_lists=n_lists,
                metric='sqeuclidean',
                kmeans_n_iters=10,
                kmeans_trainset_fraction=min(
                    0.5, max(0.01, 200_000 / max(midpoints.shape[0], 1))
                ),
                # The midpoint index is immutable and discarded after one
                # search. Dynamic-list over-allocation only wastes memory.
                conservative_memory_allocation=True,
            )
            self.knn_index = ivf_flat.build(params, midpoints)
            self._knn_search_params = ivf_flat.SearchParams(
                n_probes=actual_n_probes
            )
            self._last_n_probes = actual_n_probes
            return

        params = ivf_pq.IndexParams(
            n_lists=n_lists,
            metric='sqeuclidean',
            pq_dim=min(64, self.n_components),
            pq_bits=8,
            conservative_memory_allocation=True,
        )
        self.knn_index = ivf_pq.build(params, midpoints)
        self._last_n_probes = actual_n_probes
        self._knn_search_params = ivf_pq.SearchParams(
            n_probes=self._last_n_probes
        )

    @staticmethod
    def _neighbors_from_search_result(result):
        """Handle stable and legacy cuVS tuple ordering by output dtype."""
        first, second = result
        first = cp.asarray(first)
        second = cp.asarray(second)
        if first.dtype.kind in 'iu' and second.dtype.kind not in 'iu':
            return first
        if second.dtype.kind in 'iu':
            return second
        raise TypeError("cuVS search returned no integer neighbor array")

    def _search_knn_index(self, queries, k):
        if self._knn_kind == 'brute_force':
            return self._neighbors_from_search_result(
                brute_force.search(self.knn_index, queries, k)
            )
        module = ivf_flat if self._knn_kind == 'ivf_flat' else ivf_pq
        if self._knn_search_params is not None:
            result = module.search(
                self._knn_search_params, self.knn_index, queries, k
            )
        else:
            result = module.search(self.knn_index, queries, k)
        return self._neighbors_from_search_result(result)

    def _sample_edge_ids(self, count):
        """Return an O(count)-memory uniform systematic sample of edge ids."""
        count = min(max(int(count), 0), self.n_edges)
        if count == 0:
            return cp.empty(0, dtype=cp.int64)
        if count == self.n_edges:
            return cp.arange(self.n_edges, dtype=cp.int64)

        # A random start and a stride coprime to E define a permutation of all
        # edge ids. Taking its prefix gives every edge the same S/E inclusion
        # probability without constructing/shuffling an E-element permutation.
        start = int(np.random.randint(0, self.n_edges))
        stride = int(np.random.randint(1, self.n_edges))
        while math.gcd(stride, self.n_edges) != 1:
            stride = int(np.random.randint(1, self.n_edges))
        offsets = cp.arange(count, dtype=cp.int64)
        return (np.int64(start) + offsets * np.int64(stride)) % self.n_edges

    def _midpoints_for_edges(self, edge_ids, positions):
        selected = self.edges if edge_ids is None else self.edges[edge_ids]
        return cp.ascontiguousarray(
            0.5 * (
                positions[selected[:, 0]] + positions[selected[:, 1]]
            ),
            dtype=cp.float32,
        )

    def _locate_knn_midpoints_cuvs(
        self,
        midpoints,
        k,
        positions=None,
    ):
        """
        Locate k nearest neighbors using cuVS.

        Parameters
        ----------
        midpoints : cupy.ndarray
            Edge midpoints.
        k : int
            Number of nearest neighbors.

        Returns
        -------
        Tuple[cupy.ndarray, cupy.ndarray]
            KNN indices and sampled indices.
        """
        E = self.n_edges
        if midpoints is not None and int(midpoints.shape[0]) != E:
            raise ValueError("midpoints must contain one row per edge")
        if midpoints is None and positions is None:
            raise ValueError("positions are required when midpoints are not supplied")
        if E <= 1 or k <= 0 or self.sample_size == 0:
            return cp.empty((0, 0), dtype=cp.int64), cp.empty(0, dtype=cp.int64)
        sample_size = min(self.sample_size, E)
        reference_target = min(self.midpoint_reference_size, E)
        reference_is_full = reference_target == E
        if reference_is_full:
            reference_indices = None
            sampled_indices = self._sample_edge_ids(sample_size)
        else:
            # Draw one exact-size reference permutation and use its prefix as
            # queries. This makes every query a valid reference row and gives the
            # E/S scale a clear directed-query sampling interpretation.
            reference_indices = self._sample_edge_ids(reference_target)
            sampled_indices = reference_indices[:sample_size]

        reference_count = E if reference_is_full else int(reference_indices.shape[0])
        k = min(k, E - 1, max(reference_count - 1, 0))
        if k <= 0:
            return cp.empty((sample_size, 0), dtype=cp.int64), sampled_indices

        if midpoints is None:
            sampled_midpoints = self._midpoints_for_edges(sampled_indices, positions)
            reference = self._midpoints_for_edges(reference_indices, positions)
        else:
            sampled_midpoints = cp.ascontiguousarray(
                midpoints[sampled_indices], dtype=cp.float32
            )
            reference = (
                cp.ascontiguousarray(midpoints, dtype=cp.float32)
                if reference_is_full
                else cp.ascontiguousarray(
                    midpoints[reference_indices], dtype=cp.float32
                )
            )
        self._last_query_sample_size = sample_size
        self._last_midpoint_reference_size = reference_count

        try:
            # Positions change every iteration, so midpoint candidates are
            # refreshed. Auto mode indexes only a bounded stochastic reference;
            # local cuVS rows are mapped back to global edge ids below.
            self._build_knn_index(reference)
            local_neighbors = self._search_knn_index(sampled_midpoints, k + 1)
            self.knn_index = None
            self._knn_search_params = None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.knn_index = None
            self._knn_search_params = None
            self._knn_fallbacks += 1
            self._knn_last_error = repr(exc)
            exact_pairs = int(sampled_midpoints.shape[0]) * int(reference.shape[0])
            if (
                int(reference.shape[0]) > self._MAX_EXACT_FALLBACK_VECTORS
                or exact_pairs > self._MAX_EXACT_FALLBACK_PAIRS
            ):
                raise RuntimeError(
                    "cuVS midpoint search failed and exact fallback is disabled "
                    f"for {int(reference.shape[0]):,} vectors / "
                    f"{exact_pairs:,} query-reference pairs"
                ) from exc
            self.logger.warning(
                "cuVS midpoint search failed; using bounded tiled exact search: %s",
                exc,
            )
            local_neighbors = self._fallback_knn_search(
                sampled_midpoints, reference, k + 1
            )

        valid_local = (
            (local_neighbors >= 0) &
            (local_neighbors < reference_count)
        )
        neighbors = cp.full(local_neighbors.shape, -1, dtype=cp.int64)
        if reference_is_full:
            neighbors[valid_local] = local_neighbors[valid_local]
        else:
            neighbors[valid_local] = reference_indices[local_neighbors[valid_local]]
        usable = (neighbors >= 0) & (neighbors != sampled_indices[:, None])
        order = cp.argsort(cp.logical_not(usable), axis=1)
        neighbors = cp.take_along_axis(neighbors, order, axis=1)
        return neighbors[:, :k], sampled_indices

    def _manual_knn_search(
        self,
        query_points,
        reference_points,
        k
    ):
        """Fallback manual KNN search using CuPy."""
        query_norm = cp.sum(query_points * query_points, axis=1, keepdims=True)
        reference_norm = cp.sum(reference_points * reference_points, axis=1)[None, :]
        distances = query_norm + reference_norm - 2.0 * (query_points @ reference_points.T)
        return cp.argpartition(distances, kth=k - 1, axis=1)[:, :k]

    def _fallback_knn_search(
        self,
        sampled_midpoints,
        midpoints,
        k,
    ):
        """Fallback KNN search with chunked processing."""
        n_samples = sampled_midpoints.shape[0]
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        memory_chunk = max(1, int((free_bytes * 0.15) // max(8 * midpoints.shape[0], 1)))
        chunk_size = min(self.batch_size, memory_chunk, n_samples)

        all_indices = []
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunk = sampled_midpoints[i:end_idx]

            indices = self._manual_knn_search(chunk, midpoints, k)
            all_indices.append(indices)

        return cp.concatenate(all_indices, axis=0)

    def _compute_spring_forces_cuvs(
        self,
        positions,
        edges
    ):
        """
        Compute spring forces using CuPy operations.

        Parameters
        ----------
        positions : cupy.ndarray
            Current vertex positions.
        edges : cupy.ndarray
            Edge list.

        Returns
        -------
        cupy.ndarray
            Spring forces for each vertex.
        """
        forces = cp.zeros_like(positions)
        if edges.shape[0] == 0 or self.k_attr == 0:
            return forces
        if positions.dtype != cp.float32:
            raise TypeError("the fused spring kernel requires float32 positions")
        if positions.shape != (self.n, self.n_components):
            raise ValueError(
                "positions must have shape (n_vertices, n_components)"
            )
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape (n_edges, 2)")
        if edges.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
            raise TypeError("the fused spring kernel requires int32 or int64 edges")
        if not positions.flags.c_contiguous or not edges.flags.c_contiguous:
            raise ValueError("the fused spring kernel requires contiguous arrays")

        device_id = cp.cuda.Device().id
        module = self._spring_modules.get(device_id)
        if module is None:
            module = cp.RawModule(
                code=_SPRING_KERNEL_SOURCE,
                options=('--std=c++11',),
            )
            self._spring_modules[device_id] = module
        kernel_name = (
            'spring_forces_i32' if edges.dtype == np.dtype(np.int32)
            else 'spring_forces_i64'
        )
        kernel = module.get_function(kernel_name)
        threads = 256
        blocks = (int(edges.shape[0]) + threads - 1) // threads
        direction = 1.0 if self.force_mode == 'attractive' else -1.0
        kernel(
            (blocks,),
            (threads,),
            (
                positions,
                edges,
                np.int64(edges.shape[0]),
                np.int32(self.n_components),
                np.float32(self.L_min),
                np.float32(self.k_attr),
                np.float32(direction),
                forces,
            ),
        )
        return forces

    def _compute_intersection_forces_cuvs(
        self,
        positions,
        edges,
        knn_indices,
        sampled_indices
    ):
        """
        Compute intersection forces using CuPy operations.

        Similar to PyTorch version but using CuPy for GPU acceleration.
        """
        if knn_indices.size == 0:
            return cp.zeros_like(positions)
        _, n_neighbors = knn_indices.shape
        candidate_i = cp.repeat(sampled_indices, n_neighbors)
        candidate_j = knn_indices.reshape(-1)
        valid_mask = (
            (candidate_j >= 0) &
            (candidate_j < edges.shape[0]) &
            (candidate_i != candidate_j)
        )
        if not bool(cp.any(valid_mask)):
            return cp.zeros_like(positions)

        # KNN nominations are a directed-query objective: mutual neighbors
        # intentionally contribute twice at every sample size, while E/S is the
        # inverse query-inclusion weight. Canonical pair orientation makes both
        # nominations produce the same physical separation direction.
        first_ids = cp.minimum(
            candidate_i[valid_mask], candidate_j[valid_mask]
        )
        second_ids = cp.maximum(
            candidate_i[valid_mask], candidate_j[valid_mask]
        )
        edges_i = edges[first_ids]
        edges_j = edges[second_ids]
        share_mask = (
            (edges_i[:, 0] == edges_j[:, 0]) |
            (edges_i[:, 0] == edges_j[:, 1]) |
            (edges_i[:, 1] == edges_j[:, 0]) |
            (edges_i[:, 1] == edges_j[:, 1])
        )
        interaction_mask = ~share_mask
        if not bool(cp.any(interaction_mask)):
            return cp.zeros_like(positions)
        edges_i = edges_i[interaction_mask]
        edges_j = edges_j[interaction_mask]

        p1 = positions[edges_i[:, 0]]
        p2 = positions[edges_i[:, 1]]
        q1 = positions[edges_j[:, 0]]
        q2 = positions[edges_j[:, 1]]
        intersect_mask = self._check_line_intersections_cuvs(p1, p2, q1, q2)
        if not bool(cp.any(intersect_mask)):
            return cp.zeros_like(positions)
        edges_i = edges_i[intersect_mask]
        edges_j = edges_j[intersect_mask]
        p1, p2 = p1[intersect_mask], p2[intersect_mask]
        q1, q2 = q1[intersect_mask], q2[intersect_mask]
        # Parameterize the crossing point on both segments. Translate both
        # endpoints of one segment together and the other segment oppositely,
        # along the normal that moves the shallower parameter toward its nearest
        # endpoint. This conserves total force and reduces crossing depth; unlike
        # radial four-endpoint repulsion it also breaks a symmetric X.
        first_direction = p2 - p1
        second_direction = q2 - q1
        relative_origin = q1 - p1

        def cross_2d(first, second):
            return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]

        denominator = cross_2d(first_direction, second_direction)
        first_parameter = cross_2d(relative_origin, second_direction) / denominator
        second_parameter = cross_2d(relative_origin, first_direction) / denominator
        first_depth = cp.minimum(first_parameter, 1.0 - first_parameter)
        second_depth = cp.minimum(second_parameter, 1.0 - second_parameter)
        move_first_parameter = first_depth <= second_depth

        first_escape = cp.stack(
            (-second_direction[:, 1], second_direction[:, 0]), axis=1
        )
        second_escape = cp.stack(
            (-first_direction[:, 1], first_direction[:, 0]), axis=1
        )
        escape = cp.where(
            move_first_parameter[:, None], first_escape, second_escape
        )
        escape /= cp.linalg.norm(escape, axis=1, keepdims=True) + 1.0e-12
        selected_parameter = cp.where(
            move_first_parameter, first_parameter, second_parameter
        )
        selected_depth = cp.where(move_first_parameter, first_depth, second_depth)
        endpoint_direction = cp.where(selected_parameter >= 0.5, 1.0, -1.0)
        orientation_direction = cp.where(denominator >= 0.0, 1.0, -1.0)
        escape *= (endpoint_direction * orientation_direction)[:, None]

        inclusion_scale = np.float32(self.intersection_inclusion_scale)
        segment_force = (
            np.float32(self.k_inter) * inclusion_scale
            * (1.0 + selected_depth)[:, None] * escape
        )
        endpoint_force = 0.5 * segment_force

        return self._scatter_intersection_forces(
            edges_i, edges_j, endpoint_force, positions
        )

    def _scatter_intersection_forces(
        self, first_edges, second_edges, endpoint_force, positions
    ):
        """Accumulate 2D crossing interactions with one fused FP32 kernel."""
        forces = cp.zeros_like(positions)
        interaction_count = int(endpoint_force.shape[0])
        if interaction_count == 0:
            return forces
        if self.n_components != 2 or positions.dtype != cp.float32:
            raise TypeError("intersection scatter requires 2D float32 positions")
        if positions.shape != (self.n, 2) or not positions.flags.c_contiguous:
            raise ValueError("intersection positions must be contiguous (n_vertices, 2)")
        expected_shape = (interaction_count, 2)
        if any(
            array.ndim != 2 or array.shape != expected_shape
            for array in (first_edges, second_edges, endpoint_force)
        ):
            raise ValueError("intersection inputs must have matching (n, 2) shapes")
        first_edges = cp.ascontiguousarray(first_edges)
        second_edges = cp.ascontiguousarray(second_edges)
        endpoint_force = cp.ascontiguousarray(endpoint_force, dtype=cp.float32)
        if first_edges.dtype != second_edges.dtype or first_edges.dtype not in (
            np.dtype(np.int32), np.dtype(np.int64)
        ):
            raise TypeError("intersection edges must share int32 or int64 dtype")

        device_id = cp.cuda.Device().id
        module = self._spring_modules.get(device_id)
        if module is None:
            module = cp.RawModule(
                code=_SPRING_KERNEL_SOURCE,
                options=('--std=c++11',),
            )
            self._spring_modules[device_id] = module
        suffix = "i32" if first_edges.dtype == np.dtype(np.int32) else "i64"
        kernel = module.get_function(f"intersection_scatter_{suffix}")
        threads = 256
        blocks = (interaction_count + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                first_edges,
                second_edges,
                endpoint_force,
                np.int64(interaction_count),
                forces,
            ),
        )
        return forces

    def _check_line_intersections_cuvs(
        self,
        p1,
        p2,
        q1,
        q2
    ):
        """Check line segment intersections using CuPy."""
        def orientation(a, b, c):
            return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - \
                   (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)

        return (o1 * o2 < 0) & (o3 * o4 < 0)

    def update_positions(self):
        """Update vertex positions using cuVS-accelerated computations."""
        started = self._stage_start()
        spring_forces = self._compute_spring_forces_cuvs(self.positions, self.edges)
        self._stage_end('spring', started)

        intersection_forces = cp.zeros_like(self.positions)
        use_intersections = (
            self.k_inter > 0 and
            self.n_components == 2 and
            self.n_edges > 1
        )
        if use_intersections:
            refresh_neighbors = (
                self._cached_knn_indices is None
                or self._iteration % self.intersection_interval == 0
            )
            if refresh_neighbors:
                started = self._stage_start()
                self._cached_knn_indices, self._cached_sampled_indices = (
                    self._locate_knn_midpoints_cuvs(
                        None,
                        self.n_neighbors,
                        positions=self.positions,
                    )
                )
                self._stage_end('midpoint_knn', started)
            started = self._stage_start()
            intersection_forces = self._compute_intersection_forces_cuvs(
                self.positions,
                self.edges,
                self._cached_knn_indices,
                self._cached_sampled_indices,
            )
            self._stage_end('intersections', started)

        total_forces = spring_forces + intersection_forces
        total_forces[~self._active_mask] = 0.0
        if self.max_displacement is not None:
            norms = cp.linalg.norm(total_forces, axis=1, keepdims=True) + 1e-12
            total_forces *= cp.minimum(1.0, self.max_displacement / norms)

        started = self._stage_start()
        new_positions = self.positions + self.learning_rate * total_forces
        new_positions[~self._active_mask] = 0.0
        new_positions = self._center_active_positions(new_positions)
        if self.n_active_vertices:
            squared_radius = cp.sum(new_positions * new_positions) / np.float32(
                self.n_active_vertices
            )
            current_radius = cp.sqrt(cp.maximum(squared_radius, 0.0))
            new_positions *= np.float32(self._normalization_radius) / (
                current_radius + np.float32(1.0e-12)
            )
        new_positions[~self._active_mask] = 0.0
        self.positions = new_positions.astype(cp.float32, copy=False)
        self._stage_end('normalization', started)
        self._iteration += 1

    def run_layout(self, num_iterations=100):
        """
        Run the force-directed layout algorithm.

        Parameters
        ----------
        num_iterations : int, default=100
            Number of iterations to run.

        Returns
        -------
        cupy.ndarray
            Final vertex positions.
        """
        self.logger.info("Running cuVS-accelerated layout for %d iterations", num_iterations)

        if num_iterations < 0:
            raise ValueError("num_iterations must be non-negative")
        for iteration in tqdm(
            range(num_iterations), desc="Layout iterations", disable=not self.verbose
        ):
            self.update_positions()
            if self.verbose and (iteration + 1) % 10 == 0:
                self.logger.info("Completed iteration %d/%d", iteration + 1, num_iterations)

        self.logger.info("cuVS layout computation completed")
        return self.positions

    def get_positions(self):
        """Get vertex positions as numpy array."""
        return cp.asnumpy(self.positions)

    def get_scores(self, as_numpy=True):
        """Return radial node scores, optionally leaving them on the GPU."""
        scores = cp.linalg.norm(self.positions, axis=1)
        return cp.asnumpy(scores) if as_numpy else scores

    def topk_nodes(self, k):
        """Return top radial node ids while transferring only ``k`` integers."""
        if not 0 <= k <= self.n_active_vertices:
            raise ValueError(
                "k must be between zero and the number of non-isolated vertices"
            )
        if k == 0:
            return []
        scores = self.get_scores(as_numpy=False)
        scores = cp.where(self._active_mask, scores, -cp.inf)
        candidates = cp.argpartition(scores, self.n - k)[-k:]
        candidates = candidates[cp.argsort(scores[candidates])[::-1]]
        return cp.asnumpy(candidates).astype(np.int64).tolist()

    def diverse_topk_nodes(self, k, diversity=0.2, candidate_pool_size=None):
        """Select high-score nodes while spreading them in embedding space.

        This addresses a common influence-maximization failure mode in which the
        highest radial nodes are geometrically redundant. Work is O(kCd), where C
        is a bounded candidate pool, and remains entirely on the GPU.
        """
        if not 0 <= k <= self.n_active_vertices:
            raise ValueError(
                "k must be between zero and the number of non-isolated vertices"
            )
        if not 0.0 <= diversity <= 1.0:
            raise ValueError("diversity must be between zero and one")
        if k == 0:
            return []
        if candidate_pool_size is None:
            candidate_pool_size = min(
                self.n_active_vertices, max(10_000, 200 * k)
            )
        candidate_pool_size = min(
            self.n_active_vertices, int(candidate_pool_size)
        )
        if candidate_pool_size < k:
            raise ValueError("candidate_pool_size must be at least k")

        scores = self.get_scores(as_numpy=False)
        scores = cp.where(self._active_mask, scores, -cp.inf)
        candidates = cp.argpartition(
            scores, self.n - candidate_pool_size
        )[-candidate_pool_size:]
        candidate_scores = scores[candidates]
        relevance = (candidate_scores - cp.min(candidate_scores)) / (
            cp.max(candidate_scores) - cp.min(candidate_scores) + 1e-12
        )
        candidate_positions = self.positions[candidates]
        chosen = cp.zeros(candidate_pool_size, dtype=cp.bool_)
        first = int(cp.argmax(candidate_scores).item())
        selected_local = [first]
        chosen[first] = True
        minimum_distance = cp.sum(
            (candidate_positions - candidate_positions[first]) ** 2, axis=1
        )

        while len(selected_local) < k:
            distance_score = minimum_distance / (cp.max(minimum_distance) + 1e-12)
            utility = (1.0 - diversity) * relevance + diversity * distance_score
            utility[chosen] = -cp.inf
            selected = int(cp.argmax(utility).item())
            selected_local.append(selected)
            chosen[selected] = True
            distance = cp.sum(
                (candidate_positions - candidate_positions[selected]) ** 2, axis=1
            )
            minimum_distance = cp.minimum(minimum_distance, distance)
        return cp.asnumpy(candidates[cp.asarray(selected_local)]).astype(np.int64).tolist()

    def get_diagnostics(self):
        """Return algorithm choices and synchronized stage timings."""
        return {
            'n_vertices': self.n,
            'n_active_vertices': self.n_active_vertices,
            'n_isolated_vertices': self.n - self.n_active_vertices,
            'n_edges': self.n_edges,
            'initialization': (
                'spectral'
                if self.initialization == 'auto'
                and self.adjacency is not None
                and self.n <= self.spectral_max_vertices
                else 'randomized' if self.initialization == 'auto'
                else self.initialization
            ),
            'force_mode': self.force_mode,
            'index_type': (
                self._knn_kind
                or self._select_index_type(self.midpoint_reference_size)
            ),
            'configured_index_type': self.index_type,
            'configured_ivf_n_probes': self.ivf_n_probes,
            'actual_ivf_n_probes': self._last_n_probes,
            'last_n_probes': self._last_n_probes,
            'sample_size': self.sample_size,
            'last_query_sample_size': self._last_query_sample_size,
            'query_inclusion_scale': self.intersection_inclusion_scale,
            'query_inclusion_policy': (
                'directed_knn_nominations_inverse_query_probability_E_over_S_'
                'before_per_vertex_max_displacement'
            ),
            'intersection_geometry': '2d_equal_opposite_segment_separation',
            'intersection_accumulator': 'fused_fp32_atomic',
            'max_candidate_pairs': self.max_candidate_pairs,
            'midpoint_reference_mode': self._midpoint_reference_mode,
            'midpoint_reference_size': self.midpoint_reference_size,
            'configured_midpoint_reference_fraction': (
                float(self.midpoint_reference_size) / float(self.n_edges)
                if self.n_edges else 0.0
            ),
            'last_midpoint_reference_size': self._last_midpoint_reference_size,
            'midpoint_reference_fraction': (
                float(self._last_midpoint_reference_size) / float(self.n_edges)
                if self.n_edges else 0.0
            ),
            'spring_accumulator': 'fused_fp32_atomic',
            'normalization': 'active_isotropic_rms',
            'initial_median_edge_length_before_scale': (
                self._initial_edge_length_before_scale
            ),
            'initial_median_edge_length_after_scale': (
                self._initial_edge_length_after_scale
            ),
            'edge_chunk_size': self.edge_chunk_size,
            'edge_chunk_size_role': 'compatibility_only_fused_spring_uses_one_launch',
            'batch_size_role': 'bounded_exact_knn_fallback_only',
            'intersection_interval': self.intersection_interval,
            'midpoint_index_refresh_interval': self.intersection_interval,
            'iterations_completed': self._iteration,
            'knn_fallbacks': self._knn_fallbacks,
            'knn_last_error': self._knn_last_error,
            'timings_seconds': dict(self.timings),
        }

    def display_layout(
        self,
        edge_width=1,
        node_size=3,
        node_colors=None
    ):
        """Display the graph embedding using Plotly."""
        self.logger.info("Displaying cuVS layout")

        if self.n_components == 2:
            self._display_layout_2d(edge_width, node_size, node_colors)
        elif self.n_components == 3:
            self._display_layout_3d(edge_width, node_size, node_colors)
        else:
            raise ValueError("Can only display 2D or 3D layouts")

    def _display_layout_2d(self, edge_width, node_size, node_colors):
        """Display 2D layout using Plotly."""
        pos = self.get_positions()
        edges_np = cp.asnumpy(self.edges)

        # Create traces (same as PyTorch version)
        x_edges, y_edges = [], []
        for i, j in edges_np:
            x_edges.extend([pos[i, 0], pos[j, 0], None])
            y_edges.extend([pos[i, 1], pos[j, 1], None])

        edge_trace = go.Scatter(
            x=x_edges, y=y_edges, mode='lines',
            line={'color': 'gray', 'width': edge_width}, hoverinfo='none'
        )

        node_trace = go.Scatter(
            x=pos[:, 0], y=pos[:, 1], mode='markers',
            marker={
                'color': node_colors if node_colors is not None else 'red',
                'colorscale': 'Bluered', 'size': node_size,
                'colorbar': {'title': 'Node Label'},
                'showscale': node_colors is not None
            }, hoverinfo='none'
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="2D Graph Embedding (cuVS Rapids)",
            xaxis={'title': 'X', 'showgrid': False, 'zeroline': False},
            yaxis={'title': 'Y', 'showgrid': False, 'zeroline': False},
            showlegend=False, width=800, height=800
        )
        fig.show()

    def _display_layout_3d(self, edge_width, node_size, node_colors):
        """Display 3D layout using Plotly."""
        pos = self.get_positions()
        edges_np = cp.asnumpy(self.edges)

        x_edges, y_edges, z_edges = [], [], []
        for i, j in edges_np:
            x_edges.extend([pos[i, 0], pos[j, 0], None])
            y_edges.extend([pos[i, 1], pos[j, 1], None])
            z_edges.extend([pos[i, 2], pos[j, 2], None])

        edge_trace = go.Scatter3d(
            x=x_edges, y=y_edges, z=z_edges, mode='lines',
            line={'color': 'gray', 'width': edge_width}, hoverinfo='none'
        )

        node_trace = go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode='markers',
            marker={
                'color': node_colors if node_colors is not None else 'red',
                'colorscale': 'Bluered', 'size': node_size,
                'colorbar': {'title': 'Node Label'},
                'showscale': node_colors is not None
            }, hoverinfo='none'
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="3D Graph Embedding (cuVS Rapids)",
            scene={'xaxis': {'title': 'X'}, 'yaxis': {'title': 'Y'}, 'zaxis': {'title': 'Z'}},
            showlegend=False, width=800, height=800
        )
        fig.show()

    def __repr__(self):
        """String representation."""
        return (f"GraphEmbedderCuVS(n_vertices={self.n}, n_components={self.n_components}, "
                f"index_type={self._select_index_type()})")
