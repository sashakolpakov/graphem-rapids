"""
RAPIDS cuVS-based implementation of GraphEmbedder for large-scale datasets.

This module provides graph embedding using RAPIDS cuVS for efficient
large-scale nearest neighbor computations and GPU-accelerated processing.
"""

import logging
import warnings

import numpy as np
import torch
import plotly.graph_objects as go
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import laplacian
from tqdm import tqdm

# RAPIDS imports
try:
    import cupy as cp
    import cuvs
    from cuvs.neighbors import brute_force, ivf_flat, ivf_pq
    CUVS_AVAILABLE = True
except ImportError:
    CUVS_AVAILABLE = False
    warnings.warn(
        "RAPIDS cuVS not available. This backend requires RAPIDS cuVS installation.",
        ImportWarning
    )

from ..utils.memory_management import (
    MemoryManager,
    get_optimal_chunk_size,
    cleanup_gpu_memory,
    monitor_memory_usage
)

logger = logging.getLogger(__name__)


class GraphEmbedderCuVS:
    """
    RAPIDS cuVS-based graph embedder for large-scale datasets.

    This class provides graph embedding using RAPIDS cuVS for efficient
    large-scale KNN computations and GPU-accelerated force computations.
    Optimized for datasets with >100K vertices.

    Attributes
    ----------
    edges : cupy.ndarray
        Edge list as (n_edges, 2) array.
    n : int
        Number of vertices in the graph.
    dimension : int
        Dimension of the embedding space.
    positions : cupy.ndarray
        Current vertex positions as (n_vertices, dimension) array.
    """

    def __init__(
        self,
        edges,
        n_vertices,
        dimension=2,
        L_min=1.0,
        k_attr=0.2,
        k_inter=0.5,
        knn_k=10,
        sample_size=1024,
        batch_size=4096,
        index_type='auto',
        dtype=np.float32,
        verbose=True,
        logger_instance=None
    ):
        """
        Initialize the cuVS GraphEmbedder.

        Parameters
        ----------
        edges : array-like
            Array of edge pairs (i, j) with i < j.
        n_vertices : int
            Number of vertices in the graph.
        dimension : int, default=2
            Dimension of the embedding.
        L_min : float, default=1.0
            Minimum spring length.
        k_attr : float, default=0.2
            Attraction force constant.
        k_inter : float, default=0.5
            Intersection repulsion force constant.
        knn_k : int, default=10
            Number of nearest neighbors for intersection detection.
        sample_size : int, default=1024
            Sample size for kNN computation (larger for cuVS).
        batch_size : int, default=4096
            Batch size for processing (larger for cuVS).
        index_type : str, default='auto'
            cuVS index type ('brute_force', 'ivf_flat', 'ivf_pq', 'auto').
        dtype : numpy.dtype, default=np.float32
            Data type for computations.
        verbose : bool, default=True
            Enable verbose logging.
        logger_instance : logging.Logger, optional
            Custom logger instance.
        """
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

        # Store parameters
        self.n = n_vertices
        self.dimension = dimension
        self.dtype = dtype
        self.L_min = L_min
        self.k_attr = k_attr
        self.k_inter = k_inter
        self.knn_k = knn_k
        self.sample_size = min(sample_size, n_vertices)
        self.batch_size = batch_size
        self.index_type = index_type
        self.verbose = verbose

        # Convert edges to cupy array
        if isinstance(edges, torch.Tensor):
            edges_np = edges.detach().cpu().numpy()
        else:
            edges_np = np.array(edges)

        self.edges = cp.asarray(edges_np, dtype=cp.int32)

        # Memory management for large datasets
        self.chunk_size = get_optimal_chunk_size(self.n, self.dimension)

        if self.verbose:
            self.logger.info("Initialized GraphEmbedderCuVS")
            self.logger.info("Graph: %d vertices, %d edges, %dD", self.n, len(self.edges), self.dimension)
            self.logger.info("Index type: %s", self.index_type)
            self.logger.info("Chunk size: %d", self.chunk_size)

        # Compute initial embedding
        self.positions = self._compute_laplacian_embedding()

        # Initialize cuVS index for KNN searches
        self.knn_index = None
        self._build_knn_index()

    def _compute_laplacian_embedding(self):
        """
        Compute the Laplacian embedding using scipy then transfer to GPU.

        Returns
        -------
        cupy.ndarray
            Initial positions from Laplacian embedding.
        """
        self.logger.info("Computing Laplacian embedding")

        # Use scipy for eigendecomposition (CPU-based, more stable)
        edges_np = cp.asnumpy(self.edges)
        row = edges_np[:, 0]
        col = edges_np[:, 1]
        data = np.ones(len(edges_np))

        # Build adjacency matrix
        A = sp.csr_matrix((data, (row, col)), shape=(self.n, self.n))
        A = A + A.transpose()

        # Compute normalized Laplacian
        L = laplacian(A, normed=True)

        # Compute eigenvectors
        k = self.dimension + 1
        try:
            _, eigenvectors = spla.eigsh(L, k, which='SM')
            lap_embedding = eigenvectors[:, 1:k]
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.warning("Eigendecomposition failed: %s", e)
            lap_embedding = np.random.randn(self.n, self.dimension) * 0.1

        # Transfer to GPU
        positions = cp.asarray(lap_embedding, dtype=self.dtype)

        self.logger.info("Laplacian embedding computed and transferred to GPU")
        return positions

    def _select_index_type(self):
        """
        Automatically select optimal cuVS index type based on data size.

        Returns
        -------
        str
            Selected index type.
        """
        if self.index_type != 'auto':
            return self.index_type

        # Selection logic based on dataset size
        if self.n > 1_000_000:
            return 'ivf_pq'  # Best for very large datasets
        if self.n > 100_000:
            return 'ivf_flat'  # Good balance for large datasets
        return 'brute_force'  # Exact for smaller datasets

    def _build_knn_index(self):
        """Build cuVS index for efficient KNN searches."""
        if self.positions is None:
            return

        index_type = self._select_index_type()
        self.logger.info("Building cuVS index of type: %s", index_type)

        with MemoryManager():
            try:
                if index_type == 'brute_force':
                    self.knn_index = brute_force.build(self.positions, metric='l2')
                elif index_type == 'ivf_flat':
                    # Build IVF-Flat index with automatic parameter selection
                    n_lists = min(int(np.sqrt(self.n)), 16384)
                    self.knn_index = ivf_flat.build(
                        self.positions,
                        metric='l2',
                        n_lists=n_lists
                    )
                elif index_type == 'ivf_pq':
                    # Build IVF-PQ index for memory efficiency
                    n_lists = min(int(np.sqrt(self.n)), 16384)
                    pq_dim = min(self.dimension, 64)
                    self.knn_index = ivf_pq.build(
                        self.positions,
                        metric='l2',
                        n_lists=n_lists,
                        pq_dim=pq_dim,
                        pq_bits=8
                    )
                else:
                    raise ValueError(f"Unknown index type: {index_type}")

                self.logger.info("cuVS index built successfully")

            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.error("Failed to build cuVS index: %s", e)
                self.logger.info("Falling back to brute force search")
                self.knn_index = None

    def _locate_knn_midpoints_cuvs(
        self,
        midpoints,
        k
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
        self.logger.info("Computing kNN using cuVS")

        E = midpoints.shape[0]
        sample_size = min(self.sample_size, E)

        with MemoryManager():
            # Sample midpoints
            if sample_size < E:
                sampled_indices = cp.random.choice(E, size=sample_size, replace=False)
                sampled_midpoints = midpoints[sampled_indices]
            else:
                sampled_indices = cp.arange(E)
                sampled_midpoints = midpoints

            # Rebuild index if needed (positions may have changed)
            if self.knn_index is None:
                self._build_knn_index()

            try:
                # Use cuVS for KNN search
                if self.knn_index is not None:
                    if hasattr(brute_force, 'search'):
                        indices, _ = brute_force.search(
                            self.knn_index,
                            sampled_midpoints,
                            k + 1
                        )
                    else:
                        # Alternative search method
                        indices, _ = cuvs.neighbors.search(
                            self.knn_index,
                            sampled_midpoints,
                            k + 1
                        )
                else:
                    # Fallback to manual distance computation
                    indices = self._manual_knn_search(sampled_midpoints, midpoints, k + 1)

                # Remove self-neighbors
                knn_indices = indices[:, 1:]

                self.logger.info("cuVS kNN computation completed")
                return knn_indices, sampled_indices

            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.error("cuVS KNN search failed: %s", e)
                # Fallback to chunked distance computation
                return self._fallback_knn_search(sampled_midpoints, midpoints, k)

    def _manual_knn_search(
        self,
        query_points,
        reference_points,
        k
    ):
        """Fallback manual KNN search using CuPy."""
        # Compute all pairwise distances (memory intensive)
        distances = cp.linalg.norm(
            query_points[:, None, :] - reference_points[None, :, :],
            axis=2
        )

        # Find k nearest neighbors
        knn_indices = cp.argpartition(distances, k, axis=1)[:, :k]

        return knn_indices

    def _fallback_knn_search(
        self,
        sampled_midpoints,
        midpoints,
        k
    ):
        """Fallback KNN search with chunked processing."""
        n_samples = sampled_midpoints.shape[0]
        chunk_size = min(self.chunk_size, n_samples)

        all_indices = []
        sampled_indices = cp.arange(n_samples)

        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            chunk = sampled_midpoints[i:end_idx]

            # Compute distances
            distances = cp.linalg.norm(
                chunk[:, None, :] - midpoints[None, :, :],
                axis=2
            )

            # Find k+1 nearest (including self)
            indices = cp.argpartition(distances, k + 1, axis=1)[:, 1:k + 1]
            all_indices.append(indices)

        knn_indices = cp.concatenate(all_indices, axis=0)
        return knn_indices, sampled_indices

    @monitor_memory_usage
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
        with MemoryManager():
            # Get edge endpoints
            p1 = positions[edges[:, 0]]
            p2 = positions[edges[:, 1]]

            # Compute edge vectors and distances
            diff = p2 - p1
            dist = cp.linalg.norm(diff, axis=1, keepdims=True) + 1e-6

            # Compute force magnitude
            force_magnitude = -self.k_attr * (dist - self.L_min)

            # Compute force vectors
            edge_forces = force_magnitude * (diff / dist)

            # Accumulate forces on vertices using advanced indexing
            forces = cp.zeros_like(positions)
            cp.add.at(forces, edges[:, 0], edge_forces)
            cp.add.at(forces, edges[:, 1], -edge_forces)

            return forces

    @monitor_memory_usage
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
        with MemoryManager():
            # Generate edge pairs
            _, n_neighbors = knn_indices.shape
            candidate_i = cp.repeat(sampled_indices, n_neighbors)
            candidate_j = knn_indices.flatten()

            # Filter valid pairs
            valid_mask = candidate_i < candidate_j

            if not valid_mask.any():
                return cp.zeros_like(positions)

            # Get valid pairs
            valid_i = candidate_i[valid_mask]
            valid_j = candidate_j[valid_mask]

            edges_i = edges[valid_i]
            edges_j = edges[valid_j]

            # Check for shared vertices
            share_mask = (
                (edges_i[:, 0] == edges_j[:, 0]) |
                (edges_i[:, 0] == edges_j[:, 1]) |
                (edges_i[:, 1] == edges_j[:, 0]) |
                (edges_i[:, 1] == edges_j[:, 1])
            )

            interaction_mask = ~share_mask

            if not interaction_mask.any():
                return cp.zeros_like(positions)

            # Filter to interacting pairs
            edges_i = edges_i[interaction_mask]
            edges_j = edges_j[interaction_mask]

            # Get endpoints
            p1 = positions[edges_i[:, 0]]
            p2 = positions[edges_i[:, 1]]
            q1 = positions[edges_j[:, 0]]
            q2 = positions[edges_j[:, 1]]

            # Check intersections
            intersect_mask = self._check_line_intersections_cuvs(p1, p2, q1, q2)

            if not intersect_mask.any():
                return cp.zeros_like(positions)

            # Compute repulsion forces
            edges_i = edges_i[intersect_mask]
            edges_j = edges_j[intersect_mask]
            p1, p2 = p1[intersect_mask], p2[intersect_mask]
            q1, q2 = q1[intersect_mask], q2[intersect_mask]

            inter_midpoints = (p1 + p2 + q1 + q2) / 4.0

            # Compute forces
            forces = cp.zeros_like(positions)

            for vertex_pos, edge_vertices in [(p1, edges_i[:, 0]), (p2, edges_i[:, 1]),
                                            (q1, edges_j[:, 0]), (q2, edges_j[:, 1])]:
                diff = vertex_pos - inter_midpoints
                dist = cp.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
                repulsion = self.k_inter * diff / (dist ** 2)

                cp.add.at(forces, edge_vertices, repulsion)

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
        self.logger.info("Updating positions using cuVS backend")

        with MemoryManager():
            # Compute spring forces
            spring_forces = self._compute_spring_forces_cuvs(self.positions, self.edges)

            # Compute edge midpoints
            midpoints = (self.positions[self.edges[:, 0]] + self.positions[self.edges[:, 1]]) / 2.0

            # Find nearest neighbors using cuVS
            knn_indices, sampled_indices = self._locate_knn_midpoints_cuvs(midpoints, self.knn_k)

            # Compute intersection forces
            inter_forces = self._compute_intersection_forces_cuvs(
                self.positions, self.edges, knn_indices, sampled_indices
            )

            # Combine and apply forces
            total_forces = spring_forces + inter_forces
            new_positions = self.positions + total_forces

            # Normalize positions
            new_positions = new_positions - cp.mean(new_positions, axis=0, keepdims=True)
            std = cp.std(new_positions, axis=0, keepdims=True) + 1e-6
            self.positions = new_positions / std

        self.logger.info("Position update completed")

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

        with MemoryManager(cleanup_on_exit=True):
            for iteration in tqdm(range(num_iterations), desc="Layout iterations"):
                self.update_positions()

                # Rebuild index periodically for better accuracy
                if (iteration + 1) % 20 == 0:
                    self._build_knn_index()

                if self.verbose and (iteration + 1) % 10 == 0:
                    self.logger.info("Completed iteration %d/%d", iteration + 1, num_iterations)

        self.logger.info("cuVS layout computation completed")
        return self.positions

    def get_positions(self):
        """Get vertex positions as numpy array."""
        return cp.asnumpy(self.positions)

    def display_layout(
        self,
        edge_width=1,
        node_size=3,
        node_colors=None
    ):
        """Display the graph embedding using Plotly."""
        self.logger.info("Displaying cuVS layout")

        if self.dimension == 2:
            self._display_layout_2d(edge_width, node_size, node_colors)
        elif self.dimension == 3:
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
        return (f"GraphEmbedderCuVS(n_vertices={self.n}, dimension={self.dimension}, "
                f"index_type={self._select_index_type()})")

    def __del__(self):
        """Cleanup GPU resources."""
        try:
            cleanup_gpu_memory()
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.warning("GPU memory cleanup failed: %s", e)