"""
PyTorch-based implementation of GraphEmbedder with CUDA acceleration.

This module provides the main graph embedding functionality using PyTorch
as the computational backend, with optional CUDA acceleration.
"""

import logging

import numpy as np
import torch
import plotly.graph_objects as go
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import laplacian
from tqdm import tqdm

from ..utils.memory_management import (
    MemoryManager,
    get_optimal_chunk_size,
    monitor_memory_usage
)

logger = logging.getLogger(__name__)


class GraphEmbedderPyTorch:
    """
    PyTorch-based graph embedder with CUDA acceleration.

    This class provides graph embedding using Laplacian initialization
    followed by force-directed layout optimization, implemented with PyTorch
    for GPU acceleration and memory efficiency.

    Attributes
    ----------
    edges : torch.Tensor
        Edge list as (n_edges, 2) tensor.
    n : int
        Number of vertices in the graph.
    dimension : int
        Dimension of the embedding space.
    device : torch.device
        Computing device (CPU or CUDA).
    positions : torch.Tensor
        Current vertex positions as (n_vertices, dimension) tensor.
    """

    def __init__(
        self,
        edges,
        n_vertices,
        dimension=2,
        device=None,
        dtype=torch.float32,
        L_min=1.0,
        k_attr=0.2,
        k_inter=0.5,
        knn_k=10,
        sample_size=256,
        batch_size=1024,
        memory_efficient=True,
        verbose=True,
        logger_instance=None
    ):
        """
        Initialize the PyTorch GraphEmbedder.

        Parameters
        ----------
        edges : array-like
            Array of edge pairs (i, j) with i < j.
        n_vertices : int
            Number of vertices in the graph.
        dimension : int, default=2
            Dimension of the embedding.
        device : str or torch.device, optional
            Computing device. If None, automatically selects GPU if available.
        dtype : torch.dtype, default=torch.float32
            Data type for computations.
        L_min : float, default=1.0
            Minimum spring length.
        k_attr : float, default=0.2
            Attraction force constant.
        k_inter : float, default=0.5
            Intersection repulsion force constant.
        knn_k : int, default=10
            Number of nearest neighbors for intersection detection.
        sample_size : int, default=256
            Sample size for kNN computation.
        batch_size : int, default=1024
            Batch size for processing.
        memory_efficient : bool, default=True
            Use memory-efficient algorithms for large graphs.
        verbose : bool, default=True
            Enable verbose logging.
        logger_instance : logging.Logger, optional
            Custom logger instance.
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

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
        self.memory_efficient = memory_efficient

        # Validate parameters
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        if n_vertices <= 0:
            raise ValueError(f"Number of vertices must be positive, got {n_vertices}")
        if k_attr < 0:
            raise ValueError(f"Attractive force constant k_attr must be non-negative, got {k_attr}")
        self.verbose = verbose

        # Convert edges to tensor
        if isinstance(edges, torch.Tensor):
            self.edges = edges.to(device=self.device, dtype=torch.long)
        else:
            self.edges = torch.tensor(np.array(edges), device=self.device, dtype=torch.long)

        # Memory management
        self.chunk_size = get_optimal_chunk_size(self.n, self.dimension)
        if self.verbose:
            self.logger.info("Initialized GraphEmbedderPyTorch on %s", self.device)
            self.logger.info("Graph: %d vertices, %d edges, %dD", self.n, len(self.edges), self.dimension)
            self.logger.info("Chunk size: %d", self.chunk_size)

        # Compute initial embedding
        self._positions = self._compute_laplacian_embedding()

    @property
    def positions(self):
        """Get positions as numpy array for API consistency."""
        return self._positions.detach().cpu().numpy()

    @positions.setter
    def positions(self, value):
        """Set positions from numpy array or tensor."""
        if isinstance(value, np.ndarray):
            self._positions = torch.tensor(value, dtype=self.dtype, device=self.device)
        else:
            self._positions = value.to(device=self.device, dtype=self.dtype)

    def _compute_laplacian_embedding(self):
        """
        Compute the Laplacian embedding of the graph using scipy.

        Returns
        -------
        torch.Tensor
            Initial positions from Laplacian embedding.
        """
        self.logger.info("Computing Laplacian embedding")

        with MemoryManager(cleanup_on_exit=True):
            # Use scipy for eigendecomposition (more stable than PyTorch)
            edges_np = self.edges.cpu().numpy()
            row = edges_np[:, 0]
            col = edges_np[:, 1]
            data = np.ones(len(edges_np))

            # Build adjacency matrix
            A = sp.csr_matrix((data, (row, col)), shape=(self.n, self.n))
            A = A + A.transpose()  # Make symmetric

            # Compute normalized Laplacian
            L = laplacian(A, normed=True)

            # Compute eigenvectors
            k = self.dimension + 1
            try:
                _, eigenvectors = spla.eigsh(L, k, which='SM')
                lap_embedding = eigenvectors[:, 1:k]  # Skip first eigenvector
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.warning("Eigendecomposition failed: %s", e)
                # Fallback to random initialization
                lap_embedding = np.random.randn(self.n, self.dimension) * 0.1

            # Convert to tensor
            positions = torch.tensor(
                lap_embedding,
                device=self.device,
                dtype=self.dtype
            )

            self.logger.info("Laplacian embedding computed")
            return positions

    def _locate_knn_midpoints(
        self,
        midpoints,
        k
    ):
        """
        Locate k nearest neighbors for edge midpoints.

        Parameters
        ----------
        midpoints : torch.Tensor
            Edge midpoints as (n_edges, dimension) tensor.
        k : int
            Number of nearest neighbors.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            KNN indices and sampled indices.
        """
        self.logger.info("Computing kNN for midpoints")

        E = midpoints.shape[0]
        sample_size = min(self.sample_size, E)

        with MemoryManager():
            # Sample midpoints
            if sample_size < E:
                sampled_indices = torch.randperm(E, device=self.device)[:sample_size]
                sampled_midpoints = midpoints[sampled_indices]
            else:
                sampled_indices = torch.arange(E, device=self.device)
                sampled_midpoints = midpoints

            # Compute pairwise distances using chunking for memory efficiency
            knn_indices = self._compute_knn_chunked(
                sampled_midpoints, midpoints, k + 1
            )

            # Remove self-neighbors (first column)
            knn_indices = knn_indices[:, 1:]

            self.logger.info("kNN computation completed")
            return knn_indices, sampled_indices

    def _compute_knn_chunked(
        self,
        query_points,
        reference_points,
        k
    ):
        """
        Compute k-nearest neighbors using chunked processing.

        Parameters
        ----------
        query_points : torch.Tensor
            Query points as (n_query, dimension) tensor.
        reference_points : torch.Tensor
            Reference points as (n_ref, dimension) tensor.
        k : int
            Number of nearest neighbors.

        Returns
        -------
        torch.Tensor
            KNN indices as (n_query, k) tensor.
        """
        n_query = query_points.shape[0]

        # Determine chunk size based on memory constraints
        max_chunk_size = min(self.chunk_size, n_query)

        all_knn_indices = []

        for i in range(0, n_query, max_chunk_size):
            end_idx = min(i + max_chunk_size, n_query)
            query_chunk = query_points[i:end_idx]

            # Compute distances for this chunk
            distances = torch.cdist(query_chunk, reference_points, p=2)

            # Find k nearest neighbors
            _, knn_indices = torch.topk(distances, k, dim=1, largest=False)
            all_knn_indices.append(knn_indices)

            # Clean up intermediate tensors
            del distances

        return torch.cat(all_knn_indices, dim=0)

    @monitor_memory_usage
    def _compute_spring_forces(
        self,
        positions,
        edges
    ):
        """
        Compute spring forces between connected vertices.

        Parameters
        ----------
        positions : torch.Tensor
            Current vertex positions.
        edges : torch.Tensor
            Edge list.

        Returns
        -------
        torch.Tensor
            Spring forces for each vertex.
        """
        with MemoryManager():
            # Get edge endpoints
            p1 = positions[edges[:, 0]]
            p2 = positions[edges[:, 1]]

            # Compute edge vectors and distances
            diff = p2 - p1
            dist = torch.norm(diff, dim=1, keepdim=True) + 1e-6

            # Compute force magnitude (spring law)
            force_magnitude = -self.k_attr * (dist - self.L_min)

            # Compute force vectors
            edge_forces = force_magnitude * (diff / dist)

            # Accumulate forces on vertices
            forces = torch.zeros_like(positions)
            forces.index_add_(0, edges[:, 0], edge_forces)
            forces.index_add_(0, edges[:, 1], -edge_forces)

            return forces

    @monitor_memory_usage
    def _compute_intersection_forces(
        self,
        positions,
        edges,
        knn_indices,
        sampled_indices
    ):
        """
        Compute intersection repulsion forces between nearby edge pairs.

        Parameters
        ----------
        positions : torch.Tensor
            Current vertex positions.
        edges : torch.Tensor
            Edge list.
        knn_indices : torch.Tensor
            KNN indices for edge midpoints.
        sampled_indices : torch.Tensor
            Indices of sampled edges.

        Returns
        -------
        torch.Tensor
            Intersection forces for each vertex.
        """
        with MemoryManager():
            # Generate edge pairs from KNN results
            _, n_neighbors = knn_indices.shape
            candidate_i = sampled_indices.unsqueeze(1).expand(-1, n_neighbors).flatten()
            candidate_j = knn_indices.flatten()

            # Filter valid pairs (i < j)
            valid_mask = candidate_i < candidate_j

            if not valid_mask.any():
                return torch.zeros_like(positions)

            # Get valid edge pairs
            valid_i = candidate_i[valid_mask]
            valid_j = candidate_j[valid_mask]

            edges_i = edges[valid_i]
            edges_j = edges[valid_j]

            # Check for shared vertices (skip connected edges)
            share_mask = (
                (edges_i[:, 0] == edges_j[:, 0]) |
                (edges_i[:, 0] == edges_j[:, 1]) |
                (edges_i[:, 1] == edges_j[:, 0]) |
                (edges_i[:, 1] == edges_j[:, 1])
            )
            interaction_mask = valid_mask[valid_mask].clone()
            interaction_mask[share_mask] = False

            if not interaction_mask.any():
                return torch.zeros_like(positions)

            # Filter to interacting pairs
            edges_i = edges_i[interaction_mask]
            edges_j = edges_j[interaction_mask]

            # Get edge endpoints
            p1 = positions[edges_i[:, 0]]
            p2 = positions[edges_i[:, 1]]
            q1 = positions[edges_j[:, 0]]
            q2 = positions[edges_j[:, 1]]

            # Check for line segment intersections using orientation test
            intersect_mask = self._check_line_intersections(p1, p2, q1, q2)

            if not intersect_mask.any():
                return torch.zeros_like(positions)

            # Filter to actually intersecting edges
            edges_i = edges_i[intersect_mask]
            edges_j = edges_j[intersect_mask]
            p1 = p1[intersect_mask]
            p2 = p2[intersect_mask]
            q1 = q1[intersect_mask]
            q2 = q2[intersect_mask]

            # Compute intersection midpoints
            inter_midpoints = (p1 + p2 + q1 + q2) / 4.0

            # Compute repulsion forces
            forces = torch.zeros_like(positions)

            for vertex_pos, edge_vertices in [(p1, edges_i[:, 0]), (p2, edges_i[:, 1]),
                                            (q1, edges_j[:, 0]), (q2, edges_j[:, 1])]:
                # Compute repulsion from intersection points
                diff = vertex_pos - inter_midpoints
                dist = torch.norm(diff, dim=1, keepdim=True) + 1e-6
                repulsion = self.k_inter * diff / (dist ** 2)

                forces.index_add_(0, edge_vertices, repulsion)

            return forces

    def _check_line_intersections(
        self,
        p1,
        p2,
        q1,
        q2
    ):
        """
        Check if line segments (p1,p2) and (q1,q2) intersect.

        Parameters
        ----------
        p1, p2 : torch.Tensor
            Endpoints of first line segment.
        q1, q2 : torch.Tensor
            Endpoints of second line segment.

        Returns
        -------
        torch.Tensor
            Boolean mask indicating which pairs intersect.
        """
        def orientation(a, b, c):
            """Compute orientation of ordered triplet (a, b, c)."""
            return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - \
                   (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

        # Compute orientations
        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)

        # Check intersection condition
        intersect = (o1 * o2 < 0) & (o3 * o4 < 0)

        return intersect

    def update_positions(self):
        """Update vertex positions based on computed forces."""
        self.logger.info("Updating vertex positions")

        with MemoryManager():
            # Compute spring forces
            spring_forces = self._compute_spring_forces(self._positions, self.edges)

            # Compute edge midpoints
            midpoints = (self._positions[self.edges[:, 0]] + self._positions[self.edges[:, 1]]) / 2.0

            # Find nearest neighbors for intersection detection
            knn_indices, sampled_indices = self._locate_knn_midpoints(midpoints, self.knn_k)

            # Compute intersection forces
            inter_forces = self._compute_intersection_forces(
                self._positions, self.edges, knn_indices, sampled_indices
            )

            # Combine forces
            total_forces = spring_forces + inter_forces

            # Update positions
            new_positions = self._positions + total_forces

            # Normalize positions (center and scale)
            new_positions = new_positions - torch.mean(new_positions, dim=0, keepdim=True)
            std = torch.std(new_positions, dim=0, keepdim=True) + 1e-6
            self._positions = new_positions / std

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
        torch.Tensor
            Final vertex positions.
        """
        self.logger.info("Running layout for %d iterations", num_iterations)

        with MemoryManager(cleanup_on_exit=True):
            for iteration in tqdm(range(num_iterations), desc="Layout iterations"):
                self.update_positions()

                # Optional: log progress
                if self.verbose and (iteration + 1) % 10 == 0:
                    self.logger.info("Completed iteration %d/%d", iteration + 1, num_iterations)

        self.logger.info("Layout computation completed")
        return self.positions

    def get_positions(self):
        """
        Get vertex positions as numpy array.

        Returns
        -------
        np.ndarray
            Vertex positions.
        """
        return self.positions  # Uses property which already returns numpy array

    def display_layout(
        self,
        edge_width=1,
        node_size=3,
        node_colors=None
    ):
        """
        Display the graph embedding using Plotly.

        Parameters
        ----------
        edge_width : float, default=1
            Width of the edges.
        node_size : float, default=3
            Size of the nodes.
        node_colors : array-like, optional
            Colors for each vertex.
        """
        self.logger.info("Displaying layout")

        if self.dimension == 2:
            self._display_layout_2d(edge_width, node_size, node_colors)
        elif self.dimension == 3:
            self._display_layout_3d(edge_width, node_size, node_colors)
        else:
            raise ValueError("Can only display 2D or 3D layouts")

    def _display_layout_2d(
        self,
        edge_width,
        node_size,
        node_colors
    ):
        """Display 2D graph embedding."""
        pos = self.get_positions()
        edges_np = self.edges.cpu().numpy()

        # Create edge traces
        x_edges, y_edges = [], []
        for i, j in edges_np:
            x_edges.extend([pos[i, 0], pos[j, 0], None])
            y_edges.extend([pos[i, 1], pos[j, 1], None])

        edge_trace = go.Scatter(
            x=x_edges, y=y_edges,
            mode='lines',
            line={'color': 'gray', 'width': edge_width},
            hoverinfo='none'
        )

        # Create node trace
        node_trace = go.Scatter(
            x=pos[:, 0], y=pos[:, 1],
            mode='markers',
            marker={
                'color': node_colors if node_colors is not None else 'red',
                'colorscale': 'Bluered',
                'size': node_size,
                'colorbar': {'title': 'Node Label'},
                'showscale': node_colors is not None
            },
            hoverinfo='none'
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="2D Graph Embedding (PyTorch)",
            xaxis={'title': 'X', 'showgrid': False, 'zeroline': False},
            yaxis={'title': 'Y', 'showgrid': False, 'zeroline': False},
            showlegend=False,
            width=800,
            height=800
        )
        fig.show()

    def _display_layout_3d(
        self,
        edge_width,
        node_size,
        node_colors
    ):
        """Display 3D graph embedding."""
        pos = self.get_positions()
        edges_np = self.edges.cpu().numpy()

        # Create edge traces
        x_edges, y_edges, z_edges = [], [], []
        for i, j in edges_np:
            x_edges.extend([pos[i, 0], pos[j, 0], None])
            y_edges.extend([pos[i, 1], pos[j, 1], None])
            z_edges.extend([pos[i, 2], pos[j, 2], None])

        edge_trace = go.Scatter3d(
            x=x_edges, y=y_edges, z=z_edges,
            mode='lines',
            line={'color': 'gray', 'width': edge_width},
            hoverinfo='none'
        )

        # Create node trace
        node_trace = go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode='markers',
            marker={
                'color': node_colors if node_colors is not None else 'red',
                'colorscale': 'Bluered',
                'size': node_size,
                'colorbar': {'title': 'Node Label'},
                'showscale': node_colors is not None
            },
            hoverinfo='none'
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="3D Graph Embedding (PyTorch)",
            scene={'xaxis': {'title': 'X'}, 'yaxis': {'title': 'Y'}, 'zaxis': {'title': 'Z'}},
            showlegend=False,
            width=800,
            height=800
        )
        fig.show()

    def __repr__(self):
        """String representation of the embedder."""
        return (f"GraphEmbedderPyTorch(n_vertices={self.n}, dimension={self.dimension}, "
                f"device={self.device}, memory_efficient={self.memory_efficient})")