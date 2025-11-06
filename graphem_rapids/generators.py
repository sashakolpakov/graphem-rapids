"""
Graph generators for Graphem.

This module provides various functions to generate different types of graphs.
All generators return sparse adjacency matrices instead of simple edge lists.
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.spatial import Delaunay


def _nx_to_sparse_adjacency(G):
    """Convert NetworkX graph to sparse adjacency matrix."""
    return nx.adjacency_matrix(G, dtype=int)


def _edges_to_sparse_adjacency(edges, n):
    """Convert edge list to sparse adjacency matrix."""
    if len(edges) == 0:
        return sp.csr_matrix((n, n), dtype=int)

    # Create undirected adjacency matrix from edges
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(rows), dtype=int)

    adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return adj


def generate_er(n, p, seed=0):
    """
    Generate a random undirected graph using the Erdős–Rényi G(n, p) model.

    Parameters:
      n: int
         Number of vertices.
      p: float
         Probability that an edge exists between any pair of vertices.
      seed: int
         Random seed for reproducibility.

    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return _nx_to_sparse_adjacency(G)


def compute_vertex_degrees(adjacency):
    """
    Compute the degree of each vertex from the adjacency matrix.

    Parameters:
      adjacency: scipy.sparse matrix
         Sparse adjacency matrix

    Returns:
      degrees: np.array of shape (n,) with degree of each vertex
    """
    # For undirected graphs, degree is sum of each row
    return np.array(adjacency.sum(axis=1)).flatten()


def generate_sbm(n_per_block=75, num_blocks=4, p_in=0.15, p_out=0.01, labels=False, seed=0):
    """
    Generate a stochastic block model graph.
    
    Parameters:
      n_per_block: int
         Number of vertices per block.
      num_blocks: int
         Number of blocks.
      p_in: float
         Probability of edge within a block.
      p_out: float
         Probability of edge between blocks.
      labels: bool
         If True, return vertex labels.
      seed: int
         Random seed for reproducibility.

    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
      labels: np.ndarray of shape (n,) (only if labels=True)
         Block labels for each vertex.
    """
    # Use NetworkX to generate the SBM
    sizes = [n_per_block] * num_blocks
    p_matrix = np.ones((num_blocks, num_blocks)) * p_out
    np.fill_diagonal(p_matrix, p_in)

    # Generate the graph
    np.random.seed(seed)
    G = nx.stochastic_block_model(sizes, p_matrix, seed=seed)

    # Convert to sparse adjacency matrix
    adjacency = _nx_to_sparse_adjacency(G)

    # Return labels if requested
    if labels:
        # Generate labels (block IDs for each vertex)
        vertex_labels = np.repeat(np.arange(num_blocks), n_per_block)
        return adjacency, vertex_labels

    return adjacency


def generate_ba(n=300, m=3, seed=0):
    """
    Generate a Barabási-Albert preferential attachment graph.
    
    Parameters:
      n: int
         Number of vertices.
      m: int
         Number of edges to attach from a new vertex to existing vertices.
      seed: int
         Random seed for reproducibility.

    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return _nx_to_sparse_adjacency(G)


def generate_ws(n=1000, k=6, p=0.3, seed=0):
    """
    Generate a Watts-Strogatz small-world graph.
    
    Parameters:
      n: int
         Number of vertices.
      k: int
         Each vertex is connected to k nearest neighbors in ring topology.
      p: float
         Probability of rewiring each edge.
      seed: int
         Random seed for reproducibility.

    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return _nx_to_sparse_adjacency(G)


def generate_power_cluster(n=1000, m=3, p=0.5, seed=0):
    """
    Generate a powerlaw cluster graph.
    
    Parameters:
      n: int
         Number of vertices.
      m: int
         Number of random edges to add per new vertex.
      p: float
         Probability of adding a triangle after adding a random edge.
      seed: int
         Random seed for reproducibility.

    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    G = nx.powerlaw_cluster_graph(n, m, p, seed=seed)
    return _nx_to_sparse_adjacency(G)


def generate_road_network(width=30, height=30):
    """
    Generate a 2D grid graph representing a road network.
    
    Parameters:
      width: int
         Width of the grid.
      height: int
         Height of the grid.

    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    G = nx.grid_2d_graph(width, height)

    # Convert node labels from (x,y) tuples to integers
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    return _nx_to_sparse_adjacency(G)


def generate_bipartite_graph(n_top=50, n_bottom=100, p=0.1, seed=0):
    """
    Generate a random bipartite graph.

    Parameters:
      n_top: int
         Number of vertices in the top set.
      n_bottom: int
         Number of vertices in the bottom set.
      p: float
         Probability of edge between any vertex in top set and any vertex in bottom set.
      seed: int
         Random seed for reproducibility.

    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    G = nx.bipartite.random_graph(n_top, n_bottom, p, seed=seed)
    return _nx_to_sparse_adjacency(G)


def generate_complete_bipartite_graph(n_top=50, n_bottom=100):
    """
    Generate a complete bipartite graph.

    In a complete bipartite graph, every vertex in the top set is connected
    to every vertex in the bottom set, resulting in n_top * n_bottom edges.

    Parameters:
      n_top: int
         Number of vertices in the top set.
      n_bottom: int
         Number of vertices in the bottom set.

    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    n_top, n_bottom = int(n_top), int(n_bottom)
    G = nx.complete_bipartite_graph(n_top, n_bottom)
    return _nx_to_sparse_adjacency(G)


def generate_balanced_tree(r=2, h=10):
    """
    Generate a balanced r-ary tree of height h.
    
    Parameters:
      r: int
         Branching factor of the tree.
      h: int
         Height of the tree.

    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    G = nx.balanced_tree(r, h)
    return _nx_to_sparse_adjacency(G)


def generate_random_regular(n=100, d=3, seed=0):
    """
    Generate a random regular graph where each node has degree d.

    Parameters:
      n: int
         Number of vertices.
      d: int
         Degree of each vertex.
      seed: int
         Random seed for reproducibility.

    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    G = nx.random_regular_graph(d, n, seed=seed)
    return _nx_to_sparse_adjacency(G)


def generate_scale_free(n=100, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0, seed=0):
    """
    Generate a scale-free graph using Holme and Kim algorithm.
    
    Parameters:
      n: int
         Number of vertices.
      alpha, beta, gamma, delta_in, delta_out: float
         Parameters for the scale-free graph generation.
      seed: int
         Random seed for reproducibility.
         
    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    G = nx.scale_free_graph(n, alpha, beta, gamma, delta_in, delta_out, seed=seed)
    # Convert to undirected graph by dropping edge directions
    G = G.to_undirected()
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))

    return _nx_to_sparse_adjacency(G)


def generate_geometric(n=100, radius=0.2, dim=2, seed=0):
    """
    Generate a random geometric graph in a unit cube.
    
    Parameters:
      n: int
         Number of vertices.
      radius: float
         Distance threshold for connecting vertices.
      dim: int
         Dimension of the space.
      seed: int
         Random seed for reproducibility.
         
    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    G = nx.random_geometric_graph(n, radius, dim=dim, seed=seed)
    return _nx_to_sparse_adjacency(G)


def generate_caveman(l=10, k=10):
    """
    Generate a caveman graph with l cliques of size k.
    
    Parameters:
      l: int
         Number of cliques.
      k: int
         Size of each clique.
         
    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    G = nx.caveman_graph(l, k)
    return _nx_to_sparse_adjacency(G)


def generate_relaxed_caveman(l=10, k=10, p=0.1, seed=0):
    """
    Generate a relaxed caveman graph with l cliques of size k,
    and a rewiring probability p.

    Parameters:
      l: int
         Number of cliques.
      k: int
         Size of each clique.
      p: float
         Rewiring probability.
      seed: int
         Random seed for reproducibility.

    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    np.random.seed(seed)
    G = nx.relaxed_caveman_graph(l, k, p)
    return _nx_to_sparse_adjacency(G)


def generate_delaunay_triangulation(n=100, seed=0):
    """
    Generate a Delaunay triangulation graph.

    Vertices are randomly placed in a 2D unit square, and edges are created
    based on the Delaunay triangulation of these points. The resulting graph
    has planar structure with triangular faces.

    Parameters:
      n: int
         Number of vertices.
      seed: int
         Random seed for reproducibility.

    Returns:
      adjacency: scipy.sparse.csr_matrix
         Sparse adjacency matrix (n × n).
    """
    # Generate random points in 2D unit square
    rng = np.random.RandomState(seed)
    pts = rng.rand(n, 2)

    # Compute Delaunay triangulation
    tri = Delaunay(pts)

    # Build graph from triangulation
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Each simplex is a triangle (i, j, k)
    for simplex in tri.simplices:
        i, j, k = simplex
        G.add_edges_from([(i, j), (j, k), (k, i)])

    return _nx_to_sparse_adjacency(G)
