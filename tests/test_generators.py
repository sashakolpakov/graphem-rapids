"""Unit tests for graph generators."""

import pytest
import numpy as np
import scipy.sparse as sp
from graphem_rapids.generators import (
    generate_er,
    generate_random_regular,
    generate_scale_free,
    generate_geometric,
    generate_caveman,
    generate_relaxed_caveman,
    generate_ws,
    generate_ba,
    generate_sbm,
    generate_bipartite_graph,
    generate_complete_bipartite_graph,
    generate_delaunay_triangulation
)


class TestGenerators:
    """Test graph generators."""

    @pytest.mark.fast
    def test_generate_er(self):
        """Test Erdős-Rényi graph generator."""
        n, p = 50, 0.1
        adjacency = generate_er(n=n, p=p, seed=42)

        assert sp.issparse(adjacency)
        assert adjacency.shape == (n, n)
        assert adjacency.dtype in [np.int32, np.int64]

        # Check adjacency matrix is symmetric for undirected graph
        assert (adjacency != adjacency.T).nnz == 0

        # Check no self loops (diagonal should be zero)
        assert adjacency.diagonal().sum() == 0

    @pytest.mark.fast
    def test_random_regular_graph(self):
        """Test random regular graph generator."""
        n, d = 20, 3
        adjacency = generate_random_regular(n=n, d=d, seed=42)

        assert sp.issparse(adjacency)
        assert adjacency.shape == (n, n)
        assert adjacency.dtype in [np.int32, np.int64]

        # Check each node has degree d
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        assert np.all(degrees == d)

        # Check symmetry
        assert (adjacency != adjacency.T).nnz == 0

    @pytest.mark.fast
    def test_scale_free_graph(self):
        """Test scale-free graph generator."""
        n = 50
        adjacency = generate_scale_free(n=n, seed=42)

        assert sp.issparse(adjacency)
        assert adjacency.shape == (n, n)
        assert adjacency.dtype in [np.int32, np.int64]

        # Check symmetry (should be undirected)
        assert (adjacency != adjacency.T).nnz == 0

    @pytest.mark.fast
    def test_geometric_graph(self):
        """Test geometric graph generator."""
        n = 30
        adjacency = generate_geometric(n=n, radius=0.3, seed=42)

        assert sp.issparse(adjacency)
        assert adjacency.shape == (n, n)
        assert adjacency.dtype in [np.int32, np.int64]

        # Check symmetry
        assert (adjacency != adjacency.T).nnz == 0

    @pytest.mark.fast
    def test_caveman_graph(self):
        """Test caveman graph generator."""
        l, k = 5, 4
        adjacency = generate_caveman(l=l, k=k)
        expected_n = l * k

        assert sp.issparse(adjacency)
        assert adjacency.shape == (expected_n, expected_n)
        assert adjacency.dtype in [np.int32, np.int64]

    @pytest.mark.fast
    def test_relaxed_caveman_graph(self):
        """Test relaxed caveman graph generator."""
        l, k = 5, 4
        adjacency = generate_relaxed_caveman(l=l, k=k, p=0.1, seed=42)
        expected_n = l * k

        assert sp.issparse(adjacency)
        assert adjacency.shape == (expected_n, expected_n)
        assert adjacency.dtype in [np.int32, np.int64]

    @pytest.mark.fast
    def test_watts_strogatz_graph(self):
        """Test Watts-Strogatz graph generator."""
        n, k, p = 20, 4, 0.3
        adjacency = generate_ws(n=n, k=k, p=p, seed=42)

        assert sp.issparse(adjacency)
        assert adjacency.shape == (n, n)
        assert adjacency.dtype in [np.int32, np.int64]

    @pytest.mark.fast
    def test_barabasi_albert_graph(self):
        """Test Barabási-Albert graph generator."""
        n, m = 30, 3
        adjacency = generate_ba(n=n, m=m, seed=42)

        assert sp.issparse(adjacency)
        assert adjacency.shape == (n, n)
        assert adjacency.dtype in [np.int32, np.int64]

    @pytest.mark.fast
    def test_stochastic_block_model(self):
        """Test stochastic block model generator."""
        n_per_block, num_blocks = 15, 3
        adjacency = generate_sbm(
            n_per_block=n_per_block,
            num_blocks=num_blocks,
            p_in=0.8,
            p_out=0.1,
            seed=42
        )
        expected_n = n_per_block * num_blocks

        assert sp.issparse(adjacency)
        assert adjacency.shape == (expected_n, expected_n)
        assert adjacency.dtype in [np.int32, np.int64]

        # Test with labels
        adjacency, labels = generate_sbm(
            n_per_block=n_per_block,
            num_blocks=num_blocks,
            labels=True,
            seed=42
        )
        assert len(labels) == expected_n
        assert len(np.unique(labels)) == num_blocks

    @pytest.mark.fast
    def test_reproducible_results(self):
        """Test that generators produce reproducible results with same seed."""
        adj1 = generate_er(n=20, p=0.2, seed=123)
        adj2 = generate_er(n=20, p=0.2, seed=123)

        # Should be identical
        assert (adj1 != adj2).nnz == 0

    @pytest.mark.fast
    def test_different_seeds(self):
        """Test that different seeds produce different graphs."""
        adj1 = generate_er(n=30, p=0.2, seed=1)
        adj2 = generate_er(n=30, p=0.2, seed=2)

        # Should be different (with high probability)
        assert (adj1 != adj2).nnz > 0

    @pytest.mark.fast
    def test_bipartite_graph(self):
        """Test random bipartite graph generator."""
        n_top, n_bottom = 20, 30
        adjacency = generate_bipartite_graph(n_top=n_top, n_bottom=n_bottom, p=0.2, seed=42)
        expected_n = n_top + n_bottom

        assert sp.issparse(adjacency)
        assert adjacency.shape == (expected_n, expected_n)
        assert adjacency.dtype in [np.int32, np.int64]

        # Check symmetry
        assert (adjacency != adjacency.T).nnz == 0

        # Check bipartite structure: no edges within top or bottom sets
        # Top set: nodes 0 to n_top-1, Bottom set: nodes n_top to n_top+n_bottom-1
        top_submatrix = adjacency[:n_top, :n_top]
        bottom_submatrix = adjacency[n_top:, n_top:]
        assert top_submatrix.nnz == 0, "Top set should have no internal edges"
        assert bottom_submatrix.nnz == 0, "Bottom set should have no internal edges"

        # Check that cross edges exist (with p=0.2, very likely)
        cross_edges = adjacency[:n_top, n_top:].nnz
        assert cross_edges > 0, "Should have edges between top and bottom sets"

    @pytest.mark.fast
    def test_bipartite_graph_reproducibility(self):
        """Test that bipartite graph generator produces reproducible results."""
        adj1 = generate_bipartite_graph(n_top=15, n_bottom=25, p=0.3, seed=123)
        adj2 = generate_bipartite_graph(n_top=15, n_bottom=25, p=0.3, seed=123)

        # Should be identical
        assert (adj1 != adj2).nnz == 0

    @pytest.mark.fast
    def test_bipartite_graph_different_densities(self):
        """Test bipartite graph with different edge probabilities."""
        n_top, n_bottom = 10, 15

        # Sparse graph
        adj_sparse = generate_bipartite_graph(n_top=n_top, n_bottom=n_bottom, p=0.1, seed=42)

        # Dense graph
        adj_dense = generate_bipartite_graph(n_top=n_top, n_bottom=n_bottom, p=0.8, seed=42)

        # Dense graph should have more edges
        assert adj_dense.nnz > adj_sparse.nnz

    @pytest.mark.fast
    def test_complete_bipartite_graph(self):
        """Test complete bipartite graph generator."""
        n_top, n_bottom = 10, 15
        adjacency = generate_complete_bipartite_graph(n_top=n_top, n_bottom=n_bottom)
        expected_n = n_top + n_bottom
        expected_edges = n_top * n_bottom * 2  # Each edge counted twice in undirected graph

        assert sp.issparse(adjacency)
        assert adjacency.shape == (expected_n, expected_n)
        assert adjacency.dtype in [np.int32, np.int64]

        # Check symmetry
        assert (adjacency != adjacency.T).nnz == 0

        # Check bipartite structure: no edges within top or bottom sets
        top_submatrix = adjacency[:n_top, :n_top]
        bottom_submatrix = adjacency[n_top:, n_top:]
        assert top_submatrix.nnz == 0, "Top set should have no internal edges"
        assert bottom_submatrix.nnz == 0, "Bottom set should have no internal edges"

        # Check complete: every node in top connects to every node in bottom
        cross_matrix = adjacency[:n_top, n_top:]
        assert cross_matrix.nnz == n_top * n_bottom, "Should have all possible cross edges"

        # Total edges (counting both directions)
        assert adjacency.nnz == expected_edges

    @pytest.mark.fast
    def test_complete_bipartite_graph_degrees(self):
        """Test that complete bipartite graph has correct degrees."""
        n_top, n_bottom = 8, 12
        adjacency = generate_complete_bipartite_graph(n_top=n_top, n_bottom=n_bottom)

        degrees = np.array(adjacency.sum(axis=1)).flatten()

        # Top set nodes should each have degree n_bottom
        top_degrees = degrees[:n_top]
        assert np.all(top_degrees == n_bottom)

        # Bottom set nodes should each have degree n_top
        bottom_degrees = degrees[n_top:]
        assert np.all(bottom_degrees == n_top)

    @pytest.mark.fast
    def test_delaunay_triangulation(self):
        """Test Delaunay triangulation graph generator."""
        n = 50
        adjacency = generate_delaunay_triangulation(n=n, seed=42)

        assert sp.issparse(adjacency)
        assert adjacency.shape == (n, n)
        assert adjacency.dtype in [np.int32, np.int64]

        # Check symmetry
        assert (adjacency != adjacency.T).nnz == 0

        # Check no self loops
        assert adjacency.diagonal().sum() == 0

        # Delaunay triangulation should be connected for random points
        # Check that there are edges
        assert adjacency.nnz > 0

    @pytest.mark.fast
    def test_delaunay_triangulation_reproducibility(self):
        """Test that Delaunay triangulation is reproducible with same seed."""
        adj1 = generate_delaunay_triangulation(n=30, seed=123)
        adj2 = generate_delaunay_triangulation(n=30, seed=123)

        # Should be identical
        assert (adj1 != adj2).nnz == 0

    @pytest.mark.fast
    def test_delaunay_triangulation_planarity(self):
        """Test that Delaunay triangulation produces planar graph."""
        n = 40
        adjacency = generate_delaunay_triangulation(n=n, seed=42)

        # For a planar graph with n vertices: edges <= 3n - 6
        num_edges = adjacency.nnz // 2  # Divide by 2 for undirected graph
        assert num_edges <= 3 * n - 6, "Delaunay triangulation should produce planar graph"

    @pytest.mark.fast
    def test_adjacency_format(self):
        """Test that all generators return sparse adjacency matrices."""
        generators_and_params = [
            (generate_er, {'n': 10, 'p': 0.3}),
            (generate_random_regular, {'n': 10, 'd': 3}),
            (generate_scale_free, {'n': 20}),
            (generate_geometric, {'n': 15, 'radius': 0.4}),
            (generate_ws, {'n': 12, 'k': 4, 'p': 0.2}),
            (generate_ba, {'n': 15, 'm': 2}),
        ]

        for generator, params in generators_and_params:
            adjacency = generator(seed=42, **params)

            # Check it's sparse
            assert sp.issparse(adjacency)

            # Check it's square
            assert adjacency.shape[0] == adjacency.shape[1]

            # Check data type
            assert adjacency.dtype in [np.int32, np.int64]