"""Unit tests for graph generators."""

import pytest
import numpy as np
import networkx as nx
from graphem_rapids.generators import (
    erdos_renyi_graph,
    generate_random_regular,
    generate_scale_free,
    generate_geometric,
    generate_caveman,
    generate_relaxed_caveman,
    generate_ws,
    generate_ba,
    generate_sbm
)


class TestGenerators:
    """Test graph generators."""

    @pytest.mark.fast
    def test_erdos_renyi_graph(self):
        """Test Erdős-Rényi graph generator."""
        n, p = 50, 0.1
        edges = erdos_renyi_graph(n=n, p=p, seed=42)
        
        assert isinstance(edges, np.ndarray)
        assert edges.shape[1] == 2
        assert edges.dtype in [np.int32, np.int64]
        
        if len(edges) > 0:
            assert np.min(edges) >= 0
            assert np.max(edges) < n
        
        assert not np.any(edges[:, 0] == edges[:, 1])
        assert 0 <= len(edges) <= n * (n - 1) // 2

    @pytest.mark.fast
    def test_random_regular_graph(self):
        """Test random regular graph generator."""
        n, d = 20, 3
        edges = generate_random_regular(n=n, d=d, seed=42)
        
        assert isinstance(edges, np.ndarray)
        assert edges.shape[1] == 2
        
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        
        degrees = dict(G.degree())
        for node in range(n):
            assert degrees[node] == d

    @pytest.mark.fast
    def test_scale_free_graph(self):
        """Test scale-free graph generator."""
        n = 50
        edges = generate_scale_free(n=n, seed=42)
        
        assert isinstance(edges, np.ndarray)
        assert edges.shape[1] == 2
        
        if len(edges) > 0:
            assert np.min(edges) >= 0
            assert np.max(edges) < n

    @pytest.mark.fast
    def test_geometric_graph(self):
        """Test random geometric graph generator."""
        n, radius = 30, 0.3
        edges = generate_geometric(n=n, radius=radius, seed=42)
        
        assert isinstance(edges, np.ndarray)
        assert edges.shape[1] == 2
        
        if len(edges) > 0:
            assert np.min(edges) >= 0
            assert np.max(edges) < n

    @pytest.mark.fast
    def test_caveman_graph(self):
        """Test caveman graph generator."""
        l, k = 3, 5
        edges = generate_caveman(l=l, k=k)
        
        assert isinstance(edges, np.ndarray)
        assert edges.shape[1] == 2
        
        total_vertices = l * k
        if len(edges) > 0:
            assert np.min(edges) >= 0
            assert np.max(edges) < total_vertices

    @pytest.mark.fast
    def test_relaxed_caveman_graph(self):
        """Test relaxed caveman graph generator."""
        l, k, p = 3, 5, 0.1
        edges = generate_relaxed_caveman(l=l, k=k, p=p, seed=42)
        
        assert isinstance(edges, np.ndarray)
        assert edges.shape[1] == 2
        
        total_vertices = l * k
        if len(edges) > 0:
            assert np.min(edges) >= 0
            assert np.max(edges) < total_vertices

    @pytest.mark.fast
    def test_watts_strogatz_graph(self):
        """Test Watts-Strogatz small-world graph generator."""
        n, k, p = 20, 4, 0.3
        edges = generate_ws(n=n, k=k, p=p, seed=42)
        
        assert isinstance(edges, np.ndarray)
        assert edges.shape[1] == 2
        
        if len(edges) > 0:
            assert np.min(edges) >= 0
            assert np.max(edges) < n

    @pytest.mark.fast
    def test_barabasi_albert_graph(self):
        """Test Barabási-Albert graph generator."""
        n, m = 50, 2
        edges = generate_ba(n=n, m=m, seed=42)
        
        assert isinstance(edges, np.ndarray)
        assert edges.shape[1] == 2
        
        if len(edges) > 0:
            assert np.min(edges) >= 0
            assert np.max(edges) < n

    @pytest.mark.fast
    def test_stochastic_block_model(self):
        """Test Stochastic Block Model generator."""
        n_per_block, num_blocks = 10, 3
        p_in, p_out = 0.8, 0.1
        edges = generate_sbm(
            n_per_block=n_per_block, 
            num_blocks=num_blocks, 
            p_in=p_in, 
            p_out=p_out, 
            seed=42
        )
        
        assert isinstance(edges, np.ndarray)
        assert edges.shape[1] == 2
        
        total_vertices = n_per_block * num_blocks
        if len(edges) > 0:
            assert np.min(edges) >= 0
            assert np.max(edges) < total_vertices


    @pytest.mark.fast
    def test_reproducible_results(self):
        """Test that generators produce reproducible results with same seed."""
        n, p = 30, 0.2
        
        edges1 = erdos_renyi_graph(n=n, p=p, seed=123)
        edges2 = erdos_renyi_graph(n=n, p=p, seed=123)
        
        np.testing.assert_array_equal(edges1, edges2)

    @pytest.mark.fast
    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        n, p = 30, 0.3
        
        edges1 = erdos_renyi_graph(n=n, p=p, seed=123)
        edges2 = erdos_renyi_graph(n=n, p=p, seed=456)
        
        if len(edges1) > 0 and len(edges2) > 0:
            assert not np.array_equal(edges1, edges2)

    @pytest.mark.fast
    def test_edge_format(self):
        """Test that all generators return edges in consistent format."""
        generators_params = [
            (erdos_renyi_graph, {"n": 20, "p": 0.1, "seed": 42}),
            (generate_random_regular, {"n": 20, "d": 3, "seed": 42}),
            (generate_ws, {"n": 20, "k": 4, "p": 0.3, "seed": 42}),
            (generate_ba, {"n": 20, "m": 2, "seed": 42}),
        ]
        
        for generator, params in generators_params:
            edges = generator(**params)
            
            assert isinstance(edges, np.ndarray)
            if len(edges) > 0:
                assert edges.shape[1] == 2
                assert edges.ndim == 2
                assert np.issubdtype(edges.dtype, np.integer)