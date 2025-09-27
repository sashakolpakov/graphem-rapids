"""Unit tests for graph embedder."""

import numpy as np
from graphem_rapids import create_graphem
from graphem_rapids.generators import erdos_renyi_graph, generate_random_regular


class TestEmbedder:
    """Test graph embedder functionality."""

    def test_embedder_initialization(self):
        """Test embedder initialization."""
        edges = generate_random_regular(n=50, d=4, seed=42)
        
        embedder = create_graphem(
            edges=edges,
            n_vertices=50,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(256, len(edges)),
            batch_size=1024
        )
        
        assert embedder.n == 50
        assert embedder.dimension == 2
        assert embedder.positions.shape == (50, 2)
        assert embedder.positions is not None

    def test_embedder_dimensions(self):
        """Test embedder with different dimensions."""
        edges = generate_random_regular(n=40, d=4, seed=42)
        
        for dim in [2, 3, 4]:
            embedder = create_graphem(
                edges=edges,
                n_vertices=40,
                dimension=dim,
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                knn_k=15,
                sample_size=min(200, len(edges)),
                batch_size=1024
            )
            
            assert embedder.dimension == dim
            assert embedder.positions.shape == (40, dim)

    def test_layout_execution(self):
        """Test layout algorithm execution."""
        edges = generate_random_regular(n=40, d=4, seed=42)
        
        embedder = create_graphem(
            edges=edges,
            n_vertices=40,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=10,
            sample_size=min(128, len(edges)),
            batch_size=1024
        )
        
        initial_positions = embedder.positions.copy()
        embedder.run_layout(num_iterations=3)
        
        assert not np.array_equal(initial_positions, embedder.positions)
        assert embedder.positions.shape == (40, 2)
        assert np.all(np.isfinite(embedder.positions))

    def test_disconnected_graph(self):
        """Test embedder with disconnected graph."""
        # Create two disconnected triangles
        edges = np.array([
            [0, 1], [1, 2], [2, 0],  # Triangle 1
            [3, 4], [4, 5], [5, 3]   # Triangle 2
        ])
        
        embedder = create_graphem(
            edges=edges,
            n_vertices=6,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=5,
            sample_size=min(6, len(edges)),
            batch_size=64
        )
        
        embedder.run_layout(num_iterations=2)
        assert embedder.positions.shape == (6, 2)

    def test_layout_stability(self):
        """Test that layout runs are numerically stable."""
        edges = generate_random_regular(n=30, d=4, seed=42)
        
        embedder = create_graphem(
            edges=edges,
            n_vertices=30,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(64, len(edges)),
            batch_size=1024
        )
        
        for _ in range(3):
            embedder.run_layout(num_iterations=2)
            
            assert np.all(np.isfinite(embedder.positions))
            
            max_coord = np.max(np.abs(embedder.positions))
            assert max_coord < 1000  # Reasonable bound

    def test_large_graphs(self):
        """Test embedder with large graphs."""
        edges = erdos_renyi_graph(n=200, p=0.02, seed=42)
        
        embedder = create_graphem(
            edges=edges,
            n_vertices=200,
            dimension=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            knn_k=15,
            sample_size=min(512, len(edges)),
            batch_size=1024
        )
        
        assert embedder.positions.shape == (200, 2)
        assert np.all(np.isfinite(embedder.positions))