"""Unit tests for graph embedder."""

import pytest
import numpy as np
from graphem_rapids import create_graphem
from graphem_rapids.generators import generate_er, generate_random_regular


class TestEmbedder:
    """Test graph embedder functionality."""

    @pytest.mark.fast
    def test_embedder_initialization(self):
        """Test embedder initialization."""
        adjacency = generate_random_regular(n=50, d=4, seed=42)
        
        embedder = create_graphem(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=256,
        )
        
        assert embedder.n == 50
        assert embedder.n_components == 2
        assert embedder.positions.shape == (50, 2)
        assert embedder.positions is not None

    @pytest.mark.fast
    def test_embedder_dimensions(self):
        """Test embedder with different dimensions."""
        adjacency = generate_random_regular(n=40, d=4, seed=42)
        
        for dim in [2, 3, 4]:
            embedder = create_graphem(
                adjacency=adjacency,
                n_components=dim,
                L_min=10.0,
                k_attr=0.5,
                k_inter=0.1,
                n_neighbors=15,
                sample_size=200,
                )
            
            assert embedder.n_components == dim
            assert embedder.positions.shape == (40, dim)

    @pytest.mark.fast
    def test_layout_execution(self):
        """Test layout algorithm execution."""
        adjacency = generate_random_regular(n=40, d=4, seed=42)
        
        embedder = create_graphem(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=10,
            sample_size=128,
        )
        
        initial_positions = embedder.positions.copy()
        embedder.run_layout(num_iterations=3)
        
        assert not np.array_equal(initial_positions, embedder.positions)
        assert embedder.positions.shape == (40, 2)
        assert np.all(np.isfinite(embedder.positions))

    @pytest.mark.fast
    def test_disconnected_graph(self):
        """Test embedder with disconnected graph."""
        # Create two disconnected triangles using dense adjacency matrix
        adjacency = np.array([
            [0, 1, 1, 0, 0, 0],  # vertex 0: connected to 1,2
            [1, 0, 1, 0, 0, 0],  # vertex 1: connected to 0,2
            [1, 1, 0, 0, 0, 0],  # vertex 2: connected to 0,1
            [0, 0, 0, 0, 1, 1],  # vertex 3: connected to 4,5
            [0, 0, 0, 1, 0, 1],  # vertex 4: connected to 3,5
            [0, 0, 0, 1, 1, 0],  # vertex 5: connected to 3,4
        ])

        embedder = create_graphem(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=5,
            sample_size=6,
        )
        
        embedder.run_layout(num_iterations=2)
        assert embedder.positions.shape == (6, 2)

    @pytest.mark.fast
    def test_layout_stability(self):
        """Test that layout runs are numerically stable."""
        adjacency = generate_random_regular(n=30, d=4, seed=42)
        
        embedder = create_graphem(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=64,
        )
        
        for _ in range(3):
            embedder.run_layout(num_iterations=2)
            
            assert np.all(np.isfinite(embedder.positions))
            
            max_coord = np.max(np.abs(embedder.positions))
            assert max_coord < 1000  # Reasonable bound

    @pytest.mark.slow
    def test_large_graphs(self):
        """Test embedder with large graphs."""
        adjacency = generate_er(n=200, p=0.02, seed=42)
        
        embedder = create_graphem(
            adjacency=adjacency,
            n_components=2,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=512,
        )
        
        assert embedder.positions.shape == (200, 2)
        assert np.all(np.isfinite(embedder.positions))