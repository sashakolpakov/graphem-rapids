"""Comprehensive integration tests for GraphEm Rapids."""

import pytest
import numpy as np
import torch
from graphem_rapids import create_graphem
from graphem_rapids.backends.embedder_pytorch import GraphEmbedderPyTorch
from graphem_rapids.generators import (
    erdos_renyi_graph,
    generate_random_regular,
    generate_scale_free
)


class TestEndToEndIntegration:
    """Test complete end-to-end workflows."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_pytorch_pipeline(self):
        """Test complete PyTorch pipeline from graph generation to embedding."""
        # Generate a medium-sized graph
        adjacency =erdos_renyi_graph(n=100, p=0.05, seed=42)

        # Create embedder
        embedder = create_graphem(
            adjacency=adjacency,
            n_components=3,
            backend='pytorch',
            L_min=5.0,
            k_attr=0.3,
            k_inter=0.1,
            verbose=False
        )

        # Run complete layout
        final_positions = embedder.run_layout(num_iterations=10)

        # Validate results
        assert final_positions.shape == (100, 3)
        assert np.all(np.isfinite(final_positions))

        # Check that embedding has reasonable properties
        distances = np.linalg.norm(final_positions, axis=1)
        assert np.std(distances) > 0.1  # Points are not all at origin
        assert np.max(distances) < 100  # Points are not too far apart

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.gpu
    def test_complete_cuda_pipeline(self):
        """Test complete CUDA pipeline."""
        adjacency =generate_random_regular(n=80, d=6, seed=42)

        embedder = create_graphem(
            adjacency=adjacency,
            n_components=2,
            backend='pytorch',
            device='cuda',
            verbose=False
        )

        final_positions = embedder.run_layout(num_iterations=8)

        assert final_positions.shape == (80, 2)
        assert np.all(np.isfinite(final_positions))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_different_graph_types(self):
        """Test embedding different types of graphs."""
        test_cases = [
            ("Random Regular", generate_random_regular, {"n": 50, "d": 4, "seed": 42}),
            ("Scale-Free", generate_scale_free, {"n": 60, "seed": 42}),
            ("Erdős-Rényi", erdos_renyi_graph, {"n": 40, "p": 0.1, "seed": 42}),
        ]

        for graph_name, generator, params in test_cases:
            adjacency =generator(**params)
            n_vertices = params.get('n', adjacency.shape[0])

            embedder = create_graphem(
                adjacency=adjacency,
                n_components=2,
                backend='pytorch',
                verbose=False
            )

            positions = embedder.run_layout(num_iterations=5)

            # Basic validation
            assert positions.shape[0] == n_vertices
            assert positions.shape[1] == 2
            assert np.all(np.isfinite(positions)), f"Non-finite positions in {graph_name}"

            # Check that embedding preserves some graph structure
            # Extract edges from adjacency matrix
            edges = np.array(np.nonzero(adjacency)).T
            if len(edges) > 0:
                edge_lengths = []
                for i, j in edges[:10]:  # Sample first 10 edges
                    if i < n_vertices and j < n_vertices:
                        dist = np.linalg.norm(positions[i] - positions[j])
                        edge_lengths.append(dist)

                if edge_lengths:
                    mean_edge_length = np.mean(edge_lengths)
                    assert mean_edge_length > 0, f"Zero edge lengths in {graph_name}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_dimension_consistency(self):
        """Test that embeddings work consistently across dimensions."""
        adjacency =erdos_renyi_graph(n=50, p=0.08, seed=42)

        dimensions = [2, 3, 4, 5]
        embeddings = {}

        for dim in dimensions:
            embedder = create_graphem(
                adjacency=adjacency,
                n_components=dim,
                backend='pytorch',
                verbose=False
            )

            positions = embedder.run_layout(num_iterations=5)
            embeddings[dim] = positions

            # Check dimensionality
            assert positions.shape == (50, dim)
            assert np.all(np.isfinite(positions))

            # Check that higher dimensions don't collapse to lower ones
            if dim > 2:
                variance_per_dim = np.var(positions, axis=0)
                assert np.all(variance_per_dim > 1e-6), f"Dimension collapse in {dim}D"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_parameter_sensitivity(self):
        """Test that different parameters produce different but valid embeddings."""
        adjacency =generate_random_regular(n=40, d=4, seed=42)

        parameter_sets = [
            {"L_min": 1.0, "k_attr": 0.1, "k_inter": 0.05},
            {"L_min": 5.0, "k_attr": 0.5, "k_inter": 0.2},
            {"L_min": 10.0, "k_attr": 1.0, "k_inter": 0.5},
        ]

        embeddings = []

        for i, params in enumerate(parameter_sets):
            embedder = create_graphem(
                adjacency=adjacency,
                n_components=2,
                backend='pytorch',
                **params,
                verbose=False
            )

            positions = embedder.run_layout(num_iterations=5)
            embeddings.append(positions)

            assert positions.shape == (40, 2)
            assert np.all(np.isfinite(positions))

        # Check that different parameters produce different results
        for i, embedding_i in enumerate(embeddings):
            for j in range(i + 1, len(embeddings)):
                # Embeddings should be different (not identical)
                diff = np.mean(np.abs(embedding_i - embeddings[j]))
                assert diff > 1e-3, f"Parameters {i} and {j} produce too similar results"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_small_graphs(self):
        """Test integration with small graphs."""
        # 6-vertex graph (two connected triangles) - large enough for k-NN
        adjacency = np.array([
            [0, 1, 1, 0, 0, 1],  # vertex 0: connected to 1,2,5
            [1, 0, 1, 1, 0, 0],  # vertex 1: connected to 0,2,3
            [1, 1, 0, 0, 1, 0],  # vertex 2: connected to 0,1,4
            [0, 1, 0, 0, 1, 1],  # vertex 3: connected to 1,4,5
            [0, 0, 1, 1, 0, 1],  # vertex 4: connected to 2,3,5
            [1, 0, 0, 1, 1, 0]   # vertex 5: connected to 0,3,4
        ])

        embedder = create_graphem(
            adjacency=adjacency,
            n_components=2,
            backend='pytorch',
            n_neighbors=3,  # Safe k for 6-vertex graph
            verbose=False
        )

        positions = embedder.run_layout(num_iterations=3)

        assert positions.shape == (6, 2)
        assert np.all(np.isfinite(positions))

        # Check that vertices are reasonably spaced
        distances = []
        for i in range(6):
            for j in range(i + 1, 6):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)

        mean_distance = np.mean(distances)
        assert mean_distance > 1e-3  # Not collapsed to a point

    @pytest.mark.integration
    @pytest.mark.slow
    def test_reproducibility_integration(self):
        """Test end-to-end reproducibility."""
        adjacency = generate_scale_free(n=60, seed=42)

        # Run same embedding twice with consistent seeding
        results = []
        for _ in range(2):
            # Set seeds before creating embedder (critical for reproducibility)
            np.random.seed(123)
            torch.manual_seed(123)

            embedder = create_graphem(
                adjacency=adjacency,
                n_components=2,
                backend='pytorch',
                L_min=5.0,
                k_attr=0.3,
                verbose=False
            )

            positions = embedder.run_layout(num_iterations=5)
            results.append(positions)

        # Results should be very similar up to reflections in each axis
        # Check all 4 possible sign combinations: [1,1], [1,-1], [-1,1], [-1,-1]
        min_diff = float('inf')
        for sign_x in [1, -1]:
            for sign_y in [1, -1]:
                reflected = results[1].copy()
                reflected[:, 0] *= sign_x
                reflected[:, 1] *= sign_y
                diff = np.mean(np.abs(results[0] - reflected))
                min_diff = min(min_diff, diff)

        assert min_diff < 1e-2, f"Embeddings are not reproducible (best match diff: {min_diff})"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_memory_efficiency(self):
        """Test that memory-efficient mode works for larger graphs."""
        adjacency =erdos_renyi_graph(n=200, p=0.02, seed=42)

        embedder = create_graphem(
            adjacency=adjacency,
            n_components=3,
            backend='pytorch',
            memory_efficient=True,
            batch_size=64,  # Small batch for memory efficiency
            sample_size=100,  # Smaller sample size
            verbose=False
        )

        positions = embedder.run_layout(num_iterations=5)

        assert positions.shape == (200, 3)
        assert np.all(np.isfinite(positions))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_disconnected_components_integration(self):
        """Test integration with graphs having multiple components."""
        # Create two separate components as edge list (12 vertices total)
        # Component 1: hexagon (6 vertices)
        component1 = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])
        # Component 2: hexagon (6 vertices)
        component2 = np.array([[6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 6]])
        edges = np.vstack([component1, component2])

        # Convert to adjacency matrix
        import scipy.sparse as sp  # pylint: disable=import-outside-toplevel
        n_vertices = 12
        adjacency = sp.csr_matrix(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
            shape=(n_vertices, n_vertices)
        )
        # Make symmetric for undirected graph
        adjacency = adjacency + adjacency.T

        embedder = create_graphem(
            adjacency=adjacency,
            n_components=2,
            backend='pytorch',
            verbose=False
        )

        positions = embedder.run_layout(num_iterations=8)

        assert positions.shape == (12, 2)
        assert np.all(np.isfinite(positions))

        # Check that components are reasonably separated
        comp1_center = np.mean(positions[:6], axis=0)
        comp2_center = np.mean(positions[6:], axis=0)
        separation = np.linalg.norm(comp1_center - comp2_center)

        # Components should be somewhat separated
        assert separation > 1e-2


class TestCrossBackendConsistency:
    """Test consistency across different backends (when available)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pytorch_vs_auto_selection(self):
        """Test that auto-selection produces reasonable results."""
        adjacency =generate_random_regular(n=50, d=4, seed=42)

        # Explicit PyTorch
        embedder_explicit = GraphEmbedderPyTorch(
            adjacency=adjacency,
            n_components=2,
            verbose=False
        )
        positions_explicit = embedder_explicit.run_layout(num_iterations=5)

        # Auto selection (should pick PyTorch for this size)
        embedder_auto = create_graphem(
            adjacency=adjacency,
            n_components=2,
            backend='auto',
            verbose=False
        )
        positions_auto = embedder_auto.run_layout(num_iterations=5)

        # Both should produce valid results
        assert positions_explicit.shape == (50, 2)
        assert positions_auto.shape == (50, 2)
        assert np.all(np.isfinite(positions_explicit))
        assert np.all(np.isfinite(positions_auto))


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""

    @pytest.mark.integration
    @pytest.mark.fast
    def test_invalid_graph_data(self):
        """Test handling of invalid graph data."""
        # Test with invalid adjacency matrix (non-square)
        invalid_adjacency = np.array([[0, 1], [1, 0], [0, 1]])  # 3x2 matrix (non-square)

        with pytest.raises((ValueError, IndexError)):
            embedder = create_graphem(
                adjacency=invalid_adjacency,
                n_components=2,
                backend='pytorch',
                verbose=False
            )
            embedder.run_layout(num_iterations=1)

    @pytest.mark.integration
    @pytest.mark.fast
    def test_empty_graph_handling(self):
        """Test handling of empty graphs."""
        # Empty adjacency matrix (all zeros - no edges)
        empty_adjacency = np.zeros((5, 5))

        # Empty graphs should raise ValueError as invalid input
        with pytest.raises((ValueError, RuntimeError)):
            embedder = create_graphem(
                adjacency=empty_adjacency,
                n_components=2,
                backend='pytorch',
                verbose=False
            )
            # If embedder creation succeeds, running layout should fail
            embedder.run_layout(num_iterations=1)

    @pytest.mark.integration
    @pytest.mark.fast
    def test_parameter_edge_cases(self):
        """Test parameter edge cases."""
        adjacency =generate_random_regular(n=20, d=3, seed=42)

        # Very small parameters
        embedder = create_graphem(
            adjacency=adjacency,
            n_components=2,
            backend='pytorch',
            L_min=0.1,
            k_attr=0.01,
            k_inter=0.01,
            verbose=False
        )

        positions = embedder.run_layout(num_iterations=2)
        assert positions.shape == (20, 2)
        assert np.all(np.isfinite(positions))