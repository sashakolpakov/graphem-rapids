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
        edges = erdos_renyi_graph(n=100, p=0.05, seed=42)

        # Create embedder
        embedder = create_graphem(
            edges=edges,
            n_vertices=100,
            dimension=3,
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
        edges = generate_random_regular(n=80, d=6, seed=42)

        embedder = create_graphem(
            edges=edges,
            n_vertices=80,
            dimension=2,
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
            edges = generator(**params)
            n_vertices = params.get('n', len(np.unique(edges)))

            embedder = create_graphem(
                edges=edges,
                n_vertices=n_vertices,
                dimension=2,
                backend='pytorch',
                verbose=False
            )

            positions = embedder.run_layout(num_iterations=5)

            # Basic validation
            assert positions.shape[0] == n_vertices
            assert positions.shape[1] == 2
            assert np.all(np.isfinite(positions)), f"Non-finite positions in {graph_name}"

            # Check that embedding preserves some graph structure
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
        edges = erdos_renyi_graph(n=50, p=0.08, seed=42)

        dimensions = [2, 3, 4, 5]
        embeddings = {}

        for dim in dimensions:
            embedder = create_graphem(
                edges=edges,
                n_vertices=50,
                dimension=dim,
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
        edges = generate_random_regular(n=40, d=4, seed=42)

        parameter_sets = [
            {"L_min": 1.0, "k_attr": 0.1, "k_inter": 0.05},
            {"L_min": 5.0, "k_attr": 0.5, "k_inter": 0.2},
            {"L_min": 10.0, "k_attr": 1.0, "k_inter": 0.5},
        ]

        embeddings = []

        for i, params in enumerate(parameter_sets):
            embedder = create_graphem(
                edges=edges,
                n_vertices=40,
                dimension=2,
                backend='pytorch',
                **params,
                verbose=False
            )

            positions = embedder.run_layout(num_iterations=5)
            embeddings.append(positions)

            assert positions.shape == (40, 2)
            assert np.all(np.isfinite(positions))

        # Check that different parameters produce different results
        for i, _ in enumerate(embeddings):
            for j in range(i + 1, len(embeddings)):
                # Embeddings should be different (not identical)
                diff = np.mean(np.abs(embeddings[i] - embeddings[j]))
                assert diff > 1e-3, f"Parameters {i} and {j} produce too similar results"

    @pytest.mark.integration
    @pytest.mark.fast
    def test_small_graphs(self):
        """Test integration with very small graphs."""
        # Triangle
        edges = np.array([[0, 1], [1, 2], [2, 0]])

        embedder = create_graphem(
            edges=edges,
            n_vertices=3,
            dimension=2,
            backend='pytorch',
            knn_k=2,  # Small k for small graph
            verbose=False
        )

        positions = embedder.run_layout(num_iterations=3)

        assert positions.shape == (3, 2)
        assert np.all(np.isfinite(positions))

        # Check that triangle vertices are reasonably spaced
        distances = []
        for i in range(3):
            for j in range(i + 1, 3):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)

        mean_distance = np.mean(distances)
        assert mean_distance > 1e-3  # Not collapsed to a point

    @pytest.mark.integration
    @pytest.mark.slow
    def test_reproducibility_integration(self):
        """Test end-to-end reproducibility."""
        edges = generate_scale_free(n=60, seed=42)

        # Run same embedding twice
        results = []
        for _ in range(2):
            embedder = create_graphem(
                edges=edges,
                n_vertices=60,
                dimension=2,
                backend='pytorch',
                L_min=5.0,
                k_attr=0.3,
                verbose=False
            )

            positions = embedder.run_layout(num_iterations=5)
            results.append(positions)

        # Results should be very similar (allowing for minor numerical differences)
        diff = np.mean(np.abs(results[0] - results[1]))
        assert diff < 1e-2, "Embeddings are not reproducible"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_memory_efficiency(self):
        """Test that memory-efficient mode works for larger graphs."""
        edges = erdos_renyi_graph(n=200, p=0.02, seed=42)

        embedder = create_graphem(
            edges=edges,
            n_vertices=200,
            dimension=3,
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
        # Create two separate components
        component1 = np.array([[0, 1], [1, 2], [2, 0]])  # Triangle
        component2 = np.array([[3, 4], [4, 5], [5, 6], [6, 3]])  # Square
        edges = np.vstack([component1, component2])

        embedder = create_graphem(
            edges=edges,
            n_vertices=7,
            dimension=2,
            backend='pytorch',
            verbose=False
        )

        positions = embedder.run_layout(num_iterations=8)

        assert positions.shape == (7, 2)
        assert np.all(np.isfinite(positions))

        # Check that components are reasonably separated
        comp1_center = np.mean(positions[:3], axis=0)
        comp2_center = np.mean(positions[3:], axis=0)
        separation = np.linalg.norm(comp1_center - comp2_center)

        # Components should be somewhat separated
        assert separation > 1e-2


class TestCrossBackendConsistency:
    """Test consistency across different backends (when available)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pytorch_vs_auto_selection(self):
        """Test that auto-selection produces reasonable results."""
        edges = generate_random_regular(n=50, d=4, seed=42)

        # Explicit PyTorch
        embedder_explicit = GraphEmbedderPyTorch(
            edges=edges,
            n_vertices=50,
            dimension=2,
            backend='pytorch',
            verbose=False
        )
        positions_explicit = embedder_explicit.run_layout(num_iterations=5)

        # Auto selection (should pick PyTorch for this size)
        embedder_auto = create_graphem(
            edges=edges,
            n_vertices=50,
            dimension=2,
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
        # Test with invalid edges
        invalid_edges = np.array([[0, 1], [1, 0], [2, 5]])  # 5 > n_vertices

        with pytest.raises((ValueError, IndexError)):
            embedder = create_graphem(
                edges=invalid_edges,
                n_vertices=3,
                dimension=2,
                backend='pytorch',
                verbose=False
            )
            embedder.run_layout(num_iterations=1)

    @pytest.mark.integration
    @pytest.mark.fast
    def test_empty_graph_handling(self):
        """Test handling of empty graphs."""
        empty_edges = np.array([]).reshape(0, 2)

        # Empty graphs should raise ValueError as invalid input
        with pytest.raises((ValueError, RuntimeError)):
            embedder = create_graphem(
                edges=empty_edges,
                n_vertices=5,  # 5 isolated vertices
                dimension=2,
                backend='pytorch',
                verbose=False
            )
            # If embedder creation succeeds, running layout should fail
            embedder.run_layout(num_iterations=1)

    @pytest.mark.integration
    @pytest.mark.fast
    def test_parameter_edge_cases(self):
        """Test parameter edge cases."""
        edges = generate_random_regular(n=20, d=3, seed=42)

        # Very small parameters
        embedder = create_graphem(
            edges=edges,
            n_vertices=20,
            dimension=2,
            backend='pytorch',
            L_min=0.1,
            k_attr=0.01,
            k_inter=0.01,
            verbose=False
        )

        positions = embedder.run_layout(num_iterations=2)
        assert positions.shape == (20, 2)
        assert np.all(np.isfinite(positions))