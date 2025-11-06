#!/usr/bin/env python
"""
Test script for various graph generators in Graphem.

This script demonstrates how to generate and visualize different types of graphs,
with a focus on the random regular graphs and other recently added generators.
"""

import networkx as nx

from graphem_rapids.backends.embedder_pytorch import GraphEmbedderPyTorch
from graphem_rapids.generators import (
    generate_er,
    generate_random_regular,
    generate_scale_free,
    generate_geometric,
    generate_caveman,
    generate_relaxed_caveman,
    generate_ws,
    generate_ba,
    generate_sbm
)


def test_graph_generator(generator, params, name, dim=3, num_iterations=30):
    """
    Test a graph generator function and visualize the resulting embedding.
    
    Parameters:
        generator: function
            Graph generator function
        params: dict
            Parameters for the graph generator
        name: str
            Name of the graph type for display
        dim: int
            Dimension of the embedding
        num_iterations: int
            Number of layout iterations
    """
    print(f"\n{'='*50}")
    print(f"Testing {name} graph")
    print(f"{'='*50}")
    
    # Generate graph
    adjacency = generator(**params)

    # Extract edges from adjacency matrix
    edges = list(zip(*adjacency.nonzero()))
    edges = [(int(i), int(j)) for i, j in edges if i < j]

    # Determine number of vertices
    if len(edges) > 0:
        n = int(max(max(i for i, j in edges), max(j for i, j in edges)) + 1)
    else:
        n = params.get('n', 0)

    print(f"Generated graph with {n} vertices and {adjacency.nnz//2} edges")

    # Create NetworkX graph for visualization
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    
    print("Graph statistics:")
    print(f"- Density: {2 * adjacency.nnz//2 / (n * (n - 1)):.4f}")
    print(f"- Average degree: {2 * adjacency.nnz//2 / n:.2f}")
    
    try:
        print(f"- Average shortest path length: {nx.average_shortest_path_length(G):.2f}")
    except nx.NetworkXError as e:
        print("- Average shortest path length: N/A")
        print(e)
    
    try:
        print(f"- Average clustering coefficient: {nx.average_clustering(G):.4f}")
    except nx.NetworkXError as e:
        print("- Average clustering coefficient: N/A")
        print(e)
    
    # Create and run embedder
    embedder = GraphEmbedderPyTorch(
        adjacency=adjacency,
        n_components=dim,
        L_min=10.0,
        k_attr=0.5,
        k_inter=0.1,
        n_neighbors=15,
        sample_size=min(512, adjacency.nnz//2),
        verbose=True
    )
    
    print(f"Running layout for {num_iterations} iterations...")
    embedder.run_layout(num_iterations=num_iterations)
    
    # Display the graph
    print("Displaying graph layout...")
    embedder.display_layout(edge_width=1, node_size=5)
    
    return embedder


def main():
    """
    Test various graph generators.
    """
    # Test Random Regular Graph
    test_graph_generator(
        generate_random_regular,
        {'n': 100, 'd': 3, 'seed': 42},
        'Random Regular'
    )
    
    # Test Scale-Free Graph
    test_graph_generator(
        generate_scale_free,
        {'n': 100, 'seed': 42},
        'Scale-Free'
    )
    
    # Test Random Geometric Graph
    test_graph_generator(
        generate_geometric,
        {'n': 100, 'radius': 0.15, 'seed': 42},
        'Random Geometric'
    )
    
    # Test Caveman Graph
    test_graph_generator(
        generate_caveman,
        {'l': 5, 'k': 20},
        'Caveman'
    )
    
    # Test Relaxed Caveman Graph
    test_graph_generator(
        generate_relaxed_caveman,
        {'l': 5, 'k': 20, 'p': 0.1, 'seed': 42},
        'Relaxed Caveman'
    )
    
    # Test Erdős–Rényi Graph
    test_graph_generator(
        generate_er,
        {'n': 100, 'p': 0.05, 'seed': 42},
        'Erdős–Rényi'
    )
    
    # Test Watts-Strogatz Small-World Graph
    test_graph_generator(
        generate_ws,
        {'n': 100, 'k': 4, 'p': 0.1, 'seed': 42},
        'Watts-Strogatz Small-World'
    )
    
    # Test Barabási-Albert Graph
    test_graph_generator(
        generate_ba,
        {'n': 100, 'm': 2, 'seed': 42},
        'Barabási-Albert'
    )
    
    # Test Stochastic Block Model
    test_graph_generator(
        generate_sbm,
        {'n_per_block': 25, 'num_blocks': 4, 'p_in': 0.3, 'p_out': 0.01, 'seed': 42},
        'Stochastic Block Model'
    )


if __name__ == "__main__":
    main()
