#!/usr/bin/env python
"""
Focused test script for random regular graphs in Graphem.

This script performs detailed testing and analysis of random regular graphs
with varying degrees and sizes.
"""

import time
import numpy as np
import networkx as nx
import pandas as pd
import plotly.express as px

from graphem_rapids.backends.embedder_pytorch import GraphEmbedderPyTorch
from graphem_rapids.benchmark import run_benchmark
from graphem_rapids.visualization import report_full_correlation_matrix
from graphem_rapids.generators import (
    erdos_renyi_graph,
    generate_random_regular,
    generate_ws,
    generate_ba
)


def test_random_regular_varying_degree(n=100, degrees=None, dim=3, num_iterations=30):
    """
    Test random regular graphs with varying degrees.
    
    Parameters:
        n: int
            Number of vertices
        degrees: list
            The list of degrees to test
        dim: int
            Dimension of the embedding
        num_iterations: int
            Number of layout iterations
    """
    print(f"\n{'='*75}")
    print(f"Testing Random Regular Graphs with Varying Degrees (n={n})")
    print(f"{'='*75}")
    
    results = []

    if degrees is None:
        degrees = [3, 4, 5, 6]
    
    for d in degrees:
        print(f"\n{'-'*25}")
        print(f"Random Regular Graph with degree d={d}")
        print(f"{'-'*25}")
        
        # Generate graph
        start_time = time.time()
        adjacency = generate_random_regular(n=n, d=d, seed=42)
        gen_time = time.time() - start_time
        
        print(f"Generated graph with {n} vertices, {adjacency.nnz//2} edges in {gen_time:.2f}s")
        
        # Create NetworkX graph for analysis
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        
        # Analyze graph properties
        density = 2 * adjacency.nnz//2 / (n * (n - 1))
        avg_degree = 2 * adjacency.nnz//2 / n
        
        print("Graph statistics:")
        print(f"- Density: {density:.4f}")
        print(f"- Average degree: {avg_degree:.2f}")
        
        try:
            avg_path_length = nx.average_shortest_path_length(G)
            print(f"- Average shortest path length: {avg_path_length:.2f}")
        except nx.NetworkXError as e:
            avg_path_length = float('nan')
            print("- Average shortest path length: N/A")
            print(e)
        
        try:
            avg_clustering = nx.average_clustering(G)
            print(f"- Average clustering coefficient: {avg_clustering:.4f}")
        except nx.NetworkXError as e:
            avg_clustering = float('nan')
            print("- Average clustering coefficient: N/A")
            print(e)
            
        # Compute graph diameter
        try:
            diameter = nx.diameter(G)
            print(f"- Diameter: {diameter}")
        except nx.NetworkXError as e:
            diameter = float('nan')
            print("- Diameter: N/A")
            print(e)
        
        # Create and run embedder
        embedder = GraphEmbedderPyTorch(
            adjacency=edges,
            n_vertices=n,
            n_components=dim,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=min(512, adjacency.nnz//2),
            verbose=True
        )
        
        print(f"Running layout for {num_iterations} iterations...")
        layout_start = time.time()
        embedder.run_layout(num_iterations=num_iterations)
        layout_time = time.time() - layout_start
        
        # Display the graph
        print("Displaying graph layout...")
        embedder.display_layout(edge_width=1, node_size=5)
        
        # Store results
        results.append({
            'degree': d,
            'n': n,
            'density': density,
            'avg_degree': avg_degree,
            'avg_path_length': avg_path_length,
            'avg_clustering': avg_clustering,
            'diameter': diameter,
            'layout_time': layout_time,
            'gen_time': gen_time
        })
        
        # Calculate centrality measures
        positions = np.array(embedder.positions)
        radii = np.linalg.norm(positions, axis=1)
        
        degree_centrality = np.array([d for _, d in G.degree()])
        betweenness_centrality = np.array(list(nx.betweenness_centrality(G).values()))
        eigenvector_centrality = np.array(list(nx.eigenvector_centrality_numpy(G).values()))
        pagerank = np.array(list(nx.pagerank(G).values()))
        closeness_centrality = np.array(list(nx.closeness_centrality(G).values()))
        
        # Display correlation with radial distances
        print("\nCorrelation between embedding radii and centrality measures:")
        _ = report_full_correlation_matrix(
            radii, 
            degree_centrality,
            betweenness_centrality,
            eigenvector_centrality,
            pagerank,
            closeness_centrality,
            np.zeros_like(radii)  # placeholder for edge betweenness
        )
        
    # Summarize results
    df = pd.DataFrame(results)
    print("\nSummary of results for all degrees:")
    print(df)
    
    # Plot layout time vs degree
    fig = px.line(
        df, x='degree', y='layout_time', 
        title='Layout Time vs. Degree',
        markers=True
    )
    fig.update_layout(
        xaxis_title='Degree (d)',
        yaxis_title='Layout Time (seconds)'
    )
    fig.show()
    
    # Plot average path length vs degree
    fig = px.line(
        df, x='degree', y='avg_path_length', 
        title='Average Path Length vs. Degree',
        markers=True
    )
    fig.update_layout(
        xaxis_title='Degree (d)',
        yaxis_title='Average Path Length'
    )
    fig.show()
    
    return df


def test_random_regular_varying_size(degree=3, sizes=None, dim=3, num_iterations=30):
    """
    Test random regular graphs with varying sizes.
    
    Parameters:
        degree: int
            Degree of each vertex
        sizes: list
            The list of graph sizes to test
        dim: int
            Dimension of the embedding
        num_iterations: int
            Number of layout iterations
    """
    print(f"\n{'='*75}")
    print(f"Testing Random Regular Graphs with Varying Sizes (d={degree})")
    print(f"{'='*75}")
    
    results = []

    if sizes is None:
        sizes = [50, 100, 200, 500]
    
    for n in sizes:
        print(f"\n{'-'*25}")
        print(f"Random Regular Graph with size n={n}")
        print(f"{'-'*25}")
        
        # Generate graph
        start_time = time.time()
        adjacency = generate_random_regular(n=n, d=degree, seed=42)
        gen_time = time.time() - start_time
        
        print(f"Generated graph with {n} vertices, {adjacency.nnz//2} edges in {gen_time:.2f}s")
        
        # Create NetworkX graph for analysis
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        
        # Analyze graph properties
        density = 2 * adjacency.nnz//2 / (n * (n - 1))
        avg_degree = 2 * adjacency.nnz//2 / n
        
        print("Graph statistics:")
        print(f"- Density: {density:.4f}")
        print(f"- Average degree: {avg_degree:.2f}")
        
        try:
            avg_path_length = nx.average_shortest_path_length(G)
            print(f"- Average shortest path length: {avg_path_length:.2f}")
        except nx.NetworkXError as e:
            avg_path_length = float('nan')
            print("- Average shortest path length: N/A")
            print(e)
        
        try:
            avg_clustering = nx.average_clustering(G)
            print(f"- Average clustering coefficient: {avg_clustering:.4f}")
        except nx.NetworkXError as e:
            avg_clustering = float('nan')
            print("- Average clustering coefficient: N/A")
            print(e)
            
        try:
            diameter = nx.diameter(G)
            print(f"- Diameter: {diameter}")
        except nx.NetworkXError as e:
            diameter = float('nan')
            print("- Diameter: N/A")
            print(e)
        
        # Create and run embedder
        embedder = GraphEmbedderPyTorch(
            adjacency=edges,
            n_vertices=n,
            n_components=dim,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=min(512, adjacency.nnz//2),
            verbose=True
        )
        
        print(f"Running layout for {num_iterations} iterations...")
        layout_start = time.time()
        embedder.run_layout(num_iterations=num_iterations)
        layout_time = time.time() - layout_start
        
        # Display the graph
        print("Displaying graph layout...")
        embedder.display_layout(edge_width=1, node_size=5)
        
        # Store results
        results.append({
            'degree': degree,
            'n': n,
            'density': density,
            'avg_degree': avg_degree,
            'avg_path_length': avg_path_length,
            'avg_clustering': avg_clustering,
            'diameter': diameter,
            'layout_time': layout_time,
            'gen_time': gen_time
        })
    
    # Summarize results
    df = pd.DataFrame(results)
    print("\nSummary of results for all sizes:")
    print(df)
    
    # Plot layout time vs size
    fig = px.line(
        df, x='n', y='layout_time', 
        title='Layout Time vs. Graph Size',
        markers=True
    )
    fig.update_layout(
        xaxis_title='Number of Vertices (n)',
        yaxis_title='Layout Time (seconds)'
    )
    fig.show()
    
    # Plot average path length vs size
    fig = px.line(
        df, x='n', y='avg_path_length', 
        title='Average Path Length vs. Graph Size',
        markers=True,
        log_x=True  # Use log scale for x-axis
    )
    fig.update_layout(
        xaxis_title='Number of Vertices (n)',
        yaxis_title='Average Path Length'
    )
    fig.show()
    
    return df


def compare_with_benchmark():
    """
    Compare random regular graphs with other graph types using the benchmark module.
    """
    print(f"\n{'='*75}")
    print("Comparing Random Regular Graphs with Other Graph Types")
    print(f"{'='*75}")

    # Define parameters for each graph type to test
    graph_configs = [
        (generate_random_regular, {'n': 100, 'd': 3, 'seed': 42}, 'Random Regular (d=3)'),
        (generate_random_regular, {'n': 100, 'd': 5, 'seed': 42}, 'Random Regular (d=5)'),
        (erdos_renyi_graph, {'n': 100, 'p': 0.03, 'seed': 42}, 'Erdős–Rényi'),
        (generate_ws, {'n': 100, 'k': 4, 'p': 0.1, 'seed': 42}, 'Watts-Strogatz'),
        (generate_ba, {'n': 100, 'm': 2, 'seed': 42}, 'Barabási-Albert')
    ]

    benchmark_results = []

    for generator, params, name in graph_configs:
        print(f"\n{'-'*25}")
        print(f"Benchmarking {name} graph")
        print(f"{'-'*25}")

        # Run benchmark
        result = run_benchmark(
            generator,
            params,
            dim=3,
            L_min=10.0,
            k_attr=0.5,
            k_inter=0.1,
            n_neighbors=15,
            sample_size=512,
            num_iterations=30
        )

        # Add graph type name
        result['display_name'] = name
        benchmark_results.append(result)

    # Compare results
    for result in benchmark_results:
        print(f"\nResults for {result['display_name']}:")
        print(f"- Number of vertices: {result['n']}")
        print(f"- Number of edges: {result['m']}")
        print(f"- Average degree: {result['avg_degree']:.2f}")
        print(f"- Layout time: {result['layout_time']:.2f}s")

    return benchmark_results


def main():
    """
    Main function to run all tests.
    """
    # Test varying degrees
    test_random_regular_varying_degree(n=100, degrees=[3, 4, 6, 8])
    
    # Test varying sizes
    test_random_regular_varying_size(degree=3, sizes=[50, 100, 200, 300])
    
    # Compare with other graph types
    compare_with_benchmark()


if __name__ == "__main__":
    main()
